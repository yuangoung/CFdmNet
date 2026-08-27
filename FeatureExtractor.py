# ==============================================================
# FeatureExtractor.py
# Description:
#   Dual-Branch Feature Extractor for MSI and DEM
#   MSI: Mamba-enhanced Spectral-Spatial Encoder
#   DEM: Wavelet-Convolutional Morphology Encoder (DWT)
#   Outputs spatially aligned features (same H×W per level)
# ==============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
# --------------------------------------------------------------
# Basic Conv Block
# --------------------------------------------------------------
class BasicConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1):
        super().__init__()
        out_ch = int(out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)

# --------------------------------------------------------------
# Gated Linear
# --------------------------------------------------------------
class GLUBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return self.fc1(x) * torch.sigmoid(self.fc2(x))
# --------------------------------------------------------------
# Mamba SSM Block: SS2D 4-direction scan
# --------------------------------------------------------------
class SS2DMambaBlock(nn.Module):
    """
    Vision SS2D (4 directions), but slower than fused CUDA kernels.
    """
    def __init__(self, dim, expansion=1, dropout=0.1, d_state=16):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)

        self.dim = int(dim)
        self.inner_dim = int(self.dim * expansion)
        self.d_state = int(d_state)

        self.in_proj = nn.Conv2d(self.dim, 2 * self.inner_dim, 1, bias=False)
        self.dwconv  = nn.Conv2d(self.inner_dim, self.inner_dim, 3, padding=1, groups=self.inner_dim, bias=False)

        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.B_proj  = nn.Linear(self.inner_dim, self.d_state, bias=True)
        self.C_proj  = nn.Linear(self.inner_dim, self.d_state, bias=True)

        A_init = torch.arange(1, self.d_state + 1, dtype=torch.float32).view(1, -1)
        A_init = A_init.repeat(self.inner_dim, 1)  # [D, N]
        self.A_log = nn.Parameter(torch.log(A_init))  # A = -exp(A_log)

        self.D = nn.Parameter(torch.ones(self.inner_dim, dtype=torch.float32))

        self.out_proj = nn.Conv2d(self.inner_dim, self.dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)

        nn.init.constant_(self.dt_proj.bias, -2.0)

    def _ssm_scan_1d(self, u_seq: torch.Tensor) -> torch.Tensor:
        """
        u_seq: [B2, L, D] -> y_seq: [B2, L, D]
        """
        B2, L, D = u_seq.shape
        N = self.d_state

        dt = F.softplus(self.dt_proj(u_seq))       # [B2, L, D]
        Bp = self.B_proj(u_seq)                    # [B2, L, N]
        Cp = self.C_proj(u_seq)                    # [B2, L, N]

        A = -torch.exp(self.A_log).to(u_seq.dtype) # [D, N]
        D_skip = self.D.to(u_seq.dtype)            # [D]

        state = torch.zeros((B2, D, N), device=u_seq.device, dtype=u_seq.dtype)
        ys = []

        for t in range(L):
            dt_t = dt[:, t, :]                     # [B2, D]
            u_t  = u_seq[:, t, :]                  # [B2, D]
            B_t  = Bp[:, t, :]                     # [B2, N]
            C_t  = Cp[:, t, :]                     # [B2, N]

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))                 # [B2, D, N]
            dB_u = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) * u_t.unsqueeze(-1)    # [B2, D, N]

            state = dA * state + dB_u
            y_t = (state * C_t.unsqueeze(1)).sum(-1) + D_skip.unsqueeze(0) * u_t
            ys.append(y_t)

        return torch.stack(ys, dim=1)

    def _ss2d(self, u_2d: torch.Tensor) -> torch.Tensor:
        """
        u_2d: [B, D, H, W] -> [B, D, H, W]
        """
        B, D, H, W = u_2d.shape

        # L->R
        u_lr = u_2d.permute(0, 2, 3, 1).contiguous().view(B * H, W, D)
        y_lr = self._ssm_scan_1d(u_lr).view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # R->L
        u_rl = torch.flip(u_lr, dims=[1])
        y_rl = self._ssm_scan_1d(u_rl)
        y_rl = torch.flip(y_rl, dims=[1]).view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # T->B
        u_tb = u_2d.permute(0, 3, 2, 1).contiguous().view(B * W, H, D)
        y_tb = self._ssm_scan_1d(u_tb).view(B, W, H, D).permute(0, 3, 2, 1).contiguous()

        # B->T
        u_bt = torch.flip(u_tb, dims=[1])
        y_bt = self._ssm_scan_1d(u_bt)
        y_bt = torch.flip(y_bt, dims=[1]).view(B, W, H, D).permute(0, 3, 2, 1).contiguous()

        return (y_lr + y_rl + y_tb + y_bt) * 0.25

    def forward(self, x):
        residual = x
        x = self.norm(x)

        u, z = self.in_proj(x).chunk(2, dim=1)   # [B, inner, H, W]
        u = self.dwconv(u)

        y = self._ss2d(u)
        y = y * F.silu(z)

        y = self.out_proj(y)
        return residual + self.drop(y)

# --------------------------------------------------------------
# Mamba Lightweight SSM Block (Selective lightweight 2-direction scan)
# --------------------------------------------------------------

class MambaSSMBlock(nn.Module):
    """
    Mamba Lightweight Block (conv-gated)
    """
    def __init__(self, dim, expansion=2, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        hidden_dim = int(dim * expansion)

        self.in_proj = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        self.gate = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.out_proj = nn.Conv2d(hidden_dim, dim, 1, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        u = self.in_proj(x)
        v = self.dwconv(u)
        g = torch.sigmoid(self.gate(u))
        y = v * g

        y = self.out_proj(y)
        return residual + self.drop(y)

class GatedConvBlock(nn.Module):
    """
    local modeling:
    """
    def __init__(self, dim, expansion=2, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        hidden_dim = int(dim * expansion)

        self.in_proj = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.dwconv  = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        self.gate    = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.out_proj= nn.Conv2d(hidden_dim, dim, 1, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        u = self.in_proj(x)
        v = self.dwconv(u)
        g = torch.sigmoid(self.gate(u))
        y = v * g

        y = self.out_proj(y)
        return residual + self.drop(y)

# --------------------------------------------------------------
# MSI Encoder (two-level: H/2, H/4)
# --------------------------------------------------------------
class MSIEncoder(nn.Module):
    """
    MSI Encoder with Bi-level output:
      - low:  [B, 2*base_ch, H/2, W/2]
      - high: [B, 8*base_ch, H/4, W/4]
    """
    def __init__(self, in_ch=4, base_ch=64):
        super().__init__()
        self.low = nn.Sequential(
            BasicConvBlock(in_ch, base_ch * 2, k=3, s=2, p=1),  # H/2
            SS2DMambaBlock(base_ch * 2, expansion=1, dropout=0.1, d_state=16),
        )

        self.high = nn.Sequential(
            BasicConvBlock(base_ch * 2, base_ch * 8, k=3, s=2, p=1),  # H/4
            SS2DMambaBlock(base_ch * 8, expansion=1, dropout=0.1, d_state=16),
        )

        # --------------------------------------------------------------
        # Selective Lightweight Mamba SSM Block ( lightweight 2-direction scan)
        # --------------------------------------------------------------

        # self.low = nn.Sequential(
        #     BasicConvBlock(in_ch, base_ch * 2, k=3, s=2, p=1),  # H/2
        #     GatedConvBlock(base_ch * 2),
        # )
        #
        # self.high = nn.Sequential(
        #     BasicConvBlock(base_ch * 2, base_ch * 8, k=3, s=2, p=1),  # H/4
        #     MambaSSMBlock(base_ch * 8),
        # )

    def forward(self, msi):
        low = self.low(msi)
        high = self.high(low)
        return low, high

# --------------------------------------------------------------
# DEM Encoder (DWT downsample + conv)
# --------------------------------------------------------------

class DWTFeatureExtractor(nn.Module):
    """
    Haar DWT for arbitrary channel tensors via grouped conv.
    Input:  [B, C, H, W]
    Output: [B, out_ch, H/2, W/2]
    """
    def __init__(self, in_ch, out_ch, wavelet="haar"):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)

        w = pywt.Wavelet(wavelet)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)

        self.register_buffer("dec_lo", dec_lo)
        self.register_buffer("dec_hi", dec_hi)

        self.fuse = BasicConvBlock(4 * self.in_ch, self.out_ch, k=3, s=1, p=1)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert C == self.in_ch, f"DWTFeatureExtractor expected C={self.in_ch}, got C={C}"

        # 1D filters
        lo = self.dec_lo.to(dtype=x.dtype, device=x.device)
        hi = self.dec_hi.to(dtype=x.dtype, device=x.device)

        # weight: [C, 1, k, 1] for vertical; groups=C
        k = lo.numel()
        lo_v = lo.view(1, 1, k, 1).repeat(C, 1, 1, 1)
        hi_v = hi.view(1, 1, k, 1).repeat(C, 1, 1, 1)

        # weight: [C, 1, 1, k] for horizontal; groups=C
        lo_h = lo.view(1, 1, 1, k).repeat(C, 1, 1, 1)
        hi_h = hi.view(1, 1, 1, k).repeat(C, 1, 1, 1)


        x_lo = F.conv2d(x, lo_v, stride=(2, 1), padding=(0, 0), groups=C)
        x_hi = F.conv2d(x, hi_v, stride=(2, 1), padding=(0, 0), groups=C)

        ll = F.conv2d(x_lo, lo_h, stride=(1, 2), padding=(0, 0), groups=C)
        lh = F.conv2d(x_lo, hi_h, stride=(1, 2), padding=(0, 0), groups=C)
        hl = F.conv2d(x_hi, lo_h, stride=(1, 2), padding=(0, 0), groups=C)
        hh = F.conv2d(x_hi, hi_h, stride=(1, 2), padding=(0, 0), groups=C)

        out = torch.cat([ll, lh, hl, hh], dim=1)  # [B, 4C, H/2, W/2]
        return self.fuse(out)

class DEMEncoder(nn.Module):
    """
    DEM Encoder with Bi-level output aligned to MSI:
      - low:  [B, 2*base_ch, H/2, W/2]
      - high: [B, 8*base_ch, H/4, W/4]
    """
    def __init__(self, in_ch=1, base_ch=64, wavelet="haar"):
        super().__init__()
        base_half = base_ch // 2

        self.low_pre = BasicConvBlock(in_ch, base_half, k=3, s=1, p=1)
        self.low_dwt = DWTFeatureExtractor(in_ch=base_half, out_ch=base_ch * 2, wavelet=wavelet)

        self.high_pre = BasicConvBlock(base_ch * 2, base_ch * 2, k=3, s=1, p=1)
        self.high_dwt = DWTFeatureExtractor(in_ch=base_ch * 2, out_ch=base_ch * 8, wavelet=wavelet)

    def forward(self, dem):
        x = self.low_pre(dem)        # [B, base/2, H, W]
        low = self.low_dwt(x)        # [B, 2*base, H/2, W/2]

        y = self.high_pre(low)       # [B, 2*base, H/2, W/2]
        high = self.high_dwt(y)      # [B, 8*base, H/4, W/4]
        return low, high


# --------------------------------------------------------------
# Combined Feature Extractor
# --------------------------------------------------------------

class MDFeatureExtractor(nn.Module):
    def __init__(self, msi_channels=4, dem_channels=1, base_ch=64, wavelet="haar"):
        super().__init__()
        self.msi_encoder = MSIEncoder(in_ch=msi_channels, base_ch=base_ch)
        self.dem_encoder = DEMEncoder(in_ch=dem_channels, base_ch=base_ch, wavelet=wavelet)

    def forward(self, msi, dem):
        msi_low, msi_high = self.msi_encoder(msi)
        dem_low, dem_high = self.dem_encoder(dem)
        return (msi_low, msi_high), (dem_low, dem_high)


# --------------------------------------------------------------
# Test
# --------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MDFeatureExtractor(msi_channels=4, dem_channels=1, base_ch=64, wavelet="haar").to(device)

    msi = torch.randn(1, 4, 256, 256, device=device)
    dem = torch.randn(1, 1, 256, 256, device=device)

    (msi_low, msi_high), (dem_low, dem_high) = model(msi, dem)

    print(f"MSI low: {msi_low.shape}, high: {msi_high.shape}")
    print(f"DEM low: {dem_low.shape}, high: {dem_high.shape}")
