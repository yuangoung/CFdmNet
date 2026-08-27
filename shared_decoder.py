# ==============================================================
# shared_decoder.py
# Description:
#   -Feature fusion of MSI and DEM
# ==============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ConvBNAct, DepthwiseSeparableConv, UpSample2x, GatedFusion

# ==============================================================

class SharedDecoder(nn.Module):
    """
    Expected inputs (from FeatureExtractor):
      (msi_low, msi_high), (dem_low, dem_high)
    where:
      msi_low/dem_low   : [B, 2*base_ch, H/2, W/2]
      msi_high/dem_high : [B, 8*base_ch, H/4, W/4]
    """

    def __init__(
        self,
        base_ch: int = 64,
        out_classes: int = 1,
        gate_type: str = "channel",   # "channel" or "spatial"
        decoder_ch: int = 256,        # internal decoder width
        use_dwconv: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        base_ch = int(base_ch)
        self.low_ch = 2 * base_ch
        self.high_ch = 8 * base_ch
        self.decoder_ch = int(decoder_ch)
        self.out_classes = int(out_classes)

        # 1) modality fusion per scale
        self.fuse_low = GatedFusion(self.low_ch, gate_type=gate_type, hidden_ratio=0.5, dropout=0.0)
        self.fuse_high = GatedFusion(self.high_ch, gate_type=gate_type, hidden_ratio=0.5, dropout=0.0)

        # 2) project to unified decoder channels
        self.proj_high = ConvBNAct(self.high_ch, self.decoder_ch, k=1, s=1, p=0, act="gelu")
        self.proj_low  = ConvBNAct(self.low_ch,  self.decoder_ch, k=1, s=1, p=0, act="gelu")

        # 3) decode blocks
        block = DepthwiseSeparableConv if use_dwconv else ConvBNAct

        self.refine_high = block(self.decoder_ch, self.decoder_ch, act="gelu") if use_dwconv else ConvBNAct(self.decoder_ch, self.decoder_ch)
        self.up2 = UpSample2x()

        # skip fusion between (upsampled high) and (low)
        self.fuse_skip = GatedFusion(self.decoder_ch, gate_type=gate_type, hidden_ratio=0.5, dropout=0.0)

        self.refine_low = block(self.decoder_ch, self.decoder_ch, act="gelu") if use_dwconv else ConvBNAct(self.decoder_ch, self.decoder_ch)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # 4) final head at full resolution
        self.proj_full = ConvBNAct(self.decoder_ch, self.decoder_ch // 2, k=3, s=1, p=1, act="gelu")
        self.head = nn.Conv2d(self.decoder_ch // 2, self.out_classes, kernel_size=1, bias=True)

    def forward(self, msi_feats, dem_feats):
        """
        msi_feats: (msi_low, msi_high)
        dem_feats: (dem_low, dem_high)
        """
        msi_low, msi_high = msi_feats
        dem_low, dem_high = dem_feats

        # --- modality fusion (same scale) ---
        low_f  = self.fuse_low(msi_low, dem_low)         # [B, 128,  H/2, W/2]
        high_f = self.fuse_high(msi_high, dem_high)      # [B, 512, H/4, W/4]

        # --- decode from high ---
        x = self.proj_high(high_f)                       # [B, dec, H/4, W/4]
        x = self.refine_high(x)

        # --- up to H/2 and gated-sum with low skip ---
        x = self.up2(x)                                  # [B, dec, H/2, W/2]
        low_p = self.proj_low(low_f)                     # [B, dec, H/2, W/2]
        x = self.fuse_skip(x, low_p)                     # [B, dec, H/2, W/2]
        x = self.refine_low(x)
        x = self.drop(x)

        # --- up to full resolution H/W ---
        x = self.up2(x)                                  # [B, dec, H, W]
        x = self.proj_full(x)                            # [B, dec/2, H, W]
        logits = self.head(x)                            # [B, out_classes, H, W]
        return logits


# ------------------------------
# Test
# ------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_ch = 64
    dec = SharedDecoder(base_ch=base_ch, out_classes=1, gate_type="channel", decoder_ch=256, use_dwconv=True).to(device)

    B, H, W = 2, 256, 256
    msi_low  = torch.randn(B, 2*base_ch, H//2, W//2, device=device)
    msi_high = torch.randn(B, 8*base_ch, H//4, W//4, device=device)
    dem_low  = torch.randn(B, 2*base_ch, H//2, W//2, device=device)
    dem_high = torch.randn(B, 8*base_ch, H//4, W//4, device=device)

    logits = dec((msi_low, msi_high), (dem_low, dem_high))
    print("logits:", logits.shape)  # [B, 1, H, W]
