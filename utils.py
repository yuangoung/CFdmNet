import os
import numpy as np
from osgeo import gdal, osr
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, act="gelu"):
        super().__init__()
        out_ch = int(out_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Lightweight conv: DW 3x3 + PW 1x1"""
    def __init__(self, in_ch, out_ch, act="gelu"):
        super().__init__()
        out_ch = int(out_ch)
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

class UpSample2x(nn.Module):
    """Bilinear upsample x2"""
    def __init__(self, mode="bilinear", align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=self.align_corners)


class MCSF(nn.Module):
    """
    Multi-Scale Channel-Spatial Fusion
    CMX (CVPR 2022), MMFNet (TGRS 2023)
    Input:
        msi_feat: [B, C1, H, W]
        dem_feat: [B, C2, H, W]
    Output:
        fused: [B, out_ch, H, W]
    """
    def __init__(self, msi_ch=256, dem_ch=128, out_ch=256, reduction=16):
        super().__init__()
        self.msi_proj = nn.Conv2d(msi_ch, out_ch, 1)
        self.dem_proj = nn.Conv2d(dem_ch, out_ch, 1)

        # 通道注意力
        self.channel_msi = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // reduction, out_ch, 1),
            nn.Sigmoid()
        )

        self.channel_dem = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // reduction, out_ch, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 融合卷积
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, msi_feat, dem_feat):
        if dem_feat.shape[2:] != msi_feat.shape[2:]:
            dem_feat = F.interpolate(dem_feat, size=msi_feat.shape[2:], mode='bilinear', align_corners=False)

        msi = self.msi_proj(msi_feat)
        dem = self.dem_proj(dem_feat)

        # 通道注意力
        msi = msi * self.channel_msi(msi)
        dem = dem * self.channel_dem(dem)

        # 融合
        fused = (msi + dem) / 2

        # 空间注意力
        avg_out = torch.mean(fused, dim=1, keepdim=True)
        max_out, _ = torch.max(fused, dim=1, keepdim=True)
        spatial = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        fused = fused * spatial + fused  # 残差增强

        return self.out_conv(fused)

# ==============================================================
# Gated Fusion Module
# ==============================================================
class GatedFusion(nn.Module):
    """
    Gated Sum Fusion (weighted sum):
      y = g * x1 + (1 - g) * x2
    where g = sigmoid(Conv([x1, x2])).

    Supports:
      - channel-wise gate: g shape [B, C, H, W] (default)
      - spatial gate:      g shape [B, 1, H, W]
    """
    def __init__(
        self,
        channels: int,
        gate_type: str = "channel",   # "channel" or "spatial"
        hidden_ratio: float = 0.5,
        dropout: float = 0.0
    ):
        super().__init__()
        channels = int(channels)
        assert gate_type in ["channel", "spatial"]
        self.channels = channels
        self.gate_type = gate_type

        hidden = max(8, int(channels * hidden_ratio))
        out_gate_ch = channels if gate_type == "channel" else 1

        # gate net uses concatenated features -> gate map
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_gate_ch, 1, bias=True)
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # optional post-proj for stability (keeps channels unchanged)
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x1, x2):
        """
        x1, x2: [B, C, H, W] and same shape
        """
        if x1.shape != x2.shape:
            raise ValueError(f"GatedFusion requires same shape, got {x1.shape} vs {x2.shape}")
        if x1.shape[1] != self.channels:
            raise ValueError(f"GatedFusion channels mismatch: expected {self.channels}, got {x1.shape[1]}")

        gate_in = torch.cat([x1, x2], dim=1)
        g = torch.sigmoid(self.gate_net(gate_in))  # [B,C,H,W] or [B,1,H,W]
        y = g * x1 + (1.0 - g) * x2
        y = self.post(self.drop(y))
        return y

# ==============================================================
# Edge Preservation Loss
# ==============================================================
class EdgePreservationLoss(nn.Module):
    """
    Combines BCE with gradient difference (Sobel) for edge preservation.
    """
    def __init__(self, alpha=1.0, beta=0.5):
        super(EdgePreservationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, gt):

        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        loss_bce = self.bce(pred, gt)

        # Gradient (edge) difference
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        gt_grad_x = F.conv2d(gt, self.sobel_x, padding=1)
        gt_grad_y = F.conv2d(gt, self.sobel_y, padding=1)

        pred_edge = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)
        gt_edge = torch.sqrt(gt_grad_x ** 2 + gt_grad_y ** 2 + 1e-6)
        loss_edge = torch.mean(torch.abs(pred_edge - gt_edge))

        return loss_bce, loss_edge

# ==============================================================
def reconstruct_probability_map(patches, coords, img_shape, tile_size, stride):
    """
    Reconstruct continuous probability map from patch predictions.

    Args:
        patches (list or np.ndarray): list of patch probability maps, shape [N, h, w]
        coords (list): top-left coordinates [(x,y), ...] of patches
        img_shape (tuple): (H, W)
        tile_size (int): patch size
        stride (int): stride used during sliding window
    Returns:
        prob_map (np.ndarray): reconstructed probability map [H, W]
    """
    H, W = img_shape
    prob_sum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        patch = patches[i]
        h, w = patch.shape
        prob_sum[y:y + h, x:x + w] += patch
        weight[y:y + h, x:x + w] += 1.0

    # Avoid division by zero
    weight[weight == 0] = 1
    prob_map = prob_sum / weight

    # Stretch to 0-1 range (2–98 percentile)
    p2, p98 = np.percentile(prob_map, [2, 98])
    prob_map = np.clip((prob_map - p2) / (p98 - p2 + 1e-6), 0, 1)

    return prob_map


# ==============================================================
# GeoTIFF
# ==============================================================
def save_geotiff(array, meta, output_path, nodata=0):
    """
    Save 2D array as GeoTIFF file with spatial metadata.

    Args:
        array (np.ndarray): 2D data array [H, W]
        meta (dict): contains geotransform and projection from dataset.get_meta()
        output_path (str): save path for GeoTIFF
        nodata (float): no data value (default 0)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    H, W = array.shape
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, W, H, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(meta["geotransform"])
    out_ds.SetProjection(meta["projection"])

    band = out_ds.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    out_ds.FlushCache()
    out_ds = None

    print(f" Saved GeoTIFF: {output_path}")

# ==============================================================
def visualize_probability_map(prob_map, output_png="prob_map.png"):
    """
    Save visualization of probability map (for inspection).
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(prob_map, cmap="bwr", vmin=0, vmax=1)
    plt.colorbar(label="Landslide Probability")
    plt.title("Predicted Landslide Probability Map")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f" Saved visualization: {output_png}")


# ==============================================================
# Test
# ==============================================================
if __name__ == "__main__":
    # Example synthetic test
    patches = [np.random.rand(256, 256) for _ in range(4)]
    coords = [(0, 0), (256, 0), (0, 256), (256, 256)]
    prob_map = reconstruct_probability_map(patches, coords, (512, 512), 256, 128)

    meta = {
        "geotransform": (0, 1, 0, 0, 0, -1),
        "projection": osr.SRS_WKT_WGS84,
        "width": 512,
        "height": 512
    }

    save_geotiff(prob_map, meta, "example_prob_map.tif")
    visualize_probability_map(prob_map, "example_prob_map.png")
