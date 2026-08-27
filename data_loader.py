# ==============================================================
# data_loader.py
# Description:
#   Data loading for landslide susceptibility analysis
#   with per-epoch random sampling.
#   Band-wise robust normalization to [0, 1] (percentile based)
#   Without NoData/NaN/Inf
# ==============================================================
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from osgeo import gdal
# ==============================================================
#  GeoTIFF
# ==============================================================

def load_tif(path):
    dataset = gdal.Open(path)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open file: {path}")
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount

    data = []
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        arr = band.ReadAsArray()
        data.append(arr)

    arr = np.stack(data, axis=-1) if bands > 1 else data[0]  # HWC if multi-band
    geo = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    dataset = None
    return arr, geo, proj


# ==============================================================
# normalization
# ==============================================================
def _to_float_and_clean(arr, fill_value=0.0):
    """Convert to float32 and replace NaN/Inf with fill_value."""
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return arr


def bandwise_robust_norm_01(tile_chw, clip_percent=(2.0, 98.0), fill_value=0.0, eps=1e-6):
    """
    Band-wise robust normalization to [0, 1] using percentiles.

    Args:
        tile_chw: np.ndarray, shape [C, H, W]
        clip_percent: (low, high) percentiles per band (robust against outliers)
        fill_value: value used to replace NaN/Inf (and potentially NoData if pre-masked)
        eps: numeric stability epsilon

    Returns:
        np.ndarray float32, shape [C, H, W], values in [0, 1]
    """
    tile_chw = _to_float_and_clean(tile_chw, fill_value=fill_value)

    C, H, W = tile_chw.shape
    out = np.empty_like(tile_chw, dtype=np.float32)
    low_p, high_p = clip_percent

    for c in range(C):
        band = tile_chw[c]

        vmin = np.percentile(band, low_p)
        vmax = np.percentile(band, high_p)

        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin) < eps:
            out[c] = 0.0
            continue

        norm = (band - vmin) / (vmax - vmin + eps)
        out[c] = np.clip(norm, 0.0, 1.0).astype(np.float32)

    return out

# ==============================================================
# Dataset
# ==============================================================

class LandslideDataset(Dataset):
    def __init__(
        self,
        msi_path,
        dem_path,
        gt_path=None,
        tile_size=128,
        stride=64,
        clip_percent=(2.0, 98.0),
        fill_value=0.0,
    ):
        self.msi, self.geo, self.proj = load_tif(msi_path)   # HWC (multi-band)
        self.dem, _, _ = load_tif(dem_path)                  # HW or HWC(1)
        self.gt = None

        if gt_path is not None:
            self.gt, _, _ = load_tif(gt_path)
            if self.gt.ndim == 3:
                self.gt = self.gt[:, :, 0]

        self.tile_size = tile_size
        self.stride = stride
        self.clip_percent = clip_percent
        self.fill_value = fill_value

        H, W = self.msi.shape[:2]
        self.coords = []
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                self.coords.append((x, y))

        print(f"MSI dimensions: {H}×{W}×{self.msi.shape[-1]}")
        print(f"DEM dimensions: {self.dem.shape}")
        print(f"Total patches: {len(self.coords)}")
        print(f"Robust norm percentiles: {self.clip_percent} -> [0,1] per band")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]

        # ---- MSI: HWC -> CHW, band-wise robust norm to [0,1] ----
        tile_msi = self.msi[y:y + self.tile_size, x:x + self.tile_size, :]  # HWC
        tile_msi = tile_msi.astype(np.float32)
        tile_msi = np.transpose(tile_msi, (2, 0, 1))  # CHW
        tile_msi = bandwise_robust_norm_01(
            tile_msi,
            clip_percent=self.clip_percent,
            fill_value=self.fill_value
        )

        # ---- DEM: ensure [1,H,W], band-wise robust norm to [0,1] ----
        tile_dem = self.dem[y:y + self.tile_size, x:x + self.tile_size]
        if tile_dem.ndim == 2:
            tile_dem = np.expand_dims(tile_dem, axis=0)  # [1, H, W]
        elif tile_dem.ndim == 3 and tile_dem.shape[2] == 1:
            tile_dem = np.transpose(tile_dem, (2, 0, 1))  # [1, H, W]
        else:
            raise ValueError(f"Unexpected DEM shape: {tile_dem.shape}")

        tile_dem = tile_dem.astype(np.float32)
        tile_dem = bandwise_robust_norm_01(
            tile_dem,
            clip_percent=self.clip_percent,
            fill_value=self.fill_value
        )

        # ---- GT ----
        if self.gt is not None:
            tile_gt = self.gt[y:y + self.tile_size, x:x + self.tile_size].astype(np.float32)
        else:
            tile_gt = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)

        return (
            torch.from_numpy(tile_msi),
            torch.from_numpy(tile_dem),
            torch.from_numpy(tile_gt),
            (x, y),
        )

    def get_meta(self):
        return {"geo": self.geo, "proj": self.proj, "height": self.msi.shape[0], "width": self.msi.shape[1]}


# ==============================================================
# Dataloader
# ==============================================================

def get_dataloader(
    msi_path,
    dem_path,
    gt_path=None,
    tile_size=256,
    stride=128,
    batch_size=8,
    num_workers=0,
    clip_percent=(2.0, 98.0),
    fill_value=0.0,
):
    dataset = LandslideDataset(
        msi_path=msi_path,
        dem_path=dem_path,
        gt_path=gt_path,
        tile_size=tile_size,
        stride=stride,
        clip_percent=clip_percent,
        fill_value=fill_value,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader, dataset


# ==============================================================
# Label Statistics
# ==============================================================

def get_label_balance_info(dataset):
    """Compute label balance from GT."""
    if dataset.gt is None:
        print("No GT available, skipping label balance computation.")
        return None

    gt = dataset.gt.flatten()
    pos = np.sum(gt > 0)
    neg = np.sum(gt == 0)
    total = len(gt)

    print("------------------------------------------------------------")
    print(" GT Label Balance Information")
    print("------------------------------------------------------------")
    print(f" Total samples       : {total}")
    print(f" Positive (landslide): {pos} ({pos / total * 100:.2f}%)")
    print(f" Negative (stable)   : {neg} ({neg / total * 100:.2f}%)")
    print("------------------------------------------------------------")

    return {"num_positive": int(pos), "num_negative": int(neg), "total": int(total)}


# ==============================================================
# Randomly sample a subset with equal positive/negative samples
# ==============================================================
def get_balanced_subset(dataset, positive_ratio=0.2, negative_ratio=0.2):
    """
    Randomly sample a subset with equal positive/negative samples.
    Default: 20% positive + 20% negative.
    """
    gt = dataset.gt
    if gt is None:
        raise ValueError("GT is required for balanced sampling.")

    labels = np.array([
        gt[y:y + dataset.tile_size, x:x + dataset.tile_size].mean() > 0.8
        for x, y in dataset.coords
    ], dtype=np.uint8)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("No positive or negative samples found in GT.")

    n_pos = max(1, int(len(pos_idx) * positive_ratio))
    n_neg = max(1, int(len(neg_idx) * negative_ratio))

    pos_sel = np.random.choice(pos_idx, n_pos, replace=False)
    neg_sel = np.random.choice(neg_idx, n_neg, replace=False)

    selected_idx = np.concatenate([pos_sel, neg_sel])
    np.random.shuffle(selected_idx)

    subset = Subset(dataset, selected_idx)
    return subset
