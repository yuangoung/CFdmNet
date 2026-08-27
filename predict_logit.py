# ==============================================================
# predict_logit.py
# Description:
#   - Data reading & normalization fully consistent with training
#   - Sliding window coords cover full raster without padding
#   - After stitching: global stretch to [0,1] in probability space
#       p_low/p_high: percentile anchors for 0 and 1
#       alpha in [0,1]: strength of percentile anchors
#   - Output: GeoTIFF (stretched 0-1)
# ==============================================================
import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as tud
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from osgeo import gdal
from FeatureExtractor import MDFeatureExtractor
from shared_decoder import SharedDecoder
from data_loader import LandslideDataset

# ==============================================================
# Config
# ==============================================================
class Config:
    model_ckpt = r"F:...\checkpoints_MDDB\best_logs_MDDB.pth"

    msi_path = r"...\MSI.tif"
    dem_path = r"...\DEM.tif"
    gt_path = None

    output_tif = r"...\output.tif"
    output_png = r"...\output.png"

    tile_size  = 256
    stride     = 64
    batch_size = 21
    num_workers = 0  # Windows safe

    # global stretch (after stitching)
    stretch_p_low  = 1.0
    stretch_p_high = 99.0
    stretch_alpha  = 1.0

    # stats sampling for percentiles
    stat_block_h   = 1024
    stat_stride    = 8
    stat_max_vals  = 2_000_000

    # visualization
    cmap = "jet"       # or "turbo"
    viz_max_side = 2200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================
def sliding_positions(length, tile, stride):
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    last = length - tile
    if pos[-1] != last:
        pos.append(last)
    return pos


def gaussian_window(tile_size, sigma_ratio=0.25):
    center = (tile_size - 1) / 2.0
    sigma = tile_size * sigma_ratio
    x = np.arange(tile_size, dtype=np.float32)
    y = np.arange(tile_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    g = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))
    g /= (g.max() + 1e-8)
    return g.astype(np.float32)


def sigmoid_np_stable(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float32, copy=False)
    out = np.empty_like(z, dtype=np.float32)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def save_geotiff_float32(path, meta, write_block_fn, nodata=-9999.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        path,
        meta["width"],
        meta["height"],
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=YES"],
    )
    out_ds.SetGeoTransform(meta["geo"])
    out_ds.SetProjection(meta["proj"])
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata))

    write_block_fn(out_ds)

    band.FlushCache()
    out_ds.FlushCache()
    out_ds = None


def coords_to_numpy(coords) -> np.ndarray:

    if torch.is_tensor(coords):
        arr = coords.detach().cpu().numpy()
        if arr.ndim == 1 and arr.size == 2:
            arr = arr[None, :]
        return arr.astype(np.int64)

    if isinstance(coords, (list, tuple)):
        # case 2: (tensor_x, tensor_y)
        if len(coords) == 2 and torch.is_tensor(coords[0]) and torch.is_tensor(coords[1]):
            xs = coords[0].detach().cpu().numpy().astype(np.int64)
            ys = coords[1].detach().cpu().numpy().astype(np.int64)
            return np.stack([xs, ys], axis=1)

        # case 3: [(x,y), ...]
        if len(coords) > 0 and isinstance(coords[0], (list, tuple)) and len(coords[0]) == 2:
            return np.array(coords, dtype=np.int64)

    # fallback
    return np.array(coords, dtype=np.int64)


def estimate_stretch_range_from_memmap(
    logit_sum_mm,
    weight_sum_mm,
    p_low=1.0,
    p_high=99.0,
    alpha=1.0,
    block_h=1024,
    stride=8,
    max_vals=2_000_000,
    eps=1e-6,
):
    H, W = logit_sum_mm.shape
    vals_list = []
    total = 0
    s_min, s_max = None, None

    for y in tqdm(range(0, H, block_h), desc="Estimate stretch stats"):
        h = min(block_h, H - y)
        ls = np.array(logit_sum_mm[y:y+h, :], dtype=np.float32, copy=False)
        ws = np.array(weight_sum_mm[y:y+h, :], dtype=np.float32, copy=False)

        valid = ws > 0
        if not np.any(valid):
            continue

        z = np.zeros_like(ls, dtype=np.float32)
        np.divide(ls, ws, out=z, where=valid)
        p = sigmoid_np_stable(z)

        p_s = p[::stride, ::stride]
        v_s = valid[::stride, ::stride]
        chunk = p_s[v_s]
        if chunk.size == 0:
            continue

        if s_min is None:
            s_min = float(np.min(chunk))
            s_max = float(np.max(chunk))
        else:
            s_min = min(s_min, float(np.min(chunk)))
            s_max = max(s_max, float(np.max(chunk)))

        vals_list.append(chunk.astype(np.float32, copy=False))
        total += int(chunk.size)
        if total >= max_vals:
            break

    if total == 0:
        return 0.0, 1.0

    vals = np.concatenate(vals_list, axis=0)
    if vals.size > max_vals:
        vals = vals[:max_vals]

    q_low = float(np.percentile(vals, p_low))
    q_high = float(np.percentile(vals, p_high))

    if s_min is None or s_max is None or (s_max - s_min) < eps:
        s_min, s_max = 0.0, 1.0

    a = float(np.clip(alpha, 0.0, 1.0))
    vmin = (1.0 - a) * s_min + a * q_low
    vmax = (1.0 - a) * s_max + a * q_high
    if (vmax - vmin) < eps:
        vmin, vmax = 0.0, 1.0

    return float(vmin), float(vmax)


# ==============================================================
# Load Model
# ==============================================================
def load_model(ckpt_path, device, msi_channels=4, dem_channels=1):
    print(f"\n>>> Loading model: {ckpt_path}")
    model_fea = MDFeatureExtractor(msi_channels=msi_channels, dem_channels=dem_channels).to(device)
    model_dec = SharedDecoder(
        base_ch=64, out_classes=1, gate_type="channel",
        decoder_ch=256, use_dwconv=True, dropout=0.1
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    fea_state = ckpt.get("model_feature", ckpt)
    dec_state = ckpt.get("model_decoder", ckpt)

    model_fea.load_state_dict(fea_state, strict=False)
    model_dec.load_state_dict(dec_state, strict=False)

    model_fea.eval()
    model_dec.eval()
    print(">>> Model loaded.")
    return model_fea, model_dec


@torch.no_grad()
def forward_logits(model_fea, model_dec, msi_bchw, dem_bchw):
    msi_feats, dem_feats = model_fea(msi_bchw, dem_bchw)
    logits = model_dec(msi_feats, dem_feats)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if logits.ndim == 4:
        logits = logits[:, 0, :, :]
    return logits  # [B,H,W]


# ==============================================================
class InferenceDataset(LandslideDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        H, W = self.msi.shape[:2]
        xs = sliding_positions(W, self.tile_size, self.stride)
        ys = sliding_positions(H, self.tile_size, self.stride)
        self.coords = [(x, y) for y in ys for x in xs]
        print(f"[InferenceDataset] Full-coverage patches: {len(self.coords)}")


# ==============================================================
def inference(cfg: Config):
    torch.backends.cudnn.benchmark = True

    dataset = InferenceDataset(
        msi_path=cfg.msi_path,
        dem_path=cfg.dem_path,
        gt_path=cfg.gt_path,
        tile_size=cfg.tile_size,
        stride=cfg.stride,
        clip_percent=(2.0, 98.0),
        fill_value=0.0,
    )
    meta = dataset.get_meta()  # {"geo","proj","height","width"}
    H, W = meta["height"], meta["width"]

    loader = tud.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print(f"\nRaster: {W}×{H}, tile={cfg.tile_size}, stride={cfg.stride}, batch={cfg.batch_size}")
    print(f"Device: {cfg.device}")
    print(f"Global stretch: p_low={cfg.stretch_p_low}, p_high={cfg.stretch_p_high}, alpha={cfg.stretch_alpha}")

    model_fea, model_dec = load_model(cfg.model_ckpt, cfg.device, msi_channels=4, dem_channels=1)

    w_tile = gaussian_window(cfg.tile_size)

    tmp_dir = os.path.join(os.path.dirname(cfg.output_tif), "_tmp_memmap")
    os.makedirs(tmp_dir, exist_ok=True)
    logit_path = os.path.join(tmp_dir, "logit_sum.dat")
    wsum_path  = os.path.join(tmp_dir, "weight_sum.dat")

    logit_sum = np.memmap(logit_path, dtype=np.float32, mode="w+", shape=(H, W))
    weight_sum = np.memmap(wsum_path, dtype=np.float32, mode="w+", shape=(H, W))
    logit_sum[:] = 0.0
    weight_sum[:] = 0.0

    t0 = time.time()

    for msi_b, dem_b, _gt, coords in tqdm(loader, desc="Inference (logit blend)"):
        coords_np = coords_to_numpy(coords)  # [B,2] int64

        msi_b = msi_b.to(cfg.device, non_blocking=True).float()
        dem_b = dem_b.to(cfg.device, non_blocking=True).float()

        logits = forward_logits(model_fea, model_dec, msi_b, dem_b)
        logits = logits.detach().cpu().numpy().astype(np.float32)

        B = logits.shape[0]
        for i in range(B):
            x = int(coords_np[i, 0])
            y = int(coords_np[i, 1])
            z = logits[i]  # [tile,tile]
            logit_sum[y:y+cfg.tile_size, x:x+cfg.tile_size] += z * w_tile
            weight_sum[y:y+cfg.tile_size, x:x+cfg.tile_size] += w_tile

        del msi_b, dem_b, logits

    logit_sum.flush()
    weight_sum.flush()
    print(f"\n>>> Done tiling. Elapsed: {(time.time() - t0)/60:.2f} min")

    vmin, vmax = estimate_stretch_range_from_memmap(
        logit_sum, weight_sum,
        p_low=cfg.stretch_p_low,
        p_high=cfg.stretch_p_high,
        alpha=cfg.stretch_alpha,
        block_h=cfg.stat_block_h,
        stride=cfg.stat_stride,
        max_vals=cfg.stat_max_vals,
    )
    print(f">>> Stretch range (prob space): vmin={vmin:.6f}, vmax={vmax:.6f}")

    nodata = -9999.0
    eps = 1e-6

    def _write_blocks(out_ds):
        band = out_ds.GetRasterBand(1)
        block_h = 1024

        for y in tqdm(range(0, H, block_h), desc="Write GeoTIFF"):
            h = min(block_h, H - y)
            ls = np.array(logit_sum[y:y+h, :], dtype=np.float32, copy=False)
            ws = np.array(weight_sum[y:y+h, :], dtype=np.float32, copy=False)

            out = np.full_like(ls, nodata, dtype=np.float32)
            valid = ws > 0
            if np.any(valid):
                z = np.zeros_like(ls, dtype=np.float32)
                np.divide(ls, ws, out=z, where=valid)
                p = sigmoid_np_stable(z)
                s = (p - vmin) / (vmax - vmin + eps)
                s = np.clip(s, 0.0, 1.0).astype(np.float32)
                out[valid] = s[valid]

            band.WriteArray(out, xoff=0, yoff=y)

    save_geotiff_float32(cfg.output_tif, meta, _write_blocks, nodata=nodata)
    print(f">>> Saved GeoTIFF: {cfg.output_tif}")

    out_ds = gdal.Open(cfg.output_tif, gdal.GA_ReadOnly)
    if out_ds is not None:
        max_side = int(cfg.viz_max_side)
        scale = max(out_ds.RasterXSize / max_side, out_ds.RasterYSize / max_side, 1.0)
        out_w = int(out_ds.RasterXSize / scale)
        out_h = int(out_ds.RasterYSize / scale)

        img = out_ds.ReadAsArray(
            xoff=0, yoff=0,
            xsize=out_ds.RasterXSize, ysize=out_ds.RasterYSize,
            buf_xsize=out_w, buf_ysize=out_h
        ).astype(np.float32)

        img = np.where(img == nodata, np.nan, img)

        os.makedirs(os.path.dirname(cfg.output_png), exist_ok=True)
        plt.figure(figsize=(9, 8))
        im = plt.imshow(img, cmap=cfg.cmap, norm=Normalize(vmin=0.0, vmax=1.0))
        plt.title("Landslide Susceptibility (Logit Blend + Global Stretch)", fontsize=14)
        plt.axis("off")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Stretched Risk (0-1)", rotation=90, fontsize=11)
        plt.tight_layout()
        plt.savefig(cfg.output_png, dpi=220, bbox_inches="tight")
        plt.close()
        print(f">>> Saved PNG: {cfg.output_png}")

    print("\n>>> All done.")


# ==============================================================
def build_cfg_from_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=Config.model_ckpt)
    p.add_argument("--msi", type=str, default=Config.msi_path)
    p.add_argument("--dem", type=str, default=Config.dem_path)
    p.add_argument("--out_tif", type=str, default=Config.output_tif)
    p.add_argument("--out_png", type=str, default=Config.output_png)
    p.add_argument("--tile", type=int, default=Config.tile_size)
    p.add_argument("--stride", type=int, default=Config.stride)
    p.add_argument("--batch", type=int, default=Config.batch_size)

    p.add_argument("--p_low", type=float, default=Config.stretch_p_low)
    p.add_argument("--p_high", type=float, default=Config.stretch_p_high)
    p.add_argument("--alpha", type=float, default=Config.stretch_alpha)

    p.add_argument("--cmap", type=str, default=Config.cmap)
    p.add_argument("--viz_max_side", type=int, default=Config.viz_max_side)

    args = p.parse_args()

    cfg = Config()
    cfg.model_ckpt = args.ckpt
    cfg.msi_path = args.msi
    cfg.dem_path = args.dem
    cfg.output_tif = args.out_tif
    cfg.output_png = args.out_png
    cfg.tile_size = args.tile
    cfg.stride = args.stride
    cfg.batch_size = args.batch

    cfg.stretch_p_low = args.p_low
    cfg.stretch_p_high = args.p_high
    cfg.stretch_alpha = float(np.clip(args.alpha, 0.0, 1.0))

    cfg.cmap = args.cmap
    cfg.viz_max_side = args.viz_max_side

    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return cfg


if __name__ == "__main__":
    cfg = build_cfg_from_args()
    inference(cfg)
