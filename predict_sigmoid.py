# ==============================================================
# predict_sigmoid.py
# Description:
#   landslide susceptibility mapping
#   - Gaussian-weighted sliding window (probability domain)
#   - GeoTIFF output (0-1)
# ==============================================================
import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize
from osgeo import gdal
from FeatureExtractor import MDFeatureExtractor
from shared_decoder import SharedDecoder
from data_loader import get_dataloader

# ==============================================================
# Configuration
# ==============================================================

class Config:

    model_ckpt = r"F:...\checkpoints_MDDB\best_logs_MDDB.pth"

    msi_path = r"...\MSI.tif"
    dem_path = r"...\DEM.tif"
    gt_path = None

    output_tif = r"...\output.tif"
    output_png = r"...\output.png"


    tile_size = 256
    stride = 100
    batch_size = 21

    # robust normalization (match your new data_loader.py)
    clip_percent = (2.0, 98.0)
    fill_value = 0.0

    # visualization (PNG only)
    cmap = "jet"            # or "turbo"
    viz_gamma = 0.75        # <1 expands low-prob details; 1 = linear
    viz_percent_clip = (1.0, 99.0)  # (low, high) percentiles for visualization stretch
    viz_max_side = 2200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================
def save_geotiff_float32(path, meta, write_block_fn, nodata=-9999.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        path,
        meta["cols"],
        meta["rows"],
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=YES"],
    )
    out_ds.SetGeoTransform(meta["geotrans"])
    out_ds.SetProjection(meta["proj"])
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(float(nodata))

    write_block_fn(out_ds)

    out_band.FlushCache()
    out_ds.FlushCache()
    out_ds = None

# ==============================================================
def gaussian_window(tile_size, sigma_ratio=0.25):
    center = (tile_size - 1) / 2.0
    sigma = tile_size * sigma_ratio
    x = np.arange(tile_size, dtype=np.float32)
    y = np.arange(tile_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    g = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))
    g /= (g.max() + 1e-8)
    return g.astype(np.float32)


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
def forward_prob(model_fea, model_dec, msi_bchw, dem_bchw):
    msi_feats, dem_feats = model_fea(msi_bchw, dem_bchw)
    logits = model_dec(msi_feats, dem_feats)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if logits.ndim == 4:
        logits = logits[:, 0, :, :]
    prob = torch.sigmoid(logits)
    return prob  # [B,H,W]


# ==============================================================
def inference(cfg: Config):
    model_fea, model_dec = load_model(cfg.model_ckpt, cfg.device, msi_channels=4, dem_channels=1)

    loader, dataset = get_dataloader(
        msi_path=cfg.msi_path,
        dem_path=cfg.dem_path,
        gt_path=None,
        tile_size=cfg.tile_size,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        num_workers=0,
        clip_percent=cfg.clip_percent,
        fill_value=cfg.fill_value,
    )

    dmeta = dataset.get_meta()
    H, W = int(dmeta["height"]), int(dmeta["width"])
    meta = {"proj": dmeta["proj"], "geotrans": dmeta["geo"], "cols": W, "rows": H}

    print(f"Raster: {W}×{H}, tile={cfg.tile_size}, stride={cfg.stride}, batch={cfg.batch_size}")
    print(f"Total tiles: {len(dataset)}")
    print(f"Robust norm: {cfg.clip_percent} -> [0,1] per band")

    w_tile = gaussian_window(cfg.tile_size)

    tmp_dir = os.path.join(os.path.dirname(cfg.output_tif), "_tmp_memmap")
    os.makedirs(tmp_dir, exist_ok=True)
    prob_path = os.path.join(tmp_dir, "prob_sum.dat")
    wsum_path = os.path.join(tmp_dir, "weight_sum.dat")

    prob_sum = np.memmap(prob_path, dtype=np.float32, mode="w+", shape=(H, W))
    weight_sum = np.memmap(wsum_path, dtype=np.float32, mode="w+", shape=(H, W))
    prob_sum[:] = 0.0
    weight_sum[:] = 0.0

    t0 = time.time()

    for batch in tqdm(loader, desc="Inference"):
        msi_b, dem_b, _, coords = batch  # coords from dataset

        msi_b = msi_b.to(cfg.device, non_blocking=True)  # [B,C,H,W] already normalized by data_loader
        dem_b = dem_b.to(cfg.device, non_blocking=True)  # [B,1,H,W] already normalized by data_loader

        prob = forward_prob(model_fea, model_dec, msi_b, dem_b).detach().cpu().numpy().astype(np.float32)

        # coords handling: default_collate -> (xs_tensor, ys_tensor) OR list of tuples
        if isinstance(coords, (list, tuple)) and len(coords) == 2 and torch.is_tensor(coords[0]) and torch.is_tensor(coords[1]):
            xs = coords[0].cpu().numpy()
            ys = coords[1].cpu().numpy()
        else:
            xs = np.array([c[0] for c in coords], dtype=np.int64)
            ys = np.array([c[1] for c in coords], dtype=np.int64)

        B = prob.shape[0]
        for b in range(B):
            x = int(xs[b])
            y = int(ys[b])

            p = prob[b]  # [tile,tile]
            prob_sum[y:y + cfg.tile_size, x:x + cfg.tile_size] += p * w_tile
            weight_sum[y:y + cfg.tile_size, x:x + cfg.tile_size] += w_tile

        del msi_b, dem_b, prob

    prob_sum.flush()
    weight_sum.flush()
    print(f"\n>>> Done tiling. Elapsed: {(time.time() - t0)/60:.2f} min")

    nodata = -9999.0
    eps = 1e-6

    global_max = 0.0
    block_h = 1024
    for y in tqdm(range(0, H, block_h), desc="Scan global max"):
        h = min(block_h, H - y)
        ps = np.array(prob_sum[y:y + h, :], dtype=np.float32, copy=False)
        ws = np.array(weight_sum[y:y + h, :], dtype=np.float32, copy=False)
        valid = ws > 0
        if np.any(valid):
            out0 = ps[valid] / (ws[valid] + eps)
            m = float(np.max(out0))
            if m > global_max:
                global_max = m

    if not np.isfinite(global_max) or global_max < eps:
        global_max = 1.0
    print(f">>> Global max before stretch: {global_max:.6f}  (will scale to make max=1.0)")

    def _write_blocks(out_ds):
        band = out_ds.GetRasterBand(1)
        for y in tqdm(range(0, H, block_h), desc="Write GeoTIFF"):
            h = min(block_h, H - y)
            ps = np.array(prob_sum[y:y + h, :], dtype=np.float32, copy=False)
            ws = np.array(weight_sum[y:y + h, :], dtype=np.float32, copy=False)

            out = np.full_like(ps, nodata, dtype=np.float32)
            valid = ws > 0

            out0 = ps[valid] / (ws[valid] + eps)
            out0 = np.clip(out0, 0.0, 1.0)

            # stretch to full [0,1] (max->1)
            out0 = out0 / (global_max + eps)
            out0 = np.clip(out0, 0.0, 1.0)

            out[valid] = out0
            band.WriteArray(out, xoff=0, yoff=y)

    save_geotiff_float32(cfg.output_tif, meta, _write_blocks, nodata=nodata)
    print(f">>> Saved GeoTIFF: {cfg.output_tif}")

    # PNG with colorbar (visual-only stretch)
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

        finite = np.isfinite(img)
        if finite.any():
            p_low, p_high = cfg.viz_percent_clip
            vmin = float(np.nanpercentile(img, p_low))
            vmax = float(np.nanpercentile(img, p_high))
            if vmax - vmin < 1e-6:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0

        norm = PowerNorm(gamma=float(cfg.viz_gamma), vmin=vmin, vmax=vmax, clip=True) if cfg.viz_gamma != 1.0 \
            else Normalize(vmin=vmin, vmax=vmax, clip=True)

        os.makedirs(os.path.dirname(cfg.output_png), exist_ok=True)
        plt.figure(figsize=(9, 8))
        im = plt.imshow(img, cmap=cfg.cmap, norm=norm)
        plt.title("Predicted Landslide Susceptibility Map", fontsize=16)
        plt.axis("off")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Landslide Probability", rotation=90, fontsize=12)
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
    p.add_argument("--pclip_low", type=float, default=Config.clip_percent[0])
    p.add_argument("--pclip_high", type=float, default=Config.clip_percent[1])
    p.add_argument("--cmap", type=str, default=Config.cmap)
    p.add_argument("--gamma", type=float, default=Config.viz_gamma)
    p.add_argument("--viz_low", type=float, default=Config.viz_percent_clip[0])
    p.add_argument("--viz_high", type=float, default=Config.viz_percent_clip[1])
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
    cfg.clip_percent = (args.pclip_low, args.pclip_high)
    cfg.cmap = args.cmap
    cfg.viz_gamma = args.gamma
    cfg.viz_percent_clip = (args.viz_low, args.viz_high)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return cfg


if __name__ == "__main__":
    cfg = build_cfg_from_args()
    inference(cfg)
