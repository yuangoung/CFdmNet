# ==============================================================
# main_loss.py
# Description:
#   Supervised training for landslide susceptibility mapping
#   Compatible with Light FeatureExtractor (multi-scale)
#   Loss = BCE + SSIM + TV with Pareto weighting
# ==============================================================
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import inspect
from data_loader import get_dataloader, get_balanced_subset
from FeatureExtractor import MDFeatureExtractor
from shared_decoder import SharedDecoder
# ==============================================================
# Configuration
# ==============================================================
class Config:
    """Training configuration """

    msi_path = r"...\demo_datasets\DEM_demo_Beiluhe.tif"
    dem_path = r"...\demo_datasets\DEM_demo_Beiluhe.tif"
    gt_path  = r"...\demo_datasets\DEM_demo_Beiluhe.tif"

    tile_size = 256
    stride = 200
    batch_size = 21
    num_workers = 0
    num_epochs = 150
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    positive_ratio = 1
    negative_ratio = 1

    save_dir = "checkpoints_MDDB"
    log_dir = "best_logs_MDDB"
    csv_log = "best_training_log.csv"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
# ==============================================================
# Composite Loss: SSIM + BCE + TV
# ==============================================================
def gaussian_window(window_size, sigma, channel):
    gauss = torch.Tensor(
        [np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)]
    )
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(pred, target, window_size=11, sigma=1.5, data_range=1.0):

    channel = pred.size(1)
    window = gaussian_window(window_size, sigma, channel).to(pred.device)
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class SSIMTVLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def tv_loss(self, pred):
        diff_x = torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]))
        diff_y = torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]))
        return diff_x + diff_y

    def forward(self, pred, gt):
        pred_sig = torch.sigmoid(pred)
        loss_bce = self.bce(pred, gt)
        loss_ssim = 1 - ssim(pred_sig, gt)
        loss_tv = self.tv_loss(pred_sig)
        total = self.alpha * loss_bce + self.beta * loss_ssim + self.gamma * loss_tv
        return total, loss_bce, loss_ssim, loss_tv


# ==============================================================
# Pareto Dynamic Weight Update
# ==============================================================
def pareto_update(grad1, grad2):
    g1 = torch.norm(grad1)
    g2 = torch.norm(grad2)
    total = g1 + g2 + 1e-8
    w1 = 1 - (g1 / total)
    w2 = 1 - (g2 / total)
    return w1.item(), w2.item()

# ==============================================================
# Training Loop
# ==============================================================
def train_one_epoch(model_fea, model_dec, loader, optimizer, criterion, device):
    model_fea.train()
    model_dec.train()

    total_loss, epoch_acc = 0.0, 0.0

    for msi, dem, gt, _ in tqdm(loader, desc="Training"):
        msi, dem, gt = msi.to(device), dem.to(device), gt.to(device)
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)

        # ---- Forward ----
        msi_feats, dem_feats = model_fea(msi, dem)     # multi-scale features
        pred = model_dec(msi_feats, dem_feats)         # [B,1,H,W] (full-res logits)

        # ---- Align GT size (safe) ----
        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(gt.float(), size=pred.shape[-2:], mode="nearest")

        # ---- Loss ----
        total, loss_bce, loss_ssim, loss_tv = criterion(pred, gt.float())

        grad_bce = torch.autograd.grad(loss_bce, pred, retain_graph=True, create_graph=True)[0]
        grad_ssim = torch.autograd.grad(loss_ssim, pred, retain_graph=True, create_graph=True)[0]
        w1, w2 = pareto_update(grad_bce, grad_ssim)
        total_loss_batch = w1 * loss_bce + w2 * (loss_ssim + loss_tv)

        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()

        # ---- Accuracy ----
        with torch.no_grad():
            preds = torch.sigmoid(pred)
            correct = ((preds > 0.5) == gt.bool()).float().mean()
            epoch_acc += correct.item()

    return total_loss / len(loader), epoch_acc / len(loader), loss_bce.item(), loss_ssim.item(), loss_tv.item()


# ==============================================================
# Training Plot
# ==============================================================
class LiveLossPlotter:
    """Visualization of loss and accuracy"""

    def __init__(self):
        plt.ion()
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
        self.losses, self.accs = [], []

    def update(self, loss, acc):
        self.losses.append(loss)
        self.accs.append(acc)
        self.plot()

    def plot(self):
        if len(self.losses) > 1:
            self.ax_loss.clear()
            self.ax_acc.clear()
            epochs = range(1, len(self.losses) + 1)
            self.ax_loss.plot(epochs, gaussian_filter1d(self.losses, sigma=1), color='red', label='Train Loss')
            self.ax_acc.plot(epochs, gaussian_filter1d(self.accs, sigma=1), color='blue', label='Accuracy')
            self.ax_loss.set_xlabel('Epoch')
            self.ax_loss.set_ylabel('Loss')
            self.ax_acc.set_ylabel('Accuracy')
            self.ax_loss.legend()
            self.ax_acc.legend()
            self.ax_loss.grid(True)
            self.ax_acc.grid(True)
            plt.pause(0.01)

    def save(self, path="training_curve.png"):
        self.fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(self.fig)


# ==============================================================
# Main Training
# ==============================================================
def main():
    cfg = Config()
    print("\n=== Loading Dataset ===")
    _, dataset = get_dataloader(cfg.msi_path, cfg.dem_path, cfg.gt_path,
                                tile_size=cfg.tile_size, stride=cfg.stride,
                                batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    total_patches = len(dataset.coords)
    pos_samples = int(total_patches * cfg.positive_ratio)
    neg_samples = int(total_patches * cfg.negative_ratio)
    total_used = pos_samples + neg_samples
    print(f"Total patches: {total_patches}")
    print(f"Selected for training → Positive: {pos_samples}, Negative: {neg_samples}, Total Used: {total_used}")

    # ---- Build Models ----
    print("\n=== Building Models ===")
    model_fea = MDFeatureExtractor(msi_channels=4, dem_channels=1).to(cfg.device)
    model_dec = SharedDecoder(base_ch=64, out_classes=1, gate_type="channel", decoder_ch=256, use_dwconv=True, dropout=0.1).to(cfg.device)

    optimizer = torch.optim.Adam(list(model_fea.parameters()) +
                                 list(model_dec.parameters()), lr=cfg.lr)
    criterion = SSIMTVLoss(alpha=1.0, beta=0.5, gamma=0.1)

    # ---- Logging ----
    writer = SummaryWriter(log_dir=cfg.log_dir)
    csv_file = open(os.path.join(cfg.log_dir, cfg.csv_log), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Train_Loss", "Accuracy", "BCE", "SSIM", "TV"])
    plotter = LiveLossPlotter()

    print("\n=== Starting Training ===")
    for epoch in range(cfg.num_epochs):
        subset = get_balanced_subset(dataset, cfg.positive_ratio, cfg.negative_ratio)
        loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

        train_loss, acc, loss_bce, loss_ssim, loss_tv = train_one_epoch(
            model_fea, model_dec, loader, optimizer, criterion, cfg.device)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", acc, epoch)
        csv_writer.writerow([epoch, train_loss, acc, loss_bce, loss_ssim, loss_tv])
        csv_file.flush()
        plotter.update(train_loss, acc)

        print(f"[Epoch {epoch}] Loss={train_loss:.4f}, Acc={acc:.4f}, BCE={loss_bce:.4f}, SSIM={loss_ssim:.4f}, TV={loss_tv:.4f}")

        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            torch.save({
                "model_feature": model_fea.state_dict(),
                "model_decoder": model_dec.state_dict(),
                "epoch": epoch
            }, os.path.join(cfg.save_dir, f"epoch_{epoch}.pth"))

    csv_file.close()
    writer.close()
    plotter.save(os.path.join(cfg.log_dir, "training_curve.png"))
    print("\n=== Training Completed Successfully ===")


# ==============================================================
if __name__ == "__main__":
    main()