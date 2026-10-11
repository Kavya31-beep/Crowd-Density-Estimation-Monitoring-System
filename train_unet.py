"""Train the U-Net density model. Run this on the GPU laptop.

Dataset: ShanghaiTech crowd counting, Part B (folder layout):
    <data>/train_data/images/IMG_1.jpg
    <data>/train_data/ground-truth/GT_IMG_1.mat
    <data>/test_data/images/...   <data>/test_data/ground-truth/...

Example:
    python train_unet.py --data data/ShanghaiTech/part_B --epochs 150
Output: models/unet_density.pth  (best model by test MAE)
"""
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from src.density_unet import MEAN, STD
from src.unet_model import DENSITY_SCALE, UNet


class ShanghaiTech(Dataset):
    def __init__(self, root, split, crop=256, sigma=4.0):
        self.split, self.crop, self.sigma = split, crop, sigma
        self.img_dir = Path(root) / f"{split}_data" / "images"
        self.gt_dir = Path(root) / f"{split}_data" / "ground-truth"
        self.files = sorted(self.img_dir.glob("*.jpg"))
        if not self.files:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.files)

    def _load(self, p):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        pts = loadmat(str(self.gt_dir / f"GT_{p.stem}.mat"))["image_info"][0, 0][0, 0][0]
        h, w = img.shape[:2]
        dens = np.zeros((h, w), np.float32)
        for x, y in pts:
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                dens[yi, xi] += 1
        return img, cv2.GaussianBlur(dens, (0, 0), self.sigma)

    def __getitem__(self, i):
        img, dens = self._load(self.files[i])
        h, w = img.shape[:2]
        if self.split == "train":
            c = self.crop
            if h < c or w < c:  # upscale tiny images, keeping the total count
                s = c / min(h, w) + 1e-3
                nh, nw = int(h * s) + 1, int(w * s) + 1
                total = dens.sum()
                img, dens = cv2.resize(img, (nw, nh)), cv2.resize(dens, (nw, nh))
                if dens.sum() > 0:
                    dens *= total / dens.sum()
                h, w = nh, nw
            y, x = random.randint(0, h - c), random.randint(0, w - c)
            img, dens = img[y:y + c, x:x + c], dens[y:y + c, x:x + c]
            if random.random() < 0.5:
                img, dens = img[:, ::-1], dens[:, ::-1]
        else:  # test: whole image, trimmed to multiples of 8
            h8, w8 = h // 8 * 8, w // 8 * 8
            img, dens = img[:h8, :w8], dens[:h8, :w8]
        img = (img.astype(np.float32) / 255 - MEAN) / STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        dens = torch.from_numpy(np.ascontiguousarray(dens))[None] * DENSITY_SCALE
        return img, dens


def evaluate(model, loader, device):
    model.eval()
    errs = []
    with torch.no_grad():
        for img, dens in loader:
            pred = model(img.to(device)).sum().item() / DENSITY_SCALE
            errs.append(abs(pred - dens.sum().item() / DENSITY_SCALE))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to ShanghaiTech part_B folder")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="models/unet_density.pth")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)
    train_ds, test_ds = ShanghaiTech(a.data, "train"), ShanghaiTech(a.data, "test")
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=2, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=1)

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.MSELoss()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for ep in range(1, a.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for img, dens in train_dl:
            img, dens = img.to(device), dens.to(device)
            opt.zero_grad()
            loss = loss_fn(model(img), dens)
            loss.backward()
            opt.step()
            total += loss.item()
        msg = f"epoch {ep}/{a.epochs}  loss {total / len(train_dl):.4f}  ({time.time() - t0:.0f}s)"
        if ep % 5 == 0 or ep == a.epochs:
            mae = evaluate(model, test_dl, device)
            msg += f"  test MAE {mae:.1f}"
            if mae < best:
                best = mae
                torch.save(model.state_dict(), a.out)
                msg += "  <- saved best"
        print(msg, flush=True)
    print(f"Done. Best test MAE: {best:.1f}. Weights: {a.out}")


if __name__ == "__main__":
    main()
