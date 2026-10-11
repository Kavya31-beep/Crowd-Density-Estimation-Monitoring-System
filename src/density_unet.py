"""Density-map estimation (best for dense, occluded crowds) plus a fallback density builder."""
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def density_from_boxes(boxes, shape, sigma=None):
    """Builds a density map from detected boxes (one Gaussian blob per person's head)."""
    h, w = shape[:2]
    d = np.zeros((h, w), np.float32)
    for x1, y1, x2, y2 in boxes:
        cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
        cy = int(np.clip(y1 + 0.2 * (y2 - y1), 0, h - 1))
        d[cy, cx] += 1.0
    sigma = sigma or max(6.0, 0.02 * max(h, w))
    return cv2.GaussianBlur(d, (0, 0), sigma)


class UNetDensity:
    """Loads trained weights if they exist. If not, `available` stays False and the
    system falls back to tiled YOLO so the demo still works."""

    def __init__(self, weights, device=None):
        self.available = False
        self.model = None
        self.device = "cpu"
        if not Path(weights).exists():
            return
        try:
            import torch
            from .unet_model import UNet
        except ImportError:
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(str(weights), map_location=self.device))
        self.model.eval()
        self.available = True

    def predict(self, frame_bgr, max_side=1024):
        """Returns (density_map_at_original_size, estimated_count)."""
        import torch
        from .unet_model import DENSITY_SCALE

        h, w = frame_bgr.shape[:2]
        s = min(1.0, max_side / max(h, w))
        nh, nw = max(8, int(h * s) // 8 * 8), max(8, int(w * s) // 8 * 8)
        img = cv2.cvtColor(cv2.resize(frame_bgr, (nw, nh)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        img = (img - MEAN) / STD
        x = torch.from_numpy(img.transpose(2, 0, 1).copy())[None].to(self.device)
        with torch.no_grad():
            d = self.model(x)[0, 0].cpu().numpy() / DENSITY_SCALE
        count = float(d.sum())
        full = cv2.resize(d, (w, h))
        if full.sum() > 0:
            full *= count / full.sum()  # keep the total count unchanged after resizing
        return full, count
