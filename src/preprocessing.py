"""Step 1: preprocessing and edge filtering (noise removal, edges, background subtraction)."""
import cv2
import numpy as np


def denoise(frame, method="gaussian"):
    if method == "median":
        return cv2.medianBlur(frame, 5)
    return cv2.GaussianBlur(frame, (5, 5), 0)


def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 80, 160)


class BackgroundSubtractor:
    """Separates moving pedestrians from static trees, walls and floor patterns (video only)."""

    def __init__(self):
        self.mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)

    def apply(self, frame):
        mask = self.mog.apply(frame)
        mask[mask == 127] = 0  # drop shadow pixels
        return mask


def preprocess(frame, bg=None):
    """Returns (clean_frame, edge_map, motion_mask_or_None)."""
    clean = denoise(frame)
    edges = detect_edges(clean)
    mask = bg.apply(clean) if bg is not None else None
    return clean, edges, mask
