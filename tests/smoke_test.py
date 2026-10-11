"""Quick check that grid + alerts + pipeline work WITHOUT any model or GPU.
Run from the project root:  python tests/smoke_test.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import Settings, analyze_frame  # noqa: E402


class StubYolo:
    def detect(self, frame, imgsz=640):
        h, w = frame.shape[:2]
        rng = np.random.default_rng(0)
        xs, ys = rng.integers(0, w // 3 - 30, 40), rng.integers(0, h // 3 - 60, 40)   # crowd in top-left cell
        xo, yo = rng.integers(w // 2, w - 30, 5), rng.integers(h // 2, h - 60, 5)      # a few people elsewhere
        x, y = np.concatenate([xs, xo]), np.concatenate([ys, yo])
        return np.stack([x, y, x + 30, y + 60], 1).astype(float)

    def detect_tiled(self, frame, tiles=3, imgsz=640):
        return self.detect(frame)


class StubUnet:
    available = False


frame = np.full((720, 1280, 3), 120, np.uint8)
out = analyze_frame(frame, StubYolo(), StubUnet(), Settings(threshold_n=30))
print("mode:", out["result"].mode, "| count:", round(out["result"].count))
print("overcrowded cells:\n", out["flags"])
assert out["flags"][0, 0], "top-left cell should be flagged"
assert out["flags"].sum() == 1, "only the crowded cell should be flagged"
print("SMOKE TEST PASSED")
