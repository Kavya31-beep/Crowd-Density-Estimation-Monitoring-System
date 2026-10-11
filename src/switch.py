"""Step 2: Smart Density Switch. YOLO below threshold N, density map at or above N."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .density_unet import density_from_boxes


@dataclass
class DensityResult:
    mode: str
    count: float
    boxes: Optional[np.ndarray]
    density: np.ndarray


def smart_density(frame, yolo, unet, threshold_n, force="auto", tiles=1):
    """force: 'auto' (use the switch), 'yolo' or 'density'.
    tiles: 1 = normal YOLO; 2 or 3 = run YOLO on tiles so small/distant people are found."""
    h, w = frame.shape[:2]

    def detect():
        return yolo.detect(frame) if tiles <= 1 else yolo.detect_tiled(frame, tiles=tiles)

    if force == "density":
        use_dense, boxes0 = True, None
    else:
        boxes0 = detect()
        use_dense = force == "auto" and len(boxes0) >= threshold_n

    if not use_dense:
        label = "YOLO box detection" + (f" ({tiles}x{tiles} tiles)" if tiles > 1 else "")
        return DensityResult(label, len(boxes0), boxes0, density_from_boxes(boxes0, (h, w)))

    if unet.available:
        dens, count = unet.predict(frame)
        return DensityResult("U-Net density map", count, None, dens)

    # No trained weights yet: tiled YOLO keeps the demo working for crowded scenes.
    boxes = boxes0 if (boxes0 is not None and tiles >= 3) else yolo.detect_tiled(frame, tiles=max(tiles, 3))
    return DensityResult("Tiled YOLO (fallback, no U-Net weights found)", len(boxes), boxes,
                         density_from_boxes(boxes, (h, w)))
