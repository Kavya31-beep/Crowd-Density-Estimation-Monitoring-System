"""Glues all steps together: preprocess -> smart switch -> grid -> overlay."""
from dataclasses import dataclass

from . import config
from .grid import classify_cells, draw_overlay, grid_counts
from .preprocessing import preprocess
from .switch import smart_density


@dataclass
class Settings:
    threshold_n: int = config.SWITCH_THRESHOLD_N
    rows: int = config.GRID_ROWS
    cols: int = config.GRID_COLS
    cell_area_m2: float = config.CELL_AREA_M2
    density_limit: float = config.DENSITY_LIMIT
    force: str = "auto"  # 'auto' | 'yolo' | 'density'
    tiles: int = 2       # YOLO tiling: 1 = off, 2 = 2x2, 3 = 3x3


def analyze_frame(frame, yolo, unet, s, bg=None):
    clean, edges, mask = preprocess(frame, bg)
    res = smart_density(clean, yolo, unet, s.threshold_n, s.force, s.tiles)
    counts = grid_counts(res.density, s.rows, s.cols)
    dens, flags = classify_cells(counts, s.cell_area_m2, s.density_limit)
    overlay = draw_overlay(frame, res.density, counts, flags, s.rows, s.cols, res.boxes,
                          s.cell_area_m2, s.density_limit)
    return {"result": res, "counts": counts, "density": dens, "flags": flags,
            "overlay": overlay, "edges": edges, "mask": mask}
