"""Step 3: spatial grid. Density per cell so we can pinpoint the exact bottleneck."""
import cv2
import numpy as np


def grid_counts(density, rows, cols):
    h, w = density.shape
    counts = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            counts[r, c] = density[y0:y1, x0:x1].sum()
    return counts


def classify_cells(counts, cell_area_m2, limit):
    """Returns (people_per_m2 per cell, boolean 'overcrowded' flag per cell)."""
    dens = counts / cell_area_m2
    return dens, dens > limit


def draw_overlay(frame, density, counts, flags, rows, cols, boxes=None,
                 cell_area_m2=3.0, limit=4.0, alpha=0.4):
    out = frame.copy()
    h, w = out.shape[:2]

    # Heat map on an ABSOLUTE scale: colour = local people/m2 compared with the capacity limit.
    # (blue/green = well below the limit, yellow = near it, red = at or above 2x the limit),
    # so a few scattered people never show up as red.
    smooth = cv2.GaussianBlur(density, (0, 0), 0.04 * max(h, w))
    px_per_m2 = (h * w / (rows * cols)) / cell_area_m2
    dn = np.clip(smooth * px_per_m2 / (2 * limit), 0, 1)
    heat = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(out, 1 - alpha, heat, alpha, 0)
    out = np.where((dn > 0.08)[..., None], blend, out).astype(np.uint8)

    if boxes is not None:
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    red = out.copy()
    scale = max(0.5, w / 1200)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            if flags[r, c]:
                cv2.rectangle(red, (x0, y0), (x1, y1), (0, 0, 255), -1)
    out = cv2.addWeighted(red, 0.35, out, 0.65, 0)

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            color = (0, 0, 255) if flags[r, c] else (255, 255, 255)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2 if flags[r, c] else 1)
            label = f"{counts[r, c]:.0f}" + (" OVERCROWDED" if flags[r, c] else "")
            cv2.putText(out, label, (x0 + 6, y0 + int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x0 + 6, y0 + int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * scale, color, 1, cv2.LINE_AA)
    return out
