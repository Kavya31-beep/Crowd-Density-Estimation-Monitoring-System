"""Central settings. Change values here instead of hunting through the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LOG_FILE = ROOT / "logs" / "events.csv"

YOLO_WEIGHTS = "yolov8n.pt"                  # downloads automatically the first time
UNET_WEIGHTS = MODELS_DIR / "unet_density.pth"  # produced by train_unet.py (GPU laptop)

SWITCH_THRESHOLD_N = 30   # headcount at which we switch from YOLO to the density map
GRID_ROWS = 3
GRID_COLS = 3
CELL_AREA_M2 = 3.0        # real-world area covered by ONE grid cell (set per camera)
DENSITY_LIMIT = 4.0       # people per m^2 above which a cell turns red
