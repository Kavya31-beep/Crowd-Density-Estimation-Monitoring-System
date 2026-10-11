# Automated Crowd Density Estimation & Management System (Group 8)

Pipeline: Video/Image -> Preprocessing -> Smart Density Switch (YOLOv8 or U-Net) -> Spatial Grid -> Dashboard + Alarm + Event log

## Project structure
| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (run this) |
| `src/preprocessing.py` | Gaussian/median filter, edges, background subtraction |
| `src/detector_yolo.py` | YOLOv8 people detection / counting |
| `src/unet_model.py`, `src/density_unet.py` | U-Net density map model and loader |
| `src/switch.py` | Smart density switch (threshold N) |
| `src/grid.py` | 3x3 / 4x4 zones, red overcrowded cells, heat map overlay |
| `src/alerts.py` | Alarm sound + event log (timestamp, camera ID) |
| `src/pipeline.py` | Connects all steps |
| `train_unet.py` | Trains the U-Net (GPU laptop) |
| `tests/smoke_test.py` | Checks the logic without any model |

## Setup (everyone)
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test.py     # should print SMOKE TEST PASSED
streamlit run app.py
```
Put a few crowd images/videos in `samples/` (or just upload from the dashboard).
Without U-Net weights the app still works: crowded scenes use tiled YOLO (shown in the sidebar).

## Training the U-Net (GPU teammate)
1. Install the CUDA build of PyTorch from https://pytorch.org (pick your CUDA version), then `pip install -r requirements.txt`.
2. Download the ShanghaiTech crowd counting dataset, Part B (search "ShanghaiTech crowd counting dataset" on Kaggle) and unzip it to `data/ShanghaiTech/part_B` (it must contain `train_data` and `test_data`).
3. Run: `python train_unet.py --data data/ShanghaiTech/part_B --epochs 150`
4. The best model is saved to `models/unet_density.pth`. It is NOT in git (too large). Upload it to the shared Drive folder, and everyone downloads it into their own `models/` folder.

## Git rules
- Never commit datasets, `venv/` or `.pt/.pth` files (already in `.gitignore`).
- `git pull` before you start. Work on a branch: `git checkout -b feature/your-task`.
- `git add .` -> `git commit -m "message"` -> `git push -u origin feature/your-task` -> open a Pull Request on GitHub.
