# Automated Crowd Density Estimation & Management System

A computer-vision system that estimates crowd density from CCTV footage or images, locates high-density zones, and raises automated alerts. Developed as a group project (Group 8).

## 1. Problem Statement

Traditional CCTV-based crowd monitoring relies on continuous human observation. Operators become fatigued, overall head counts do not show *where* people are concentrated, and heavy occlusion in dense crowds reduces the accuracy of conventional detectors. As a result, critical crowd conditions may be noticed too late for preventive action.

## 2. Objectives

1. **Count** – estimate the number of people in a frame.
2. **Detect** – identify dense zones inside the camera view.
3. **Visualize** – generate density heat maps.
4. **Alert** – flag critical zones with a visual cue, an audio alarm and a logged event.

## 3. Proposed Approach

The system is an adaptive hybrid pipeline:

```
Video/Image -> Preprocessing -> Smart Density Switch -> Spatial Grid -> Dashboard + Alarm + Event Log
```

| Stage | Description |
|---|---|
| Preprocessing | Gaussian noise reduction, Canny edge detection, and MOG2 background subtraction (video) to separate moving people from static background. |
| Smart Density Switch | A threshold *N* selects the counting method. Sparse scenes use **YOLOv8** person detection; dense scenes use a **U-Net density map**, which does not depend on detecting every individual body. |
| Spatial Grid | The view is divided into a 3x3 or 4x4 grid. Density (people/m²) is computed per cell, and only the cells above the capacity limit are marked as overcrowded. |
| Dashboard and Alerts | A Streamlit dashboard shows the heat map and zone table. Overcrowded zones turn red, an audio alarm plays, and each event is logged with timestamp and camera ID. |

If trained U-Net weights are not available, the system falls back to tiled YOLO detection so that it remains usable.

## 4. Technology Stack

- Python 3.11
- OpenCV, NumPy, Pandas, SciPy, Matplotlib
- PyTorch and Ultralytics YOLOv8
- Streamlit (dashboard)

## 5. Project Structure

```
.
├── app.py                  # Streamlit dashboard (entry point)
├── train_unet.py           # U-Net training script
├── requirements.txt
├── src/
│   ├── config.py           # Default settings
│   ├── preprocessing.py    # Filtering, edge detection, background subtraction
│   ├── detector_yolo.py    # YOLOv8 detection and tiled detection
│   ├── unet_model.py       # U-Net architecture
│   ├── density_unet.py     # Density estimation and model loading
│   ├── switch.py           # Smart density switch
│   ├── grid.py             # Grid analysis and heat map overlay
│   ├── alerts.py           # Alarm sound and event logging
│   └── pipeline.py         # Combines all stages
├── tests/smoke_test.py     # Logic test that needs no trained model
├── models/                 # Trained weights (not tracked by git)
├── data/                   # Datasets (not tracked by git)
└── samples/                # Sample images and videos
```

## 6. Installation

Requires Python 3.11 and Git.

```bash
git clone https://github.com/Kavya31-beep/Crowd-Density-Estimation-Monitoring-System.git
cd Crowd-Density-Estimation-Monitoring-System

python -m venv venv
# Windows (PowerShell):  .\venv\Scripts\Activate.ps1
# Linux / macOS:         source venv/bin/activate

pip install -r requirements.txt
python tests/smoke_test.py
```

The smoke test should finish with `SMOKE TEST PASSED`.

## 7. Usage

```bash
streamlit run app.py
```

Open `http://localhost:8501`, then upload an image or a video. The sidebar provides these settings:

| Setting | Meaning |
|---|---|
| Detection mode | Automatic switch, force YOLO, or force density map |
| Switch threshold N | Headcount at which the system changes method |
| Grid | 3x3 or 4x4 zones |
| Area per grid cell (m²) | Real-world area covered by one cell (depends on the camera) |
| Capacity limit (people/m²) | Density above which a zone is flagged (default 4) |
| YOLO confidence / tiling | Detection sensitivity; tiling helps find small or distant people |

Each alert is appended to `logs/events.csv` with timestamp, camera ID, source, method, total count, overcrowded zones and maximum density. The log can also be viewed and downloaded from the **Event log** tab.

## 8. Training the Density Model

The U-Net is trained on the ShanghaiTech crowd counting dataset (Part A recommended for dense scenes). A CUDA-enabled GPU is advised.

1. Install the CUDA build of PyTorch from <https://pytorch.org>, then run `pip install -r requirements.txt`.
2. Place the dataset in `data/ShanghaiTech/part_A` (it must contain `train_data` and `test_data`).
3. Run:
   ```bash
   python train_unet.py --data data/ShanghaiTech/part_A --epochs 200
   ```
4. The best model (lowest test MAE) is saved to `models/unet_density.pth`. Copy this file into the `models/` folder of every machine that will run the dashboard. Model weights are not committed to the repository because of their size.

## 9. Results

Fill in after training and testing.

| Item | Value |
|---|---|
| Dataset | ShanghaiTech Part A |
| Epochs | |
| Test MAE | |
| Test hardware | |

## 10. Limitations

- Pretrained YOLO detects people poorly in overhead views and in very dense crowds, which is why the density model is used for such scenes.
- The density model is trained from scratch on a single dataset, so accuracy drops on camera angles and crowd types unlike the training data.
- Density in people/m² depends on the configured cell area, which must be set for each camera. The system does not calibrate camera geometry automatically.
- Heat map and zone values are estimates intended for early warning, not exact counts.

## 11. Future Work

- Camera calibration to convert pixels to real-world area automatically.
- Live RTSP/CCTV stream input and multi-camera support.
- Fine-tuning on locally collected footage.
- SMS or e-mail alerts to control-room staff.

## 12. Team (Group 8)

- Aardra M R
- Anusree S
- Kavya K J
- Meenakshi S
- Parvathy V Nair
- Roshna Jayan

## 13. Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- ShanghaiTech crowd counting dataset (Zhang et al., CVPR 2016)
- [Streamlit](https://streamlit.io), [OpenCV](https://opencv.org), [PyTorch](https://pytorch.org)
