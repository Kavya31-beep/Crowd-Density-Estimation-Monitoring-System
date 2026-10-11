"""YOLOv8 person detection and counting (best for sparse / moderate crowds)."""
import cv2
import numpy as np


class YoloCounter:
    def __init__(self, weights="yolov8n.pt", conf=0.25):
        from ultralytics import YOLO  # imported here so the rest of the code works without it
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame, imgsz=640):
        """Returns an (N, 4) array of person boxes [x1, y1, x2, y2]."""
        res = self.model.predict(frame, classes=[0], conf=self.conf, imgsz=imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4))
        return res.boxes.xyxy.cpu().numpy()

    def detect_tiled(self, frame, tiles=3, imgsz=640):
        """Splits the frame into tiles x tiles pieces so small, distant people are still found."""
        h, w = frame.shape[:2]
        th, tw = h // tiles, w // tiles
        found = []
        for i in range(tiles):
            for j in range(tiles):
                y0, x0 = i * th, j * tw
                y1 = h if i == tiles - 1 else y0 + th
                x1 = w if j == tiles - 1 else x0 + tw
                b = self.detect(frame[y0:y1, x0:x1], imgsz)
                if len(b):
                    found.append(b + np.array([x0, y0, x0, y0]))
        if not found:
            return np.zeros((0, 4))
        return self._nms(np.vstack(found))

    @staticmethod
    def _nms(boxes, thr=0.5):
        xywh = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in boxes]
        idx = np.array(cv2.dnn.NMSBoxes(xywh, [1.0] * len(xywh), 0.0, thr)).flatten()
        return boxes[idx] if len(idx) else np.zeros((0, 4))
