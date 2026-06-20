"""
Live inference with bounding boxes.

Sources (swap via --source):
  0, 1, 2 ...         webcam device index
  rtsp://...          RTSP stream
  http://...          MJPEG/HTTP stream
  path/to/video.mp4   video file

Usage:
  python infer.py --model best.pt --source 0
  python infer.py --model best.pt --source rtsp://192.168.1.10/stream
  python infer.py --model best.pt --source footage.mp4 --conf 0.3
"""

import argparse
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Source abstraction
# ---------------------------------------------------------------------------

class FrameSource(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return next BGR frame, or None if stream ended."""
        ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


class CVSource(FrameSource):
    """Wraps any OpenCV-compatible source: webcam index, RTSP URL, video file."""

    def __init__(self, source):
        # Convert numeric strings to int so cv2 treats them as device indices
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self._source = source
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self._source!r}")

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        if self._cap:
            self._cap.release()

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_boxes(frame: np.ndarray, results, class_names: list[str]) -> np.ndarray:
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run(
    model_path: str,
    source,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    show: bool = True,
    window_name: str = "DroneYOLO",
) -> None:
    model = YOLO(model_path)
    class_names = list(model.names.values())

    with CVSource(source) as src:
        prev_time = time.time()
        while True:
            frame = src.read()
            if frame is None:
                break

            results = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
            )

            if show:
                annotated = draw_boxes(frame.copy(), results, class_names)

                # FPS overlay
                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 live inference")
    parser.add_argument("model", nargs="?", default="best.pt", help="Path to .pt or .onnx model")
    parser.add_argument("--source", default="0", help="Webcam index, RTSP URL, or video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--no-show", action="store_true", help="Suppress display window")
    args = parser.parse_args()

    run(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        show=not args.no_show,
    )
