from ultralytics import YOLO
from datetime import datetime
import os
import torch

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

def select_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = select_device()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  
def train(data_yaml, subset=False):

    model = YOLO("yolov8n.pt") #n for nano-fast iteration

    model.train(
        data = data_yaml,
        epochs=10,
        imgsz=1280,
        batch=4,
        device=device,
        workers=4,
        project=os.path.abspath('runs/train'),
        name=f'visdrone-nano-{_ts}',
        exist_ok=False,
        fraction=0.1 if subset else 1.0,
        max_det=300,
        conf=0.001,
        lr0=0.02,
        lrf=0.1,
        flipud=0.5,
        mosaic=0.0,
        degrees=0.0,
    )

if __name__ == "__main__":
    train(
        data_yaml="/Users/sihnat13/dev/projects/droneYOLOv8/dataset.yml",
        subset=True,
    )