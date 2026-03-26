from ultralytics import YOLO
from datetime import datetime
import os

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  
def train(data_yaml, subset=False):

    model = YOLO("yolov8n.pt") #n for nano-fast iteration

    model.train(
        data = data_yaml,
        epochs=10,
        imgsz=640,
        batch=16,
        device='mps',
        workers=4, #CPU threads for loading images in parallel.
        project=os.path.abspath('runs/train'),
        name=f'visdrone-nano-{_ts}',
        exist_ok=False,
        fraction=0.1 if subset else 1.0,  #fraction of dataset
        max_det=300, # cap detections per image
        conf=0.001, # filter garbage candidates before NMS
        close_mosaic=0, # keep mosaic active all 10 epochs (default=10 disables it immediately)
    )

if __name__ == "__main__":
    train(
        data_yaml="/Users/sihnat13/dev/projects/droneYOLOv8/dataset.yml",
        subset=True,
    )