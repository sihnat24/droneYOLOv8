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
        imgsz=1280, #higher improves obj detecetion but multiplies mem and compute
        batch=8, #images per grad update (reduced from 16 for imgsz=1280)
        device='mps',
        workers=4, #CPU threads for loading images in parallel.
        project='runs/train',
        name=f'visdrone-nano-{_ts}',
        exist_ok=False,
        fraction=0.1 if subset else 1.0,  #fraction of dataset 
        max_det=300, # cap detections per image
        conf=0.001, # filter garbage candidates before NMS
        flipud=0.5, # aerial has no canonical up/down orientation
    )

if __name__ == "__main__":
    train(
        data_yaml="/Users/sihnat13/dev/projects/droneYOLOv8/dataset.yml",
        subset=True,
    )