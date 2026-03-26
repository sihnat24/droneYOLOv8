from ultralytics import YOLO


def train(data_yaml, subset=False):

    model = YOLO("yolov8n.pt") #n for nano-fast iteration

    model.train(
        data = data_yaml,
        epochs=10,
        imgsz=640, #higher improves obj detecetion but multiplies mem and compute
        batch=16, #images per grad update
        device='mps',
        workers=4, #CPU threads for loading images in parallel.
        project='runs/train',
        name='visdrone-nano-subset',
        exist_ok=True,
        fraction=0.1 if subset else 1.0,  #fraction of dataset 
        max_det=300, # cap detections per image
        conf=0.001, # filter garbage candidates before NMS
    )

if __name__ == "__main__":
    train(
        data_yaml="/Users/sihnat13/dev/projects/droneYOLOv8/dataset.yml",
        subset=True,
    )