import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

model = YOLO("yolov8-SlimNeck+GSConv.yaml")
model.train(data = "../autodl-tmp/ObstacleDataset/dataset/obstacle.yaml",
            cache=False,
            imgsz=640,
            epochs=100,
            batch=64,
            close_mosaic=10,
            workers=0,
            device='0',
            optimizer='SGD', # using SGD
            # resume='', # last.pt path
            # amp=False, # close amp
            # fraction=0.2,
            project='runs/train',
            name='SlimNeck+WIOU',
            multi_scale=True
            )