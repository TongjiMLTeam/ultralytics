import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(r'/home/zgjgroup/wwd/ultralytics/ultralytics/cfg/models/v8/yolov8-SlimNeck+GSConv.yaml')
    # model.load(r'E:\BaiduNetdiskDownload\yolov8-main\dataset\WiderPerson_yo\best.pt') # loading pretrain weights
    model = YOLO(r'/home/zgjgroup/wwd/ultralytics/yolov8-SlimNeck+GSConv+GAM.yaml')

    model.train(data=r'/home/zgjgroup/wwd/ultralytics/ObstacleDataset/ObstacleDataset/dataset/obstacle.yaml',
                cache=False,
                imgsz=640,
                epochs=250,
                batch=64,
                close_mosaic=10,
                workers=0,
                device='1',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='SlimNeck+GAM+WIOU',
                multi_scale=True

                )