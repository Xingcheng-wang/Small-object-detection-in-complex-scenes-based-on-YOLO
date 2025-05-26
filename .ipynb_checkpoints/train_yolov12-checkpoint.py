import sys
import argparse
import os

# https://www.codewithgpu.com/u/iscyy

# sys.path.append('/root/ultralyticsPro/') # Path

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg 
    weights = opt.weights 
    model = YOLO(yaml) # 直接加载yaml文件训练
    # model = YOLO(weights)  # 直接加载权重文件进行训练
    # model = YOLO(yaml).load(weights) # 加载yaml配置文件的同时，加载权重进行训练

    # print(model)
    model.info()

    results = model.train(data='/root/autodl-tmp/yolo_dataset/data.yaml',  # 训练参数均可以重新设置
                        epochs=300, 
                        imgsz=640, 
                        workers=2, 
                        batch=2,
                        # ...在这里添加需要修改的参数
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=r'/root/ultralyticsPro-YOLOv12/ultralytics/cfg/models/v12/yolov12.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='yolov12n.pt', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)