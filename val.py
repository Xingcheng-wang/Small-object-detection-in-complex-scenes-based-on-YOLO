import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model_path = 'runs/train/exp57/weights/best.pt'
    model = YOLO(model_path) # 选择训练好的权重路径
    result = model.val(data='/root/autodl-tmp/yolo_dataset/data.yaml',
                        split='val', # split可以选择train、val、test 
                        imgsz=800,
                        batch=16,
                        # iou=0.7,
                        # rect=False,
                        # save_json=True, # if you need to cal coco metrice
                        project='runs/val',
                        name='exp',
                        )