import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 模型配置文件ultralyticsPro-YOLOv12
model_yaml_path = r'/root/ultralyticsPro-YOLOv12/ultralytics/cfg_yolov12/YOLOV12-EGAVF.yaml'
#数据集配置文件
data_yaml_path = r'/root/autodl-tmp/yolo_dataset/data.yaml'

#只在当前文件执行
if __name__ == '__main__':
	#加载模型
    model = YOLO(model_yaml_path)
    #训练模型
    model.train(data=data_yaml_path,
                          imgsz=800,
                          epochs=200,
                          batch=8,
                          device='0',
                          workers=4,
                          optimizer='AdamW',
                          lr0=0.001,
                          amp=False,  # 如果出现训练损失为Nan可以关闭amp
                          project='runs/train',
                          name='exp',
                          )