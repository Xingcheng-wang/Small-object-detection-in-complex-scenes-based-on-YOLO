# 目录
- 基本介绍
- 实验环境
- 项目结构
- 实验步骤
- 实验结果分析
- 开源许可证
----
# 基本介绍
《基于深度学习的小目标检测研究》项目是西安电子科技大学空间科学与技术学院本科生毕业设计。项目基于YOLO算法，针对复杂场景小目标检测任务进行算法改进。在两个改进方向上分别进行了多次实验。
- 特征增强策略：SuperYOLO在VEDAI数据集上的mAP50指标，相比于改进前提升==**12.44%**==
- 网络架构优化：LSKV-YOLO在AITOD数据集上的mAP50指标，相比于改进前提升==**4.83%**==
---
# 实验环境
### **SuperYOLO**
super-resolution reconstructionYOLO：多模态-超分辨率辅助YOLO
硬件环境：[AutoDL算力云](https://www.autodl.com/home)

| 项目  | 名称                              |
| --- | ------------------------------- |
| CPU | Intel(R) Xeon(R) Platinum 8474C |
| GPU | RTX 4090D                       |
| 显存  | 24GB                            |
| 内存  | 80GB                            |

软件环境：requirements.txt
```txt
absl-py==0.13.0
anyio==3.3.1
argon2-cffi==21.1.0
attrs==21.2.0
Babel==2.9.1
backcall==0.2.0
bleach==4.1.0
brotlipy==0.7.0
cachetools==4.2.2
certifi==2021.5.30
cffi @ file:///tmp/build/80754af9/cffi_1625807838443/work
chardet @ file:///tmp/build/80754af9/chardet_1607706746162/work
conda==4.10.3
conda-package-handling @ file:///tmp/build/80754af9/conda-package-handling_1618262148928/work
cryptography @ file:///tmp/build/80754af9/cryptography_1616769286105/work
cycler==0.10.0
dataclasses==0.6
debugpy==1.4.3
decorator==5.1.0
defusedxml==0.7.1
entrypoints==0.3
filelock==3.16.1
fsspec==2025.3.0
future==0.18.2
google-auth==1.35.0
google-auth-oauthlib==0.4.6
grpcio==1.40.0
hf-xet==1.1.0
huggingface-hub==0.31.1
idna @ file:///home/linux1/recipes/ci/idna_1610986105248/work
importlib-metadata==8.5.0
ipykernel==6.4.1
ipython==7.27.0
ipython-genutils==0.2.0
ipywidgets==7.6.5
jedi==0.18.0
Jinja2==3.0.1
json5==0.9.6
jsonschema==3.2.0
jupyter-client==7.0.3
jupyter-core==4.8.1
jupyter-server==1.11.0
jupyterlab==3.1.12
jupyterlab-language-pack-zh-CN @ http://autodl-public.ks3-cn-beijing.ksyun.com/instance/jupyterlab_language_pack_zh_CN-0.0.1.dev0-py2.py3-none-any.whl
jupyterlab-pygments==0.1.2
jupyterlab-server==2.8.1
jupyterlab-widgets==1.0.2
kiwisolver==1.3.2
llvmlite==0.41.1
Markdown==3.3.4
MarkupSafe==2.0.1
matplotlib==3.4.3
matplotlib-inline==0.1.3
mistune==0.8.4
nbclassic==0.3.2
nbclient==0.5.4
nbconvert==6.1.0
nbformat==5.1.3
nest-asyncio==1.5.1
notebook==6.4.4
numba==0.58.1
numpy==1.22.0
oauthlib==3.1.1
opencv-python==4.11.0.86
packaging==21.0
pandas==2.0.3
pandocfilters==1.5.0
parso==0.8.2
pexpect==4.8.0
pickleshare==0.7.5
Pillow==8.3.2
prometheus-client==0.11.0
prompt-toolkit==3.0.20
protobuf==3.18.0
ptyprocess==0.7.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycocotools==2.0.7
pycosat==0.6.3
pycparser @ file:///tmp/build/80754af9/pycparser_1594388511720/work
Pygments==2.10.0
pyOpenSSL @ file:///tmp/build/80754af9/pyopenssl_1608057966937/work
pyparsing==2.4.7
pyrsistent==0.18.0
PySocks @ file:///tmp/build/80754af9/pysocks_1605305779399/work
python-dateutil==2.8.2
pytz==2021.1
PyYAML==6.0.2
pyzmq==22.3.0
requests @ file:///tmp/build/80754af9/requests_1608241421344/work
requests-oauthlib==1.3.0
requests-unixsocket==0.2.0
rsa==4.7.2
ruamel-yaml-conda @ file:///tmp/build/80754af9/ruamel_yaml_1616016699510/work
safetensors==0.5.3
scipy==1.10.1
seaborn==0.13.2
Send2Trash==1.8.0
six @ file:///tmp/build/80754af9/six_1623709665295/work
sniffio==1.2.0
supervisor==4.2.2
tensorboard==2.6.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
terminado==0.12.1
testpath==0.5.0
thop==0.0.31.post2005241907
timm==1.0.15
torch @ http://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp38-cp38-linux_x86_64.whl
torchvision @ http://download.pytorch.org/whl/cu110/torchvision-0.8.1%2Bcu110-cp38-cp38-linux_x86_64.whl
tornado==6.1
tqdm @ file:///tmp/build/80754af9/tqdm_1625563689033/work
traitlets==5.1.0
typing-extensions==3.10.0.2
tzdata==2025.2
urllib3 @ file:///tmp/build/80754af9/urllib3_1625084269274/work
wcwidth==0.2.5
webencodings==0.5.1
websocket-client==1.2.1
Werkzeug==2.0.1
widgetsnbextension==3.5.1
XlsxWriter==3.2.3
zipp==3.20.2
```
### **LSKV-YOLO**
Large Selective Kernel Varifocal Loss YOLO：动态感受野-变焦距损失YOLO
硬件环境：[AutoDL算力云](https://www.autodl.com/home)

| 项目  | 名称                               |
| --- | -------------------------------- |
| CPU | AMD EPYC 9754 128-Core Processor |
| GPU | RTX 4090D                        |
| 显存  | 24GB                             |
| 内存  | 80GB                             |
软件环境：requirements.txt
```txt
addict==2.4.0
aiofiles==23.2.1
albucore==0.0.23
albumentations==2.0.4
aliyun-python-sdk-core==2.16.0
aliyun-python-sdk-kms==2.16.5
annotated-types==0.7.0
anyio==4.9.0
certifi==2025.4.26
cffi==1.17.1
charset-normalizer==3.4.2
click==8.1.8
colorama==0.4.6
coloredlogs==15.0.1
contourpy==1.3.2
crcmod==1.7
cryptography==44.0.3
cycler==0.12.1
dill==0.4.0
einops==0.8.1
fastapi==0.115.12
ffmpy==0.5.0
filelock==3.14.0
flatbuffers==25.2.10
fonttools==4.57.0
fsspec==2025.3.2
gradio==4.44.1
gradio_client==1.3.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.30.2
humanfriendly==10.0
idna==3.10
importlib_resources==6.5.2
Jinja2==3.1.6
jmespath==0.10.0
kiwisolver==1.4.8
Markdown==3.8
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.10.1
mdurl==0.1.2
mmcv==2.2.0
mmengine==0.10.7
model-index==0.1.11
mpmath==1.3.0
networkx==3.4.2
numpy==1.26.4
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
onnx==1.14.0
onnxruntime==1.15.1
onnxruntime-gpu==1.18.0
onnxslim==0.1.31
opencv-python==4.9.0.80
opencv-python-headless==4.11.0.86
opendatalab==0.0.10
openmim==0.3.9
openxlab==0.1.2
ordered-set==4.1.0
orjson==3.10.18
oss2==2.17.0
packaging==24.2
pandas==2.2.3
pillow==11.2.1
platformdirs==4.3.8
protobuf==6.30.2
psutil==5.9.8
py-cpuinfo==9.0.0
pycocotools==2.0.7
pycparser==2.22
pycryptodome==3.22.0
pydantic==2.11.4
pydantic_core==2.33.2
pydub==0.25.1
Pygments==2.19.1
pyparsing==3.2.3
python-dateutil==2.9.0.post0
python-multipart==0.0.20
pytorch-wavelets==1.3.0
pytz==2023.4
PyWavelets==1.8.0
PyYAML==6.0.2
requests==2.28.2
rich==13.4.2
ruff==0.11.8
safetensors==0.5.3
scipy==1.13.0
seaborn==0.13.2
semantic-version==2.10.0
shellingham==1.5.4
simsimd==6.2.1
six==1.17.0
sniffio==1.3.1
starlette==0.46.2
stringzilla==3.12.5
sympy==1.13.1
tabulate==0.9.0
termcolor==3.1.0
timm==0.6.7
tomlkit==0.12.0
torch==2.6.0
torchaudio==2.6.0
torchvision==0.21.0
tqdm==4.65.2
triton==3.2.0
typer==0.15.3
typing-inspection==0.4.0
typing_extensions==4.13.2
tzdata==2025.2
# Editable install with no version control (ultralytics==8.3.13)
-e /root/ultralyticsPro-YOLOv12
ultralytics-thop==2.0.14
urllib3==1.26.20
uvicorn==0.34.2
websockets==12.0
yapf==0.43.0

```
---
# 项目结构
### **SpuerYOLO**
```bash
├─data
│  └─images
├─dataset
│  ├─VEDAI
│  │  ├─images
│  │  └─labels
│  └─VEDAI_1024
│      ├─images
│      └─labels
├─Fig
├─models
├─utils
└─weights
```

### **LSKV-YOLO**
```
├─docker
├─docs
├─examples
├─tests
└─ultralytics
    ├─assets
    ├─cfg
    ├─cfg_yolov12
    ├─data
    ├─engine
    ├─hub
    ├─models
    ├─nn
    ├─script
    ├─solutions
    ├─trackers
    └─utils
```
---
# 实验步骤
- 配置环境依赖
- 配置数据集
- 配置YOLO
- 训练
- 验证
- 结果分析
----
## 配置环境依赖
软件依赖配置命令：详情阅读requirements.txt，此处使用清华镜像源安装
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
其他可能用到的命令：Bash和Conda基础命令
```bash
ls 查看当下所有目录
pwd 查看当前工作位置
mkdir dirname 新建工作目录
cd Path 工作目录转到工作下
conda create -n envname python=x.xx 创建名字为envname的环境
conda info -e 查看所有的环境
conda init 环境初始化
conda activate envname 激活envname环境
conda list 查看该环境中的依赖项
```
---
## 配置数据集
将自制的或者是公开的数据集**转化**成YOLO格式，并书写yaml格式的数据集**配置**。以下代码用于展示，需自行修改使用。
##### 格式转化
因人而异，常见的数据集格式有三种：
- YOLO格式
- COCO格式
- VOC格式
YOLO格式内容如下：
```
#class    xcenter      ycenter        width        height
3 0.567087833984375 0.985569609375 0.052734375 0.025390625
```
YOLO格式数据集文件结构如下：
```bash
├─images
│  ├─test
│  ├─train
│  └─val
└─labels
    ├─test
    ├─train
    └─val
```

VEDAI数据集是YOLO格式，不需要转化
需要将AITOD数据集（COCO格式）转化成YOLO格式，命令如下：
```bash
python coco2yolo.py
```
##### 数据集配置（数据增强）
```yaml
path: /root/autodl-tmp/yolo_dataset/data.yaml
train: /root/autodl-tmp/yolo_dataset/images/train
val: /root/autodl-tmp/yolo_dataset/images/val
test: /root/autodl-tmp/yolo_dataset/images/test

nc: 9  # 类别数量
names:
  0: none
  1: airplane
  2: bridge
  3: storage-tank
  4: ship
  5: swimming-pool
  6: vehicle
  7: person
  8: wind-mill

augment:
  flipud: 0.5      # 50% 概率进行垂直翻转
  fliplr: 0.5      # 50% 概率进行水平翻转
  mosaic: 1.0      # 启用 Mosaic 数据增强
  mixup: 0.5       # 启用 Mixup 数据增强
  hsv_h: 0.015     # 色调增强，范围为 ±0.015
  hsv_s: 0.7       # 饱和度增强，范围为 ±0.7
  hsv_v: 0.4       # 亮度增强，范围为 ±0.4
  scale: 0.5       # 随机缩放，范围为 ±50%
  shear: 0.0       # 随机剪切，设置为 0 禁用
  perspective: 0.0 # 随机透视变换，设置为 0 禁用
```
---
## 配置YOLO
配置YOLO网络结构 ，修改YOLO源代码
##### 网络结构
模型参数，可修改类别，基础模型规模
```yaml
# Parameters
nc: 80 # 类别数量
scales: # 模型尺寸nsmlx五个类型 model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'

  # [depth, width, max_channels]
  # 深度 宽度 最大通道数
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs
```

主干部分，可修改组件名称和组件需要的参数
```yaml
# YOLO11n backbone
backbone:
  # 输入  重复次数  组件名称  参数[输入通道数 卷积核大小 步长]
  # -1表示从上一层输入
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2 第一特征图是输入1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4 第二特征图是输入1/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8 第三特征图是输入1/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10
```

颈部和头部，可修改多尺度策略，检测头
```yaml
  # YOLO11n head
  # 输入  重复次数  组件名称  参数[输入通道数 卷积核大小 步长]
  # [from, repeats, module, args]
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13
 
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)
 
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)
 
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)
 
  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
  #nn.Upsample上采样
  #Concat拼接
```
##### 代码修改
添加YOLO组件的一种方法：
- 核心代码参考论文，或者[Papers With Code](https://paperswithcode.com/)，或者开源社区
- ultralytics/nn/modules/conv.py第一步添加核心代码到这或者导入到这
- ultralytics/nn/modules/conv.py第二步修改__all__内容
- ultralytics/nn/modules/__init__.py第三步从conv导入模块
- ultralytics/nn/modules/__init__.py第四步修改__all__内容
- ultralytics/nn/tasks.py 第五步导入模块
- ultralytics/nn/tasks.py 第六步修改task下==**parse_model**==函数加入模块
- 根据函数参数的不同，选择不同的elseif分支
---
## 训练
- 修改路径，选用模型配置文件和数据集配置文件
- 可修改超参数，不修改则使用默认参数
- 执行`python train.py`开始训练
- 分析训练日志，反复调节超参数
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 模型配置文件ultralyticsPro-YOLOv12
model_yaml_path = r'/root/ultralyticsPro-YOLOv12/ultralytics/cfg/models/v12/yolov12.yaml'
#数据集配置文件
data_yaml_path = r'/root/autodl-tmp/yolo_dataset/data.yaml'

#只在当前文件执行
if __name__ == '__main__':
	#加载模型
    model = YOLO(model_yaml_path)
	model.load('yolov12n.pt')
    #训练模型
    model.train(data=data_yaml_path,
                          imgsz=800,
                          epochs=200,
                          batch=16,
                          device='0',
                          workers=4,
                          optimizer='AdamW',
                          lr0=0.001,
                          amp=False,  # 如果出现训练损失为Nan可以关闭amp
                          project='runs/train',
                          name='exp',
                          )
```
## 验证
- 一般执行完训练会自动执行一次验证
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model_path = 'runs/train/exp/weights/best.pt'
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
```
---
# 实验结果分析
### 常用指标

| 指标                                  | 作用                                             |
| :---------------------------------- | :--------------------------------------------- |
| `精确率P（Precision）`                   | 所有检测为正样本的结果中，正确检测的比例，反映检测的准确性                  |
| `召回率R（Recall）`                      | 所有实际正样本中，被正确检测出的比例，反映检测的完整性                    |
| `交并比IoU（Intersection over Union）​​` | 预测框与真实框的重叠面积占两者并集面积的比例，用于衡量定位精度                |
| `平均精度AP（Average Precision）`         | 在不同召回率阈值下的平均精确率，对应 Precision-Recall（PR）曲线下的面积  |
| `平均精度均值mAP（mean Average Precision）` | 所有类别的 AP 的平均值，用于多类别检测任务的整体评估                   |
| `mAP@0.5​​`                         | IoU 阈值为 0.5 时的 mAP                             |
| `mAP@0.5:0.95​​`                    | IoU 阈值从 0.5 到 0.95（步长 0.05）的平均 mAP，更严格评估模型定位能力 |
| `​​F1 分数（F1-Score）`                 | 精确率与召回率的调和平均数，平衡两者的权衡关系                        |
| `FPS（Frames Per Second）`            | 每秒处理的图像帧数，衡量实时性                                |
| `混淆矩阵`                              | 展示 TP、FP、FN、TN 的矩阵，辅助分析模型误检和漏检情况               |
| `params (M)`                        | 参数量，百万级                                        |
| `FLOPs (G)`                         | 每秒浮点运算次数。G表示10亿，可以用来衡量算法/模型复杂度，理论上该数值越高越好      |

### 常见问题和解决方法
以下是常见问题，确诊问题只需看指标的异常行为
- 欠拟合
	- 增加轮数
	- 增加学习率
- 过拟合
	- 减小学习率
	- 使用数据增强
	- 增加轮数
- 显存爆炸
	- 减少线程
	- 减少批大小
- 内存不足
	- 扩容
- 梯度爆炸
	- 减少线程
	- 禁用amp
### 训练中日志分析
分析运行`train.py`运行中的结果，主要内容是实时的预测值和损失值。可以及时止损，节约时间
### 训练后结果分析
分析`runs/train/exp`中的结果，主要内容是训练后的训练过程记录，超参数记录，最终训练结果，测试结果
# 开源许可证
                    GNU AFFERO GENERAL PUBLIC LICENSE
                       Version 3, 19 November 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU Affero General Public License is a free, copyleft license for
software and other kinds of works, specifically designed to ensure
cooperation with the community in the case of network server software.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
our General Public Licenses are intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  Developers that use our General Public Licenses protect your rights
with two steps: (1) assert copyright on the software, and (2) offer
you this License which gives you legal permission to copy, distribute
and/or modify the software.

  A secondary benefit of defending all users' freedom is that
improvements made in alternate versions of the program, if they
receive widespread use, become available for other developers to
incorporate.  Many developers of free software are heartened and
encouraged by the resulting cooperation.  However, in the case of
software used on network servers, this result may fail to come about.
The GNU General Public License permits making a modified version and
letting the public access it on a server without ever releasing its
source code to the public.

  The GNU Affero General Public License is designed specifically to
ensure that, in such cases, the modified source code becomes available
to the community.  It requires the operator of a network server to
provide the source code of the modified version running there to the
users of that server.  Therefore, public use of a modified version, on
a publicly accessible server, gives the public access to the source
code of the modified version.

  An older license, called the Affero General Public License and
published by Affero, was designed to accomplish similar goals.  This is
a different license, not a version of the Affero GPL, but Affero has
released a new version of the Affero GPL which permits relicensing under
this license.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU Affero General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Remote Network Interaction; Use with the GNU General Public License.

  Notwithstanding any other provision of this License, if you modify the
Program, your modified version must prominently offer all users
interacting with it remotely through a computer network (if your version
supports such interaction) an opportunity to receive the Corresponding
Source of your version by providing access to the Corresponding Source
from a network server at no charge, through some standard or customary
means of facilitating copying of software.  This Corresponding Source
shall include the Corresponding Source for any work covered by version 3
of the GNU General Public License that is incorporated pursuant to the
following paragraph.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the work with which it is combined will remain governed by version
3 of the GNU General Public License.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU Affero General Public License from time to time.  Such new versions
will be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU Affero General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU Affero General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU Affero General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If your software can interact with users remotely through a computer
network, you should also make sure that it provides a way for users to
get its source.  For example, if your program is a web application, its
interface could display a "Source" link that leads users to an archive
of the code.  There are many ways you could offer source, and different
solutions will be better for different programs; see section 13 for the
specific requirements.

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU AGPL, see
<https://www.gnu.org/licenses/>.