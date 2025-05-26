import timm
print(timm.__version__)  # 应输出0.9.16
from timm.models.efficientnet_builder import _parse_ksize  # 确认可导入