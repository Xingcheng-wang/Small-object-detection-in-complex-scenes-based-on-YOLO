import math
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath, trunc_normal_

# https://arxiv.org/pdf/2503.14012

class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])
    def forward(self, x):
        out = self.block(x)
        return out


class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

class LFEA(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(LFEA, self).__init__()
        self.channel = channel
        # t = int(abs((log(channel, 2) + 1) / 2))
        # k = t if t % 2 else t + 1
        self.conv2d = self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer())
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        # self.sigmoid = nn.Sigmoid()
        # self.norm = build_norm_layer(norm_layer, channel)[1]

    def forward(self, c, att):
        att = c * att + c
        att = self.conv2d(att)
        # wei = self.avg_pool(att)
        # wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # wei = self.sigmoid(wei)
        # x = self.norm(c + att * wei)

        return att

class Scharr(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Scharr, self).__init__()
        # 定义Scharr滤波器
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        # 将Sobel滤波器分配给卷积层
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, channel)[1]
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel, norm_layer, act_layer)

    def forward(self, x):
        # show_feature(x)
        # 应用卷积操作
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        # 计算边缘和高斯分布强度（可以选择不同的方式进行融合，这里使用平方和开根号）
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        scharr_edge = self.act(self.norm(scharr_edge))
        out = self.conv_extra(x + scharr_edge)
        # show_feature(out)

        return out

class EGA(nn.Module):
    def __init__(self,
                 dim,
                 stage=1,
                 mlp_ratio=2.,
                 drop_path=0.1,
                 act_layer=nn.ReLU,
                 norm_layer=dict(type='BN', requires_grad=True),
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        # mlp_layer: List[nn.Module] = [
        #     nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
        #     build_norm_layer(norm_layer, mlp_hidden_dim)[1],
        #     act_layer(),
        #     nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)]

        # self.mlp = nn.Sequential(*mlp_layer)
        self.LFEA = LFEA(dim, norm_layer, act_layer)

        if stage == 0:
            self.Scharr_edge = Scharr(dim, norm_layer, act_layer)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, dim)[1]

    def forward(self, x):
        # show_feature(x)
        if self.stage == 0:
            att = self.Scharr_edge(x)
        else:
            att = self.gaussian(x)
        x_att = self.LFEA(x, att)

        return x_att


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_EGA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = EGA(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_EGA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_HSMSSD特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_EGA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


def main():
    # 实例化LFE_Module，设置参数
    dim = 64  # 特征维度
    model = EGA(dim)

    # 创建随机输入张量
    batch_size = 4
    height = 27
    width = 31
    x = torch.randn(batch_size, dim, height, width)

    # 调用模型的forward方法
    output = model(x)

    # 打印输出张量的形状
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()