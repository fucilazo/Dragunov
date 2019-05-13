import torch.nn as nn
import torch.nn.functional as F


class SiamRPN(nn.Module):
    def __init__(self, size=2, feature_out=512, anchor=5):
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x==3 else x*size, configs))   # [3, 192, 512, 768, 768, 512]
        feat_in = configs[-1]
        super(SiamRPN, self).__init__()
        self.featureExtract = nn.Sequential(    # torch.nn.Sequential是顺序容器。模块将按照它们在构造函数中传递的顺序添加到它中。或者，也可以传入模块的有序字典。
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),    # 卷积层
            nn.BatchNorm2d(configs[1]),                                     # 归一化
            nn.MaxPool2d(kernel_size=3, stride=2),                          # 池化层
            nn.ReLU(inplace=True),                                          # 激活函数
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )

        self.anchor = anchor
        self.feature_out = feature_out

        # conv_r1是模板分支的坐标回归，conv_cls1是其分类
        # conv_r2和conv_cls2作用于检测分支，只是普通的3x3卷积
        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)  # regress_adjust为1x1卷积，在论文中未有提及。

        self.r1_kernel = []
        self.cls1_kernel = []

        self.cfg = {}

    # torch.nn.Module.forward 定义每次调用时执行的计算。所有子类应该覆盖该函数
    def forward(self, x):
        # 坐标回归接regress_adjust1x1卷积
        x_f = self.featureExtract(x)
        return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    # 由于孪生网的两个分支在跟踪推理阶段是异步运行，所以 SiamRPNBIG 专门定义了一个 temple 函数来处理模板。
    def temple(self, z):
        # 调用featureExtract提取模板z的特征
        z_f = self.featureExtract(z)
        # 由z_f得到坐标回归量r1_kernel_raw和分类得分cls1_kernel_raw
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        # kernel_size为模板宽度
        kernel_size = r1_kernel_raw.data.size()[-1]
        # torch.Tensor.view 返回一个新的张量，其数据与自身张量相同，但大小不同。
        # 修改r1_kernel和cls1_kernel的形状
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)


class SiamRPNBIG(SiamRPN):
    def __init__(self):
        super(SiamRPNBIG, self).__init__(size=2)
        self.cfg = {'lr': 0.295, 'window_influence': 0.42, 'penalty_k': 0.055, 'instance_size': 271, 'adaptive': True} # 0.383


class SiamRPNvot(SiamRPN):
    def __init__(self):
        super(SiamRPNvot, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 271, 'adaptive': False} # 0.355


class SiamRPNotb(SiamRPN):
    def __init__(self):
        super(SiamRPNotb, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'instance_size': 271, 'adaptive': False} # 0.655
