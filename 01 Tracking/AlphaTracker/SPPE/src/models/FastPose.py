import torch.nn as nn
from torch.autograd import Variable

from .layers.SE_Resnet import SEResnet
from .layers.DUC import DUC
from opt import opt
from .layers.MobileNetV2 import MobileNet


def createModel():
    return FastPose()
    # return FastPose_Mobile()


class FastPose(nn.Module):
    DIM = 128

    def __init__(self):
        super(FastPose, self).__init__()

        self.preact = SEResnet('resnet101')

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(
            self.DIM, opt.nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Variable):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        return out


class FastPose_Mobile(nn.Module):
    conv_dim = 80

    def __init__(self):
        super(FastPose_Mobile, self).__init__()

        self.preact = MobileNet()
        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(320, 640, upscale_factor=2)
        self.duc2 = DUC(160, 320, upscale_factor=2)

        self.conv_out = nn.Conv2d(
            self.conv_dim, opt.nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        return out
