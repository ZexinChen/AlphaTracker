import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import numpy as np
from SPPE.src.utils.img import flip_v, shuffleLR
from SPPE.src.utils.eval import getPrediction
from SPPE.src.models.FastPose import createModel

import visdom
import time
import sys

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class InferenNet(nn.Module):
    def __init__(self, kernel_size, dataset, opt):
        super(InferenNet, self).__init__()
        self.opt = opt

        model = createModel().cuda()
        print('Loading pose model from {}'.format(opt.pose_model_path))
        sys.stdout.flush()
        model.load_state_dict(torch.load(opt.pose_model_path))
        model.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, self.opt.nClasses)

        flip_out = self.pyranet(flip_v(x))
        flip_out = flip_out.narrow(1, 0, self.opt.nClasses)
        
        flip_out = flip_v(shuffleLR(
            flip_out, self.dataset))

        out = (flip_out + out) / 2

        return out


class InferenNet_fast(nn.Module):
    def __init__(self, kernel_size, dataset, opt):
        super(InferenNet_fast, self).__init__()

        model = createModel().cuda()
        mobile_pth = opt.pose_model_path
        print('Loading pose model from {}'.format(mobile_pth))
        model.load_state_dict(torch.load(mobile_pth))

        model.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        return out
