import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from layers import wavelet
from mamba_ssm import Mamba


class WTMamba(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1', mamba_configs=None):
        super(WTMamba, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter = nn.Parameter(torch.randn(1, in_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.iwt_filter = nn.Parameter(torch.randn(1, in_channels, in_channels, kernel_size, kernel_size), requires_grad=True)

        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.mamba = Mamba(
            d_model=mamba_configs.d_model,
            d_state=mamba_configs.d_ff,
            d_conv=mamba_configs.d_conv,
            expand=mamba_configs.expand,
        )

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet.wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            curr_x_ll_reshaped = curr_x_ll.permute(0, 1, 3, 2).reshape(curr_x_ll.size(0), curr_x_ll.size(1), -1).transpose(1, 2)
            curr_x_ll_mamba = self.mamba(curr_x_ll_reshaped)

            curr_x_ll_mamba = curr_x_ll_mamba.transpose(1, 2).view(curr_x_ll.size(0), curr_x_ll.size(1), curr_x_ll.size(3), curr_x_ll.size(2)).permute(0, 1, 3, 2)

            x_ll_in_levels.append(curr_x_ll_mamba)

            curr_x_h = curr_x[:, :, 1:4, :, :]
            x_h_in_levels.append(curr_x_h)

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        if self.do_stride is not None:
            x_tag = self.do_stride(x_tag)

        return x_tag


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)
