#--just practice--
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import collections
import numpy as np


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, weight, bias):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        print(input.dtype, self.weight.t().dtype)
        output = input.matmul(self.weight.t())
        output += self.bias
        return output


class Conv2DLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 weight,
                 bias,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1) -> None:
        super(Conv2DLayer, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def max_pool2d(input, kernel_size, stride, padding):
    # print("input", input.dtype)
    input = input.numpy()
    N, C, H, W = input.shape
    pool_w, pool_h = kernel_size
    stride_w, stride_h = stride
    out_h = int(1 + (H - pool_h) / stride_h)
    out_w = int(1 + (W - pool_w) / stride_w)

    col = im2col(input, pool_h, pool_w, stride, padding)
    col = col.astype(np.float32)
    col = col.reshape(-1, pool_h * pool_w)

    out = np.max(col, axis=1)
    out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    return torch.from_numpy(out)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    stride_w, stride_h = stride
    out_h = (H + 2*pad - filter_h)//stride_h + 1
    out_w = (W + 2*pad - filter_w)//stride_w + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride_h*out_h
        for x in range(filter_w):
            x_max = x + stride_w*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride_h, x:x_max:stride_w]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


