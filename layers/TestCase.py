#--just practice--
import torch.nn as nn
import torch.nn.functional as F
import torch
from Layer import *


class Conv2DTesting(nn.Module):
    def __init__(self, input_channels, mid_channels, output_channels):
        super(Conv2DTesting, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=5, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(mid_channels, output_channels, kernel_size=5)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, input_data):
        x = self.conv1(input_data)
        print("conv1", x.shape)
        y = self.conv2(x)
        print("conv2", y.shape)
        return y
    
def main():
    input = torch.randn(1, 1, 5, 1)
    print(input)
    max_pool2d(input, kernel_size=(1,4), stride=(1,4), padding=0)

    # model = Conv2DTesting(input_channels=3, mid_channels=5, output_channels=2)
    # input_data = torch.randn(1, 3, 10, 11)
    # a_real = model(input_data)
    #
    # param_list = collections.OrderedDict()
    # for name, param in model.named_parameters():
    #     param_list[name] = param
    #     print(name ,param.shape)
    #     print('------------------')
    #
    # weight_file0 = open('/Users/huanghuangtao/Desktop/weight_0.txt', 'w')
    # # weight_file1 = open('/Users/huanghuangtao/Desktop/weight_1.txt', 'w')
    #
    # Conv0 = Conv2DLayer(in_channels=3, out_channels=5, kernel_size=5, padding=2, weight=param_list['conv1.weight'], bias=param_list['conv1.bias'])
    # Conv1 = Conv2DLayer(in_channels=5, out_channels=2, kernel_size=5, weight=param_list['conv2.weight'], bias= param_list['conv2.bias'])
    # a_ = Conv0(input_data)
    # a = Conv1(a_)
    #
    # print(a_real.shape, file=weight_file0)
    # print(a_real, file=weight_file0)
    # print('-------------------', file=weight_file0)
    # print(a.shape, file=weight_file0)
    # print(a, file=weight_file0)