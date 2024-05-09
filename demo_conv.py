import torch.nn as nn
from layers.Layer import *
import tensorflow as tf
import numpy as np

class Conv2DDemo(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, weight, bias, stride=1, padding=(1,1)):
        super(Conv2DDemo, self).__init__()
        self.conv1 = Conv2DLayer(in_channel, out_channel, kernel_size, weight, bias, stride, padding)
        

    def forward(self, input_data):
        print("input_data ", input_data.dtype)
        x = self.conv1(input_data)
        print("conv1 ", x.shape, x.dtype)
        return x
    


model_path = 'model/model_383_3.tflite'
input_h  = 383
input_w  = 3
kernel_c = 1
input_dtype=np.int8

interpreter = tf.lite.Interpreter(model_path=model_path, experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

weight_tensor = torch.from_numpy(np.transpose(interpreter.tensor(5)(),(0, 3, 1, 2))).to(torch.int32)
bias_tensor = torch.from_numpy(interpreter.tensor(4)()).to(torch.int32)
input_data = np.random.randint(-128, 127, (1, kernel_c, input_h, input_w), dtype=input_dtype)
input_tensor = torch.from_numpy(input_data).to(torch.int32)

model = Conv2DDemo(in_channel=1, out_channel=kernel_c, kernel_size=(3,3), weight=weight_tensor, bias=bias_tensor, stride=2)

out = model(input_tensor)
out_int8 = out.clamp(torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max).to(torch.int8)

