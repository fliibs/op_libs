import torch.nn as nn
from layers.Layer import *
import tensorflow as tf
import numpy as np
import onnx
import onnxruntime

from scipy.stats import ks_2samp

class Conv2DDemo(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, weight, bias, stride=1, padding=(1,1)):
        super(Conv2DDemo, self).__init__()
        self.conv1 = Conv2DLayer(in_channel, out_channel, kernel_size, weight, bias, stride, padding)

    def forward(self, input_data):
        x = self.conv1(input_data)
        return x
    

def parse_model(model_path, input_shape, input_dtype=np.int8):
    interpreter = tf.lite.Interpreter(model_path=model_path, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_data = np.random.randint(np.iinfo(input_dtype).min, np.iinfo(input_dtype).max, input_shape, dtype=input_dtype)
    input_tensor = torch.from_numpy(input_data).to(torch.int32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    return interpreter, input_tensor


def easy_quanti(tensor):
    flat_tensor = tensor.numpy().ravel()
    max = np.max(flat_tensor)
    scale_max_floor = np.floor(max/127)
    scale_max_ceil  = np.ceil(max/127)
    min = np.min(flat_tensor)
    scale_min_floor = np.floor(min/-128)
    scale_min_ceil  = np.ceil(min/-128)
    scale_max = np.max([scale_max_floor, scale_max_ceil, scale_min_floor, scale_min_ceil])
    scale_min = np.min([scale_max_floor, scale_max_ceil, scale_min_floor, scale_min_ceil])
    scale_max = int(np.max([tranform_log2(scale_max), tranform_log2(scale_min)]))
    scale_min = int(np.min([tranform_log2(scale_max), tranform_log2(scale_min)]))
    print(scale_min, scale_max)
    return tensor//scale_min, tensor//scale_max
    

def tranform_log2(value):
    scale_exp_ceil  = 2**np.ceil(np.log2(value))
    scale_exp_floor = 2**np.floor(np.log2(value))
    return scale_exp_ceil, scale_exp_floor


def distribution_compare(tensor1, tensor2):
    flat_tensor1 = tensor1.numpy().ravel()
    flat_tensor2 = tensor2.numpy().ravel()

    ks_statistic, p_value = ks_2samp(flat_tensor1,flat_tensor2)
    print("Kolmogorov-Smirnov统计量:", ks_statistic)
    print("p值:", p_value)


def calculate_mse(original, quantized):
    original = np.array(original)
    quantized = np.array(quantized)
    squared_diff = (original - quantized) ** 2
    mse = np.mean(squared_diff)
    print("mse: ", mse)
    return mse

def calculate_nmse(original, quantized):

    original = np.array(original, dtype=np.float32)
    quantized = np.array(quantized, dtype=np.float32)
    squared_diff = (original - quantized) ** 2
    variance = np.var(original)
    nmse = np.mean(squared_diff) / variance
    print("nmse: ", nmse)
    
    return nmse

def norm_zero_score_tensor(tensor):
    tensor_float = tensor.float()
    mean_torch = tensor_float.mean()
    std_torch = tensor_float.std()
    x_standardized_torch = (tensor_float - mean_torch) / std_torch
    return x_standardized_torch

def norm_min_max_tensor(tensor):
    tensor_float = tensor.float()
    x_min_torch = tensor_float.min()
    x_max_torch = tensor_float.max()
    x_normalized_torch = (tensor_float - x_min_torch) / (x_max_torch - x_min_torch)
    return x_normalized_torch


if __name__=="__main__":
    model_path = 'model/model_diff.tflite'
    onnx_path  = 'model/model_diff.onnx'
    input_h  = 383
    input_w  = 3
    in_channel  = 5
    out_channel = 1
    kernel_shape = (2,4)
    input_dtype=np.int8
    cal_dtype=torch.int32

    interpreter, input_tensor = parse_model(model_path, (1, in_channel, input_h, input_w), input_dtype)
    weight_tensor = torch.from_numpy(np.transpose(interpreter.tensor(5)(),(0, 3, 1, 2))).to(cal_dtype)
    bias_tensor = torch.from_numpy(interpreter.tensor(4)()).to(cal_dtype)

    # ref model
    model = Conv2DDemo(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_shape, weight=weight_tensor, bias=bias_tensor, stride=1)
    out = model(input_tensor)

    # onnx model
    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: input_tensor.to(torch.float32).numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    tensor_ort = torch.from_numpy(ort_outs[0])

    tflite_data = interpreter.tensor(9)()
    tflite_tensor_int32 = torch.from_numpy(tflite_data)

    tensor_out_scale_min, tensor_out_scale_max = easy_quanti(out)
    out_min_int8 = tensor_out_scale_min.clamp(torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max).to(torch.int8)
    out_max_int8 = tensor_out_scale_max.clamp(torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max).to(torch.int8)
    print(tensor_out_scale_min[0][0][0], tensor_out_scale_max[0][0][0], tflite_tensor_int32[0][0][0],tensor_ort[0][0][0], out[0][0][0])

    out_min_int8_norm = norm_min_max_tensor(out_min_int8)
    out_max_int8_norm = norm_min_max_tensor(out_max_int8)
    tensor_ort_norm   = norm_min_max_tensor(tensor_ort)
    tflite_tensor_int32 = norm_min_max_tensor(tflite_tensor_int32)

    print(tensor_ort_norm[0][0][0], tflite_tensor_int32[0][0][0], out_min_int8_norm[0][0][0], out_max_int8_norm[0][0][0])
    calculate_mse(out_min_int8_norm, out_max_int8_norm)
    calculate_mse(out_min_int8_norm, tensor_ort_norm)
    calculate_mse(out_max_int8_norm, tensor_ort_norm)
    calculate_mse(tflite_tensor_int32, tensor_ort_norm)




