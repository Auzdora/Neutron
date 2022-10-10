"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved
    Filename: _Operator.py
    Description: This file provides tons of operators for the Neutron.
    Created by Melrose-Lbt 2022-8-22
"""
from math import floor
from sys import hash_info
from typing import List, Tuple
from ._Tensor import Tensor
from ._CUDA_OP import *
from .utils import *
import numpy as np


class Operator(object):
    def __call__(self, inputs: List[Tensor], device, shape: Tuple) -> Tensor:
        data = np.zeros(shape, dtype=np.float32)
        output_Tensor = Tensor(data, device=device, require_grad=True)
        output_Tensor.op = self
        for father in inputs:
            output_Tensor.father.append(father)
            father.children.append(output_Tensor)

        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor, *arg) -> Tensor:
        raise NotImplementedError
    
    def gradient(self, inputs: List[Tensor], output: Tensor, *arg) -> Tensor:
        raise NotImplementedError
    
    def infer_shape(self, inputs: List[Tensor], *arg) -> Tuple:
        raise NotImplementedError

 
class Add(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaAdd(inputs[0], inputs[1], output)
        else:
            output.handle = inputs[0].handle + inputs[1].handle
    
    def gradient(self, inputs: List[Tensor], output: Tensor):
        _g = np.array(np.eye(output.shape[0] * output.shape[1]))
        if inputs[0].require_grad: inputs[0].grad = np.dot(output.grad, _g)
        if inputs[1].require_grad: inputs[1].grad = np.dot(output.grad, _g)

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class Sub(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaSub(inputs[0], inputs[1], output)
        else:
            output.handle = inputs[0].handle - inputs[1].handle

    def gradient(self, inputs: List[Tensor], output: Tensor):
        _g = np.array(np.eye(output.shape[0] * output.shape[1]))
        if inputs[0].require_grad: inputs[0].grad = np.dot(output.grad, _g)
        if inputs[1].require_grad: inputs[1].grad = np.dot(output.grad, _g)

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class AddConst(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaAddConst(inputs[0], inputs[1], output)
        else:
            output.handle = inputs[0].handle + inputs[1].handle

    def gradient(self, inputs: List[Tensor], output: Tensor):
        pass

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class SubConst(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaSubConst(inputs[0], inputs[1], output)
        else:
            output.handle = inputs[0].handle - inputs[1].handle

    def gradient(self, inputs: List[Tensor], output: Tensor):
        pass

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class DivConst(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaDivConst(inputs[0], inputs[1], output)
        else:
            output.handle = inputs[0].handle - inputs[1].handle

    def gradient(self, inputs: List[Tensor], output: Tensor):
        pass

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class MulConst(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaMulConst(inputs[0], inputs[1], output)
        else:
            output.handle = inputs[0].handle - inputs[1].handle

    def gradient(self, inputs: List[Tensor], output: Tensor):
        pass

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class ElemMul(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaElementMul(inputs[0], inputs[1], output)
        else:
            output.handle = inputs[0].handle * inputs[1].handle

    def gradient(self, inputs: List[Tensor], output: Tensor):
        pass

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class ElemDiv(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaElementDiv(inputs[0], inputs[1], output)
        else:
            output.handle = inputs[0].handle / inputs[1].handle

    def gradient(self, inputs: List[Tensor], output: Tensor):
        pass

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class Exp(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaElementExp(inputs[0], output)
        else:
            output.handle = np.exp(inputs[0])

    def gradient(self, inputs: List[Tensor], output: Tensor):
        pass

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class Sqrt(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaElementSqrt(inputs[0], output)
        else:
            output.handle = np.sqrt(inputs[0].handle)

    def gradient(self, inputs: List[Tensor], output: Tensor):
        pass

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class ReLU(Operator):
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        shape = self.infer_shape(inputs)
        output_Tensor = Operator.__call__(self, inputs, inputs[0].device, shape)
        self.compute(inputs, output_Tensor)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], output: Tensor):
        if output.device == GPU:
            cudaReLU(inputs[0], output)
        else:
            output.handle = inputs[0].handle - inputs[1].handle

    def gradient(self, inputs: List[Tensor], output: Tensor):
        pass

    def infer_shape(self, inputs: List[Tensor]) -> Tuple:
        return inputs[0].shape


class Convolution2D(Operator):
    def __call__(self, Inputs: List[Tensor],
                       padding: Tuple,
                       stride: Tuple,
                       dilation: Tuple) -> Tensor:
        shape = self.infer_shape(Inputs, padding, stride, dilation)
        print(shape)
        output_Tensor = Operator.__call__(self, Inputs, Inputs[0].device, shape)
        self.compute(Inputs, output_Tensor, padding, stride, dilation)
        return output_Tensor
    
    def compute(self, inputs: List[Tensor], 
                      output: Tensor, 
                      padding: Tuple, 
                      stride: Tuple, 
                      dilation: Tuple):

        if output.device == GPU:
            cudnnConv2D(inputs[0], inputs[1], output, padding, stride, dilation)
        else:
            raise NotImplementedError("The CPU version of Convolution2D OP has not been implemented yet")

    def gradient(self, inputs: List[Tensor], 
                       output: Tensor,
                       padding: Tuple, 
                       stride: Tuple, 
                       dilation: Tuple) -> Tensor:
        pass
    
    def infer_shape(self, inputs: List[Tensor], padding: Tuple, stride: Tuple, dilation: Tuple) -> Tuple:
        input_shape = inputs[0].shape
        filter_shape = inputs[1].shape
        n_in, c_in, h_in, w_in = input_shape
        c_out, k_c_in, k_h_in, k_w_in = filter_shape

        h_out = floor((h_in + 2*padding[0] - dilation[0] * (k_h_in - 1))/stride[0])
        w_out = floor((w_in + 2*padding[1] - dilation[1] * (k_w_in - 1))/stride[1])
        
        return (n_in, c_out, h_out, w_out)