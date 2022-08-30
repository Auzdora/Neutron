"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved
    Filename: _CUDA_OP.py
    Description: GPU operators package.
    Created by Melrose-Lbt 2022-8-20
"""
from ctypes import c_float
from numpy import ndarray
from .utils import CUDALib
from ._Tensor import Tensor


def cudaAdd(arr1, arr2, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(arr2, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuAdd(arr1.handle, arr2.handle, target.handle)

def cudaSub(arr1, arr2, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(arr2, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuSub(arr1.handle, arr2.handle, target.handle)

def cudaAddConst(arr1, arr2, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(arr2, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuAddConst(arr1.handle, arr2.handle, target.handle)

def cudaSubConst(arr1, arr2, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(arr2, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuSubConst(arr1.handle, arr2.handle, target.handle)

def cudaDivConst(arr1, arr2, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(arr2, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuDivConst(arr1.handle, arr2.handle, target.handle)

def cudaMulConst(arr1, arr2, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(arr2, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuMulConst(arr1.handle, arr2.handle, target.handle)

def cudaElementMul(arr1, arr2, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(arr2, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuElemMul(arr1.handle, arr2.handle, target.handle)

def cudaElementDiv(arr1, arr2, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(arr2, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuElemDiv(arr1.handle, arr2.handle, target.handle)

def cudaElementSqrt(arr1, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuElemSqrt(arr1.handle, target.handle)

def cudaElementExp(arr1, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuElemExp(arr1.handle, target.handle)

def cudaReLU(arr1, target):
    assert isinstance(arr1, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuReLU(arr1.handle, target.handle)

def cudaConv1D(arr, filter, target):
    assert isinstance(arr, Tensor)
    assert isinstance(filter, Tensor)
    assert isinstance(target, Tensor)
    CUDALib.GpuConv1D(arr.handle, filter.handle, target.handle)