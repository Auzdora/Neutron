"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved
    Filename: _Tensor.py
    Description: This file provides basic data structure for the Neutron.
    Created by Melrose-Lbt 2022-8-19
"""
import ctypes
from ctypes import *
from tkinter import NONE
from turtle import backward
from .utils import CPU, GPU, getShape, CUDALib
import numpy as np
from numpy import ndarray
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)


class Quark(ctypes.Structure):
    """
        C++ back-end data structure. Contains data pointer (numpy data type has to
        be float32, otherwise it'll raise calculate error), device, data shape poin
        ter and dimension.
    """
    _fields_ = [('data', ctypes.POINTER(c_float)),
                ('device', ctypes.c_int),
                ('shape', ctypes.POINTER(ctypes.c_int)),
                ('dim', ctypes.c_int)]


class Tensor:
    """
        Python fore-end data structure.
        The most important attr is handle. Handle is a pointer to the real data str
        -ucture. It manages GPU data structure (Quark) and CPU data structure (numpy).
        
        When you instantiate the Tensor, you need to give parameters as follows:
        1. data: numpy array, dtype is np.float32.
        2. device: on cpu or on gpu.
        3. require_grad: require calculate gradient or not.
    """

    def __init__(self, data, device=CPU, require_grad=False):
        self.children = []
        self.father = []
        self.op = None
        self.grad = None
        self.device = device
        self.require_grad = require_grad
        self.handle = self.configureHandle(self, data, device)
        

    @property
    def shape(self):  # get data shape
        if isinstance(self.handle, ndarray):
            return self.handle.shape
        return tuple([self.handle.shape[idx] for idx in range(self.handle.dim)])
    
    @property
    def data(self):  # get data
        assert(self.device == GPU), "the data on the gpu instead of cpu"
        return np.ctypeslib.as_array(self.handle.data, shape=self.shape)
    
    def __str__(self):

        return "Tensor({}, shape={}, dtype=Tensor.float32)".format(np.ctypeslib.as_array(self.handle.data, shape=self.shape), self.shape)
    
    @staticmethod
    # configure the handle attribute
    def configureHandle(self, data, device):
        if isinstance(data, tuple):
            data = np.random.random(data)
        if device == GPU:
            return self.getQuarkHandle(data.astype(np.float32))
        elif device == CPU:
            return self.getNumpyHandle(data.astype(np.float32))
    
    @staticmethod
    # get the Quark data structure handle
    def getQuarkHandle(numpy_data):
        assert isinstance(numpy_data, ndarray), "input data should be numpy array"
        data = numpy_data
        arr = Quark()
        arr.data = data.ctypes.data_as(ctypes.POINTER(c_float))
        arr.device = GPU
        arr.shape = getShape(ctypes.c_int, data.shape)
        arr.dim = len(data.shape)
        
        # start to allocate and copy data to GPU
        size = CUDALib.getSize(arr.dim, arr.shape)
        dev_ptr = CUDALib.AllocateDeviceData(size)
        CUDALib.CopyDataFromTo(arr.data, dev_ptr, CPU, GPU, size)
        arr.data = dev_ptr
        return arr

    @staticmethod
    # get the numpy data structure handle4
    def getNumpyHandle(numpy_data):
        assert isinstance(numpy_data, ndarray), "input data should be numpy array"
        return numpy_data
    
    # transfer the data from the gpu to the cpu
    def cpu(self):
        if self.device == GPU:
            size = CUDALib.getSize(self.handle.dim, self.handle.shape)
            host_ptr = CUDALib.AllocateHostData(size)
            CUDALib.CopyDataFromTo(self.handle.data, host_ptr, GPU, CPU, size)
            self.handle.data = host_ptr
        return self
    
    # transfer the data from the cpu to the gpu
    def gpu(self):
        if self.device == CPU and isinstance(self.handle, ndarray):
            self.handle = self.getQuarkHandle(self.handle)
            self.device = GPU
        return self
    
    def clear(self):
        """
            In this version, you need to call clear() with your end node in compute graph,
        for example:
            for epoch in range(100):
                output = model(x)
                loss = LossMSE()
                loss.backward()
                ...
                loss.clear()
            You need to call this at every end of epoch to make sure memory won't blow up.
            This function aims at decoupling each Tensor in compute graph, by doing this,
        every object's reference count will go down to zero after each epoch, so python
        could call 'def __del__(self)' method automatically to delete release memory space.
        """
        # Leaf node decoupling
        if len(self.parents) == 0:
            self.parents = []
            self.children = []
            return

        # Other node decoupling
        if len(self.parents) > 0:
            for parent in self.parents:
                parent.children = []
                parent.clear()
            self.parents = []
    
    def backward(self):
        if self.grad is None:
            self.grad = 1
        for father in self.father:
            if father.require_grad:
                self.op.gradient(self.father, self)
                father.backward()
    


if __name__ == "__main__":
    a = np.ones((1024, 1024), dtype=np.float32)
    at = Tensor(a, device=GPU)
    print(at)