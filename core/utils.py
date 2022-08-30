import ctypes
from ctypes import *


CUDALib = CDLL('./src/runtime_api.so')

CUDALib.AllocateDeviceData.restype = POINTER(c_float)
CUDALib.AllocateHostData.restype = POINTER(c_float)

(CPU, GPU) = (0, 1)

def getShape(ctype, values):
    return (ctype * len(values))(*values)
