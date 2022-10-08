from core import Tensor, Add
from core import *
import numpy as np
import time
from ctypes import *

# 求卷积核的梯度时， 卷积核的梯度值，是batch中每一个图片的加和，要进行梯度下降时需除以batch_size
# 求数据的梯度时，得到的梯度值是output_channel中每一个channel的和，进行梯度下降时要除以out_channel
# 上述的定义还需要验证，建议采用Pytorch验证，同时需要手算以下各个维度的梯度，保证正确性再进行下一步

# MetaFlow的一个问题是，batch数据的前向计算过程直接除了batch_size，但没有计入计算图中，反向传播时忽略了除法的反向传播
# pytorch的batch训练思路是，计算过程中始终保留batch，会得到batch个损失值，这几个损失反向传播，每处计算平均梯度，然后继续各计算各的梯度传播
# 预计9月中旬之后再开始验证

x = np.ones((1, 2, 4, 4))
dy = np.ones((1, 2, 4, 4))

# x = np.array([[[[1, 2, -1, 3], [1, 3, -4, 5],[0.4, 1, 1, -1], [2, 0, -4, 2.1]]]], dtype=np.float32)
# dy = np.array([[[[0.3, 0.3], [-0.4, 0.1]]]], dtype=np.float32)
k = np.ones((1, 2, 3, 3), dtype=np.float32)
# const = np.array(1.23, dtype=np.float32)
# print(dy.shape)
input = Tensor(x, GPU, require_grad=True)
kernel = Tensor(k, GPU, require_grad=False)
Dy = Tensor(dy, GPU, require_grad=False)
# CUDALib.cudnnConv2dGetDataGradient(input.handle, kernel.handle, Dy.handle,0,0,1,1,1,1)
# input.cpu()
# print(input)
# kernel.cpu()
# print(kernel)
adop = Add()
c = adop([input, Dy])
c.backward()
print(input.grad)
