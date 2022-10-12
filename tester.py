from core import Tensor, Add, Convolution2D
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

x = np.ones((1, 2, 256, 256))
dy = np.ones((1, 1, 254, 254))
k = np.ones((1, 2, 3, 3), dtype=np.float32)

# input = Tensor(x, GPU, require_grad=True)
# kernel = Tensor(k, GPU, require_grad=False)
# Dy = Tensor(dy, GPU, require_grad=False) 
# conv1 = Convolution2D()
# out = conv1([input, kernel], (0, 0), (1, 1), (1, 1))
# conv1.gradient([input, kernel], Dy, (0, 0), (1, 1), (1, 1))
# print(input.grad, kernel.grad)
m1 = np.array([[[1, 2, 1],
                [2, 0, 1]],
                [[1, 2, 1],
                [0, 1, 1]]])
m2 = np.array([[[1, 1],
                [1, 2],
                [0, 1]],
                [[2, 1],
                [1, 2],
                [1, 1]]])
k1 = np.ones((4, 1, 256))
k2 = np.ones((1, 256, 16))
re = np.ones((4, 1,16))
M1 = Tensor(k1, GPU)
M2 = Tensor(k2, GPU)
ans = Tensor(re, GPU)
cudaMatMul3D(M1, M2, ans)
print(ans.cpu())