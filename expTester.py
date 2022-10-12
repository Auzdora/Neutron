import numpy as np
from core import Tensor, Add, Convolution2D, Flatten_op
from core import *
import numpy as np
import time
from ctypes import *

m1 = np.array([[[[1, 2, 1],
                [2, 0, 1]],
                [[1, 2, 1],
                [0, 1, 1]]]])
t_m1 = Tensor(m1, CPU, True)
print(t_m1.shape)
Fl = Flatten_op()
out = Fl([t_m1])
print(out.shape)
out.handle[0][1] = out.handle[0][1] - 0.1 * 3
out.grad = out.handle.copy()
print(out.grad)
out.backward()
print(t_m1.grad)