"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved
    Filename: F.py
    Description: This file provides basic layer function for the Neutron.
    Created by Melrose-Lbt 2022-10-11
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import *

def fully_connected_layer(input, weight, bias):
    """
        Compute logic of fully connected layer.
    :param input: Input Tensor
    :param weight: weight parameters
    :param bias: bias parameters
    :return:
    """
    assert weight.shape[2] == input.shape[1], "Weight matrix col:{} and input matrix row:{} conflict! " \
        .format(weight.shape[2], input.shape[1])
    if bias is None:
        return MatMul(weight, input)
    else:
        assert bias.shape[0] == weight.shape[1], "Bias matrix row:{} and MatMul(weight, input) matrix row:{} conflict!"\
        .format(bias.shape[0], weight.shape[1])
        return Add(MatMul(weight, input), bias)