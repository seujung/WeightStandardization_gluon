import mxnet as mx
import numpy as np
from mxnet.gluon import nn
from mxnet import nd, autograd

class Conv2D(nn.Conv2D):
    def __init__(self, channels, kernel_size, strides=(1, 1),
                 padding=(0, 0), dilation=(1, 1),groups=1, layout='NCHW', activation=None,
                 use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        super().__init__(channels, kernel_size, strides, padding, dilation,groups, layout, activation,
                                     use_bias, weight_initializer, bias_initializer, in_channels, **kwargs)

    def hybrid_forward(self, F, x, weight, bias=None):
        weight_mean = F.mean(weight, axis=(1, 2, 3), keepdims=True)
        weight_sub = F.broadcast_sub(weight, weight_mean)
        weight_std = F.square(weight_sub)
        std = F.sqrt(F.mean(weight_std, axis=(1, 2, 3), keepdims=True)) + 1e-5
        F.broadcast_div(weight_sub, std, out=weight)
        return super().hybrid_forward(F, x, weight=weight, bias=bias)