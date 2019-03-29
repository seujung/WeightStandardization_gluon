import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

class conv2D(nn.Conv2D):
    def __init__(self, channels, kernel_size, strides=(1, 1),
                                               padding=(0, 0), dilation=(1, 1),
                                               groups=1, layout='NCHW', activation=None,
                                               use_bias=True, weight_initializer=None,
                                               bias_initializer='zeros', in_channels=0, **kwargs):
        super(conv2D, self).__init__(channels, kernel_size, strides,
                                               padding, dilation,
                                               groups, layout, activation,
                                               use_bias, weight_initializer,
                                               bias_initializer, in_channels, **kwargs)
    
    def weight_standardization(self):
        weight = self.weight.data()
        #calculate mean
        weight_mean = weight.mean(axis=1, keepdims=True).mean(axis=2,
                                  keepdims=True).mean(axis=3, keepdims=True)
        weight = weight - weight_mean
        #calculate std
        weight_tmp = weight.reshape(0,-1)
        weight_tmp = weight_tmp.asnumpy()
        weight_std = weight_tmp.std(axis=1) # ndarray does not support std 
        weight_std = nd.array(weight_std).reshape(0,1,1,1) + 1e-5
        weight = weight / weight_std
        #update weight
        with autograd.pause():
            self.weight.set_data(weight)
        
    def forward(self, x):
        with x.context:
            self.weight_standardization()
        return super(conv2D, self).forward(x)
