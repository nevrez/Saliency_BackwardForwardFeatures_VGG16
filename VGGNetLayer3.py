#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F


class VGGNetLayer3(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as input
    """

    def __init__(self):
        super(VGGNetLayer3, self).__init__(           
            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
        )
        self.train = False

    def __call__(self, x):

        h = F.relu(self.conv3_1(x))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
	
	return h
