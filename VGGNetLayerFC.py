#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F


class VGGNetLayerFC(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """

    def __init__(self):
        super(VGGNetLayerFC, self).__init__(
            
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )
        self.train = False

    def __call__(self, x):        
        h = F.relu(self.fc6(x))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
	
	h = F.softmax(h)
	return h
