#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['cnn_tiny_cifar']

cfg = {
    #'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'BNN':[32,32,'M',64,64,'M',128,128,'M','M','L'],
}

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg["BNN"])
        self.classifier = nn.Sequential(
    		#nn.Linear(2048,1024),
    		#nn.Dropout(),
    		nn.Linear(256,10)
    	)
	

    def forward(self, x):
    	out = self.features(x)
    	out = out.view(out.size(0),-1)
    	out = self.classifier(out)        
    	return out

    def _make_layers(self,cfg):
    	layers = []
    	in_channels = 3
    	for x in cfg:
    		if x== 'M':
    			layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
    		elif x=='L':
    			layers += [nn.Conv2d(128,64,kernel_size=1)]
    		else:
    			layers += [nn.Conv2d(in_channels,x,kernel_size=3,padding=1),
    						nn.ReLU(inplace=True)]
    			in_channels = x
    	layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
    	return nn.Sequential(*layers)


def cnn_tiny_cifar():
    model = VGG()
    return model
