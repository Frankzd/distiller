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
import numpy as np

__all__ = ['vgg_bnn_cifar']

cfg = {
    #'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'BNN':[128,128,'M',256,256,'M',512,512,'M','M'],
}

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        #self.features = self._make_layers(cfg["BNN"])
        #self.classifier = nn.Sequential(
    	#	nn.Linear(2048,1024),
    	#	nn.ReLU(inplace=True),
    	#	nn.Dropout(),
    	#	nn.Linear(1024,10)
    	#)
        print("model init")
        self.conv1 = nn.Conv2d(3,128,kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.ave = nn.AvgPool2d(kernel_size=1,stride=1)
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,10)

    def forward(self, x):
        '''
        input_1 = x.cpu().numpy()
        input_r = input_1[0,0,:,:]
        input_r = input_r.reshape(32,32)
        input_g = input_1[0,1,:,:]
        input_g = input_g.reshape(32,32)
        input_b = input_1[0,2,:,:]
        input_b = input_b.reshape(32,32)
        
        np.savetxt('input_r.csv',input_r,delimiter=',',fmt="%f")
        np.savetxt('input_g.csv',input_g,delimiter=',',fmt="%f")
        np.savetxt('input_b.csv',input_b,delimiter=',',fmt="%f")
        '''
        #print("normal forward")
        #print("input.shape",x.shape)
        file_input = x.cpu().numpy().reshape(-1,1)
        #np.savetxt('test_input.csv',file_input,delimiter=',',fmt="%f")
        x = x.cuda()
        out = self.conv1(x)
        #out1 = out.cpu().detach().numpy()[0,0,:,:]
        #out1 = out1.reshape(32,32)
        #np.savetxt('out1.csv',out1,delimiter=',',fmt="%f")
        
        file_conv1 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv1.csv',file_conv1,delimiter=',',fmt="%f")
        out = self.relu(out)
        file_conv1_relu = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv1_relu.csv',file_conv1_relu,delimiter=',',fmt="%f")
        out = self.conv2(out)
        file_conv2 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv2.csv',file_conv2,delimiter=',',fmt="%f")
        out = self.relu(out)
        file_conv2_relu = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv2_relu.csv',file_conv2_relu,delimiter=',',fmt="%f")
        out = self.pooling(out)
        file_pooling1 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_pooling1.csv',file_pooling1,delimiter=',',fmt="%f")        
        
        out = self.conv3(out)
        file_conv3 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv3.csv',file_conv3,delimiter=',',fmt="%f")
        out = self.relu(out)
        file_conv3_relu = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv3_relu.csv',file_conv3_relu,delimiter=',',fmt="%f")
        out = self.conv4(out)
        file_conv4 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv4.csv',file_conv4,delimiter=',',fmt="%f")
        out = self.relu(out)
        file_conv4_relu = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv4_relu.csv',file_conv4_relu,delimiter=',',fmt="%f")
        out = self.pooling(out)
        file_pooling2 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_poolin2.csv',file_pooling2,delimiter=',',fmt="%f")      

        out = self.conv5(out)
        file_conv5 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv5.csv',file_conv5,delimiter=',',fmt="%f")
        out = self.relu(out)
        file_conv5_relu = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv5_relu.csv',file_conv5_relu,delimiter=',',fmt="%f")
        out = self.conv6(out)
        file_conv6 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv6.csv',file_conv6,delimiter=',',fmt="%f")
        out = self.relu(out)
        file_conv6_relu = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_conv6_relu.csv',file_conv6_relu,delimiter=',',fmt="%f")
        out = self.pooling(out)
        file_pooling3 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_poolin3.csv',file_pooling3,delimiter=',',fmt="%f")        
        out = self.pooling(out)
        file_pooling4 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_poolin4.csv',file_pooling4,delimiter=',',fmt="%f")
        
        out = self.ave(out)
        file_ave = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_ave.csv',file_ave,delimiter=',',fmt="%f")        
        
        out = out.view(out.size(0),-1)
        
        out = self.fc1(out)
        file_fc1 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_fc1.csv',file_fc1,delimiter=',',fmt="%f")         
        
        out = self.relu(out)
        #file_fc1_relu = out.cpu().numpy().reshape(-1,1)
        ##np.savetxt('test_fc1_relu.csv',file_fc1_relu,delimiter=',',fmt="%f")
        
        out = self.fc2(out)
        file_fc2 = out.cpu().detach().numpy().reshape(-1,1)
        #np.savetxt('test_fc2.csv',file_fc2,delimiter=',',fmt="%f")         
        
                
        #out = self.features(x)
        #out = out.view(out.size(0),-1)
        #out = self.classifier(out)        
        #print("out.shape:",out.shape)
        return out

    def _make_layers(self,cfg):
    	layers = []
    	in_channels = 3
    	for x in cfg:
    		if x== 'M':
    			layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
    		elif x=='L':
    			layers += [nn.Conv2d(512,256,kernel_size=1)]
    		else:
    			layers += [nn.Conv2d(in_channels,x,kernel_size=3,padding=1),
    						nn.ReLU(inplace=True)]
    			in_channels = x
    	layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
    	return nn.Sequential(*layers)


def vgg_bnn_cifar():
    model = VGG()
    return model
