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

__all__ = ['lenet_mnist']

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        #filter:5*5
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1,6,5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)        

    def forward(self, x):
        file_input = x.cpu().numpy().reshape(-1,1)
        x = x.cuda()
        
        out = self.relu(x)
        file_input_relu = out.cpu().numpy().reshape(-1,1)
        
        out = self.conv1(out)
        file_conv1 = out.cpu().detach().numpy().reshape(-1,1)
        
        out = self.relu(out)
        file_conv1_relu = out.cpu().detach().numpy().reshape(-1,1)
        
        out = F.max_pool2d(out,(2,2))
        file_pooling1 = out.cpu().detach().numpy().reshape(-1,1)
        
        out = self.conv2(out)
        file_conv2 = out.cpu().detach().numpy().reshape(-1,1)
        
        out = self.relu(out)
        file_conv2_relu = out.cpu().detach().numpy().reshape(-1,1)
        
        out = F.max_pool2d(out,(2,2))
        file_pooling2 = out.cpu().detach().numpy().reshape(-1,1)
           
        out = out.view(out.size(0),-1)
        
        out = self.fc1(out)
        file_fc1 = out.cpu().detach().numpy().reshape(-1,1)
                
        out = self.relu(out)
        file_fc1_relu = out.cpu().detach().numpy().reshape(-1,1)
        
        out = self.fc2(out)
        file_fc2 = out.cpu().detach().numpy().reshape(-1,1)
                 
        out = self.relu(out)
        file_fc2_relu = out.cpu().detach().numpy().reshape(-1,1)
        
        out = self.fc3(out)
        file_fc3 = out.cpu().detach().numpy().reshape(-1,1)
        '''
        np.savetxt('test_input.csv',file_input,delimiter=',',fmt="%f")
        np.savetxt('test_input_relu.csv',file_input_relu,delimiter=',',fmt="%f")
        np.savetxt('test_conv1.csv',file_conv1,delimiter=',',fmt="%f")
        np.savetxt('test_conv1_relu.csv',file_conv1_relu,delimiter=',',fmt="%f")
        np.savetxt('test_pooling1.csv',file_pooling1,delimiter=',',fmt="%f")    
        np.savetxt('test_conv2.csv',file_conv2,delimiter=',',fmt="%f")
        np.savetxt('test_conv2_relu.csv',file_conv2_relu,delimiter=',',fmt="%f")
        np.savetxt('test_pooling2.csv',file_pooling2,delimiter=',',fmt="%f")    
        np.savetxt('test_fc1.csv',file_fc1,delimiter=',',fmt="%f") 
        np.savetxt('test_fc1_relu.csv',file_fc1_relu,delimiter=',',fmt="%f")
        np.savetxt('test_fc2.csv',file_fc2,delimiter=',',fmt="%f")
        np.savetxt('test_fc2_relu.csv',file_fc2_relu,delimiter=',',fmt="%f")
        np.savetxt('test_fc3.csv',file_fc3,delimiter=',',fmt="%f")  
        '''
        #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        #x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))        
        #x = x.view(x.size(0),-1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = nn.Threshold(0.2, 0.0)#ActivationZeroThreshold(x)
        #x = self.fc3(x)
        return out

def lenet_mnist():
    model = Lenet()
    return model
