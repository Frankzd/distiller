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

"""
AlexNet model with batch-norm layers.
Model configuration based on the AlexNet DoReFa example in TensorPack:
https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net/alexnet-dorefa.py

Code based on the AlexNet PyTorch sample, with the required changes.
"""

import math
import torch.nn as nn

__all__ = ['AlexNet_TINY_IMG', 'alexnet_tiny_img']


class AlexNet_TINY_IMG(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNet_TINY_IMG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4,padding=2),                           # conv0 (224x224x3) -> (54x54x64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                # pool1 (54x54x64) -> (27x27x64)
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),   # conv1 (27x27x64)  -> (27x27x192)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                # pool1 (27x27x192) -> (13x13x192)
            

            nn.Conv2d(192, 384, kernel_size=3, padding=1),            # conv2 (13x13x192) -> (13x13x384)
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),         # conv3 (13x13x384) -> (13x13x256)
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),        # conv4 (13x13x256) -> (13x13x256)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                                # pool4 (13x13x256) -> (6x6x256)

        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096, bias=False),       # fc0
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=False),              # fc1
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),                   # fct
        )

        #for m in self.modules():
        #    if isinstance(m, (nn.Conv2d, nn.Linear)):
        #        fan_in, k_size = (m.in_channels, m.kernel_size[0] * m.kernel_size[1]) if isinstance(m, nn.Conv2d) \
        #            else (m.in_features, 1)
        #        n = k_size * fan_in
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
         #       if hasattr(m, 'bias') and m.bias is not None:
         #           m.bias.data.fill_(0)
         #   elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_tiny_img(**kwargs):
    r"""AlexNet model with batch-norm layers.
    Model configuration based on the AlexNet DoReFa example in `TensorPack
    <https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net/alexnet-dorefa.py>`
    """
    model = AlexNet_TINY_IMG(**kwargs)
    return model
