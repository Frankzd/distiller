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
import numpy as np

from .quantizer import Quantizer
from .q_utils import *

###
# Range-based linear quantization
###


class RangeLinearQuantWrapper(nn.Module):
    """
    Base class for module which wraps an existing module with linear range-base quantization functionality

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
        clip_acts (bool): If true, will clip activations instead of using absolute min/max. At the moment clipping is
            done by averaging over the max absolute values of samples within a batch. More methods might be added in
            the future.
    """
    def __init__(self, wrapped_module, num_bits_acts, num_bits_accum=32, clip_acts=False):
        super(RangeLinearQuantWrapper, self).__init__()

        self.wrapped_module = wrapped_module
        self.num_bits_acts = num_bits_acts
        self.num_bits_accum = num_bits_accum
        self.clip_acts = clip_acts
        self.acts_sat_val_func = get_tensor_avg_max_abs_across_batch if clip_acts else get_tensor_max_abs

        self.acts_min_q_val, self.acts_max_q_val = get_quantized_range(num_bits_acts, signed=True)
        self.accum_min_q_val, self.accum_max_q_val = get_quantized_range(num_bits_accum, signed=True)

    def forward(self, *inputs):
        in_scales = self.pre_quantized_forward(*inputs)

        # Quantize inputs
        inputs_q = []
        for idx, input in enumerate(inputs):
            input_q = linear_quantize_clamp(input.data, in_scales[idx], self.acts_min_q_val, self.acts_max_q_val,
                                            inplace=False)
            inputs_q.append(torch.autograd.Variable(input_q))

####################################print the input value of every layer################################# 
        '''
        if input_q.shape[1] == 1:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv1.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 6:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv2.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 400:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_fc1.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 120:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_fc2.csv',file_input,delimiter=',',fmt="%f")
        else:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_fc3.csv',file_input,delimiter=',',fmt="%f")
        '''
        '''
        if input_q.shape[1] == 3:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv1.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 128 and input_q.shape[2] == 32:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv2.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 128 and input_q.shape[2] == 16:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv3.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 256 and input_q.shape[2] == 16:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv4.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 256 and input_q.shape[2] == 8:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv5.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 512 and input_q.shape[2] == 8:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv6.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 2048:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_fc1.csv',file_input,delimiter=',',fmt="%f")
        else:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_fc2.csv',file_input,delimiter=',',fmt="%f")
        '''


        # Forward through wrapped module
        accum = self.wrapped_module.forward(*inputs_q)
        clamp(accum.data, self.accum_min_q_val, self.accum_max_q_val, inplace=True)

        # Re-quantize accumulator to quantized output range
        requant_scale, out_scale = self.post_quantized_forward(accum)
        out_q = linear_quantize_clamp(accum.data, requant_scale, self.acts_min_q_val, self.acts_max_q_val, inplace=True)

####################################print the input value of every layer################################# 
        '''
        if input_q.shape[1] == 1:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv1.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 6:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv2.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 400:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_fc1.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 120:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_fc2.csv',file_input,delimiter=',',fmt="%f")
        else:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_fc3.csv',file_input,delimiter=',',fmt="%f")
        '''
        '''
        if input_q.shape[1] == 3:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv1.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 128 and input_q.shape[2] == 32:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv2.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 128 and input_q.shape[2] == 16:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv3.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 256 and input_q.shape[2] == 16:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv4.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 256 and input_q.shape[2] == 8:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv5.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 512 and input_q.shape[2] == 8:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_conv6.csv',file_input,delimiter=',',fmt="%f")
        elif input_q.shape[1] == 2048:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_fc1.csv',file_input,delimiter=',',fmt="%f")
        else:
            file_input = input_q.cpu().detach().numpy().reshape(-1,1)
            np.savetxt('file_input_fc2.csv',file_input,delimiter=',',fmt="%f")
        '''


        # De-quantize back to FP32
        out_f = linear_dequantize(out_q, out_scale, inplace=True)

        return torch.autograd.Variable(out_f)

    def pre_quantized_forward(self, *inputs):
        """
        Calculate input scale factors and perform any action required before quantization of inputs.

        Should be overridden by all subclasses

        :param inputs: Current input tensors passed to forward method
        :return: List of scale factors per input
        """
        raise NotImplementedError

    def post_quantized_forward(self, accumulator):
        """
        Calculate re-quantization scale factor (for converting the intermediate integer accumulator to output range),
        and output scale factor.

        :param accumulator: Tensor with accumulator values
        :return: Tuple of (re-quantization scale factor, output scale factor)
        """
        raise NotImplementedError


class RangeLinearQuantParamLayerWrapper(RangeLinearQuantWrapper):
    """
    Linear range-based quantization wrappers for layers with weights and bias (namely torch.nn.ConvNd and
    torch.nn.Linear)

    Args:
        wrapped_module (torch.nn.Module): Module to be wrapped
        num_bits_acts (int): Number of bits used for inputs and output quantization
        num_bits_params (int): Number of bits used for parameters (weights and bias) quantization
        num_bits_accum (int): Number of bits allocated for the accumulator of intermediate integer results
        clip_acts (bool): See RangeLinearQuantWrapper
    """
    def __init__(self, wrapped_module, num_bits_acts, num_bits_params, num_bits_accum=32, clip_acts=False):
        super(RangeLinearQuantParamLayerWrapper, self).__init__(wrapped_module, num_bits_acts,
                                                                num_bits_accum, clip_acts)

        if not isinstance(wrapped_module, (nn.Conv2d, nn.Linear)):
            raise ValueError(self.__class__.__name__ + ' can wrap only Conv2D and Linear modules')

        self.num_bits_params = num_bits_params
        self.params_min_q_val, self.params_max_q_val = get_quantized_range(num_bits_params, signed=True)

        # Quantize weights - overwrite FP32 weights
        # Quantize weights - overwrite FP32 weights
        
        #cifar-10
        if wrapped_module.weight.shape[1] == 3:
            self.w_scale = 4
        elif wrapped_module.weight.shape[1] == 128 and wrapped_module.weight.shape[0] == 128:
            self.w_scale = 8
        elif wrapped_module.weight.shape[1] == 128 and wrapped_module.weight.shape[0] == 256:
            self.w_scale = 16
        elif wrapped_module.weight.shape[1] == 256 and wrapped_module.weight.shape[0] == 256:
            self.w_scale = 16
        elif wrapped_module.weight.shape[1] == 256 and wrapped_module.weight.shape[0] == 512:
            self.w_scale = 16
        elif wrapped_module.weight.shape[1] == 512 and wrapped_module.weight.shape[0] == 512:
            self.w_scale = 32
        elif wrapped_module.weight.shape[1] == 2048:
            self.w_scale = 32
        else:
            self.w_scale =16
        
        
        #mnist
        '''
        if wrapped_module.weight.shape[1] == 1:
            self.w_scale = 4
        elif wrapped_module.weight.shape[1] == 6:
            self.w_scale = 8
        elif wrapped_module.weight.shape[1] == 400:
            self.w_scale = 4
        elif wrapped_module.weight.shape[1] == 120:
            self.w_scale = 16
        else:
            self.w_scale =8        
        '''
        #self.w_scale = symmetric_linear_quantization_scale_factor(num_bits_params,
        #                                                          get_tensor_max_abs(wrapped_module.weight))
        linear_quantize_clamp(wrapped_module.weight.data, self.w_scale, self.params_min_q_val, self.params_max_q_val,
                              inplace=True)

        # Quantize bias
        self.has_bias = hasattr(wrapped_module, 'bias') and wrapped_module.bias is not None
        if self.has_bias:
            self.b_scale = symmetric_linear_quantization_scale_factor(num_bits_params,
                                                                      get_tensor_max_abs(wrapped_module.bias))
            base_b_q = linear_quantize_clamp(wrapped_module.bias.data, self.b_scale,
                                             self.params_min_q_val, self.params_max_q_val)
            # Dynamic ranges - save in auxiliary buffer, requantize each time based on dynamic input scale factor
            self.register_buffer('base_b_q', base_b_q)

        self.current_accum_scale = 1

    def pre_quantized_forward(self, input):
        super(RangeLinearQuantParamLayerWrapper, self).forward(input)

    def pre_quantized_forward(self, input):
        #print(input.shape)
        
        #cifar-10
        if input.shape[1] == 3:
            in_scale=64 #60
        elif input.shape[1] == 128 and input.shape[2] == 32:
            in_scale=32 #20
        elif input.shape[1] == 128 and input.shape[2] == 16:
            in_scale=16 #10
        elif input.shape[1] == 256 and input.shape[2] == 16:
            in_scale=16 #10
        elif input.shape[1] == 256 and input.shape[2] == 8:
            in_scale=8 #5
        elif input.shape[1] == 512 and input.shape[2] == 8:
            in_scale=4
        elif input.shape[1] == 2048:
            in_scale=4
        else:
            in_scale=16 #15
        
        '''
        #mnist
        if input.shape[1] == 1:
            in_scale=64 #45
        elif input.shape[1] == 6:
            in_scale=16 #17
        elif input.shape[1] == 400:
            in_scale=4 #4
        elif input.shape[1] == 120:
            in_scale=4 #5
        else:
            in_scale=4 #3
        '''
        #in_scale = symmetric_linear_quantization_scale_factor(self.num_bits_acts,
        #                                                      self.acts_sat_val_func(input))
        self.current_accum_scale = in_scale * self.w_scale
        if self.has_bias:
            # Re-quantize bias to match x * w scale: b_q' = (in_scale * w_scale / b_scale) * b_q
            self.wrapped_module.bias.data = linear_quantize_clamp(self.base_b_q, self.current_accum_scale / self.b_scale,
                                                                  self.accum_min_q_val, self.accum_max_q_val)
        
        #print("input.shape:",input.shape)                                                  
        #print("in_scale:",in_scale)
        return [in_scale]

    def post_quantized_forward(self, accumulator):
        accum_max_abs = self.acts_sat_val_func(accumulator)
        y_f_max_abs = accum_max_abs / self.current_accum_scale
        out_scale = symmetric_linear_quantization_scale_factor(self.num_bits_acts, y_f_max_abs)
        requant_scale = out_scale / self.current_accum_scale
        return requant_scale, out_scale

    def extra_repr(self):
        tmpstr = 'wrapped_module: ' + self.wrapped_module.__repr__() + '\n'
        tmpstr += 'num_bits_acts={0}, num_bits_params={1}, num_bits_accum={2}'.format(self.num_bits_acts,
                                                                                      self.num_bits_params,
                                                                                      self.num_bits_accum) + '\n'
        tmpstr += 'clip_acts={0}'.format(self.clip_acts)
        return tmpstr


class SymmetricLinearQuantizer(Quantizer):
    """
    Applies symmetric, range-based linear quantization to a model.
    Currently, the following Modules are supported: torch.nn.Conv2d, torch.nn.Linear

    Args:
        model (torch.nn.Module): Model to be quantized
        bits_activations/parameters/accum (int): Number of bits to be used when quantizing each tensor type
        clip_acts (bool): See RangeLinearQuantWrapper
        no_clip_layers (list): List of fully-qualified layer names for which activations clipping should not be done.
            A common practice is to not clip the activations of the last layer before softmax.
            Applicable only if clip_acts is True.
    """
    def __init__(self, model, bits_activations=8, bits_parameters=8, bits_accum=32,
                 clip_acts=False, no_clip_layers=[]):
        super(SymmetricLinearQuantizer, self).__init__(model, bits_activations=bits_activations,
                                                       bits_weights=bits_parameters, train_with_fp_copy=False)
        
        self.model.quantizer_metadata = {'type': type(self),
                                         'params': {'bits_activations': bits_activations,
                                                    'bits_parameters': bits_parameters}}
        
        def replace_fn(module, name, qbits_map):
            clip = self.clip_acts and name not in no_clip_layers
            '''
            if name=='module.conv1':
                return RangeLinearQuantParamLayerWrapper(module, 8, 4)
            elif name=='module.conv2':
                return RangeLinearQuantParamLayerWrapper(module, 8, 4)
            elif name=='module.conv3':
                return RangeLinearQuantParamLayerWrapper(module, 8, 4)
            elif name=='module.conv4':
                return RangeLinearQuantParamLayerWrapper(module, 8, 4)
            elif name=='module.conv5':
                return RangeLinearQuantParamLayerWrapper(module, 8, 4)
            elif name=='module.conv6':
                return RangeLinearQuantParamLayerWrapper(module, 8, 4)
            elif name=='module.fc1':
                return RangeLinearQuantParamLayerWrapper(module, 8, 4)
            elif name=='module.fc2':
                return RangeLinearQuantParamLayerWrapper(module, 8, 4)
            else:
            '''
            return RangeLinearQuantParamLayerWrapper(module, qbits_map[name].acts, qbits_map[name].wts,
                                                     num_bits_accum=self.bits_accum, clip_acts=clip)

        self.clip_acts = clip_acts
        self.no_clip_layers = no_clip_layers
        self.bits_accum = bits_accum
        self.replacement_factory[nn.Conv2d] = replace_fn
        self.replacement_factory[nn.Linear] = replace_fn
