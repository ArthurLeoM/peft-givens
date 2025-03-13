# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import Optional, Tuple, Union
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from transformers.pytorch_utils import Conv1D
from peft.utils.other import transpose


class GivensLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("givens_hard", "givens_soft", "givens_scaler")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("no_scaling", "strict_oft", "fast_config")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.no_scaling = {}
        self.strict_oft = {}
        self.givens_hard = nn.ParameterDict({})  # parameters of n_trans * 2
        self.givens_soft = nn.ParameterDict({})  # parameters of n_trans * 2 * 2
        self.givens_scaler = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")
        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    def update_layer(self, adapter_name, no_scaling, strict_oft, fast_config, init_givens_weights):
        self.no_scaling[adapter_name] = no_scaling
        self.strict_oft[adapter_name] = strict_oft

        # Actual trainable parameters
        if strict_oft:
            # if fast_config:
            #     angles = torch.randn(self.in_features // 2)
            # else:
            angles = torch.randn(self.in_features-1)
            self.givens_hard[adapter_name] = nn.Parameter(angles)
            if not no_scaling:
                scalings = torch.randn(self.in_features)
                self.givens_scaler[adapter_name] = nn.Parameter(scalings)
        else:
            # if fast_config:
            #     rotations = torch.randn(self.in_features // 2, 2, 2)
            # else:
            rotations = torch.randn(self.in_features-1, 2, 2)
            self.givens_soft[adapter_name] = nn.Parameter(rotations)

        if init_givens_weights:
            self.reset_givens_parameters(adapter_name)

        # weight = getattr(self, "weight", None)
        base_layer = self.get_base_layer()
        if base_layer.weight is not None:
            # the layer is already completely initialized, this is an update
            if base_layer.weight.dtype.is_floating_point or base_layer.weight.dtype.is_complex:
                self.to(base_layer.weight.device, dtype=base_layer.weight.dtype)
            else:
                self.to(base_layer.weight.device)
        self.set_adapter(self.active_adapters)


    def reset_givens_parameters(self, adapter_name):
        if adapter_name in self.givens_hard.keys():
            # initialize angles to 0
            nn.init.zeros_(self.givens_hard[adapter_name])
        if adapter_name in self.givens_scaler.keys():
            # intialize the scaler to Indentity matrix
            nn.init.ones_(self.givens_scaler[adapter_name])
        if adapter_name in self.givens_soft.keys():
            # initialize each of the givens_soft transformation to an Indentity matrix
            sz = self.givens_soft[adapter_name].data.size(0)
            for i in range(sz):
                nn.init.eye_(self.givens_soft[adapter_name][i])





class Linear(nn.Module, GivensLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        no_scaling: bool = False,     # Set this to True if you don't want to fine tune the length
        strict_oft: bool = False,  # Set this to True if the layer is strict orthogonal
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        fast_config: bool = False,
        beta: float = 1.0,
        **kwargs,
    ) -> None:
        init_givens_weights = kwargs.pop("init_givens_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super().__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        GivensLayer.__init__(self, base_layer, **kwargs)
        # Freezing the pre-trained weight matrix
        self.fan_in_fan_out = fan_in_fan_out
        self.fast_config = fast_config
        self.beta = beta

        self.update_layer(adapter_name, no_scaling, strict_oft, fast_config, init_givens_weights)
        self.set_adapter(adapter_name)

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.strict_oft.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    tmp_weights = self.get_delta_weight(active_adapter)
                    dtype = tmp_weights.dtype
                    orig_weights = tmp_weights @ (orig_weights if self.fan_in_fan_out else orig_weights.T).to(dtype)
                    orig_weights = orig_weights if self.fan_in_fan_out else orig_weights.T

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights.to(base_layer.weight.data.dtype)
                else:
                    tmp_weights = self.get_delta_weight(active_adapter)
                    print(base_layer)
                    dtype = tmp_weights.dtype
                    tmp_weights = tmp_weights @ (base_layer.weight.data if self.fan_in_fan_out else base_layer.weight.data.T).to(dtype)
                    base_layer.weight.data = (tmp_weights if self.fan_in_fan_out else tmp_weights.T).to(base_layer.weight.data.dtype)
                self.merged_adapters.append(active_adapter)
            
    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            base_layer = self.get_base_layer()
            if active_adapter in self.strict_oft.keys():
                inv_rot = torch.nn.Parameter(torch.linalg.inv(self.get_delta_weight(active_adapter)))
                if self.fan_in_fan_out:
                    base_layer.weight.data = inv_rot @ base_layer.weight.data
                else:
                    base_layer.weight.data = base_layer.weight.data @ inv_rot.T

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """

        base_layer = self.get_base_layer()
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        if self.strict_oft[adapter]:
            rotations = self.givens_hard[adapter]
            scaler = None if self.no_scaling[adapter] else torch.diag(self.givens_scaler[adapter]).to(device)
        else:
            rotations = self.givens_soft[adapter]
            scaler = None

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        output_tensor = torch.nn.Parameter(scaler if scaler is not None else torch.eye(self.in_features).to(device))
        # for i in range(self.in_features-1):
        #     output_tensor = torch.nn.Parameter(output_tensor @ weights[i])
        
        if self.strict_oft[adapter]:
            all_theta = F.tanh(self.givens_hard[adapter].float()) * math.pi
        cur_cnt = self.in_features
        step_size = 1
        flag = 0
        while True:
            left = cur_cnt % 2
            cur_cnt = cur_cnt // 2
            if self.strict_oft[adapter]:
                cos_vec = torch.ones(self.in_features, dtype=base_layer.weight.dtype, device=device)
                sin_vec = torch.zeros(self.in_features, dtype=base_layer.weight.dtype, device=device)
                cos_vec[:step_size*2*cur_cnt:step_size*2] = torch.cos(all_theta[flag:flag+cur_cnt])
                cos_vec[step_size:step_size*(2*cur_cnt+1):step_size*2] = torch.cos(all_theta[flag:flag+cur_cnt])
                sin_vec[:step_size*2*cur_cnt:step_size*2] = torch.sin(all_theta[flag:flag+cur_cnt])
                sin_vec[step_size:step_size*(2*cur_cnt+1):step_size*2] = torch.sin(all_theta[flag:flag+cur_cnt])

            else:
                cos_vec = torch.ones(self.in_features).to(output_tensor.dtype)
                sin_vec = torch.zeros(self.in_features).to(output_tensor.dtype)
                cos_vec[:step_size*2*cur_cnt:step_size*2] = self.givens_soft[adapter][flag:flag+cur_cnt,0,0].to(cos_vec.dtype)
                cos_vec[step_size:step_size*(2*cur_cnt+1):step_size*2] = self.givens_soft[adapter][flag:flag+cur_cnt,1,1].to(cos_vec.dtype)
                sin_vec[:step_size*2*cur_cnt:step_size*2] = -self.givens_soft[adapter][flag:flag+cur_cnt,0,1].to(sin_vec.dtype)
                sin_vec[step_size:step_size*(2*cur_cnt+1):step_size*2] = self.givens_soft[adapter][flag:flag+cur_cnt,1,0].to(sin_vec.dtype)
            
            step_size = step_size * 2
            flag = flag + cur_cnt
            cur_cnt = cur_cnt + left

            weight_cos = output_tensor.data
            weight_sin = torch.stack([-output_tensor.data[:,1::2], output_tensor.data[:,::2]],dim=-1).reshape(output_tensor.shape)
            output_tensor.data = weight_cos * cos_vec + weight_sin * sin_vec

            if cur_cnt == 1:
                break
        
        # for cur_rot in rot_list:
        #     output_tensor.data = torch.mm(cur_rot, output_tensor.data)


        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype)

            # cast back the weights
            if self.strict_oft[adapter]:
                self.givens_hard[adapter] = self.givens_hard[adapter].to(dtype)
                if not self.no_scaling[adapter]:
                    self.givens_scaler[adapter] = self.givens_scaler[adapter].to(dtype)
            else:
                self.givens_soft[adapter] = self.givens_soft[adapter].to(dtype)
        
        # print(output_tensor)
        return output_tensor


    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.linear(input, transpose(base_layer.weight, self.fan_in_fan_out), bias=base_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype
        batch_size = x.size(0)
        device = x.device
        base_layer = self.get_base_layer()

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.strict_oft.keys():
                    continue
                # if active_adapter in self.givens_scaler.keys():
                #     scaler = torch.diag(self.givens_scaler[active_adapter])
                # else:
                #     scaler = None
                # print(scaler)
                if self.strict_oft[active_adapter]:
                    all_theta = F.tanh(self.givens_hard[active_adapter].to(base_layer.weight.dtype)) * torch.pi

                cur_cnt = self.in_features
                step_size = 1
                flag = 0
                sin_list = []
                cos_list = []
                while True:
                    left = cur_cnt % 2
                    cur_cnt = cur_cnt // 2
                    if self.strict_oft[active_adapter]:
                        cos_vec = torch.ones(self.in_features, dtype=base_layer.weight.dtype, device=device)
                        sin_vec = torch.zeros(self.in_features, dtype=base_layer.weight.dtype, device=device)
                        cos_vec[:step_size*2*cur_cnt:step_size*2] = torch.cos(all_theta[flag:flag+cur_cnt])
                        cos_vec[step_size:step_size*(2*cur_cnt+1):step_size*2] = torch.cos(all_theta[flag:flag+cur_cnt])
                        sin_vec[:step_size*2*cur_cnt:step_size*2] = torch.sin(all_theta[flag:flag+cur_cnt])
                        sin_vec[step_size:step_size*(2*cur_cnt+1):step_size*2] = torch.sin(all_theta[flag:flag+cur_cnt])

                    else:
                        cos_vec = torch.ones(self.in_features, dtype=base_layer.weight.dtype, device=device)
                        sin_vec = torch.zeros(self.in_features, dtype=base_layer.weight.dtype, device=device)
                        cos_vec[:step_size*2*cur_cnt:step_size*2] = self.givens_soft[active_adapter][flag:flag+cur_cnt,0,0].to(cos_vec.dtype)
                        cos_vec[step_size:step_size*(2*cur_cnt+1):step_size*2] = self.givens_soft[active_adapter][flag:flag+cur_cnt,1,1].to(cos_vec.dtype)
                        sin_vec[:step_size*2*cur_cnt:step_size*2] = -self.givens_soft[active_adapter][flag:flag+cur_cnt,0,1].to(sin_vec.dtype)
                        sin_vec[step_size:step_size*(2*cur_cnt+1):step_size*2] = self.givens_soft[active_adapter][flag:flag+cur_cnt,1,0].to(sin_vec.dtype)
                    
                    step_size = step_size * 2
                    flag = flag + cur_cnt
                    cur_cnt = cur_cnt + left

                    sin_list.insert(0, sin_vec)
                    cos_list.insert(0, cos_vec)

                    # weight_cos = base_layer.weight.data
                    # weight_sin = torch.stack([-base_layer.weight.data[,1::2], base_layer.weight.data[:,::2]],dim=-1).reshape(base_layer.weight.data.shape)
                    # base_layer.weight.data = weight_cos * cos_vec + weight_sin * sin_vec

                    if cur_cnt == 1:
                        break
                
                # if scaler is not None:
                #     base_layer.weight.data = base_layer.weight.data @ scaler

                # print(rot_weight)
                # result = F.linear(x.to(rot_weight.dtype), rot_weight.T)
                # result = result.to(previous_dtype)
                for cos_v, sin_v in zip(cos_list, sin_list):
                    x_cos = x
                    x_sin = torch.stack((-x[...,1::2], x[...,::2]), dim=-1).reshape(x.shape)
                    x = x_cos * cos_v + x_sin * sin_v
                
                if active_adapter in self.givens_scaler.keys():
                    x = x * self.givens_scaler[active_adapter]

                result = F.linear(x, transpose(base_layer.weight, self.fan_in_fan_out), bias=base_layer.bias)

                    # print(result)
            # result = F.linear(x.to(new_weights.dtype), new_weights.T)
            result = result.to(previous_dtype)

        return result

