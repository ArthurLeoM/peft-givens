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
from peft.utils.other import transpose


class GivensLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("givens_hard", "givens_soft", "givens_scaler")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("no_scaling", "strict_oft", "fast_config")

    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.no_scaling = {}
        self.strict_oft = {}
        self.givens_hard = nn.ParameterDict({})  # parameters of n_trans * 2
        self.givens_soft = nn.ParameterDict({})  # parameters of n_trans * 2 * 2
        self.givens_scaler = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
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

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
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






class Linear(nn.Linear, GivensLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
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
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        GivensLayer.__init__(self, in_features=in_features, out_features=out_features)
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
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.clone()
                    tmp_weights = self.get_delta_weight(active_adapter)
                    dtype = tmp_weights.dtype
                    orig_weights = tmp_weights @ (orig_weights if self.fan_in_fan_out else orig_weights.T).to(dtype)
                    orig_weights = orig_weights if self.fan_in_fan_out else orig_weights.T

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.weight.data = orig_weights.to(self.weight.data.dtype)
                else:
                    tmp_weights = self.get_delta_weight(active_adapter)
                    dtype = tmp_weights.dtype
                    tmp_weights = tmp_weights @ (self.weight.data if self.fan_in_fan_out else self.weight.data.T).to(dtype)
                    self.weight.data = (tmp_weights if self.fan_in_fan_out else tmp_weights.T).to(self.weight.data.dtype)
                self.merged_adapters.append(active_adapter)
            
    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.strict_oft.keys():
                inv_rot = torch.nn.Parameter(torch.linalg.inv(self.get_delta_weight(active_adapter)))
                if self.fan_in_fan_out:
                    self.weight.data = inv_rot @ self.weight.data
                else:
                    self.weight.data = self.weight.data @ inv_rot.T

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """

        if self.strict_oft[adapter]:
            rotations = self.givens_hard[adapter]
            scaler = None if self.no_scaling[adapter] else torch.diag(self.givens_scaler[adapter])
        else:
            rotations = self.givens_soft[adapter]
            scaler = None
        device = rotations.device
        dtype = rotations.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        output_tensor = torch.nn.Parameter(scaler if scaler is not None else torch.eye(self.in_features))
        # for i in range(self.in_features-1):
        #     output_tensor = torch.nn.Parameter(output_tensor @ weights[i])
        
        if self.strict_oft[adapter]:
            all_theta = F.tanh(self.givens_hard[adapter].float()) * math.pi
        cur_cnt = self.in_features
        step_size = 1
        rot_list = []
        flag = 0
        while True:
            rot_mat = torch.eye(self.in_features).to(device)
            left = cur_cnt % 2
            cur_cnt = cur_cnt // 2
            idx = torch.arange(cur_cnt) * step_size * 2
            if self.strict_oft[adapter]:
                rot_mat[idx, idx] = torch.cos(all_theta[flag:flag+cur_cnt])
                rot_mat[idx, idx+step_size] = -torch.sin(all_theta[flag:flag+cur_cnt])
                rot_mat[idx+step_size, idx] = torch.sin(all_theta[flag:flag+cur_cnt])
                rot_mat[idx+step_size, idx+step_size] = torch.cos(all_theta[flag:flag+cur_cnt])
            else:
                rot_mat[idx, idx] = self.givens_soft[adapter][flag:flag+cur_cnt,0,0].to(rot_mat.dtype)
                rot_mat[idx, idx+step_size] = self.givens_soft[adapter][flag:flag+cur_cnt,0,1].to(rot_mat.dtype)
                rot_mat[idx+step_size, idx] = self.givens_soft[adapter][flag:flag+cur_cnt,1,0].to(rot_mat.dtype)
                rot_mat[idx+step_size, idx+step_size] = self.givens_soft[adapter][flag:flag+cur_cnt,1,1].to(rot_mat.dtype)
            
            step_size = step_size * 2
            flag = flag + cur_cnt
            cur_cnt = cur_cnt + left
            rot_list.append(rot_mat)
            if cur_cnt == 1:
                break
        
        for cur_rot in rot_list:
            output_tensor.data = torch.mm(cur_rot, output_tensor.data)


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
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype
        batch_size = x.size(0)
        device = x.device

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
                if active_adapter in self.givens_scaler.keys():
                    scaler = torch.diag(self.givens_scaler[active_adapter])
                else:
                    scaler = None
                # print(scaler)
                if self.strict_oft[active_adapter]:
                    all_theta = F.tanh(self.givens_hard[active_adapter].float()) * torch.pi

                cur_cnt = self.in_features
                step_size = 1
                rot_list = []
                flag = 0
                while True:
                    rot_mat = torch.eye(self.in_features).to(x.device)
                    left = cur_cnt % 2
                    cur_cnt = cur_cnt // 2
                    idx = torch.arange(cur_cnt) * step_size * 2
                    if self.strict_oft[active_adapter]:
                        rot_mat[idx, idx] = torch.cos(all_theta[flag:flag+cur_cnt])
                        rot_mat[idx, idx+step_size] = -torch.sin(all_theta[flag:flag+cur_cnt])
                        rot_mat[idx+step_size, idx] = torch.sin(all_theta[flag:flag+cur_cnt])
                        rot_mat[idx+step_size, idx+step_size] = torch.cos(all_theta[flag:flag+cur_cnt])
                    else:
                        rot_mat[idx, idx] = self.givens_soft[active_adapter][flag:flag+cur_cnt,0,0].to(rot_mat.dtype)
                        rot_mat[idx, idx+step_size] = self.givens_soft[active_adapter][flag:flag+cur_cnt,0,1].to(rot_mat.dtype)
                        rot_mat[idx+step_size, idx] = self.givens_soft[active_adapter][flag:flag+cur_cnt,1,0].to(rot_mat.dtype)
                        rot_mat[idx+step_size, idx+step_size] = self.givens_soft[active_adapter][flag:flag+cur_cnt,1,1].to(rot_mat.dtype)
                    
                    step_size = step_size * 2
                    flag = flag + cur_cnt
                    cur_cnt = cur_cnt + left
                    rot_list.append(rot_mat.to_sparse())
                    if cur_cnt == 1:
                        break

                rot_weight = scaler if scaler is not None else torch.eye(self.in_features).to(device)
                for cur_rot in rot_list:
                    # print(cur_rot)
                    rot_weight = torch.mm(cur_rot, rot_weight)
                # print(rot_weight)
                result = F.linear(x.to(rot_weight.dtype), rot_weight.T)
                result = result.to(previous_dtype)
                result = F.linear(result, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

                    # print(result)
            # result = F.linear(x.to(new_weights.dtype), new_weights.T)
            result = result.to(previous_dtype)

        return result

