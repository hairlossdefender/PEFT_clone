# Copyright 2026-present the HuggingFace Inc. team.
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

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge

from .config import GloftConfig


class GloftLayer(BaseTunerLayer):
    adapter_layer_names: tuple[str, ...] = (
        "gloft_A",
        "gloft_B",
        "gloft_diag_pre",
        "gloft_diag_post",
    )
    other_param_names: tuple[str, ...] = ("r", "init_weights")

    def __init__(self, base_layer: nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.base_layer = base_layer

        self.r: dict[str, int] = {}
        self.init_weights: dict[str, bool] = {}

        self.gloft_A = nn.ParameterDict({})
        self.gloft_B = nn.ParameterDict({})
        self.gloft_diag_pre = nn.ParameterDict({})
        self.gloft_diag_post = nn.ParameterDict({})

        self.merged_adapters: list[str] = []
        self._merged_deltas: dict[str, torch.Tensor] = {}
        self._disable_adapters = False
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        in_features, out_features = _get_in_out_features(base_layer)
        self.in_features = in_features
        self.out_features = out_features

    def _reset_parameters(
        self,
        adapter_name: str,
        init_weights: bool,
        init_method: str,
        init_scale: float,
    ) -> None:
        self.gloft_A[adapter_name] = nn.Parameter(torch.empty(self.out_features, self.r[adapter_name]))
        self.gloft_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, self.r[adapter_name]))
        self.gloft_diag_pre[adapter_name] = nn.Parameter(torch.ones(self.out_features))
        self.gloft_diag_post[adapter_name] = nn.Parameter(torch.ones(self.out_features))

        self._initialize_parameters(adapter_name, init_weights, init_method, init_scale)

    def _initialize_parameters(
        self,
        adapter_name: str,
        init_weights: bool,
        init_method: str,
        init_scale: float,
    ) -> None:
        A = self.gloft_A[adapter_name]
        B = self.gloft_B[adapter_name]

        if init_method == "normal":
            nn.init.normal_(A, mean=0.0, std=init_scale)
            nn.init.normal_(B, mean=0.0, std=init_scale)
        elif init_method == "xavier_uniform":
            nn.init.xavier_uniform_(A, gain=1.0)
            nn.init.xavier_uniform_(B, gain=1.0)
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(A, gain=1.0)
            nn.init.xavier_normal_(B, gain=1.0)
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(B, a=math.sqrt(5))
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(A, a=math.sqrt(5))
            nn.init.kaiming_normal_(B, a=math.sqrt(5))
        elif init_method == "zeros":
            nn.init.zeros_(A)
            nn.init.zeros_(B)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        if not init_weights:
            # Use a small random initialization when explicit initialization is disabled.
            nn.init.normal_(A, mean=0.0, std=init_scale)
            nn.init.normal_(B, mean=0.0, std=init_scale)

    def _get_compute_dtype(self, device: torch.device, out_dtype: torch.dtype) -> torch.dtype:
        return torch.float32 if (device.type == "cpu" and out_dtype in (torch.float16, torch.bfloat16)) else out_dtype

    def update_layer(self, adapter_name: str, config: GloftConfig, **kwargs: Any) -> None:
        r = int(config.r)
        self.r[adapter_name] = r
        self.init_weights[adapter_name] = config.init_weights

        self.gloft_A[adapter_name] = nn.Parameter(torch.empty(self.out_features, r))
        self.gloft_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, r))
        self.gloft_diag_pre[adapter_name] = nn.Parameter(torch.ones(self.out_features))
        self.gloft_diag_post[adapter_name] = nn.Parameter(torch.ones(self.out_features))

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self._reset_parameters(
            adapter_name,
            init_weights=config.init_weights,
            init_method=config.init_method,
            init_scale=config.init_scale,
        )
        self.set_adapter([adapter_name])

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        base_layer = self.get_base_layer()
        weight = base_layer.weight
        device = weight.device
        out_dtype = weight.dtype
        compute_dtype = self._get_compute_dtype(device, out_dtype)

        A = self.gloft_A[adapter_name].to(device=device, dtype=compute_dtype)
        B = self.gloft_B[adapter_name].to(device=device, dtype=compute_dtype)
        diag_pre = self.gloft_diag_pre[adapter_name].to(device=device, dtype=compute_dtype)
        diag_post = self.gloft_diag_post[adapter_name].to(device=device, dtype=compute_dtype)
        r = self.r[adapter_name]

        U = torch.cat([A, B], dim=1)
        V = torch.cat([B, -A], dim=1)
        eye = torch.eye(2 * r, device=device, dtype=compute_dtype)
        M = eye + V.t() @ U
        Z = torch.linalg.solve(M, V.t())
        K = torch.eye(self.out_features, device=device, dtype=compute_dtype) - 2.0 * Z.t() @ U.t()

        W = weight.to(device=device, dtype=compute_dtype)
        W_transformed = diag_post[:, None] * (K @ (diag_pre[:, None] * W))
        delta = W_transformed - W
        return delta.to(dtype=out_dtype)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base_layer = self.get_base_layer()

        for active_adapter in adapter_names:
            if active_adapter not in self.gloft_A:
                continue

            delta_weight = self.get_delta_weight(active_adapter)
            base_layer = self.get_base_layer()
            orig_dtype = base_layer.weight.data.dtype
            merged_delta = delta_weight.to(orig_dtype).detach().clone()

            if safe_merge:
                output_weight = base_layer.weight.data.clone()
                output_weight += merged_delta
                if not torch.isfinite(output_weight).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                base_layer.weight.data = output_weight
            else:
                base_layer.weight.data += merged_delta

            self._merged_deltas[active_adapter] = merged_delta
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            return

        weight = self.get_base_layer().weight

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            merged_delta = self._merged_deltas.pop(active_adapter, None)
            if merged_delta is None:
                continue

            weight.data -= merged_delta.to(weight.dtype)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        if self.merged:
            return result

        original_dtype = result.dtype
        out = result.float()

        for active_adapter in self.active_adapters:
            if active_adapter not in self.gloft_A:
                continue

            A = self.gloft_A[active_adapter].to(device=out.device, dtype=torch.float32)
            B = self.gloft_B[active_adapter].to(device=out.device, dtype=torch.float32)
            diag_pre = self.gloft_diag_pre[active_adapter].to(device=out.device, dtype=torch.float32)
            diag_post = self.gloft_diag_post[active_adapter].to(device=out.device, dtype=torch.float32)

            r = self.r[active_adapter]
            U = torch.cat([A, B], dim=1)
            V = torch.cat([B, -A], dim=1)

            eye = torch.eye(2 * r, device=out.device, dtype=torch.float32)
            M = eye + V.t() @ U
            Z = torch.linalg.solve(M, V.t())

            out_pre = out * diag_pre
            temp = out_pre @ Z.t()
            out_rot = out_pre - 2.0 * temp @ U.t()
            out = out_rot * diag_post

        return out.to(original_dtype)

    def __repr__(self) -> str:
        return "gloft." + super().__repr__()


class Linear(nn.Module, GloftLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: GloftConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        GloftLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config, **kwargs)

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        return False

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return GloftLayer.forward(self, x, *args, **kwargs)


def dispatch_default(
    target: nn.Module,
    adapter_name: str,
    config: GloftConfig,
    **kwargs: Any,
) -> Optional[nn.Module]:
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        return Linear(target, adapter_name, config=config, **kwargs)

    return None
