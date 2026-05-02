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

from typing import Any, Optional

from torch import nn

from peft.tuners.tuners_utils import BaseTuner, get_device_map

from .config import GloftConfig
from .layer import GloftLayer, dispatch_default


class GloftModel(BaseTuner):
    prefix: str = "gloft_"
    tuner_layer_cls = GloftLayer
    target_module_mapping: dict[str, list[str]] = {}

    def _create_and_replace(
        self,
        peft_config: GloftConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        *,
        parameter_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key must not be None.")

        if isinstance(target, GloftLayer):
            target.update_layer(adapter_name, config=peft_config, **kwargs)
            return

        device_map = get_device_map(self.model)
        new_module = self._create_new_module(peft_config, adapter_name, target, device_map=device_map, **kwargs)

        if adapter_name not in self.active_adapters:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        gloft_config: GloftConfig,
        adapter_name: str,
        target: nn.Module,
        **kwargs: Any,
    ) -> nn.Module:
        new_module = dispatch_default(target, adapter_name, config=gloft_config, **kwargs)
        if new_module is None:
            raise ValueError("Target module is not supported by GLOFT. Only torch.nn.Linear is supported.")
        return new_module
