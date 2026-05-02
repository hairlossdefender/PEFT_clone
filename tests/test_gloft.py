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

import pytest
import torch
from torch import nn

from peft import GloftConfig
from peft.tuners.gloft.layer import GloftLayer


def test_gloft_config_validation():
    config = GloftConfig(r=4, init_method="xavier_uniform", init_scale=0.02)
    assert config.r == 4
    assert config.init_method == "xavier_uniform"
    assert config.init_scale == 0.02

    with pytest.raises(ValueError, match="`init_method` must be one of"):
        GloftConfig(init_method="invalid")

    with pytest.raises(ValueError, match="`init_scale` must be positive"):
        GloftConfig(init_scale=0.0)

    with pytest.raises(ValueError, match="`r` must be a positive integer"):
        GloftConfig(r=0)


def test_gloft_layer_merge_and_unmerge():
    base_layer = nn.Linear(4, 8, bias=False)
    config = GloftConfig(r=2, init_weights=True, init_method="normal", init_scale=0.01)

    from peft.tuners.gloft.layer import Linear

    layer = Linear(base_layer, "default", config=config)

    x = torch.randn(3, 4)
    output_before = layer(x)

    layer.merge()
    assert layer.merged

    layer.unmerge()
    assert not layer.merged

    output_after = layer(x)
    torch.testing.assert_close(output_before, output_after, atol=1e-4, rtol=1e-4)
