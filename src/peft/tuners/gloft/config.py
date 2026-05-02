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

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class GloftConfig(PeftConfig):
    """
    Configuration for GLOFT (Generalized Linear Output Feature Transformation).

    GLOFT wraps a linear layer with a trainable D-Rotor transformation that applies
    a dual-scaling rotation in output feature space.

    Args:
        r (`int`):
            Rank of the rotation space.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply GLOFT to. If this is specified, only the
            matching modules will be replaced. If not specified, an error will be raised.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to exclude from GLOFT.
        init_weights (`bool`):
            Whether to initialize GLOFT parameters before training.
        modules_to_save (`Optional[List[str]]`):
            Additional modules to save in the final checkpoint.
    """

    r: int = field(default=32, metadata={"help": "GLOFT rank."})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to replace with GLOFT."},
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from GLOFT."},
    )
    init_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the GLOFT parameters."},
    )
    init_method: str = field(
        default="normal",
        metadata={
            "help": (
                "Initialization method for GLOFT parameters. "
                "Options: normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, zeros."
            )
        },
    )
    init_scale: float = field(
        default=0.01,
        metadata={"help": "Scale used for random initialization when init_method uses a normal distribution."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from GLOFT layers to save in the final checkpoint."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.GLOFT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        if self.r <= 0:
            raise ValueError(f"`r` must be a positive integer; got {self.r}.")

        allowed_init_methods = {
            "normal",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "zeros",
        }
        if self.init_method not in allowed_init_methods:
            raise ValueError(f"`init_method` must be one of {sorted(allowed_init_methods)}; got {self.init_method!r}.")
        if self.init_scale <= 0:
            raise ValueError(f"`init_scale` must be positive; got {self.init_scale}.")
