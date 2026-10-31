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
    freeze_diag: bool = field(
        default=False,
        metadata={
            "help": (
                "Keep gloft_diag_pre and gloft_diag_post at 1 and out of the optimizer, "
                "so the adapter is the rotation alone."
            )
        },
    )
    constraint: str = field(
        default="cayley",
        metadata={
            "help": (
                "Factorial-study option. 'cayley' (default): exact rotation K = (I - S)(I + S)^-1. "
                "'linear': K = I - 2S (first-order Cayley, not orthogonal). "
                "'free': K = I - 2AB^T (no skew-symmetry, same parameter count)."
            )
        },
    )
    generator: str = field(
        default="factored",
        metadata={
            "help": (
                "Factorial-study option. 'factored' (default): S = AB^T - BA^T. "
                "'angle': unit-norm directions from A, B and a learned angle per plane (starts at 0, K = I)."
            )
        },
    )
    init_mode: str = field(
        default="random",
        metadata={
            "help": (
                "'random' (default): A, B ~ init_method, which makes K a random rotation at step 0. "
                "'identity': A = 0 so K = I at step 0 (B keeps its random init)."
            )
        },
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
        if self.constraint not in {"cayley", "linear", "free"}:
            raise ValueError(f"`constraint` must be 'cayley', 'linear' or 'free'; got {self.constraint!r}.")
        if self.generator not in {"factored", "angle"}:
            raise ValueError(f"`generator` must be 'factored' or 'angle'; got {self.generator!r}.")
        if self.init_mode not in {"random", "identity"}:
            raise ValueError(f"`init_mode` must be 'random' or 'identity'; got {self.init_mode!r}.")
        if self.generator == "angle" and self.init_mode != "identity":
            raise ValueError("generator='angle' always starts at K = I (angles = 0); set init_mode='identity'.")
