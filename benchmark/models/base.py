"""Abstract base model and registry for benchmark models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn

MODEL_REGISTRY: dict[str, type] = {}


def register_model(cls: type) -> type:
    """Decorator to register a model class in the global registry."""
    MODEL_REGISTRY[cls.name] = cls
    return cls


class BaseModel(nn.Module, ABC):
    """All benchmark models must subclass this.

    Forward signature:
        input_dict = {
            "fields": Tensor (B, C_in, D, H, W),
            "scalar_params": Tensor (B, S),       # optional
        }
        Returns:
            {"fields": Tensor (B, C_out, D, H, W)}   for field prediction
            {"scalars": Tensor (B, N)}                for scalar prediction
    """

    name: str = ""

    @abstractmethod
    def forward(self, input_dict: dict) -> dict:
        ...
