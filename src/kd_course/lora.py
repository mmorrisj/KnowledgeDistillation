"""LoRA: train a small add-on and leave the original weights frozen.

`LoRALinear` is a from-scratch version for learning. The labs use the PEFT library
through `add_lora`, which does the same thing for every targeted layer of a model.
"""

from __future__ import annotations

import math
from typing import Any

import torch

STUDENT_SMALL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
STUDENT_MEDIUM_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# The attention layers of Qwen-style models. A common, safe default.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(torch.nn.Module):
    """A linear layer with a low-rank update: output = W x + (alpha / rank) * B A x.

    W is the original layer and is frozen. Only A and B are trained.
    """

    def __init__(self, base: torch.nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.scale = alpha / rank
        # A starts random and B starts at zero, so B A = 0 and the layer begins unchanged.
        self.lora_a = torch.nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.lora_a.T @ self.lora_b.T)

    def merged_weight(self) -> torch.Tensor:
        """W plus the low-rank update, as one full-size matrix."""
        return self.base.weight + self.scale * (self.lora_b @ self.lora_a)

    def merged(self) -> torch.nn.Linear:
        """An ordinary linear layer with the update folded into its weights."""
        layer = torch.nn.Linear(
            self.base.in_features, self.base.out_features, bias=self.base.bias is not None
        )
        with torch.no_grad():
            layer.weight.copy_(self.merged_weight())
            if self.base.bias is not None:
                layer.bias.copy_(self.base.bias)
        return layer


def add_lora(
    model: Any,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: list[str] | tuple[str, ...] | None = None,
) -> Any:
    """Freeze the model and attach LoRA adapters to the targeted layers."""
    if rank < 1:
        raise ValueError("rank must be at least 1.")
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules or DEFAULT_TARGETS),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def merge_lora(lora_model: Any) -> Any:
    """Fold the adapters into the base weights and return an ordinary model."""
    return lora_model.merge_and_unload()


def count_parameters_by_kind(model: Any) -> dict[str, float]:
    """How many weights a model has, and how many of them training will change."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_share": trainable / total}


def folder_size_mb(path: Any) -> float:
    """Total size of the files under a folder, in megabytes. Zero if it does not exist."""
    from pathlib import Path

    folder = Path(path)
    if not folder.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1_048_576
