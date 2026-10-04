"""显式的训练参数与调度；所有 batch 大小的单位都是序列数。"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TrainRecipe:
    total_tokens: int
    global_batch_size: int = 256
    micro_batch_size: int = 8
    learning_rate: float = 6e-4
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 2026
    precision: str = "bf16"

    def validate(self, world_size: int) -> None:
        if min(self.total_tokens, self.global_batch_size, self.micro_batch_size) <= 0:
            raise ValueError("Token budget and batch sizes must be positive")
        if self.global_batch_size % (world_size * self.micro_batch_size):
            raise ValueError("Global batch must be divisible by world size × micro batch")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid learning-rate schedule")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("Invalid gradient clipping threshold")
        if self.seed < 0 or self.precision not in ("bf16", "fp32"):
            raise ValueError("Invalid seed or precision")


def learning_rate_factor(index: int, total_steps: int, warmup: int, minimum: float) -> float:
    if index < warmup:
        return (index + 1) / warmup
    fraction = min(1.0, (index - warmup) / max(1, total_steps - warmup - 1))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * fraction))


def make_optimizer(model, recipe: TrainRecipe, cuda: bool):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": recipe.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=recipe.learning_rate,
        betas=(0.9, 0.95),
        fused=cuda,
    )
