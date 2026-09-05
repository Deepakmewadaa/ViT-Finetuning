from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Low-rank adaptation for a linear layer: W_hat = W + (alpha / rank) * (B @ A)."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")

        self.base = base
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return base_out + lora_out * self.scaling


@dataclass(frozen=True)
class LoRAReplacement:
    name: str
    module: LoRALinear


def apply_lora_to_linears(
    module: nn.Module,
    *,
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
    name_filter: Optional[str] = None,
    freeze_base: bool = True,
) -> list[LoRAReplacement]:
    """Recursively replace matching nn.Linear children with LoRALinear modules."""
    replacements: list[LoRAReplacement] = []

    for child_name, child in list(module.named_children()):
        full_match = name_filter is None or name_filter in child_name
        if isinstance(child, nn.Linear) and full_match:
            wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout, freeze_base=freeze_base)
            setattr(module, child_name, wrapped)
            replacements.append(LoRAReplacement(child_name, wrapped))
        else:
            nested = apply_lora_to_linears(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                name_filter=name_filter,
                freeze_base=freeze_base,
            )
            replacements.extend(
                LoRAReplacement(f"{child_name}.{replacement.name}", replacement.module)
                for replacement in nested
            )

    return replacements


def apply_fovi_lora_to_vit(
    vit_model: nn.Module,
    *,
    num_early_layers: int = 6,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> list[LoRAReplacement]:
    """Applies the paper's adaptation strategy to a ViT: LoRA on early layers (0..num_early_layers-1) and trainable patch embed.

    All transformer layers from num_early_layers onwards are frozen.
    The patch embedding and classification head remain trainable.
    """
    # Freeze entire backbone first
    for param in vit_model.parameters():
        param.requires_grad_(False)

    # Locate blocks / layers
    blocks = None
    for attr in ["blocks", "layers", "transformer_blocks", "encoder"]:
        if hasattr(vit_model, attr):
            cand = getattr(vit_model, attr)
            if isinstance(cand, (nn.ModuleList, nn.Sequential, list)):
                blocks = cand
                break
            elif hasattr(cand, "layers"):
                blocks = getattr(cand, "layers")
                break

    all_replacements: list[LoRAReplacement] = []

    if blocks is not None:
        for idx in range(min(num_early_layers, len(blocks))):
            replacements = apply_lora_to_linears(
                blocks[idx],
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                freeze_base=True,
            )
            all_replacements.extend(replacements)

    # Unfreeze patch embedding
    if hasattr(vit_model, "patch_embed"):
        for param in vit_model.patch_embed.parameters():
            param.requires_grad_(True)

    # Unfreeze head / classifier
    for head_name in ["head", "classifier", "fc", "heads"]:
        if hasattr(vit_model, head_name):
            for param in getattr(vit_model, head_name).parameters():
                param.requires_grad_(True)

    return all_replacements
