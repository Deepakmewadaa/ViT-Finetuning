from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import FoveatedSensor, SensorGrid
from .layers import FOVIViTPatchEmbed
from .lora import apply_fovi_lora_to_vit


class FOVIAdapter(nn.Module):
    """Adapter module to convert standard Vision Transformers (e.g. timm, DINOv2, DINOv3) into FOVI-ViTs.

    Replaces standard Cartesian Conv2d patch embedding with FOVIViTPatchEmbed,
    interpolates positional embeddings to the foveated patch count, and optionally applies
    early-layer LoRA adaptation.
    """

    def __init__(
        self,
        base_vit: nn.Module,
        *,
        sensor: Optional[SensorGrid] = None,
        patch_target_samples: int = 64,
        patch_a: float = 2.79,
        fov_degrees: float = 16.0,
        target_sensor_samples: int = 3976,
        copy_conv_weights: bool = True,
        apply_lora: bool = True,
        num_lora_layers: int = 6,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
    ) -> None:
        super().__init__()
        self.vit = base_vit

        if sensor is None:
            sensor = FoveatedSensor.from_target_samples(
                target_sensor_samples,
                fov_degrees=fov_degrees,
                a=patch_a,
            )
        self.sensor = sensor

        # Locate base patch embedding and extract dimensions
        old_patch_embed = getattr(base_vit, "patch_embed", None)
        if old_patch_embed is None:
            raise ValueError("base_vit does not have a 'patch_embed' attribute")

        embed_dim = getattr(base_vit, "embed_dim", None)
        if embed_dim is None:
            if hasattr(old_patch_embed, "proj") and hasattr(old_patch_embed.proj, "out_channels"):
                embed_dim = old_patch_embed.proj.out_channels
            elif hasattr(old_patch_embed, "embed_dim"):
                embed_dim = old_patch_embed.embed_dim
            else:
                embed_dim = 384  # default ViT-S

        # Create FOVI patch embedding
        self.fovi_patch_embed = FOVIViTPatchEmbed(
            sensor=sensor,
            embed_dim=embed_dim,
            patch_target_samples=patch_target_samples,
            patch_a=patch_a,
        )

        # Copy and interpolate existing Conv2d patch projection weights
        if copy_conv_weights and hasattr(old_patch_embed, "proj") and isinstance(old_patch_embed.proj, nn.Conv2d):
            self.fovi_patch_embed.load_patch_kernel(
                old_patch_embed.proj.weight.data,
                old_patch_embed.proj.bias.data if old_patch_embed.proj.bias is not None else None,
            )

        # Replace patch embed in base model
        self.vit.patch_embed = self.fovi_patch_embed
        self.num_patches = self.fovi_patch_embed.num_patches

        # Adjust pos_embed shape if present
        if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None:
            self._adjust_pos_embed()

        # Apply early-layer LoRA if requested
        if apply_lora:
            self.lora_replacements = apply_fovi_lora_to_vit(
                self.vit,
                num_early_layers=num_lora_layers,
                rank=lora_rank,
                alpha=lora_alpha,
            )
        else:
            self.lora_replacements = []

    def _adjust_pos_embed(self) -> None:
        """Interpolate positional embeddings to match the foveated patch sequence length."""
        old_pos = self.vit.pos_embed.data  # [1, N_old, D]
        num_new = self.num_patches
        has_cls = (old_pos.shape[1] > 1) and (hasattr(self.vit, "cls_token") and self.vit.cls_token is not None)

        if has_cls:
            cls_pos = old_pos[:, :1]
            patch_pos = old_pos[:, 1:]
            num_old_patches = patch_pos.shape[1]
        else:
            cls_pos = None
            patch_pos = old_pos
            num_old_patches = patch_pos.shape[1]

        if num_old_patches != num_new:
            # 1D interpolation of patch position embeddings
            # [1, D, N_old] -> [1, D, N_new]
            interpolated = F.interpolate(
                patch_pos.transpose(1, 2),
                size=num_new,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

            if cls_pos is not None:
                new_pos = torch.cat([cls_pos, interpolated], dim=1)
            else:
                new_pos = interpolated

            self.vit.pos_embed = nn.Parameter(new_pos)

    def forward(
        self,
        images: torch.Tensor,
        fixations: Optional[torch.Tensor] = None,
        return_fixation_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass with optional multi-fixation averaging.

        Args:
            images: [batch, 3, H, W]
            fixations: Optional [batch, 2] or [batch, num_fix, 2]
        """
        if fixations is not None and fixations.ndim >= 2 and (fixations.ndim == 3 and fixations.shape[1] > 1):
            num_fix = fixations.shape[1]
            batch_size = images.shape[0]
            fix_logits = []
            for i in range(num_fix):
                single_fix = fixations[:, i : i + 1, :]
                logits = self.vit(images, fixations=single_fix) if "fixations" in self.vit.forward.__code__.co_varnames else self.vit(images)
                fix_logits.append(logits)
            stacked = torch.stack(fix_logits, dim=1)
            if return_fixation_logits:
                return stacked
            return stacked.mean(dim=1)

        if "fixations" in self.vit.forward.__code__.co_varnames:
            return self.vit(images, fixations=fixations)
        return self.vit(images)
