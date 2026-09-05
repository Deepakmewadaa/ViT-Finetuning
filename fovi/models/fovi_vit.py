from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..geometry import FoveatedSensor, SensorGrid
from ..layers import FOVIViTPatchEmbed


class AttentionBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=qkv_bias,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class FOVIViT(nn.Module):
    """Biologically-inspired Foveated Vision Transformer (FOVI-ViT).

    Combines foveated patchification via kNN-convolution with a standard
    transformer encoder backbone and multi-fixation aggregation.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        *,
        sensor: Optional[SensorGrid] = None,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        patch_target_samples: int = 64,
        patch_a: float = 2.79,
        fov_degrees: float = 16.0,
        target_sensor_samples: int = 3976,
        in_channels: int = 3,
        use_cls_token: bool = True,
        dropout: float = 0.0,
        precompute_weights: bool = False,
    ) -> None:
        super().__init__()
        if sensor is None:
            sensor = FoveatedSensor.from_target_samples(
                target_sensor_samples,
                fov_degrees=fov_degrees,
                a=patch_a,
            )

        self.sensor = sensor
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Foveated patch embedding
        self.patch_embed = FOVIViTPatchEmbed(
            sensor=sensor,
            embed_dim=embed_dim,
            in_channels=in_channels,
            patch_target_samples=patch_target_samples,
            patch_a=patch_a,
            precompute_weights=precompute_weights,
        )
        self.num_patches = self.patch_embed.num_patches

        seq_len = self.num_patches + (1 if use_cls_token else 0)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, images: torch.Tensor, fixations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract sequence features from ambient images.

        Args:
            images: [batch, channels, H, W]
            fixations: Optional [batch, 2] or [batch, num_fix, 2]

        Returns:
            Tokens [B, seq_len, embed_dim] or [B, num_fix, seq_len, embed_dim]
        """
        # [batch, num_patches, embed_dim] or [batch, num_fix, num_patches, embed_dim]
        tokens = self.patch_embed(images, fixations)

        if tokens.ndim == 4:
            batch, num_fix, n_patches, dim = tokens.shape
            flat_tokens = tokens.reshape(batch * num_fix, n_patches, dim)
            out = self._forward_encoder(flat_tokens)
            return out.reshape(batch, num_fix, -1, dim)

        return self._forward_encoder(tokens)

    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def forward(
        self,
        images: torch.Tensor,
        fixations: Optional[torch.Tensor] = None,
        return_fixation_logits: bool = False,
    ) -> torch.Tensor:
        """Forward classification pass.

        Args:
            images: [batch, 3, H, W]
            fixations: Optional [batch, num_fix, 2]
            return_fixation_logits: If True and multiple fixations, returns [batch, num_fix, num_classes]

        Returns:
            Logits [batch, num_classes] (averaged across fixations)
        """
        feats = self.forward_features(images, fixations)
        if feats.ndim == 4:
            batch, num_fix, seq_len, dim = feats.shape
            pooled = feats[:, :, 0] if self.cls_token is not None else feats.mean(dim=2)
            logits = self.head(pooled.reshape(batch * num_fix, dim)).reshape(batch, num_fix, -1)
            if return_fixation_logits:
                return logits
            return logits.mean(dim=1)

        pooled = feats[:, 0] if self.cls_token is not None else feats.mean(dim=1)
        return self.head(pooled)


def fovi_vit_small(
    num_classes: int = 1000,
    *,
    patch_a: float = 2.79,
    target_sensor_samples: int = 3976,
    patch_target_samples: int = 64,
    **kwargs,
) -> FOVIViT:
    """FOVI-ViT-S+ configuration (dim=384, depth=12, heads=6)."""
    return FOVIViT(
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        patch_a=patch_a,
        target_sensor_samples=target_sensor_samples,
        patch_target_samples=patch_target_samples,
        **kwargs,
    )


def fovi_vit_base(
    num_classes: int = 1000,
    *,
    patch_a: float = 2.79,
    target_sensor_samples: int = 3976,
    patch_target_samples: int = 64,
    **kwargs,
) -> FOVIViT:
    """FOVI-ViT-B configuration (dim=768, depth=12, heads=12)."""
    return FOVIViT(
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_a=patch_a,
        target_sensor_samples=target_sensor_samples,
        patch_target_samples=patch_target_samples,
        **kwargs,
    )


def fovi_vit_huge(
    num_classes: int = 1000,
    *,
    patch_a: float = 2.79,
    target_sensor_samples: int = 3976,
    patch_target_samples: int = 64,
    **kwargs,
) -> FOVIViT:
    """FOVI-ViT-H+ configuration (dim=1280, depth=32, heads=16)."""
    return FOVIViT(
        num_classes=num_classes,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        patch_a=patch_a,
        target_sensor_samples=target_sensor_samples,
        patch_target_samples=patch_target_samples,
        **kwargs,
    )
