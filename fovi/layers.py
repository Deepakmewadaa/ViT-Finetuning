from __future__ import annotations

from math import ceil, sqrt
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import FoveatedSensor, SensorGrid
from .knn import covering_knn_size, knn_indices
from .sampling import FoveatedSampler


def compute_reference_kernel_grid(
    input_grid: SensorGrid,
    output_grid: SensorGrid,
    indices: torch.Tensor,
    distances: torch.Tensor,
) -> torch.Tensor:
    """Compute local normalized Cartesian grid coordinates for sampling the reference kernel.

    For each output unit, the polar angle is computed from the visual space displacement,
    and the radius is given by the manifold distance. The resulting local Cartesian
    coordinates are normalized to [-1, 1].
    """
    input_xy = input_grid.visual_xy.float()[indices]  # [num_out, k, 2]
    output_xy = output_grid.visual_xy.float()[:, None, :]  # [num_out, 1, 2]
    delta_xy = input_xy - output_xy  # [num_out, k, 2]

    theta = torch.atan2(delta_xy[..., 1], delta_xy[..., 0])  # [num_out, k]
    local_x = distances.float() * torch.cos(theta)
    local_y = distances.float() * torch.sin(theta)

    coords = torch.stack([local_x, local_y], dim=-1)  # [num_out, k, 2]
    scale = coords.norm(dim=-1).amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return (coords / scale[..., None]).clamp(-1.0, 1.0)


class KNNConv(nn.Module):
    """Kernel-mapped k-nearest-neighbor convolution on a foveated sensor manifold.

    Maps a single learned 2D Cartesian reference filter W into each irregular kNN
    neighborhood on the sensor manifold, enabling orientation-consistent convolutional
    weight sharing across varying eccentricities.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_grid: SensorGrid,
        output_grid: SensorGrid,
        *,
        kernel_size: int,
        reference_scale: float = 2.0,
        bias: bool = True,
        precompute_weights: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.kernel_size = min(kernel_size, input_grid.num_samples)
        self.reference_scale = reference_scale
        self.reference_side = max(2, int(ceil(reference_scale * sqrt(kernel_size))))

        weight = torch.empty(out_channels, in_channels, self.reference_side, self.reference_side)
        nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        indices, distances = knn_indices(
            output_grid.manifold_xyz.float(),
            input_grid.manifold_xyz.float(),
            self.kernel_size,
            include_distances=True,
        )
        reference_grid = compute_reference_kernel_grid(input_grid, output_grid, indices, distances)
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer("reference_grid", reference_grid, persistent=False)

        self._cached_weight: Optional[torch.Tensor] = None
        self.is_precomputed = precompute_weights
        if precompute_weights:
            self.precompute_weights()

    def precompute_weights(self) -> None:
        """Pre-sample reference kernel weights for accelerated forward pass without grid_sample."""
        with torch.no_grad():
            grid = self.reference_grid.to(device=self.weight.device, dtype=self.weight.dtype)[None, :, :, :]
            sampled = F.grid_sample(
                self.weight.reshape(1, self.out_channels * self.in_channels, self.reference_side, self.reference_side),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(self.out_channels, self.in_channels, self.output_grid.num_samples, self.kernel_size)
            self._cached_weight = sampled
            self.is_precomputed = True

    def clear_cached_weights(self) -> None:
        self._cached_weight = None
        self.is_precomputed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, in_channels, input_points]

        Returns:
            Output tensor of shape [batch, out_channels, output_points]
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [batch, channels, input_points], got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.shape[1]}")
        if x.shape[2] != self.input_grid.num_samples:
            raise ValueError(f"expected {self.input_grid.num_samples} input points, got {x.shape[2]}")

        batch_size = x.shape[0]
        indices = self.indices.to(x.device)

        # Gather neighbor activations: [batch, channels, num_out, k]
        gathered = x.index_select(2, indices.reshape(-1)).reshape(
            batch_size, self.in_channels, self.output_grid.num_samples, self.kernel_size
        )

        if self.is_precomputed and self._cached_weight is not None and not self.training:
            mapped_weight = self._cached_weight.to(device=x.device, dtype=x.dtype)
        else:
            grid = self.reference_grid.to(device=x.device, dtype=x.dtype)[None, :, :, :]
            mapped_weight = F.grid_sample(
                self.weight.reshape(1, self.out_channels * self.in_channels, self.reference_side, self.reference_side),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(self.out_channels, self.in_channels, self.output_grid.num_samples, self.kernel_size)

        # Convolve: out[b, o, n] = sum_{c, k} gathered[b, c, n, k] * mapped_weight[o, c, n, k]
        out = torch.einsum("bcnk,ocnk->bon", gathered, mapped_weight)
        if self.bias is not None:
            out = out + self.bias[None, :, None]
        return out


class KNNMaxPool(nn.Module):
    """Max-pooling across k-nearest neighborhoods on the sensor manifold."""

    def __init__(self, input_grid: SensorGrid, output_grid: SensorGrid, *, kernel_size: int = 9) -> None:
        super().__init__()
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.kernel_size = min(kernel_size, input_grid.num_samples)
        indices = knn_indices(output_grid.manifold_xyz.float(), input_grid.manifold_xyz.float(), self.kernel_size)
        self.register_buffer("indices", indices, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [batch, channels, input_points]

        Returns:
            Pooled features [batch, channels, output_points]
        """
        if x.ndim != 3:
            raise ValueError("x must be [batch, channels, input_points]")
        gathered = x.index_select(2, self.indices.to(x.device).reshape(-1)).reshape(
            x.shape[0], x.shape[1], self.output_grid.num_samples, self.kernel_size
        )
        return gathered.max(dim=-1).values


class FOVIViTPatchEmbed(nn.Module):
    """Foveated patch embedding projecting sensor manifold points to transformer patch tokens."""

    def __init__(
        self,
        *,
        sensor: SensorGrid,
        embed_dim: int,
        in_channels: int = 3,
        patch_target_samples: int = 64,
        patch_a: float | None = None,
        patch_kernel_size: int | None = None,
        reference_scale: float = 2.0,
        precompute_weights: bool = False,
    ) -> None:
        super().__init__()
        patch_grid = FoveatedSensor.from_target_samples(
            patch_target_samples,
            fov_degrees=sensor.fov_degrees,
            a=sensor.a if patch_a is None else patch_a,
            dtype=sensor.visual_xy.dtype,
            device=sensor.visual_xy.device,
        )
        if patch_kernel_size is None:
            patch_kernel_size = covering_knn_size(sensor.manifold_xyz.float(), patch_grid.manifold_xyz.float())

        self.sensor = sensor
        self.patch_grid = patch_grid
        self.sampler = FoveatedSampler(sensor)
        self.proj = KNNConv(
            in_channels,
            embed_dim,
            sensor,
            patch_grid,
            kernel_size=patch_kernel_size,
            reference_scale=reference_scale,
            precompute_weights=precompute_weights,
        )

    @property
    def num_patches(self) -> int:
        return self.patch_grid.num_samples

    def forward(self, images: torch.Tensor, fixations: torch.Tensor | None = None) -> torch.Tensor:
        """Embed ambient images at specified fixations into foveated token sequence.

        Args:
            images: [batch, channels, H, W]
            fixations: Optional [batch, 2] or [batch, num_fixations, 2]

        Returns:
            Tokens [batch, num_patches, embed_dim] or [batch, num_fixations, num_patches, embed_dim]
        """
        samples = self.sampler(images, fixations)
        if samples.ndim == 4:
            batch, num_fixations, channels, points = samples.shape
            flat_samples = samples.reshape(batch * num_fixations, channels, points)
            tokens = self.proj(flat_samples).transpose(1, 2)
            return tokens.reshape(batch, num_fixations, self.num_patches, -1)
        return self.proj(samples).transpose(1, 2)

    @torch.no_grad()
    def load_patch_kernel(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        """Initialize reference filter from a standard ViT Conv2d patch embedding weight [D, C, H, W]."""
        if weight.ndim != 4:
            raise ValueError("weight must be [embed_dim, in_channels, height, width]")
        if weight.shape[:2] != self.proj.weight.shape[:2]:
            raise ValueError(
                f"weight channels ({weight.shape[:2]}) must match foveated projection ({self.proj.weight.shape[:2]})"
            )
        resized = weight.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        if resized.shape[-2:] != self.proj.weight.shape[-2:]:
            resized = F.interpolate(
                resized.reshape(-1, 1, resized.shape[-2], resized.shape[-1]),
                size=self.proj.weight.shape[-2:],
                mode="bicubic",
                align_corners=True,
            ).reshape_as(self.proj.weight)
        self.proj.weight.copy_(resized)
        if bias is not None and self.proj.bias is not None:
            self.proj.bias.copy_(bias.to(device=self.proj.bias.device, dtype=self.proj.bias.dtype))
        if self.proj.is_precomputed:
            self.proj.precompute_weights()
