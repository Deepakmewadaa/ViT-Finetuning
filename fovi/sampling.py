from __future__ import annotations

from math import pi
from typing import Optional

import torch
from torch.nn import functional as F

from .geometry import SensorGrid


def random_fixations(
    batch_size: int,
    num_fixations: int = 1,
    *,
    radius: float = 0.25,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample random fixations uniformly inside a central disk in normalized [-1, 1] coordinates.

    Args:
        batch_size: Number of images in batch.
        num_fixations: Number of saccadic fixations per image.
        radius: Radius of fixation disk as fraction of image half-width (0.25 or 0.45).
        device: Target device.
        dtype: Target data type.

    Returns:
        Tensor of shape [batch_size, num_fixations, 2] in normalized [-1, 1] range.
    """
    angle = torch.rand(batch_size, num_fixations, device=device, dtype=dtype) * (2.0 * pi)
    # sqrt ensures uniform area distribution in the circular disk
    radial = torch.sqrt(torch.rand(batch_size, num_fixations, device=device, dtype=dtype)) * radius
    return torch.stack([radial * torch.cos(angle), radial * torch.sin(angle)], dim=-1)


class FoveatedSampler:
    """Samples ambient images using the visual coordinates of an isotropic foveated sensor."""

    def __init__(
        self,
        sensor: SensorGrid,
        *,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        self.sensor = sensor
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def __call__(self, images: torch.Tensor, fixations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample images at sensor visual coordinates with optional saccadic fixations.

        Args:
            images: [batch, channels, height, width] in [-1, 1] or standard image space.
            fixations: Optional [batch, 2] or [batch, num_fixations, 2] relative center offsets.

        Returns:
            Sampled points: [batch, channels, num_samples] or [batch, num_fixations, channels, num_samples].
        """
        if images.ndim != 4:
            raise ValueError(f"images must be [batch, channels, height, width], got shape {images.shape}")

        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype

        # Normalize sensor coordinates by nominal field radius so outer FOV is at edge
        norm_xy = self.sensor.visual_xy.to(device=device, dtype=dtype) / self.sensor.field_radius

        if fixations is None:
            fixations = torch.zeros(batch_size, 1, 2, device=device, dtype=dtype)
        elif fixations.ndim == 2:
            fixations = fixations[:, None, :]
        else:
            fixations = fixations.to(device=device, dtype=dtype)

        if fixations.shape[0] != batch_size or fixations.shape[-1] != 2:
            raise ValueError(f"fixations must be [batch, 2] or [batch, num_fixations, 2], got {fixations.shape}")

        num_fix = fixations.shape[1]
        all_samples = []

        is_padding = self.sensor.is_padding.to(device=device)

        for fixation_idx in range(num_fix):
            # [batch, num_points, 2]
            grid = norm_xy[None, :, :] + fixations[:, fixation_idx, None, :]
            grid = grid[:, :, None, :]  # [batch, num_points, 1, 2]

            sampled = F.grid_sample(
                images,
                grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            ).squeeze(-1)  # [batch, channels, num_points]

            # Zero out boundary padding units
            if is_padding.any():
                sampled = sampled * (~is_padding).float()[None, None, :]

            all_samples.append(sampled)

        if num_fix == 1:
            return all_samples[0]
        return torch.stack(all_samples, dim=1)
