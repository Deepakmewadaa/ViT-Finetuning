from __future__ import annotations

from math import pi
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import FoveatedSensor, SensorGrid


class LogPolarSensor:
    """Log-polar sampling grid with constant angular samples per ring (anisotropic baseline)."""

    def __init__(
        self,
        num_rings: int = 64,
        num_angles: int = 64,
        *,
        fov_degrees: float = 16.0,
        a: float = 0.5,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_rings = num_rings
        self.num_angles = num_angles
        self.fov_degrees = fov_degrees
        self.field_radius = fov_degrees / 2.0
        self.a = a

        # Logarithmic radii
        u = torch.linspace(torch.log(torch.tensor(a)), torch.log(torch.tensor(self.field_radius + a)), num_rings, device=device, dtype=dtype)
        radii = (torch.exp(u) - a).clamp_min(0)

        # Equal angles at each radius
        thetas = torch.linspace(0, 2 * pi, num_angles + 1, device=device, dtype=dtype)[:-1]

        r_grid, theta_grid = torch.meshgrid(radii, thetas, indexing="ij")
        x_grid = r_grid * torch.cos(theta_grid)
        y_grid = r_grid * torch.sin(theta_grid)

        self.visual_xy = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
        self.grid_2d = torch.stack([x_grid / self.field_radius, y_grid / self.field_radius], dim=-1)  # [rings, angles, 2]

    @property
    def num_samples(self) -> int:
        return self.num_rings * self.num_angles


class LogPolarSampler:
    """Samples images into log-polar representations (either 2D log-polar image or 1D points)."""

    def __init__(self, sensor: LogPolarSensor, *, mode: str = "bilinear", align_corners: bool = True) -> None:
        self.sensor = sensor
        self.mode = mode
        self.align_corners = align_corners

    def __call__(
        self,
        images: torch.Tensor,
        fixations: Optional[torch.Tensor] = None,
        as_2d_image: bool = True,
    ) -> torch.Tensor:
        """Sample images.

        Args:
            images: [batch, channels, H, W]
            fixations: Optional [batch, 2] or [batch, num_fix, 2]
            as_2d_image: If True, returns [batch, channels, num_rings, num_angles], else [batch, channels, N]
        """
        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype

        base_grid = self.sensor.grid_2d.to(device=device, dtype=dtype)  # [H_ring, W_ang, 2]

        if fixations is None:
            fixations = torch.zeros(batch_size, 1, 2, device=device, dtype=dtype)
        elif fixations.ndim == 2:
            fixations = fixations[:, None, :]
        else:
            fixations = fixations.to(device=device, dtype=dtype)

        num_fix = fixations.shape[1]
        all_samples = []

        for fix_idx in range(num_fix):
            # [batch, H_ring, W_ang, 2]
            grid = base_grid[None, :, :, :] + fixations[:, fix_idx, None, None, :]
            sampled = F.grid_sample(
                images,
                grid,
                mode=self.mode,
                padding_mode="zeros",
                align_corners=self.align_corners,
            )
            if not as_2d_image:
                sampled = sampled.flatten(start_dim=2)
            all_samples.append(sampled)

        if num_fix == 1:
            return all_samples[0]
        return torch.stack(all_samples, dim=1)


class WarpedCartesianSampler:
    """Warped Cartesian foveated sampling baseline (Wang et al., 2021)."""

    def __init__(
        self,
        resolution: int = 64,
        *,
        alpha: float = 0.5,
        mode: str = "bilinear",
        align_corners: bool = True,
    ) -> None:
        self.resolution = resolution
        self.alpha = alpha
        self.mode = mode
        self.align_corners = align_corners

        # Regular [-1, 1] grid
        lin = torch.linspace(-1.0, 1.0, resolution)
        grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
        r = torch.sqrt(grid_x.square() + grid_y.square()).clamp_min(1e-6)

        # Warping: r_warped = r^(1 + alpha) or exponential mapping
        r_warped = torch.pow(r, 1.0 + alpha)
        scale = r_warped / r
        warped_x = (grid_x * scale).clamp(-1.0, 1.0)
        warped_y = (grid_y * scale).clamp(-1.0, 1.0)

        self.warped_grid = torch.stack([warped_x, warped_y], dim=-1)

    def __call__(
        self,
        images: torch.Tensor,
        fixations: Optional[torch.Tensor] = None,
        as_2d_image: bool = True,
    ) -> torch.Tensor:
        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype

        base_grid = self.warped_grid.to(device=device, dtype=dtype)

        if fixations is None:
            fixations = torch.zeros(batch_size, 1, 2, device=device, dtype=dtype)
        elif fixations.ndim == 2:
            fixations = fixations[:, None, :]
        else:
            fixations = fixations.to(device=device, dtype=dtype)

        num_fix = fixations.shape[1]
        all_samples = []

        for fix_idx in range(num_fix):
            grid = base_grid[None, :, :, :] + fixations[:, fix_idx, None, None, :]
            sampled = F.grid_sample(
                images,
                grid,
                mode=self.mode,
                padding_mode="zeros",
                align_corners=self.align_corners,
            )
            if not as_2d_image:
                sampled = sampled.flatten(start_dim=2)
            all_samples.append(sampled)

        if num_fix == 1:
            return all_samples[0]
        return torch.stack(all_samples, dim=1)


class UniformDownsampleSampler:
    """Uniform Cartesian downsampling baseline (e.g. 64x64 or 224x224)."""

    def __init__(self, target_resolution: int = 64, *, mode: str = "bilinear", align_corners: bool = True) -> None:
        self.target_resolution = target_resolution
        self.mode = mode
        self.align_corners = align_corners

    def __call__(
        self,
        images: torch.Tensor,
        fixations: Optional[torch.Tensor] = None,
        as_2d_image: bool = True,
    ) -> torch.Tensor:
        if fixations is None or (fixations == 0).all():
            downsampled = F.interpolate(
                images,
                size=(self.target_resolution, self.target_resolution),
                mode=self.mode,
                align_corners=self.align_corners,
            )
            if not as_2d_image:
                return downsampled.flatten(start_dim=2)
            return downsampled

        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype

        lin = torch.linspace(-1.0, 1.0, self.target_resolution, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)

        if fixations.ndim == 2:
            fixations = fixations[:, None, :]
        else:
            fixations = fixations.to(device=device, dtype=dtype)

        num_fix = fixations.shape[1]
        all_samples = []

        for fix_idx in range(num_fix):
            grid = base_grid[None, :, :, :] + fixations[:, fix_idx, None, None, :]
            sampled = F.grid_sample(
                images,
                grid,
                mode=self.mode,
                padding_mode="zeros",
                align_corners=self.align_corners,
            )
            if not as_2d_image:
                sampled = sampled.flatten(start_dim=2)
            all_samples.append(sampled)

        if num_fix == 1:
            return all_samples[0]
        return torch.stack(all_samples, dim=1)


def make_weak_fovi_sensor(
    target_samples: int = 4096,
    fov_degrees: float = 16.0,
    a: float = 60.94,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> FoveatedSensor:
    """Create a Weak-FOVI sensor baseline with large a for near-uniform isotropic sampling."""
    return FoveatedSensor.from_target_samples(
        target_samples=target_samples,
        fov_degrees=fov_degrees,
        a=a,
        device=device,
        dtype=dtype,
    )
