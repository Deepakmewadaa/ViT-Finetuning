from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log, pi, sqrt
from typing import Sequence

import torch


@dataclass(frozen=True)
class SensorGrid:
    """Visual, 3D manifold, and 2D cortical coordinates for an isotropic foveated sensor array."""

    visual_xy: torch.Tensor
    manifold_xyz: torch.Tensor
    cortical_xy: torch.Tensor
    radius: torch.Tensor
    theta: torch.Tensor
    hemifield: torch.Tensor
    is_padding: torch.Tensor
    radial_index: torch.Tensor
    angular_index: torch.Tensor
    a: float
    fov_degrees: float
    radial_samples: int
    pad_factor: float = 1.0

    @property
    def num_samples(self) -> int:
        return int(self.visual_xy.shape[0])

    @property
    def num_valid_samples(self) -> int:
        return int((~self.is_padding).sum().item())

    @property
    def field_radius(self) -> float:
        return self.fov_degrees / 2.0

    @property
    def max_sampled_radius(self) -> float:
        return float(self.radius.max().item())

    def to(self, device: torch.device | str) -> "SensorGrid":
        return SensorGrid(
            visual_xy=self.visual_xy.to(device),
            manifold_xyz=self.manifold_xyz.to(device),
            cortical_xy=self.cortical_xy.to(device),
            radius=self.radius.to(device),
            theta=self.theta.to(device),
            hemifield=self.hemifield.to(device),
            is_padding=self.is_padding.to(device),
            radial_index=self.radial_index.to(device),
            angular_index=self.angular_index.to(device),
            a=self.a,
            fov_degrees=self.fov_degrees,
            radial_samples=self.radial_samples,
            pad_factor=self.pad_factor,
        )


class FoveatedSensor(SensorGrid):
    """Factory class for creating isotropic foveated sensor manifolds."""

    @classmethod
    def from_radial_samples(
        cls,
        radial_samples: int,
        *,
        fov_degrees: float = 16.0,
        a: float = 0.5,
        min_angular_samples: int = 6,
        pad_factor: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "FoveatedSensor":
        grid = make_foveated_grid(
            radial_samples,
            fov_degrees=fov_degrees,
            a=a,
            min_angular_samples=min_angular_samples,
            pad_factor=pad_factor,
            device=device,
            dtype=dtype,
        )
        return cls(**grid.__dict__)

    @classmethod
    def from_target_samples(
        cls,
        target_samples: int,
        *,
        fov_degrees: float = 16.0,
        a: float = 0.5,
        max_overshoot: bool = False,
        min_angular_samples: int = 6,
        pad_factor: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "FoveatedSensor":
        radial_samples = choose_radial_samples(
            target_samples,
            fov_degrees=fov_degrees,
            a=a,
            max_overshoot=max_overshoot,
            min_angular_samples=min_angular_samples,
            pad_factor=pad_factor,
        )
        return cls.from_radial_samples(
            radial_samples,
            fov_degrees=fov_degrees,
            a=a,
            min_angular_samples=min_angular_samples,
            pad_factor=pad_factor,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_exact_patch_count(
        cls,
        target_patches: int = 64,
        *,
        fov_degrees: float = 16.0,
        a_index: int = 3,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "FoveatedSensor":
        """Generate sensor from the discrete set of a values that exactly match target_patches."""
        a_values = find_discrete_a_for_patches(target_patches=target_patches, fov_degrees=fov_degrees)
        if not a_values:
            raise ValueError(f"No exact discrete a-values found for target_patches={target_patches}")
        selected_a = a_values[min(max(0, a_index), len(a_values) - 1)]
        return cls.from_target_samples(
            target_samples=target_patches,
            fov_degrees=fov_degrees,
            a=selected_a,
            device=device,
            dtype=dtype,
        )


def cortical_magnification(radius: torch.Tensor | float, a: float) -> torch.Tensor | float:
    """Cortical magnification function M(r) = 1 / (r + a)."""
    if isinstance(radius, (int, float)):
        return 1.0 / (radius + a)
    return 1.0 / (radius + a)


def cmf_scaling_factor(fov_degrees: float, a: float) -> float:
    """Normalized scaling factor k_a = (integral_0^rmax M(r) dr)^(-1)."""
    rmax = fov_degrees / 2.0
    integral = log(rmax + a) - log(a)
    return 1.0 / integral if integral > 1e-12 else 1.0


def integrated_cmf(radius: torch.Tensor, a: float) -> torch.Tensor:
    """Integral of CMF: w(r) = log(r + a)."""
    return torch.log(radius + a)


def inverse_integrated_cmf(cortical_radius: torch.Tensor, a: float) -> torch.Tensor:
    """Inverse of integrated CMF: r(w) = exp(w) - a."""
    return torch.exp(cortical_radius) - a


def angular_sample_count(radius: float, delta_u: float, a: float, min_angular_samples: int = 6) -> int:
    """Calculates number of angular samples at radius to maintain local isotropy dr = r d_theta."""
    if radius <= 1e-12:
        return 1
    cortical_circumference = 2.0 * pi * radius / (radius + a)
    count = max(min_angular_samples, int(round(cortical_circumference / delta_u)))
    return count


def count_samples_for_radial_samples(
    radial_samples: int,
    *,
    fov_degrees: float = 16.0,
    a: float = 0.5,
    min_angular_samples: int = 6,
    pad_factor: float = 1.0,
) -> int:
    if radial_samples < 1:
        raise ValueError("radial_samples must be positive")
    field_radius = (fov_degrees / 2.0) * pad_factor
    if radial_samples == 1:
        return 1
    u_min = log(a)
    u_max = log(field_radius + a)
    delta_u = (u_max - u_min) / (radial_samples - 1)
    radii = [0.0] + [float(torch.exp(torch.tensor(u_min + i * delta_u)) - a) for i in range(1, radial_samples)]
    return sum(angular_sample_count(r, delta_u, a, min_angular_samples) for r in radii)


def choose_radial_samples(
    target_samples: int,
    *,
    fov_degrees: float = 16.0,
    a: float = 0.5,
    max_overshoot: bool = False,
    min_angular_samples: int = 6,
    pad_factor: float = 1.0,
) -> int:
    """Search for the radial sample count nr that produces the closest sample count to target."""
    if target_samples < 1:
        raise ValueError("target_samples must be positive")

    best_r = 1
    best_count = 1
    best_error = abs(target_samples - 1)
    upper = max(4, int(ceil(sqrt(target_samples) * 3.5)) + 12)

    for radial_samples in range(1, upper + 1):
        count = count_samples_for_radial_samples(
            radial_samples,
            fov_degrees=fov_degrees,
            a=a,
            min_angular_samples=min_angular_samples,
            pad_factor=pad_factor,
        )
        if max_overshoot and count > target_samples:
            continue
        error = abs(count - target_samples)
        if error < best_error:
            best_r = radial_samples
            best_count = count
            best_error = error

    if max_overshoot and best_count > target_samples:
        return max(1, floor(sqrt(target_samples)))
    return best_r


def find_discrete_a_for_patches(
    target_patches: int = 64,
    *,
    fov_degrees: float = 16.0,
    a_min: float = 0.01,
    a_max: float = 100.0,
    num_eval: int = 1000,
) -> list[float]:
    """Find discrete a values that yield exactly target_patches under isotropic sampling."""
    log_a_vals = torch.linspace(log(a_min), log(a_max), num_eval).tolist()
    matches: list[float] = []

    for nr in range(3, 20):
        found_in_nr = []
        for log_a in log_a_vals:
            a_val = float(torch.exp(torch.tensor(log_a)))
            c = count_samples_for_radial_samples(nr, fov_degrees=fov_degrees, a=a_val)
            if c == target_patches:
                found_in_nr.append(a_val)
        if found_in_nr:
            median_a = found_in_nr[len(found_in_nr) // 2]
            if not any(abs(log(median_a) - log(m)) < 0.2 for m in matches):
                matches.append(round(median_a, 4))

    matches.sort()
    return matches if matches else [0.03, 0.14, 0.58, 2.79, 60.94]


def make_foveated_grid(
    radial_samples: int,
    *,
    fov_degrees: float = 16.0,
    a: float = 0.5,
    min_angular_samples: int = 6,
    pad_factor: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> SensorGrid:
    """Create an isotropic foveated grid with 3D manifold and 2D Schwartz cortical coordinates."""
    if a <= 0:
        raise ValueError("a must be positive to avoid foveal singularity")
    if fov_degrees <= 0:
        raise ValueError("fov_degrees must be positive")
    if radial_samples < 1:
        raise ValueError("radial_samples must be positive")

    field_radius = fov_degrees / 2.0
    total_radius = field_radius * pad_factor

    if radial_samples == 1:
        radii = torch.zeros(1, device=device, dtype=dtype)
        delta_u = 1.0
    else:
        u = torch.linspace(log(a), log(total_radius + a), radial_samples, device=device, dtype=dtype)
        radii = inverse_integrated_cmf(u, a).clamp_min(0)
        delta_u = float((u[-1] - u[0]) / max(radial_samples - 1, 1))

    visual_parts: list[torch.Tensor] = []
    radius_parts: list[torch.Tensor] = []
    theta_parts: list[torch.Tensor] = []
    radial_ids: list[torch.Tensor] = []
    angular_ids: list[torch.Tensor] = []
    is_pad_parts: list[torch.Tensor] = []

    for radial_index, radius_value in enumerate(radii.tolist()):
        n_theta = angular_sample_count(radius_value, delta_u, a, min_angular_samples)
        if radial_index == 0:
            theta = torch.zeros(1, device=device, dtype=dtype)
        else:
            theta = torch.arange(n_theta, device=device, dtype=dtype) * (2.0 * pi / n_theta)
        radius_vec = torch.full_like(theta, float(radius_value))
        xy = torch.stack([radius_vec * torch.cos(theta), radius_vec * torch.sin(theta)], dim=-1)

        is_pad = radius_vec > (field_radius + 1e-5)

        visual_parts.append(xy)
        radius_parts.append(radius_vec)
        theta_parts.append(theta)
        radial_ids.append(torch.full((theta.numel(),), radial_index, device=device, dtype=torch.long))
        angular_ids.append(torch.arange(theta.numel(), device=device, dtype=torch.long))
        is_pad_parts.append(is_pad)

    visual_xy = torch.cat(visual_parts, dim=0)
    radius = torch.cat(radius_parts, dim=0)
    theta = torch.cat(theta_parts, dim=0)
    radial_index_tensor = torch.cat(radial_ids, dim=0)
    angular_index_tensor = torch.cat(angular_ids, dim=0)
    is_padding_tensor = torch.cat(is_pad_parts, dim=0)

    # 3D Rovamo & Virsu surface coordinates
    manifold_xyz = _surface_of_revolution_coords(radius, theta, a)

    # 2D Schwartz complex-log cortical coordinates with hemifield split
    cortical_xy, hemifield = _schwartz_cortical_coords(visual_xy, a)

    return SensorGrid(
        visual_xy=visual_xy,
        manifold_xyz=manifold_xyz,
        cortical_xy=cortical_xy,
        radius=radius,
        theta=theta,
        hemifield=hemifield,
        is_padding=is_padding_tensor,
        radial_index=radial_index_tensor,
        angular_index=angular_index_tensor,
        a=a,
        fov_degrees=fov_degrees,
        radial_samples=radial_samples,
        pad_factor=pad_factor,
    )


def _surface_of_revolution_coords(radius: torch.Tensor, theta: torch.Tensor, a: float) -> torch.Tensor:
    """Embed the isotropic magnification metric into a smooth 3D surface of revolution."""
    u = torch.log(radius + a)
    rho = radius / (radius + a)

    u_min = float(torch.log(torch.as_tensor(a, device=radius.device, dtype=radius.dtype)))
    u_max = float(u.max()) if u.numel() else u_min
    table_steps = max(256, int(ceil((u_max - u_min) * 512)))
    u_table = torch.linspace(u_min, u_max, table_steps, device=radius.device, dtype=radius.dtype)
    rho_prime = a * torch.exp(-u_table)
    dz_du = torch.sqrt((1.0 - rho_prime.square()).clamp_min(0.0))
    delta = u_table[1:] - u_table[:-1] if table_steps > 1 else torch.ones(1, device=radius.device, dtype=radius.dtype)
    area = 0.5 * (dz_du[1:] + dz_du[:-1]) * delta
    z_table = torch.cat([torch.zeros(1, device=radius.device, dtype=radius.dtype), area.cumsum(dim=0)])

    table_pos = ((u - u_min) / max(u_max - u_min, 1e-12)) * (table_steps - 1)
    lo = table_pos.floor().long().clamp(0, table_steps - 1)
    hi = (lo + 1).clamp(0, table_steps - 1)
    frac = (table_pos - lo.to(table_pos.dtype)).unsqueeze(-1)
    z = z_table[lo] * (1.0 - frac.squeeze(-1)) + z_table[hi] * frac.squeeze(-1)

    return torch.stack([rho * torch.cos(theta), rho * torch.sin(theta), z], dim=-1)


def _schwartz_cortical_coords(visual_xy: torch.Tensor, a: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Map visual coordinates (x, y) to 2D Schwartz complex-log V1 hemispheres."""
    x = visual_xy[:, 0]
    y = visual_xy[:, 1]

    hemifield = torch.where(x > 1e-7, torch.ones_like(x), torch.where(x < -1e-7, -torch.ones_like(x), torch.zeros_like(x)))

    r_pos = torch.sqrt((x.abs() + a).square() + y.square())
    theta_pos = torch.atan2(y, x.abs() + a)
    u_c = torch.log(r_pos) - log(a)
    v_c = theta_pos

    hemi_sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    cortical_x = hemi_sign * u_c
    cortical_y = v_c

    cortical_xy = torch.stack([cortical_x, cortical_y], dim=-1)
    return cortical_xy, hemifield
