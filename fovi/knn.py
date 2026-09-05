from __future__ import annotations

import torch


def knn_indices(
    query_xyz: torch.Tensor,
    key_xyz: torch.Tensor,
    k: int,
    *,
    include_distances: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return k nearest key indices (and optionally distances) for each query point on the manifold."""
    if query_xyz.ndim != 2 or key_xyz.ndim != 2:
        raise ValueError("query_xyz and key_xyz must be [num_points, dims]")
    if query_xyz.shape[-1] != key_xyz.shape[-1]:
        raise ValueError("query_xyz and key_xyz must have the same coordinate dimension")
    if k < 1:
        raise ValueError("k must be positive")
    if k > key_xyz.shape[0]:
        raise ValueError(f"k ({k}) cannot exceed the number of key points ({key_xyz.shape[0]})")

    distances = torch.cdist(query_xyz, key_xyz)
    values, indices = torch.topk(distances, k=k, largest=False, dim=-1)
    if include_distances:
        return indices, values
    return indices


def covering_knn_size(sensor_manifold_xyz: torch.Tensor, patch_manifold_xyz: torch.Tensor) -> int:
    """Find the minimum k such that every sensor location is included in at least one patch neighborhood."""
    distances = torch.cdist(sensor_manifold_xyz, patch_manifold_xyz)  # [num_sensor, num_patches]
    # For each sensor point, find distance to its nearest patch center
    nearest_patch_for_sensor = distances.argmin(dim=-1)
    counts = torch.bincount(nearest_patch_for_sensor, minlength=patch_manifold_xyz.shape[0])
    return int(max(1, counts.max().item()))
