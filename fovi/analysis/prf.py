from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ..models.fovi_cnn import FOVICNN


def backproject_receptive_fields(model: FOVICNN) -> dict[str, torch.Tensor]:
    """Backproject receptive field neighborhoods from each FOVI-CNN layer to input sensor indices.

    Returns:
        Dictionary mapping layer names ('conv1', 'pool1', 'conv2', 'conv3', 'conv4', 'pool2', 'conv5')
        to a boolean mask tensor [num_units_in_layer, num_input_sensor_samples] indicating the
        input points comprising the receptive field of each unit.
    """
    # 1. Layer 1 indices: [grid1_pts, k1]
    rf_masks: dict[str, torch.Tensor] = {}
    num_input = model.sensor.num_samples

    # conv1: output is grid1
    k1_indices = model.conv1.indices  # [grid1_pts, k1]
    mask_conv1 = torch.zeros(model.grid1.num_samples, num_input, dtype=torch.bool)
    for i in range(model.grid1.num_samples):
        mask_conv1[i, k1_indices[i]] = True
    rf_masks["conv1"] = mask_conv1

    # pool1: input is grid1, output is grid2
    p1_indices = model.pool1.indices  # [grid2_pts, k_pool]
    mask_pool1 = torch.zeros(model.grid2.num_samples, num_input, dtype=torch.bool)
    for i in range(model.grid2.num_samples):
        # union over pool1 receptive units in grid1
        mask_pool1[i] = mask_conv1[p1_indices[i]].any(dim=0)
    rf_masks["pool1"] = mask_pool1

    # conv2: input is grid2, output is grid2
    k2_indices = model.conv2.indices  # [grid2_pts, k2]
    mask_conv2 = torch.zeros(model.grid2.num_samples, num_input, dtype=torch.bool)
    for i in range(model.grid2.num_samples):
        mask_conv2[i] = mask_pool1[k2_indices[i]].any(dim=0)
    rf_masks["conv2"] = mask_conv2

    # conv3: input is grid2, output is grid2
    k3_indices = model.conv3.indices  # [grid2_pts, k3]
    mask_conv3 = torch.zeros(model.grid2.num_samples, num_input, dtype=torch.bool)
    for i in range(model.grid2.num_samples):
        mask_conv3[i] = mask_conv2[k3_indices[i]].any(dim=0)
    rf_masks["conv3"] = mask_conv3

    # conv4: input is grid2, output is grid2
    k4_indices = model.conv4.indices  # [grid2_pts, k4]
    mask_conv4 = torch.zeros(model.grid2.num_samples, num_input, dtype=torch.bool)
    for i in range(model.grid2.num_samples):
        mask_conv4[i] = mask_conv3[k4_indices[i]].any(dim=0)
    rf_masks["conv4"] = mask_conv4

    # pool2: input is grid2, output is grid3
    p2_indices = model.pool2.indices  # [grid3_pts, k_pool]
    mask_pool2 = torch.zeros(model.grid3.num_samples, num_input, dtype=torch.bool)
    for i in range(model.grid3.num_samples):
        mask_pool2[i] = mask_conv4[p2_indices[i]].any(dim=0)
    rf_masks["pool2"] = mask_pool2

    # conv5: input is grid3, output is grid3
    k5_indices = model.conv5.indices  # [grid3_pts, k5]
    mask_conv5 = torch.zeros(model.grid3.num_samples, num_input, dtype=torch.bool)
    for i in range(model.grid3.num_samples):
        mask_conv5[i] = mask_pool2[k5_indices[i]].any(dim=0)
    rf_masks["conv5"] = mask_conv5

    return rf_masks


def compute_layer_rf_diameters(
    model: FOVICNN,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute empirical population receptive field (pRF) diameter vs eccentricity for each layer.

    Returns:
        dict mapping layer name -> (eccentricities, diameters) in degrees of visual field.
    """
    rf_masks = backproject_receptive_fields(model)
    input_visual_xy = model.sensor.visual_xy.float()  # [N_in, 2]
    is_padding = model.sensor.is_padding  # [N_in]

    results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    layer_grids = {
        "conv1": model.grid1,
        "pool1": model.grid2,
        "conv2": model.grid2,
        "conv3": model.grid2,
        "conv4": model.grid2,
        "pool2": model.grid3,
        "conv5": model.grid3,
    }

    for layer_name, mask in rf_masks.items():
        grid = layer_grids[layer_name]
        eccentricities = grid.radius.float()  # [N_layer]
        diameters = torch.zeros(grid.num_samples)

        for unit_idx in range(grid.num_samples):
            # Exclude padding units for RF diameter computation
            valid_rf_pts = input_visual_xy[mask[unit_idx] & (~is_padding)]
            if valid_rf_pts.shape[0] < 2:
                diameters[unit_idx] = 0.1
                continue

            # Equivalent circular diameter = 2 * sqrt(Area / pi) or max distance / 2 * std
            centroid = valid_rf_pts.mean(dim=0, keepdim=True)
            dists = torch.norm(valid_rf_pts - centroid, dim=-1)
            # 2 * (standard deviation * 2) or max spread
            diameters[unit_idx] = dists.max() * 2.0

        results[layer_name] = (eccentricities, diameters)

    return results


def fit_motter_isotropy_aspect_ratios(
    rf_mask: torch.Tensor,
    visual_xy: torch.Tensor,
    min_points: int = 4,
) -> torch.Tensor:
    """Compute Motter (2009) receptive field aspect ratios (short axis / long axis) via PCA/SVD.

    Args:
        rf_mask: Boolean tensor [num_units, num_input_points]
        visual_xy: Visual space coordinates [num_input_points, 2]
        min_points: Minimum points required in RF to fit Gaussian

    Returns:
        Tensor of aspect ratios in range [0, 1]. An isotropic RF has ratio ~ 1.0.
    """
    aspect_ratios = []

    for i in range(rf_mask.shape[0]):
        pts = visual_xy[rf_mask[i]]
        if pts.shape[0] < min_points:
            continue

        centered = pts - pts.mean(dim=0, keepdim=True)
        # SVD of centered points: covariance eigenvalues are singular values squared
        _, S, _ = torch.linalg.svd(centered)
        if S.numel() < 2 or S[0] < 1e-6:
            continue
        ratio = float(S[1] / S[0])
        aspect_ratios.append(min(1.0, max(0.0, ratio)))

    if not aspect_ratios:
        return torch.ones(1)
    return torch.tensor(aspect_ratios)
