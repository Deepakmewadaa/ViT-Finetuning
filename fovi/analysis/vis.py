from __future__ import annotations

from typing import Optional

import torch

from ..geometry import SensorGrid
from ..models.fovi_cnn import FOVICNN
from .prf import backproject_receptive_fields, compute_layer_rf_diameters, fit_motter_isotropy_aspect_ratios


def plot_sensor_manifold(
    sensor: SensorGrid,
    save_path: Optional[str] = None,
    title_suffix: str = "",
) -> None:
    """Plot visual space sensor coordinates, 3D Rovamo-Virsu manifold, and 2D Schwartz cortical map."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    fig = plt.figure(figsize=(15, 4.5))

    # 1. Visual space coordinates
    ax1 = fig.add_subplot(131)
    v_xy = sensor.visual_xy.cpu().numpy()
    ax1.scatter(v_xy[:, 0], v_xy[:, 1], s=8, alpha=0.7, c=sensor.radius.cpu().numpy(), cmap="viridis")
    ax1.set_title(f"Visual Space (FOV={sensor.fov_degrees}°){title_suffix}")
    ax1.set_xlabel("Horizontal visual angle (°)")
    ax1.set_ylabel("Vertical visual angle (°)")
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. 3D Rovamo-Virsu manifold
    ax2 = fig.add_subplot(132, projection="3d")
    m_xyz = sensor.manifold_xyz.cpu().numpy()
    ax2.scatter(m_xyz[:, 0], m_xyz[:, 1], m_xyz[:, 2], s=8, alpha=0.7, c=sensor.radius.cpu().numpy(), cmap="viridis")
    ax2.set_title(f"3D Sensor Manifold (Rovamo & Virsu){title_suffix}")
    ax2.set_xlabel("X (rho cos θ)")
    ax2.set_ylabel("Y (rho sin θ)")
    ax2.set_zlabel("Z (cortical depth)")

    # 3. 2D Schwartz flat complex-log cortical map
    ax3 = fig.add_subplot(133)
    c_xy = sensor.cortical_xy.cpu().numpy()
    ax3.scatter(c_xy[:, 0], c_xy[:, 1], s=8, alpha=0.7, c=sensor.hemifield.cpu().numpy(), cmap="coolwarm")
    ax3.set_title(f"2D Cortical Manifold (Schwartz Hemifields){title_suffix}")
    ax3.set_xlabel("Left Hemisphere <--- | ---> Right Hemisphere")
    ax3.set_ylabel("Polar Angle (v)")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_prf_scaling(
    model: FOVICNN,
    save_path: Optional[str] = None,
) -> None:
    """Plot population receptive field (pRF) diameter vs. visual eccentricity across layers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    results = compute_layer_rf_diameters(model)

    fig, ax = plt.subplots(figsize=(8, 5))
    for layer_name, (ecc, diams) in results.items():
        # Sort by eccentricity
        sorted_idx = torch.argsort(ecc)
        ax.plot(
            ecc[sorted_idx].cpu().numpy(),
            diams[sorted_idx].cpu().numpy(),
            label=layer_name,
            marker="o",
            markersize=3,
            alpha=0.8,
        )

    ax.set_title(f"FOVI-CNN Population Receptive Field (pRF) Scaling (a={model.sensor.a}°)")
    ax.set_xlabel("Eccentricity r (°)")
    ax.set_ylabel("RF Diameter (°)")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_motter_isotropy(
    aspect_ratios: torch.Tensor,
    save_path: Optional[str] = None,
) -> None:
    """Plot aspect ratio histogram of backprojected receptive fields."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(aspect_ratios.cpu().numpy(), bins=20, range=(0.0, 1.0), edgecolor="black", alpha=0.75, color="#1f77b4")
    ax.axvline(1.0, color="red", linestyle="--", label="Ideal Isotropy (1.0)")
    ax.set_title("Receptive Field Aspect Ratio Distribution (Motter 2009)")
    ax.set_xlabel("Short axis / Long axis ratio")
    ax.set_ylabel("Number of Units")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
