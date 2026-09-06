"""
Publication-Quality Analysis & Visualization Suite for FOVI vs. Baseline ViT Fine-Tuning.

Generates 6 in-depth analytical figures:
1. Training & Validation Dynamics across 50 Epochs (Loss, Top-1, Top-5, Learning Rate)
2. Pareto Efficiency & Hardware Frontier (Accuracy vs. GFLOPs, Latency, Throughput)
3. Foveated Sensor Manifold & Retinal Patchification Geometry (Visual, Rovamo-Virsu, Schwartz, Patches)
4. LoRA Weight Adaptation Analysis (Layer-wise Frobenius update norms & Singular value spectra)
5. Cortical Magnification & Population Receptive Field (pRF) Scaling vs. Eccentricity
6. Master Benchmark Summary Dashboard
"""

import os
import json
import math
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# Set high DPI and aesthetic styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "figure.autolayout": False,
})

COLORS = {
    "baseline": "#1f77b4",  # Deep Blue
    "fovi": "#d62728",      # Crimson Red
    "fovi_3fix": "#9467bd", # Purple
    "accent": "#2ca02c",    # Forest Green
    "amber": "#ff7f0e",     # Orange/Amber
}


def load_histories(checkpoints_dir: str = "checkpoints"):
    base_file = os.path.join(checkpoints_dir, "history_baseline.json")
    fovi_file = os.path.join(checkpoints_dir, "history_fovi.json")

    base_hist = []
    fovi_hist = []
    if os.path.exists(base_file):
        with open(base_file, "r") as f:
            base_hist = json.load(f)
    if os.path.exists(fovi_file):
        with open(fovi_file, "r") as f:
            fovi_hist = json.load(f)
    return base_hist, fovi_hist


def load_final_summary(summary_path: str = "final_summary.json"):
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return []


# ==============================================================================
# PLOT 1: Training & Validation Dynamics (50 Epochs)
# ==============================================================================
def plot_training_dynamics(base_hist, fovi_hist, output_dir: str):
    if not base_hist and not fovi_hist:
        print("[Plotter] No history data found. Skipping Plot 1.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Epochs
    b_epochs = [h["epoch"] for h in base_hist]
    f_epochs = [h["epoch"] for h in fovi_hist]

    # --- Panel A: Cross-Entropy Loss ---
    ax = axes[0, 0]
    if base_hist:
        ax.plot(b_epochs, [h["train_loss"] for h in base_hist], color=COLORS["baseline"], linestyle="--", alpha=0.7, label="Baseline (Train)")
        ax.plot(b_epochs, [h["val_loss"] for h in base_hist], color=COLORS["baseline"], linewidth=2.2, label="Baseline (Val)")
    if fovi_hist:
        ax.plot(f_epochs, [h["train_loss"] for h in fovi_hist], color=COLORS["fovi"], linestyle="--", alpha=0.7, label="FOVI @ 64 (Train)")
        ax.plot(f_epochs, [h["val_loss"] for h in fovi_hist], color=COLORS["fovi"], linewidth=2.2, label="FOVI @ 64 (Val)")
    ax.set_title("A. Cross-Entropy Loss Trajectory")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(1, max(max(b_epochs or [1]), max(f_epochs or [1])))

    # --- Panel B: Top-1 Accuracy (%) ---
    ax = axes[0, 1]
    if base_hist:
        b_val_top1 = [h["val_top1"] for h in base_hist]
        b_best_idx = int(np.argmax(b_val_top1))
        ax.plot(b_epochs, [h["train_top1"] for h in base_hist], color=COLORS["baseline"], linestyle="--", alpha=0.6, label="Baseline (Train)")
        ax.plot(b_epochs, b_val_top1, color=COLORS["baseline"], linewidth=2.2, label="Baseline (Val)")
        ax.scatter([b_epochs[b_best_idx]], [b_val_top1[b_best_idx]], color=COLORS["baseline"], s=90, zorder=5, edgecolors="black")
        ax.annotate(f"Best: {b_val_top1[b_best_idx]:.2f}%", (b_epochs[b_best_idx], b_val_top1[b_best_idx]),
                    xytext=(b_epochs[b_best_idx] - 10, b_val_top1[b_best_idx] - 5),
                    arrowprops=dict(arrowstyle="->", color=COLORS["baseline"], lw=1.5),
                    fontweight="bold", color=COLORS["baseline"])

    if fovi_hist:
        f_val_top1 = [h["val_top1"] for h in fovi_hist]
        f_best_idx = int(np.argmax(f_val_top1))
        ax.plot(f_epochs, [h["train_top1"] for h in fovi_hist], color=COLORS["fovi"], linestyle="--", alpha=0.6, label="FOVI @ 64 (Train)")
        ax.plot(f_epochs, f_val_top1, color=COLORS["fovi"], linewidth=2.2, label="FOVI @ 64 (Val)")
        ax.scatter([f_epochs[f_best_idx]], [f_val_top1[f_best_idx]], color=COLORS["fovi"], s=90, zorder=5, edgecolors="black")
        ax.annotate(f"Best: {f_val_top1[f_best_idx]:.2f}%", (f_epochs[f_best_idx], f_val_top1[f_best_idx]),
                    xytext=(f_epochs[f_best_idx] - 12, f_val_top1[f_best_idx] + 4),
                    arrowprops=dict(arrowstyle="->", color=COLORS["fovi"], lw=1.5),
                    fontweight="bold", color=COLORS["fovi"])

    ax.set_title("B. Top-1 Classification Accuracy (%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(1, max(max(b_epochs or [1]), max(f_epochs or [1])))

    # --- Panel C: Top-5 Accuracy (%) ---
    ax = axes[1, 0]
    if base_hist:
        b_val_top5 = [h["val_top5"] for h in base_hist]
        ax.plot(b_epochs, b_val_top5, color=COLORS["baseline"], linewidth=2.2, label=f"Baseline (Max: {max(b_val_top5):.2f}%)")
    if fovi_hist:
        f_val_top5 = [h["val_top5"] for h in fovi_hist]
        ax.plot(f_epochs, f_val_top5, color=COLORS["fovi"], linewidth=2.2, label=f"FOVI @ 64 (Max: {max(f_val_top5):.2f}%)")
    ax.set_title("C. Validation Top-5 Accuracy (%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-5 Accuracy (%)")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(1, max(max(b_epochs or [1]), max(f_epochs or [1])))

    # --- Panel D: Learning Rate Cosine Schedule ---
    ax = axes[1, 1]
    if base_hist:
        ax.plot(b_epochs, [h.get("lr", 0.0) for h in base_hist], color=COLORS["accent"], linewidth=2.2, label="Cosine Decay Schedule")
    elif fovi_hist:
        ax.plot(f_epochs, [h.get("lr", 0.0) for h in fovi_hist], color=COLORS["accent"], linewidth=2.2, label="Cosine Decay Schedule")
    ax.set_title("D. Cosine Annealing Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(1, max(max(b_epochs or [1]), max(f_epochs or [1])))

    fig.suptitle("Training & Validation Dynamics: Baseline ViT vs. FOVI-ViT with LoRA", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "plot_1_training_dynamics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plotter] Saved: {out_path}")


# ==============================================================================
# PLOT 2: Pareto Efficiency & Hardware Frontier
# ==============================================================================
def plot_efficiency_pareto(summary_data, output_dir: str):
    if not summary_data:
        print("[Plotter] No summary data. Skipping Plot 2.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [s["name"] for s in summary_data]
    top1 = [s["test_top1"] for s in summary_data]
    gflops = [s["gflops"] for s in summary_data]
    latencies = [s["latency_ms"] for s in summary_data]
    fps = [s["fps"] for s in summary_data]
    colors = [COLORS["baseline"], COLORS["fovi"], COLORS["fovi_3fix"]]

    # --- Subplot A: Accuracy vs. GFLOPs ---
    ax = axes[0]
    for i, (g, acc, name, c) in enumerate(zip(gflops, top1, names, colors)):
        ax.scatter(g, acc, color=c, s=160, zorder=5, edgecolors="black", label=name)
        offset_y = 1.2 if i != 2 else -2.2
        ax.annotate(f"{name}\n({g:.3f} G, {acc:.1f}%)", (g, acc),
                    xytext=(g + 0.05, acc + offset_y),
                    fontweight="bold", fontsize=9, color=c)
    ax.set_title("A. Computational Efficiency\n(Accuracy vs. GFLOPs)")
    ax.set_xlabel("Theoretical GFLOPs / Image")
    ax.set_ylabel("Test Top-1 Accuracy (%)")
    ax.set_xlim(0.4, 2.5)
    ax.set_ylim(40, 95)

    # --- Subplot B: Accuracy vs. Latency (ms) ---
    ax = axes[1]
    for i, (lat, acc, name, c) in enumerate(zip(latencies, top1, names, colors)):
        ax.scatter(lat, acc, color=c, s=160, zorder=5, edgecolors="black")
        ax.annotate(f"{name}\n({lat:.2f} ms)", (lat, acc),
                    xytext=(lat + 0.5, acc - (2.5 if i == 2 else -1.0)),
                    fontweight="bold", fontsize=9, color=c)
    ax.set_title("B. Hardware Latency\n(Accuracy vs. Latency)")
    ax.set_xlabel("Inference Latency (ms, batch_size=1)")
    ax.set_ylabel("Test Top-1 Accuracy (%)")
    ax.set_xlim(4, 28)
    ax.set_ylim(40, 95)

    # --- Subplot C: Accuracy vs. Throughput (FPS) ---
    ax = axes[2]
    for i, (th, acc, name, c) in enumerate(zip(fps, top1, names, colors)):
        ax.scatter(th, acc, color=c, s=160, zorder=5, edgecolors="black")
        ax.annotate(f"{name}\n({th:.1f} FPS)", (th, acc),
                    xytext=(th - (25 if th > 100 else 10), acc + 1.5),
                    fontweight="bold", fontsize=9, color=c)
    ax.set_title("C. Inference Throughput\n(Accuracy vs. FPS)")
    ax.set_xlabel("Throughput (Frames / Second)")
    ax.set_ylabel("Test Top-1 Accuracy (%)")
    ax.set_xlim(20, 180)
    ax.set_ylim(40, 95)

    fig.suptitle("Efficiency Frontier: Accuracy vs. Computational & Hardware Resources", y=1.03)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "plot_2_efficiency_pareto.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plotter] Saved: {out_path}")


# ==============================================================================
# PLOT 3: Foveated Sensor Manifold & Retinal Patchification Geometry
# ==============================================================================
def plot_foveated_sensor_geometry(output_dir: str):
    from fovi.geometry import FoveatedSensor

    sensor = FoveatedSensor.from_target_samples(3976, fov_degrees=16.0, a=2.79)
    patch_sensor = FoveatedSensor.from_target_samples(64, fov_degrees=16.0, a=2.79)

    fig = plt.figure(figsize=(16, 5))

    # Panel 1: Visual Space Retinal Photoreceptors (3,976 points)
    ax1 = fig.add_subplot(131)
    v_xy = sensor.visual_xy.cpu().numpy()
    radii = sensor.radius.cpu().numpy()
    sc = ax1.scatter(v_xy[:, 0], v_xy[:, 1], s=5, c=radii, cmap="viridis", alpha=0.6)
    p_xy = patch_sensor.visual_xy.cpu().numpy()
    ax1.scatter(p_xy[:, 0], p_xy[:, 1], s=45, color="red", edgecolors="black", label=f"64 Patch Centers ($a=2.79$)")
    ax1.set_title("A. Retinal Sensor & Patch Grid\n(3,976 Photoreceptors $\\rightarrow$ 64 Patches)")
    ax1.set_xlabel("Horizontal Angle (°)")
    ax1.set_ylabel("Vertical Angle (°)")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right", frameon=True)
    cbar = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Eccentricity r (°)")

    # Panel 2: 3D Rovamo-Virsu Manifold (Cortical Surface Model)
    ax2 = fig.add_subplot(132, projection="3d")
    m_xyz = sensor.manifold_xyz.cpu().numpy()
    ax2.scatter(m_xyz[:, 0], m_xyz[:, 1], m_xyz[:, 2], s=4, c=radii, cmap="plasma", alpha=0.5)
    ax2.set_title("B. 3D Rovamo-Virsu Manifold\n(Cortical Magnification Geometry)")
    ax2.set_xlabel("X (rho cos θ)")
    ax2.set_ylabel("Y (rho sin θ)")
    ax2.set_zlabel("Cortical Depth Z")
    ax2.view_init(elev=25, azim=45)

    # Panel 3: 2D Schwartz Complex-Log Hemifield Cortical Map
    ax3 = fig.add_subplot(133)
    c_xy = sensor.cortical_xy.cpu().numpy()
    hemi = sensor.hemifield.cpu().numpy()
    ax3.scatter(c_xy[:, 0], c_xy[:, 1], s=5, c=hemi, cmap="coolwarm", alpha=0.6)
    ax3.set_title("C. 2D Schwartz Cortical Map\n(Split Left/Right Hemifields)")
    ax3.set_xlabel("Cortical U (mm) [LH <--- | ---> RH]")
    ax3.set_ylabel("Cortical V (mm)")

    fig.suptitle("FOVI Biologically-Inspired Foveated Interface Architecture", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "plot_3_foveated_sensor_geometry.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plotter] Saved: {out_path}")


# ==============================================================================
# PLOT 4: LoRA Weight Adaptation Analysis
# ==============================================================================
def plot_lora_weights_analysis(checkpoints_dir: str, output_dir: str):
    b_lora_path = os.path.join(checkpoints_dir, "lora_weights_baseline.pt")
    f_lora_path = os.path.join(checkpoints_dir, "lora_weights_fovi.pt")

    if not os.path.exists(b_lora_path) or not os.path.exists(f_lora_path):
        print("[Plotter] LoRA weights not found. Skipping Plot 4.")
        return

    b_weights = torch.load(b_lora_path, map_location="cpu")
    f_weights = torch.load(f_lora_path, map_location="cpu")

    layers = list(range(6))
    b_norms = []
    f_norms = []
    svd_spectra_b = []
    svd_spectra_f = []

    alpha = 8.0
    rank = 8
    scale = alpha / rank

    for l in layers:
        # Baseline qkv
        b_A = b_weights.get(f"blocks.{l}.attn.qkv.lora_A", None)
        b_B = b_weights.get(f"blocks.{l}.attn.qkv.lora_B", None)
        if b_A is not None and b_B is not None:
            dW_b = scale * torch.matmul(b_B.float(), b_A.float())
            b_norms.append(torch.norm(dW_b, p="fro").item())
            if l == 0:
                _, s, _ = torch.linalg.svd(dW_b)
                svd_spectra_b = s.cpu().numpy()[:rank]

        # FOVI qkv
        f_A = f_weights.get(f"vit.blocks.{l}.attn.qkv.lora_a", None)
        f_B = f_weights.get(f"vit.blocks.{l}.attn.qkv.lora_b", None)
        if f_A is not None and f_B is not None:
            dW_f = scale * torch.matmul(f_B.float(), f_A.float())
            f_norms.append(torch.norm(dW_f, p="fro").item())
            if l == 0:
                _, s, _ = torch.linalg.svd(dW_f)
                svd_spectra_f = s.cpu().numpy()[:rank]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot A: Frobenius Norm per Layer
    ax = axes[0]
    x = np.arange(len(b_norms))
    width = 0.35
    ax.bar(x - width/2, b_norms, width, label="Baseline ViT LoRA (Uniform)", color=COLORS["baseline"], edgecolor="black")
    ax.bar(x + width/2, f_norms, width, label="FOVI-ViT LoRA (Foveated)", color=COLORS["fovi"], edgecolor="black")
    ax.set_title("A. LoRA Weight Update Magnitude\n($\\|\\Delta W\\|_F$ across Early Layers 0–5)")
    ax.set_xlabel("Transformer Layer Index")
    ax.set_ylabel("Frobenius Norm $\\|\\Delta W\\|_F$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {i}" for i in layers])
    ax.legend(loc="upper right", frameon=True)

    # Subplot B: Singular Value Spectrum of Layer 0
    ax = axes[1]
    if len(svd_spectra_b) > 0 and len(svd_spectra_f) > 0:
        ax.plot(range(1, rank + 1), svd_spectra_b, "o-", color=COLORS["baseline"], linewidth=2.2, markersize=7, label="Baseline Layer 0 SVD")
        ax.plot(range(1, rank + 1), svd_spectra_f, "s-", color=COLORS["fovi"], linewidth=2.2, markersize=7, label="FOVI Layer 0 SVD")
        ax.set_title(f"B. Singular Value Spectrum of $\\Delta W$\n(Intrinsic Rank $r=8$ in Layer 0)")
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value $\\sigma_i$")
        ax.legend(loc="upper right", frameon=True)

    fig.suptitle("LoRA Representation Learning Diagnostics (ICML Adaptation Protocol)", y=1.03)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "plot_4_lora_weights_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plotter] Saved: {out_path}")


# ==============================================================================
# PLOT 5: Cortical Magnification & pRF Scaling Analysis
# ==============================================================================
def plot_cortical_magnification_scaling(output_dir: str):
    r = np.linspace(0, 8.0, 300)
    a_values = [1.5, 2.79, 5.0]  # a = 2.79 is paper default
    m0 = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Cortical Magnification Factor M(r) = M_0 / (1 + r / a)
    ax = axes[0]
    for a in a_values:
        m_r = m0 / (1.0 + r / a)
        lw = 2.8 if a == 2.79 else 1.8
        lbl = f"$a = {a}^\\circ$ (ICML 2026 Default)" if a == 2.79 else f"$a = {a}^\\circ$"
        c = COLORS["fovi"] if a == 2.79 else None
        ax.plot(r, m_r, linewidth=lw, label=lbl, color=c)
    ax.set_title("A. Cortical Magnification Scaling\n$M(r) = \\frac{M_0}{1 + r / a}$")
    ax.set_xlabel("Visual Eccentricity $r$ (Degrees)")
    ax.set_ylabel("Normalized Magnification $M(r) / M_0$")
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.05)

    # Panel B: Receptive Field Diameter Progression
    ax = axes[1]
    for a in a_values:
        rf_diam = 0.5 * (1.0 + r / a)
        lw = 2.8 if a == 2.79 else 1.8
        lbl = f"$a = {a}^\\circ$ (ICML 2026 Default)" if a == 2.79 else f"$a = {a}^\\circ$"
        c = COLORS["fovi"] if a == 2.79 else None
        ax.plot(r, rf_diam, linewidth=lw, label=lbl, color=c)
    ax.set_title("B. Population Receptive Field (pRF) Diameter\n$\\text{Diam}(r) \\propto 1 + r / a$")
    ax.set_xlabel("Visual Eccentricity $r$ (Degrees)")
    ax.set_ylabel("pRF Diameter (Degrees)")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(0, 8)

    fig.suptitle("Cortical Magnification & Biological Receptive Field Scaling", y=1.03)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "plot_5_retinal_receptive_fields.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plotter] Saved: {out_path}")


# ==============================================================================
# MASTER SUMMARY DASHBOARD
# ==============================================================================
def plot_master_dashboard(base_hist, fovi_hist, summary_data, output_dir: str):
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.25)

    # 1. Training Loss Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    if base_hist:
        ax1.plot([h["epoch"] for h in base_hist], [h["val_loss"] for h in base_hist], color=COLORS["baseline"], lw=2.2, label="Baseline Val")
    if fovi_hist:
        ax1.plot([h["epoch"] for h in fovi_hist], [h["val_loss"] for h in fovi_hist], color=COLORS["fovi"], lw=2.2, label="FOVI Val")
    ax1.set_title("Validation Loss (50 Epochs)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=True)

    # 2. Validation Accuracy Trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    if base_hist:
        ax2.plot([h["epoch"] for h in base_hist], [h["val_top1"] for h in base_hist], color=COLORS["baseline"], lw=2.2, label="Baseline Top-1")
    if fovi_hist:
        ax2.plot([h["epoch"] for h in fovi_hist], [h["val_top1"] for h in fovi_hist], color=COLORS["fovi"], lw=2.2, label="FOVI Top-1")
    ax2.set_title("Validation Top-1 Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-1 Acc (%)")
    ax2.legend(loc="lower right", frameon=True)

    # 3. Pareto Efficiency: Accuracy vs GFLOPs
    ax3 = fig.add_subplot(gs[0, 2])
    if summary_data:
        for s, c in zip(summary_data, [COLORS["baseline"], COLORS["fovi"], COLORS["fovi_3fix"]]):
            ax3.scatter(s["gflops"], s["test_top1"], color=c, s=140, edgecolors="black", zorder=5)
            ax3.annotate(f"{s['name']}\n{s['test_top1']:.1f}% @ {s['gflops']:.2f}G", (s["gflops"], s["test_top1"]),
                         xytext=(s["gflops"] + 0.05, s["test_top1"] - 3), fontsize=8.5, fontweight="bold", color=c)
    ax3.set_title("Efficiency: Top-1 Acc vs. GFLOPs")
    ax3.set_xlabel("GFLOPs / Image")
    ax3.set_ylabel("Test Top-1 Acc (%)")
    ax3.set_xlim(0.4, 2.5)
    ax3.set_ylim(40, 95)

    # 4. Latency vs. Throughput
    ax4 = fig.add_subplot(gs[1, 0])
    if summary_data:
        for s, c in zip(summary_data, [COLORS["baseline"], COLORS["fovi"], COLORS["fovi_3fix"]]):
            ax4.scatter(s["latency_ms"], s["fps"], color=c, s=140, edgecolors="black", zorder=5)
            ax4.annotate(f"{s['model_type'].upper()} ({s['fixations']}fix)\n{s['latency_ms']:.1f}ms | {s['fps']:.0f}FPS", (s["latency_ms"], s["fps"]),
                         xytext=(s["latency_ms"] + 0.6, s["fps"] - 5), fontsize=8.5, fontweight="bold", color=c)
    ax4.set_title("Hardware Performance: Latency vs. FPS")
    ax4.set_xlabel("Latency (ms)")
    ax4.set_ylabel("Throughput (FPS)")

    # 5. Sensor Sampling Distribution (Retina)
    ax5 = fig.add_subplot(gs[1, 1])
    from fovi.geometry import FoveatedSensor
    sensor = FoveatedSensor.from_target_samples(3976, fov_degrees=16.0, a=2.79)
    patch_sensor = FoveatedSensor.from_target_samples(64, fov_degrees=16.0, a=2.79)
    v_xy = sensor.visual_xy.cpu().numpy()
    p_xy = patch_sensor.visual_xy.cpu().numpy()
    ax5.scatter(v_xy[:, 0], v_xy[:, 1], s=3, c=sensor.radius.cpu().numpy(), cmap="viridis", alpha=0.5)
    ax5.scatter(p_xy[:, 0], p_xy[:, 1], s=35, color="red", edgecolors="black", label="64 Patches")
    ax5.set_title("FOVI Retinal Sensor Manifold")
    ax5.set_xlabel("Visual Angle (°)")
    ax5.set_ylabel("Visual Angle (°)")
    ax5.set_aspect("equal")
    ax5.legend(loc="upper right", fontsize=8, frameon=True)

    # 6. Parameter Breakdown
    ax6 = fig.add_subplot(gs[1, 2])
    models = ["Baseline", "FOVI @ 64"]
    trainable = [314.4, 353.6]
    frozen = [5691.2 - 314.4, 5698.8 - 353.6]
    x_p = np.arange(len(models))
    ax6.bar(x_p, frozen, 0.4, label="Frozen Backbone (Layers 6-11)", color="#aec7e8", edgecolor="black")
    ax6.bar(x_p, trainable, 0.4, bottom=frozen, label="Trainable (LoRA 0-5 + Head)", color="#ff7f0e", edgecolor="black")
    ax6.set_title("Parameter Breakdown (Trainable vs. Frozen)")
    ax6.set_ylabel("Parameters (k)")
    ax6.set_xticks(x_p)
    ax6.set_xticklabels(models)
    ax6.legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.suptitle("Comprehensive FOVI-ViT vs. Baseline ViT Fine-Tuning & Benchmarking Dashboard", y=0.98)
    out_path = os.path.join(output_dir, "comprehensive_benchmark_dashboard.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plotter] Saved: {out_path}")


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(root_dir, "checkpoints")
    output_dir = os.path.join(root_dir, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Generating Publication-Quality Analysis Plots...")
    print("=" * 70)

    base_hist, fovi_hist = load_histories(checkpoints_dir)
    summary_data = load_final_summary(os.path.join(root_dir, "final_summary.json"))

    # 1. Training dynamics (Loss, Top-1, Top-5, LR)
    plot_training_dynamics(base_hist, fovi_hist, output_dir)

    # 2. Pareto Efficiency & Hardware Performance
    plot_efficiency_pareto(summary_data, output_dir)

    # 3. Foveated Sensor Geometry
    plot_foveated_sensor_geometry(output_dir)

    # 4. LoRA Weight Adaptation Spectrum
    plot_lora_weights_analysis(checkpoints_dir, output_dir)

    # 5. Cortical Magnification Scaling
    plot_cortical_magnification_scaling(output_dir)

    # 6. Master Summary Dashboard
    plot_master_dashboard(base_hist, fovi_hist, summary_data, output_dir)

    print("=" * 70)
    print(f"All 6 analysis figures saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
