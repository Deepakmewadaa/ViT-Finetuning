from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .geometry import FoveatedSensor
from .sampling import random_fixations


@dataclass
class BenchmarkResult:
    name: str
    num_fixations: int
    num_pixels: int
    num_patches_or_points: int
    gflops: float
    train_latency_ms: float
    val_latency_ms: float
    train_mem_gb: float
    val_mem_gb: float


def estimate_flops(model: nn.Module, sample_input: torch.Tensor, fixations: Optional[torch.Tensor] = None) -> float:
    """Estimate total GFLOPs for a single image forward pass.

    Uses an analytical FLOP calculation based on module types (Linear, Conv2d, KNNConv, MultiheadAttention).
    """
    total_macs = 0

    def conv_hook(self, input, output):
        nonlocal total_macs
        # KNNConv: output is [B, Out_C, Out_pts], gathered is [B, In_C, Out_pts, K]
        # MACs = Out_pts * Out_C * In_C * K
        k = getattr(self, "kernel_size", 9)
        out_pts = self.output_grid.num_samples
        in_c = self.in_channels
        out_c = self.out_channels
        macs = out_pts * out_c * in_c * k
        total_macs += macs

    def linear_hook(self, input, output):
        nonlocal total_macs
        # Linear: in_f * out_f * batch_elements
        if isinstance(input, tuple):
            inp = input[0]
        else:
            inp = input
        batch_elements = inp.numel() // self.in_features
        total_macs += batch_elements * self.in_features * self.out_features

    def mha_hook(self, input, output):
        nonlocal total_macs
        # MultiheadAttention: query @ key [B, H, S, S] -> S^2 * D, weights @ value [B, H, S, D] -> S^2 * D
        # plus in_proj and out_proj
        if isinstance(input, tuple):
            q = input[0]
        else:
            q = input
        batch_size, seq_len, dim = q.shape
        # Attention score computation: 2 * seq_len^2 * dim
        total_macs += 2 * seq_len * seq_len * dim
        # Projections: 4 * seq_len * dim^2
        total_macs += 4 * seq_len * dim * dim

    hooks = []
    for module in model.modules():
        from .layers import KNNConv
        if isinstance(module, KNNConv):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(mha_hook))

    # Run single forward pass (batch size 1)
    single_img = sample_input[:1]
    single_fix = fixations[:1] if fixations is not None else None
    model.eval()
    with torch.no_grad():
        try:
            if single_fix is not None:
                _ = model(single_img, fixations=single_fix)
            else:
                _ = model(single_img)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    # Total FLOPs = 2 * MACs, in GFLOPs (10^9)
    gflops = (2.0 * total_macs) / 1e9
    return float(gflops)


def benchmark_model(
    model: nn.Module,
    *,
    name: str = "Model",
    batch_size: int = 64,
    image_size: int = 256,
    num_fixations: int = 1,
    num_warmup: int = 3,
    num_iter: int = 10,
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
) -> BenchmarkResult:
    """Benchmark a model for FLOPs, Latency (train/val), and Peak Memory, matching Table 1 of the paper."""
    dev = torch.device(device)
    model = model.to(dev)

    # Input image and fixations
    images = torch.randn(batch_size, 3, image_size, image_size, device=dev, dtype=dtype)
    fixations = random_fixations(batch_size, num_fixations=num_fixations, device=dev, dtype=dtype) if num_fixations > 1 else None

    # Count pixels sampled and patch count
    if hasattr(model, "sensor"):
        num_pixels = model.sensor.num_samples * num_fixations
    elif hasattr(model, "patch_embed") and hasattr(model.patch_embed, "sensor"):
        num_pixels = model.patch_embed.sensor.num_samples * num_fixations
    else:
        num_pixels = (image_size * image_size) * num_fixations

    if hasattr(model, "num_patches"):
        num_patches = model.num_patches * num_fixations
    elif hasattr(model, "patch_embed") and hasattr(model.patch_embed, "num_patches"):
        num_patches = model.patch_embed.num_patches * num_fixations
    else:
        num_patches = (image_size // 16) ** 2 * num_fixations

    gflops_per_image = estimate_flops(model, images, fixations) * num_fixations

    # 1. Benchmark Validation Mode (Inference Latency & Peak VRAM)
    model.eval()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(images, fixations=fixations) if fixations is not None else model(images)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

        start_time = time.perf_counter()
        for _ in range(num_iter):
            _ = model(images, fixations=fixations) if fixations is not None else model(images)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        val_lat = (time.perf_counter() - start_time) / num_iter * 1000.0  # ms

    val_mem = (torch.cuda.max_memory_allocated(dev) / 1e9) if dev.type == "cuda" else 0.0

    # 2. Benchmark Training Mode (Forward + Backward Latency & Peak VRAM)
    model.train()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    targets = torch.randint(0, 100, (batch_size,), device=dev)

    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    for _ in range(num_warmup):
        optimizer.zero_grad()
        out = model(images, fixations=fixations) if fixations is not None else model(images)
        loss = out.sum()
        loss.backward()
        optimizer.step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    start_time = time.perf_counter()
    for _ in range(num_iter):
        optimizer.zero_grad()
        out = model(images, fixations=fixations) if fixations is not None else model(images)
        loss = out.sum()
        loss.backward()
        optimizer.step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    train_lat = (time.perf_counter() - start_time) / num_iter * 1000.0  # ms

    train_mem = (torch.cuda.max_memory_allocated(dev) / 1e9) if dev.type == "cuda" else 0.0

    return BenchmarkResult(
        name=name,
        num_fixations=num_fixations,
        num_pixels=num_pixels,
        num_patches_or_points=num_patches,
        gflops=round(gflops_per_image, 2),
        train_latency_ms=round(train_lat, 2),
        val_latency_ms=round(val_lat, 2),
        train_mem_gb=round(train_mem, 2),
        val_mem_gb=round(val_mem, 2),
    )


def format_benchmark_table(results: list[BenchmarkResult]) -> str:
    """Format benchmark results into a GitHub-flavored Markdown table matching Table 1."""
    header = "| Model | # Fix. | Pixels | Patches | GFLOPs | Train Lat. (ms) | Val Lat. (ms) | Train Mem. (GB) | Val Mem. (GB) |"
    separator = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    rows = [header, separator]
    for r in results:
        row = f"| {r.name} | {r.num_fixations} | {r.num_pixels} | {r.num_patches_or_points} | {r.gflops:.2f} | {r.train_latency_ms:.2f} | {r.val_latency_ms:.2f} | {r.train_mem_gb:.2f} | {r.val_mem_gb:.2f} |"
        rows.append(row)
    return "\n".join(rows)
