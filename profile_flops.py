"""
Complexity & FLOPs profiling script.
Compares:
- Theoretical MACs / GFLOPs (fovi & fvcore)
- Total vs Trainable Parameters
- Inference Latency (ms/image) and Throughput (FPS) on GPU
- Peak VRAM Allocation
"""

import os
import json
import time
import torch
from tabulate import tabulate

# Auto-set FOVI environment variables before importing fovi if not already set
os.environ.setdefault("FOVI_SAVE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
os.environ.setdefault("FOVI_DATASETS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

import fovi
from fvcore.nn import FlopCountAnalysis

from models import get_model, count_parameters


def profile_model_flops_and_latency(
    model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    img_size: int = 224,
    warmup_iters: int = 20,
    test_iters: int = 50
) -> dict:
    """Profiles GFLOPs, parameters, and GPU latency for a given model."""
    print(f"\n[Profiler] Profiling {model_name.upper()} model...")
    model = get_model(model_name, num_classes=100, pretrained=False)
    model.eval()
    model.to(device)

    # 1. Parameter counting
    params = count_parameters(model)

    # 2. GFLOPs Calculation
    sample_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Using FOVI analytical estimator
    try:
        fovi_flops_g = fovi.estimate_flops(model, sample_input)
    except Exception as e:
        print(f"Warning: fovi.estimate_flops failed with {e}")
        fovi_flops_g = None

    # Using fvcore FlopCountAnalysis (on CPU for universal compatibility)
    try:
        model_cpu = get_model(model_name, num_classes=100, pretrained=False).eval()
        input_cpu = torch.randn(1, 3, img_size, img_size)
        fvcore_flops = FlopCountAnalysis(model_cpu, input_cpu)
        fvcore_flops_g = fvcore_flops.total() / 1e9
    except Exception as e:
        print(f"Warning: fvcore analysis failed with {e}")
        fvcore_flops_g = None

    # 3. Latency Benchmarking (Batch Size = 1)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(sample_input)
            torch.cuda.synchronize()

        # Timed iterations
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(test_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(test_iters)]

        with torch.no_grad():
            for i in range(test_iters):
                start_events[i].record()
                _ = model(sample_input)
                end_events[i].record()
        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]  # in milliseconds
        mean_latency_ms = sum(times) / len(times)
        fps = 1000.0 / mean_latency_ms
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        # CPU Latency fallback
        for _ in range(5):
            _ = model(sample_input)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(test_iters):
                _ = model(sample_input)
        mean_latency_ms = ((time.perf_counter() - t0) / test_iters) * 1000.0
        fps = 1000.0 / mean_latency_ms
        peak_vram_mb = 0.0

    return {
        "model": model_name,
        "total_params_m": params["total"] / 1e6,
        "trainable_params_k": params["trainable"] / 1e3,
        "trainable_pct": params["trainable_pct"],
        "fovi_gflops": fovi_flops_g,
        "fvcore_gflops": fvcore_flops_g,
        "latency_ms": mean_latency_ms,
        "fps": fps,
        "peak_vram_mb": peak_vram_mb
    }


def run_benchmark_comparison(output_json: str = "flops_comparison.json"):
    results = []
    for m in ["baseline", "fovi"]:
        res = profile_model_flops_and_latency(m)
        results.append(res)

    table_data = [
        [
            r["model"].upper(),
            f"{r['total_params_m']:.2f} M",
            f"{r['trainable_params_k']:.1f} k ({r['trainable_pct']:.1f}%)",
            f"{r['fovi_gflops']:.3f}" if r['fovi_gflops'] is not None else "N/A",
            f"{r['fvcore_gflops']:.3f}" if r['fvcore_gflops'] is not None else "N/A",
            f"{r['latency_ms']:.2f} ms",
            f"{r['fps']:.1f}",
            f"{r['peak_vram_mb']:.1f} MB"
        ]
        for r in results
    ]

    headers = [
        "Model", "Total Params", "Trainable Params", 
        "FOVI GFLOPs", "fvcore GFLOPs", "Latency (bs=1)", "FPS", "Peak VRAM"
    ]

    print("\n" + "=" * 90)
    print("                      MODEL COMPLEXITY & EFFICIENCY COMPARISON")
    print("=" * 90)
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 90)

    # Calculate reductions
    base = results[0]
    fovi_res = results[1]
    if base["fovi_gflops"] and fovi_res["fovi_gflops"]:
        flop_reduction = (1.0 - (fovi_res["fovi_gflops"] / base["fovi_gflops"])) * 100.0
        print(f"\n>> FOVI achieves a {flop_reduction:.2f}% reduction in theoretical GFLOPs vs Baseline ViT.")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Profiler] Results saved to: {output_json}\n")


if __name__ == "__main__":
    run_benchmark_comparison()
