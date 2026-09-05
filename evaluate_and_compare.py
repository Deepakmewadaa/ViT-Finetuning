"""
Final Evaluation and Comparison Script.
Evaluates both Baseline ViT + LoRA and FOVI ViT + LoRA models on the
HELD-OUT 5,000-image TEST SET (Zero Data Leakage), matching the ICML paper evaluation protocol:
- Baseline ViT: 1 fixation (uniform 224x224)
- FOVI-ViT: 1 fixation (center) AND 3 fixations (saccadic logit averaging)
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tabulate import tabulate
import matplotlib.pyplot as plt
from tqdm import tqdm
# Auto-set FOVI environment variables before importing fovi if not already set
os.environ.setdefault("FOVI_SAVE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
os.environ.setdefault("FOVI_DATASETS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

import fovi
try:
    from fovi.sampling import random_fixations
except (ImportError, AttributeError):
    try:
        from fovi.geometry import random_fixations
    except (ImportError, AttributeError):
        random_fixations = getattr(fovi, "random_fixations", None)

from dataset import get_dataloaders
from models import get_model, count_parameters
from profile_flops import profile_model_flops_and_latency


def setup_logger(log_file: str) -> logging.Logger:
    """Sets up a logger that outputs to both console and a log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("eval_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def compute_test_metrics(model, test_loader, device, model_name: str = "Model", is_fovi: bool = False, num_fixations: int = 1):
    """Computes Top-1, Top-5 accuracy, and test loss on the held-out test set with tqdm."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    n_samples = 0

    t0 = time.time()
    desc_str = f"Testing [{model_name.upper()} ({num_fixations}fix)]"
    pbar = tqdm(test_loader, desc=desc_str, dynamic_ncols=True)

    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                if is_fovi and num_fixations > 1:
                    fixations = random_fixations(batch_size, num_fixations=num_fixations, radius=0.25, device=device)
                    outputs = model(images, fixations=fixations)
                else:
                    outputs = model(images)
                loss = criterion(outputs, targets)

            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            acc1 = correct[:1].reshape(-1).float().sum().item()
            acc5 = correct[:5].reshape(-1).float().sum().item()

            total_loss += loss.item() * batch_size
            top1_sum += acc1
            top5_sum += acc5
            n_samples += batch_size

            pbar.set_postfix({
                "test_loss": f"{(total_loss / n_samples):.4f}",
                "top1": f"{(top1_sum / n_samples) * 100.0:.2f}%",
                "top5": f"{(top5_sum / n_samples) * 100.0:.2f}%"
            })

    eval_time = time.time() - t0
    return {
        "test_loss": total_loss / n_samples,
        "test_top1": (top1_sum / n_samples) * 100.0,
        "test_top5": (top5_sum / n_samples) * 100.0,
        "eval_time_s": eval_time,
        "test_samples": n_samples
    }


def plot_comparison_and_history(save_dir: str = "checkpoints", output_plot: str = "comparison_results.png"):
    base_hist_path = os.path.join(save_dir, "history_baseline.json")
    fovi_hist_path = os.path.join(save_dir, "history_fovi.json")

    has_base = os.path.exists(base_hist_path)
    has_fovi = os.path.exists(fovi_hist_path)

    if not (has_base or has_fovi):
        return

    plt.figure(figsize=(14, 5))

    # Plot 1: Validation Top-1 Accuracy
    plt.subplot(1, 2, 1)
    if has_base:
        with open(base_hist_path) as f:
            h_base = json.load(f)
        epochs_b = [x["epoch"] for x in h_base]
        val_top1_b = [x["val_top1"] for x in h_base]
        plt.plot(epochs_b, val_top1_b, "o-", label="Baseline ViT + LoRA (1 fix)", color="#1f77b4", linewidth=2)

    if has_fovi:
        with open(fovi_hist_path) as f:
            h_fovi = json.load(f)
        epochs_f = [x["epoch"] for x in h_fovi]
        val_top1_f = [x["val_top1"] for x in h_fovi]
        plt.plot(epochs_f, val_top1_f, "s-", label="FOVI ViT + LoRA", color="#2ca02c", linewidth=2)

    plt.title("Validation Top-1 Accuracy vs Epoch", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Val Top-1 Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Plot 2: Validation Loss
    plt.subplot(1, 2, 2)
    if has_base:
        val_loss_b = [x["val_loss"] for x in h_base]
        plt.plot(epochs_b, val_loss_b, "o-", label="Baseline ViT + LoRA", color="#1f77b4", linewidth=2)

    if has_fovi:
        val_loss_f = [x["val_loss"] for x in h_fovi]
        plt.plot(epochs_f, val_loss_f, "s-", label="FOVI ViT + LoRA", color="#2ca02c", linewidth=2)

    plt.title("Validation Loss vs Epoch", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"[Plotter] Comparison figure saved to: {output_plot}")


def run_evaluation(
    data_dir: str = None,
    save_dir: str = None,
    output_summary_json: str = None
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = data_dir or os.path.join(base_dir, "data", "train")
    save_dir = save_dir or os.path.join(base_dir, "checkpoints")
    output_summary_json = output_summary_json or os.path.join(base_dir, "final_summary.json")

    log_file = os.path.join(save_dir, "evaluation.log")
    logger = setup_logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 95)
    logger.info("        FINAL EVALUATION ON HELD-OUT TEST SET (5,000 IMAGES - ICML PAPER PROTOCOL)")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Log File: {log_file}")
    logger.info("=" * 95)

    # 1. Load Data
    _, _, test_loader, num_classes = get_dataloaders(data_dir=data_dir, batch_size=64, num_workers=2)

    # Models & configurations to evaluate (matching Table 1 of paper)
    eval_configs = [
        {"model_type": "baseline", "name": "Baseline ViT (Uniform @ 224)", "fixations": 1},
        {"model_type": "fovi", "name": "FOVI-ViT @ 64 (a=2.79, 1-fixation)", "fixations": 1},
        {"model_type": "fovi", "name": "FOVI-ViT @ 64 (a=2.79, 3-fixations)", "fixations": 3},
    ]

    results = []

    for cfg in eval_configs:
        m_type = cfg["model_type"]
        fix_count = cfg["fixations"]
        disp_name = cfg["name"]

        best_ckpt_path = os.path.join(save_dir, f"best_{m_type}_model.pt")
        profile_res = profile_model_flops_and_latency(m_type, device="cuda" if torch.cuda.is_available() else "cpu")
        effective_flops = (profile_res["fovi_gflops"] or 0.0) * fix_count

        if os.path.exists(best_ckpt_path):
            logger.info(f"[Evaluation] Loading weights for {disp_name} from {best_ckpt_path} ...")
            model = get_model(m_type, num_classes=num_classes, pretrained=False)
            checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)

            test_res = compute_test_metrics(model, test_loader, device, model_name=m_type, is_fovi=(m_type == "fovi"), num_fixations=fix_count)
        else:
            logger.warning(f"[Evaluation] Checkpoint not found at {best_ckpt_path} (evaluating pretrained)...")
            model = get_model(m_type, num_classes=num_classes, pretrained=True)
            model.to(device)
            test_res = compute_test_metrics(model, test_loader, device, model_name=m_type, is_fovi=(m_type == "fovi"), num_fixations=fix_count)

        results.append({
            "name": disp_name,
            "model_type": m_type,
            "fixations": fix_count,
            "test_top1": test_res["test_top1"],
            "test_top5": test_res["test_top5"],
            "test_loss": test_res["test_loss"],
            "gflops": effective_flops,
            "latency_ms": profile_res["latency_ms"] * fix_count,
            "fps": profile_res["fps"] / fix_count,
            "trainable_params_k": profile_res["trainable_params_k"],
        })

    # Summary Table
    table_data = [
        [
            r["name"],
            r["fixations"],
            f"{r['test_top1']:.2f}%",
            f"{r['test_top5']:.2f}%",
            f"{r['gflops']:.3f} G",
            f"{r['latency_ms']:.2f} ms",
            f"{r['fps']:.1f} FPS",
            f"{r['trainable_params_k']:.1f} k"
        ]
        for r in results
    ]

    headers = [
        "Model Architecture", "# Fixations", "Test Top-1", "Test Top-5", 
        "Total GFLOPs", "Latency (bs=1)", "Throughput", "Trainable Params"
    ]

    table_str = tabulate(table_data, headers=headers, tablefmt="github")
    logger.info("\n" + "=" * 95)
    logger.info("                     ICML PAPER-MATCHED BENCHMARK COMPARISON MATRIX")
    logger.info("=" * 95)
    logger.info("\n" + table_str + "\n")
    logger.info("=" * 95)

    with open(output_summary_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n[Summary] Metrics saved to: {output_summary_json}")
    logger.info(f"[Summary] Evaluation log saved to: {log_file}")

    plot_comparison_and_history(save_dir=save_dir)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.environ.get("DATA_DIR", os.path.join(base_dir, "data", "train"))
    default_save_dir = os.path.join(base_dir, "checkpoints")

    parser = argparse.ArgumentParser(description="Evaluate best checkpoints on held-out test set")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Dataset directory")
    parser.add_argument("--save_dir", type=str, default=default_save_dir, help="Checkpoints directory")
    args = parser.parse_args()
    run_evaluation(data_dir=args.data_dir, save_dir=args.save_dir)
