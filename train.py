"""
Training Pipeline strictly matching the FOVI ICML paper methodology:
- Model Adaptation: LoRA on early layers (0-5), patch embedding + classification head trainable, late layers (6-11) frozen.
- Saccadic Multi-Fixation Training: 4 random fixations in central radius r=0.25 for FOVI (mean-logit loss).
- Evaluation: 1-fixation and 3-fixations evaluated for FOVI; 1-fixation for Baseline.
- Optimization: AdamW + Cosine Decay scheduler (without early stopping) + AMP FP16.
- Checkpointing: Best val Top-1, final weights, and LoRA adapter weights.
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import fovi

from dataset import get_dataloaders
from models import get_model, count_parameters


def setup_logger(log_file: str) -> logging.Logger:
    """Sets up a logger that outputs to both console and a log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
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


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """Computes Top-K accuracies."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, scheduler, device, epoch: int, total_epochs: int,
    is_fovi: bool = False, num_fixations: int = 4
):
    model.train()
    total_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    n_samples = 0

    t0 = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", dynamic_ncols=True, leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            if is_fovi and num_fixations > 1:
                # Sample random fixations within central radius 0.25 (as in paper Section 5.2 & 6.2)
                fixations = fovi.random_fixations(batch_size, num_fixations=num_fixations, radius=0.25, device=device)
                outputs = model(images, fixations=fixations)
            else:
                outputs = model(images)

            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        total_loss += loss.item() * batch_size
        top1_sum += acc1 * batch_size
        top5_sum += acc5 * batch_size
        n_samples += batch_size

        pbar.set_postfix({
            "loss": f"{(total_loss / n_samples):.4f}",
            "top1": f"{(top1_sum / n_samples):.2f}%",
            "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
        })

    scheduler.step()
    elapsed = time.time() - t0
    return {
        "loss": total_loss / n_samples,
        "top1": top1_sum / n_samples,
        "top5": top5_sum / n_samples,
        "time_s": elapsed
    }


def evaluate(model, loader, criterion, device, epoch: int, total_epochs: int, is_fovi: bool = False, num_fixations: int = 1):
    model.eval()
    total_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    n_samples = 0

    t0 = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Val ({num_fixations}fix)]", dynamic_ncols=True, leave=False)

    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                if is_fovi and num_fixations > 1:
                    fixations = fovi.random_fixations(batch_size, num_fixations=num_fixations, radius=0.25, device=device)
                    outputs = model(images, fixations=fixations)
                else:
                    outputs = model(images)
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            total_loss += loss.item() * batch_size
            top1_sum += acc1 * batch_size
            top5_sum += acc5 * batch_size
            n_samples += batch_size

            pbar.set_postfix({
                "val_loss": f"{(total_loss / n_samples):.4f}",
                "val_top1": f"{(top1_sum / n_samples):.2f}%"
            })

    elapsed = time.time() - t0
    return {
        "loss": total_loss / n_samples,
        "top1": top1_sum / n_samples,
        "top5": top5_sum / n_samples,
        "time_s": elapsed
    }


def train_model(
    model_type: str = "baseline",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    num_workers: int = 2,
    num_fixations_train: int = 4,
    data_dir: str = None,
    save_dir: str = None
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = data_dir or os.path.join(base_dir, "data", "train")
    save_dir = save_dir or os.path.join(base_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"train_{model_type}.log")
    logger = setup_logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_fovi = (model_type == "fovi")

    logger.info("=" * 75)
    logger.info(f"Starting Training: {model_type.upper()} Model (FOVI ICML Paper Recipe)")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr} | Weight Decay: {weight_decay}")
    if is_fovi:
        logger.info(f"FOVI Strategy: Saccadic sampling with {num_fixations_train} fixations in radius r=0.25 (as in paper)")
    else:
        logger.info("Baseline Strategy: Standard uniform 14x14 grid @ 224 (1 fixation)")
    logger.info(f"Log File: {log_file}")
    logger.info("=" * 75)

    # 1. Load Data
    train_loader, val_loader, _, num_classes = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 2. Build Model
    model = get_model(model_type, num_classes=num_classes, pretrained=True)
    model.to(device)

    param_info = count_parameters(model)
    logger.info(f"Parameters -> Total: {param_info['total']:,} | Trainable (LoRA+Head): {param_info['trainable']:,} ({param_info['trainable_pct']:.2f}%)")

    # 3. Optimizer & Schedulers
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    warmup_epochs = min(2, max(1, epochs // 5))
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda')

    best_val_top1 = 0.0
    best_epoch = 0
    history = []
    best_ckpt_path = os.path.join(save_dir, f"best_{model_type}_model.pt")
    final_ckpt_path = os.path.join(save_dir, f"final_{model_type}_model.pt")
    lora_weights_path = os.path.join(save_dir, f"lora_weights_{model_type}.pt")

    for epoch in range(1, epochs + 1):
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"--- Epoch [{epoch:02d}/{epochs:02d}] (LR: {curr_lr:.6f}) ---")
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch, epochs,
            is_fovi=is_fovi, num_fixations=num_fixations_train
        )
        
        # Validate (1-fixation standard)
        val_metrics = evaluate(
            model, val_loader, criterion, device, epoch, epochs,
            is_fovi=is_fovi, num_fixations=1
        )

        is_best = val_metrics["top1"] > best_val_top1
        if is_best:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            
            torch.save({
                "epoch": epoch,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
                "val_loss": val_metrics["loss"],
            }, best_ckpt_path)

            trainable_state_dict = {k: v for k, v in model.state_dict().items() if any(p_name in k for p_name, p in model.named_parameters() if p.requires_grad)}
            torch.save(trainable_state_dict, lora_weights_path)

            best_tag = " [* BEST CHECKPOINT SAVED *]"
        else:
            best_tag = ""

        logger.info(f"  Train -> Loss: {train_metrics['loss']:.4f} | Top-1: {train_metrics['top1']:.2f}% | Top-5: {train_metrics['top5']:.2f}% ({train_metrics['time_s']:.1f}s)")
        logger.info(f"  Val   -> Loss: {val_metrics['loss']:.4f} | Top-1: {val_metrics['top1']:.2f}% | Top-5: {val_metrics['top5']:.2f}% ({val_metrics['time_s']:.1f}s){best_tag}")

        history.append({
            "epoch": epoch,
            "lr": curr_lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
        })

    torch.save({
        "epoch": epochs,
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "final_val_top1": history[-1]["val_top1"],
        "final_val_top5": history[-1]["val_top5"],
        "final_val_loss": history[-1]["val_loss"],
    }, final_ckpt_path)

    history_path = os.path.join(save_dir, f"history_{model_type}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 75)
    logger.info(f"Training for {model_type.upper()} completed!")
    logger.info(f"Best Val Top-1 Accuracy : {best_val_top1:.2f}% (Epoch {best_epoch})")
    logger.info(f"Best Checkpoint File    : {best_ckpt_path}")
    logger.info(f"Final Weights File      : {final_ckpt_path}")
    logger.info(f"LoRA Adapter File       : {lora_weights_path}")
    logger.info(f"History Metrics File    : {history_path}")
    logger.info(f"Log File Saved          : {log_file}")
    logger.info("=" * 75 + "\n")

    return best_ckpt_path, history_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT with or without FOVI using LoRA (Paper aligned)")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "fovi", "all"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--fixations_train", type=int, default=4, help="Number of random fixations per training image for FOVI")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.environ.get("DATA_DIR", os.path.join(base_dir, "data", "train"))
    default_save_dir = os.path.join(base_dir, "checkpoints")

    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Dataset directory")
    parser.add_argument("--save_dir", type=str, default=default_save_dir, help="Output directory")

    args = parser.parse_args()

    if args.model == "all":
        train_model("baseline", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, num_workers=args.num_workers, data_dir=args.data_dir, save_dir=args.save_dir)
        train_model("fovi", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, num_workers=args.num_workers, num_fixations_train=args.fixations_train,
                    data_dir=args.data_dir, save_dir=args.save_dir)
    else:
        train_model(args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, num_workers=args.num_workers, num_fixations_train=args.fixations_train,
                    data_dir=args.data_dir, save_dir=args.save_dir)
