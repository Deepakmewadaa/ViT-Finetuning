from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..sampling import random_fixations


@dataclass
class TrainMetrics:
    epoch: int
    train_loss: float
    train_top1: float
    val_loss: float
    val_top1: float
    val_top5: float


class FOVITrainer:
    """End-to-end trainer for FOVI models with multi-fixation saccadic aggregation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
        num_train_fixations: int = 4,
        num_val_fixations: int = 20,
        fixation_radius: float = 0.25,
        use_amp: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        if optimizer is None:
            # Optimize only trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        self.num_train_fixations = num_train_fixations
        self.num_val_fixations = num_val_fixations
        self.fixation_radius = fixation_radius
        self.use_amp = use_amp and (self.device.type == "cuda")
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch with multi-fixation aggregation."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = images.shape[0]

            # Generate random fixations for this batch
            if self.num_train_fixations > 1:
                fixations = random_fixations(
                    batch_size,
                    num_fixations=self.num_train_fixations,
                    radius=self.fixation_radius,
                    device=self.device,
                )
            else:
                fixations = None

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Model returns average logits across fixations: [batch, num_classes]
                logits = self.model(images, fixations=fixations)
                loss = F.cross_entropy(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item()) * batch_size
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == targets).sum().item())
            total_samples += batch_size

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss / max(1, total_samples), total_correct / max(1, total_samples)

    @torch.no_grad()
    def evaluate(self, num_fixations: Optional[int] = None) -> tuple[float, float, float]:
        """Evaluate accuracy with a specified number of saccadic fixations."""
        self.model.eval()
        n_fix = num_fixations or self.num_val_fixations
        total_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total_samples = 0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = images.shape[0]

            if n_fix > 1:
                fixations = random_fixations(
                    batch_size,
                    num_fixations=n_fix,
                    radius=self.fixation_radius,
                    device=self.device,
                )
            else:
                fixations = None

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images, fixations=fixations)
                loss = F.cross_entropy(logits, targets)

            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            # Top-1 and Top-5 accuracy
            _, pred_top5 = logits.topk(min(5, logits.shape[-1]), dim=-1, largest=True, sorted=True)
            top1_correct += int((pred_top5[:, 0] == targets).sum().item())
            top5_correct += int((pred_top5 == targets[:, None]).any(dim=-1).sum().item())

        return (
            total_loss / max(1, total_samples),
            top1_correct / max(1, total_samples),
            top5_correct / max(1, total_samples),
        )

    def fit(self, epochs: int, print_freq: int = 1) -> list[TrainMetrics]:
        """Run training and validation for given number of epochs."""
        history = []
        for epoch in range(1, epochs + 1):
            train_loss, train_top1 = self.train_epoch()
            val_loss, val_top1, val_top5 = self.evaluate()

            metrics = TrainMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_top1=train_top1,
                val_loss=val_loss,
                val_top1=val_top1,
                val_top5=val_top5,
            )
            history.append(metrics)

            if epoch % print_freq == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Train Loss: {train_loss:.4f}, Top-1: {train_top1*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Top-1: {val_top1*100:.2f}%, Top-5: {val_top5*100:.2f}%"
                )

        return history
