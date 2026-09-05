from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..geometry import FoveatedSensor, SensorGrid
from ..layers import KNNConv, KNNMaxPool
from ..sampling import FoveatedSampler


class FOVICNN(nn.Module):
    """Hierarchical biologically-inspired Foveated CNN (AlexNet-like).

    Implements the 5-layer convolutional hierarchy with kNN-convolutions
    and max pooling over isotropic sensor manifolds described in Section 5.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        *,
        sensor: Optional[SensorGrid] = None,
        in_channels: int = 3,
        channels: Sequence[int] = (96, 256, 384, 384, 256),
        fov_degrees: float = 16.0,
        a: float = 0.5,
        target_samples: int = 4096,
        pad_factor: float = 1.0,
        stride1: int = 2,
    ) -> None:
        super().__init__()
        if sensor is None:
            sensor = FoveatedSensor.from_target_samples(
                target_samples,
                fov_degrees=fov_degrees,
                a=a,
                pad_factor=pad_factor,
            )
        self.sensor = sensor
        self.sampler = FoveatedSampler(self.sensor)

        # Build progressive manifold grids
        c1, c2, c3, c4, c5 = channels
        # Grid 1: downsampled by factor corresponding to stride1^2
        grid1_target = max(16, sensor.num_samples // (stride1 * stride1))
        self.grid1 = FoveatedSensor.from_target_samples(grid1_target, fov_degrees=fov_degrees, a=a, pad_factor=pad_factor)

        # Grid 2: downsampled after Pool 1 (3x3 stride 2 equivalent)
        grid2_target = max(16, self.grid1.num_samples // 4)
        self.grid2 = FoveatedSensor.from_target_samples(grid2_target, fov_degrees=fov_degrees, a=a, pad_factor=pad_factor)

        # Grid 3: downsampled after Pool 2
        grid3_target = max(16, self.grid2.num_samples // 4)
        self.grid3 = FoveatedSensor.from_target_samples(grid3_target, fov_degrees=fov_degrees, a=a, pad_factor=pad_factor)

        # Layer 1: k=11x11 = 121
        self.conv1 = KNNConv(in_channels, c1, self.sensor, self.grid1, kernel_size=121)
        self.bn1 = nn.BatchNorm1d(c1)
        self.pool1 = KNNMaxPool(self.grid1, self.grid2, kernel_size=9)

        # Layer 2: k=5x5 = 25
        self.conv2 = KNNConv(c1, c2, self.grid2, self.grid2, kernel_size=25)
        self.bn2 = nn.BatchNorm1d(c2)

        # Layer 3: k=3x3 = 9
        self.conv3 = KNNConv(c2, c3, self.grid2, self.grid2, kernel_size=9)
        self.bn3 = nn.BatchNorm1d(c3)

        # Layer 4: k=3x3 = 9
        self.conv4 = KNNConv(c3, c4, self.grid2, self.grid2, kernel_size=9)
        self.bn4 = nn.BatchNorm1d(c4)
        self.pool2 = KNNMaxPool(self.grid2, self.grid3, kernel_size=9)

        # Layer 5: k=3x3 = 9
        self.conv5 = KNNConv(c4, c5, self.grid3, self.grid3, kernel_size=9)
        self.bn5 = nn.BatchNorm1d(c5)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(c5, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate manifold representations across layers."""
        x1 = F.relu(self.bn1(self.conv1(x)))
        p1 = self.pool1(x1)
        x2 = F.relu(self.bn2(self.conv2(p1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        p2 = self.pool2(x4)
        x5 = F.relu(self.bn5(self.conv5(p2)))
        return [x1, p1, x2, x3, x4, p2, x5]

    def forward_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """Forward pass directly from pre-sampled points [batch, channels, points]."""
        features = self.extract_features(samples)
        x5 = features[-1]  # [batch, c5, num_pts_grid3]
        # Global average pooling over the sensor manifold
        gap = x5.mean(dim=-1)  # [batch, c5]
        return self.classifier(gap)

    def forward(
        self,
        images: torch.Tensor,
        fixations: Optional[torch.Tensor] = None,
        return_fixation_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass from ambient images with optional multi-fixation aggregation.

        Args:
            images: [batch, channels, H, W]
            fixations: Optional [batch, 2] or [batch, num_fixations, 2]
            return_fixation_logits: If True and multiple fixations, returns [batch, num_fix, num_classes]

        Returns:
            Logits: [batch, num_classes] (averaged across fixations)
        """
        samples = self.sampler(images, fixations)
        if samples.ndim == 4:
            batch, num_fix, channels, points = samples.shape
            flat_samples = samples.reshape(batch * num_fix, channels, points)
            logits = self.forward_samples(flat_samples).reshape(batch, num_fix, -1)
            if return_fixation_logits:
                return logits
            return logits.mean(dim=1)
        return self.forward_samples(samples)


class FOVIResidualBlock(nn.Module):
    """Residual building block on a foveated sensor manifold."""

    def __init__(self, channels: int, grid: SensorGrid, kernel_size: int = 9) -> None:
        super().__init__()
        self.conv1 = KNNConv(channels, channels, grid, grid, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = KNNConv(channels, channels, grid, grid, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class FOVIResNet(nn.Module):
    """Modular Residual Network over isotropic foveated manifolds."""

    def __init__(
        self,
        num_classes: int = 1000,
        *,
        sensor: Optional[SensorGrid] = None,
        in_channels: int = 3,
        stages: Sequence[int] = (64, 128, 256),
        num_blocks_per_stage: Sequence[int] = (2, 2, 2),
        fov_degrees: float = 16.0,
        a: float = 0.5,
        target_samples: int = 4096,
    ) -> None:
        super().__init__()
        if sensor is None:
            sensor = FoveatedSensor.from_target_samples(target_samples, fov_degrees=fov_degrees, a=a)
        self.sensor = sensor
        self.sampler = FoveatedSampler(self.sensor)

        # Stage grids
        self.grids: list[SensorGrid] = [self.sensor]
        cur_samples = sensor.num_samples
        for _ in range(len(stages) - 1):
            cur_samples = max(16, cur_samples // 4)
            self.grids.append(FoveatedSensor.from_target_samples(cur_samples, fov_degrees=fov_degrees, a=a))

        self.in_conv = KNNConv(in_channels, stages[0], self.grids[0], self.grids[0], kernel_size=25)
        self.in_bn = nn.BatchNorm1d(stages[0])

        self.stage_modules = nn.ModuleList()
        for i, (out_ch, num_blocks) in enumerate(zip(stages, num_blocks_per_stage)):
            grid = self.grids[i]
            stage_blocks = nn.ModuleList([FOVIResidualBlock(out_ch, grid) for _ in range(num_blocks)])
            if i < len(stages) - 1:
                next_grid = self.grids[i + 1]
                next_ch = stages[i + 1]
                downsample = nn.Sequential(
                    KNNConv(out_ch, next_ch, grid, next_grid, kernel_size=9),
                    nn.BatchNorm1d(next_ch),
                )
                self.stage_modules.append(nn.ModuleDict({"blocks": stage_blocks, "downsample": downsample}))
            else:
                self.stage_modules.append(nn.ModuleDict({"blocks": stage_blocks, "downsample": nn.Identity()}))

        self.classifier = nn.Linear(stages[-1], num_classes)

    def forward(self, images: torch.Tensor, fixations: Optional[torch.Tensor] = None) -> torch.Tensor:
        samples = self.sampler(images, fixations)
        if samples.ndim == 4:
            batch, num_fix, channels, points = samples.shape
            flat_samples = samples.reshape(batch * num_fix, channels, points)
            logits = self._forward_flat(flat_samples).reshape(batch, num_fix, -1)
            return logits.mean(dim=1)
        return self._forward_flat(samples)

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.in_bn(self.in_conv(x)))
        for stage in self.stage_modules:
            for block in stage["blocks"]:
                x = block(x)
            x = stage["downsample"](x)
        gap = x.mean(dim=-1)
        return self.classifier(gap)
