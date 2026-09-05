"""
Model definitions for Baseline Tiny ViT vs. FOVI Tiny ViT with LoRA.
Supports pre-trained ViT-Tiny (DeiT-Tiny) backbones.
Features:
- Self-contained, robust LoRA implementation (Equation 1 of ICML 2026 paper: W' = W + (alpha/r)*B@A)
- Early-layer adaptation on layers 0-5, late layers 6-11 frozen
- Zero external dependency failures across different FOVI builds
"""

import os
import math
import torch
import torch.nn as nn
import timm

# Auto-set FOVI environment variables before importing fovi if not already set
os.environ.setdefault("FOVI_SAVE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
os.environ.setdefault("FOVI_DATASETS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

import fovi
try:
    import fovi.adapter
except Exception:
    pass
try:
    import fovi.models
except Exception:
    pass


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Linear layer as specified in Hu et al. (2021) and ICML 2026:
    W_hat = W + (alpha / rank) * (B @ A)
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Base frozen linear layer
        self.base_layer = linear_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Trainable low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out


def count_parameters(model: nn.Module) -> dict:
    """Returns total, trainable, and frozen parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_pct": trainable_pct,
    }


def apply_early_lora_to_vit(
    model: nn.Module,
    num_early_layers: int = 6,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0
):
    """
    Applies LoRA to the first half of transformer layers (0 to num_early_layers-1).
    All subsequent layers are frozen.
    The patch embedding and classification head remain trainable.
    """
    # 1. Freeze entire backbone first
    for param in model.parameters():
        param.requires_grad = False

    # 2. Keep classification head and patch embedding trainable
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        for param in model.head.parameters():
            param.requires_grad = True
    if hasattr(model, "patch_embed") and isinstance(model.patch_embed, nn.Module):
        for param in model.patch_embed.parameters():
            param.requires_grad = True

    # 3. Locate transformer blocks (timm, FOVIAdapter, or standard ViT)
    blocks = getattr(model, "blocks", None)
    if blocks is None and hasattr(model, "vit") and hasattr(model.vit, "blocks"):
        blocks = model.vit.blocks
    elif blocks is None and hasattr(model, "base_vit") and hasattr(model.base_vit, "blocks"):
        blocks = model.base_vit.blocks

    if blocks is not None:
        for layer_idx in range(min(num_early_layers, len(blocks))):
            block = blocks[layer_idx]
            
            # Replace Attention and MLP Linear layers with LoRALinear
            if hasattr(block, "attn"):
                if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, nn.Linear):
                    block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
                elif hasattr(block.attn, "q_proj") and isinstance(block.attn.q_proj, nn.Linear):
                    block.attn.q_proj = LoRALinear(block.attn.q_proj, rank=rank, alpha=alpha, dropout=dropout)
                    block.attn.v_proj = LoRALinear(block.attn.v_proj, rank=rank, alpha=alpha, dropout=dropout)
                if hasattr(block.attn, "proj") and isinstance(block.attn.proj, nn.Linear):
                    block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha, dropout=dropout)

            if hasattr(block, "mlp"):
                if hasattr(block.mlp, "fc1") and isinstance(block.mlp.fc1, nn.Linear):
                    block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank=rank, alpha=alpha, dropout=dropout)
                if hasattr(block.mlp, "fc2") and isinstance(block.mlp.fc2, nn.Linear):
                    block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank=rank, alpha=alpha, dropout=dropout)


def build_baseline_vit_lora(
    num_classes: int = 100,
    pretrained: bool = True,
    rank: int = 8,
    alpha: float = 8.0,
    num_early_layers: int = 6,
    model_name: str = "vit_tiny_patch16_224"
) -> nn.Module:
    """
    Builds baseline standard Tiny ViT with:
    - Pretrained timm backbone (DeiT-Tiny)
    - 100-class classification head (trainable)
    - LoRA adaptation on layers 0-5
    - Layers 6-11 frozen
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    apply_early_lora_to_vit(model, num_early_layers=num_early_layers, rank=rank, alpha=alpha)
    return model


def build_fovi_vit_lora(
    num_classes: int = 100,
    pretrained: bool = True,
    rank: int = 8,
    alpha: float = 8.0,
    num_early_layers: int = 6,
    model_name: str = "vit_tiny_patch16_224"
) -> nn.Module:
    """
    Builds FOVI-adapted Tiny ViT with:
    - Pretrained timm backbone wrapped in FOVIAdapter (foveated retinal patchification)
    - LoRA adaptation applied to layers 0-5
    - 100-class classification head (trainable)
    """
    # Robust resolution of FOVIAdapter across any Python/FOVI environment
    FOVIAdapterClass = None
    if hasattr(fovi, "adapter") and hasattr(fovi.adapter, "FOVIAdapter"):
        FOVIAdapterClass = fovi.adapter.FOVIAdapter
    elif hasattr(fovi, "FOVIAdapter"):
        FOVIAdapterClass = fovi.FOVIAdapter
    else:
        try:
            from fovi.adapter import FOVIAdapter
            FOVIAdapterClass = FOVIAdapter
        except Exception:
            try:
                from fovi import FOVIAdapter
                FOVIAdapterClass = FOVIAdapter
            except Exception:
                pass

    if FOVIAdapterClass is None:
        raise ImportError("Could not find FOVIAdapter in fovi or fovi.adapter module.")

    base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    try:
        fovi_model = FOVIAdapterClass(
            base_model,
            apply_lora=True,
            num_lora_layers=num_early_layers,
            lora_rank=rank,
            lora_alpha=alpha
        )
    except TypeError:
        # Fallback if FOVIAdapter doesn't accept lora kwargs
        fovi_model = FOVIAdapterClass(base_model, apply_lora=False)
        apply_early_lora_to_vit(fovi_model, num_early_layers=num_early_layers, rank=rank, alpha=alpha)

    return fovi_model


def get_model(
    model_type: str,
    num_classes: int = 100,
    pretrained: bool = True,
    rank: int = 8,
    alpha: float = 8.0
) -> nn.Module:
    """Factory helper to build 'baseline' or 'fovi' models."""
    model_type = model_type.lower()
    if model_type == "baseline":
        return build_baseline_vit_lora(num_classes=num_classes, pretrained=pretrained, rank=rank, alpha=alpha)
    elif model_type == "fovi":
        return build_fovi_vit_lora(num_classes=num_classes, pretrained=pretrained, rank=rank, alpha=alpha)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'baseline' or 'fovi'.")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Baseline ViT + LoRA...")
    base_model = get_model("baseline", num_classes=100, pretrained=False)
    base_counts = count_parameters(base_model)
    print(f"Baseline Params -> Total: {base_counts['total']:,}, Trainable: {base_counts['trainable']:,} ({base_counts['trainable_pct']:.2f}%)")

    print("\nTesting FOVI ViT + LoRA...")
    fovi_model = get_model("fovi", num_classes=100, pretrained=False)
    fovi_counts = count_parameters(fovi_model)
    print(f"FOVI Params     -> Total: {fovi_counts['total']:,}, Trainable: {fovi_counts['trainable']:,} ({fovi_counts['trainable_pct']:.2f}%)")

    x = torch.randn(2, 3, 224, 224)
    out_base = base_model(x)
    out_fovi = fovi_model(x)
    print(f"\nForward shapes -> Baseline: {out_base.shape}, FOVI: {out_fovi.shape}")
    print("=" * 60)
