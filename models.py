"""
Model definitions for Baseline Tiny ViT vs. FOVI Tiny ViT with LoRA.
Supports pre-trained ViT-Tiny (DeiT-Tiny) backbones.
"""

import os
import torch
import torch.nn as nn
import timm

# Auto-set FOVI environment variables before importing fovi if not already set
os.environ.setdefault("FOVI_SAVE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
os.environ.setdefault("FOVI_DATASETS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

import fovi
try:
    from fovi.adapter import apply_fovi_lora_to_vit, FOVIAdapter
except (ImportError, AttributeError):
    try:
        from fovi.lora import apply_fovi_lora_to_vit
        from fovi.adapter import FOVIAdapter
    except (ImportError, AttributeError):
        apply_fovi_lora_to_vit = getattr(fovi, "apply_fovi_lora_to_vit", None)
        FOVIAdapter = getattr(fovi, "FOVIAdapter", None)


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
    - Pretrained timm backbone
    - Replaced 100-class classification head (trainable)
    - LoRA adaptation applied to early transformer layers
    - Late transformer layers frozen
    """
    # 1. Instantiate pretrained timm model
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    # 2. Freeze all backbone parameters first
    for param in model.parameters():
        param.requires_grad = False

    # 3. Head and patch embed remain trainable
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        for param in model.head.parameters():
            param.requires_grad = True
            
    if hasattr(model, "patch_embed") and isinstance(model.patch_embed, nn.Module):
        for param in model.patch_embed.parameters():
            param.requires_grad = True

    # 4. Apply LoRA adaptation to early layers
    if apply_fovi_lora_to_vit is not None:
        apply_fovi_lora_to_vit(
            model,
            num_early_layers=num_early_layers,
            rank=rank,
            alpha=alpha,
            dropout=0.0
        )
    else:
        raise ImportError("Could not find apply_fovi_lora_to_vit in fovi, fovi.adapter, or fovi.lora")

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
    - Pretrained timm backbone wrapped in FOVIAdapter
    - Foveated retinal patchification (KNN-convolution on sensor manifold)
    - Early-layer LoRA adaptation applied
    - 100-class classification head (trainable)
    """
    if FOVIAdapter is None:
        raise ImportError("Could not find FOVIAdapter in fovi or fovi.adapter")

    # 1. Instantiate base timm model
    base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    # 2. Wrap with FOVIAdapter
    fovi_model = FOVIAdapter(
        base_model,
        apply_lora=True,
        num_lora_layers=num_early_layers,
        lora_rank=rank,
        lora_alpha=alpha
    )

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
    base_model = get_model("baseline", num_classes=100)
    base_counts = count_parameters(base_model)
    print(f"Baseline Params -> Total: {base_counts['total']:,}, Trainable: {base_counts['trainable']:,} ({base_counts['trainable_pct']:.2f}%)")

    print("\nTesting FOVI ViT + LoRA...")
    fovi_model = get_model("fovi", num_classes=100)
    fovi_counts = count_parameters(fovi_model)
    print(f"FOVI Params     -> Total: {fovi_counts['total']:,}, Trainable: {fovi_counts['trainable']:,} ({fovi_counts['trainable_pct']:.2f}%)")

    x = torch.randn(2, 3, 224, 224)
    out_base = base_model(x)
    out_fovi = fovi_model(x)
    print(f"\nForward shapes -> Baseline: {out_base.shape}, FOVI: {out_fovi.shape}")
    print("=" * 60)
