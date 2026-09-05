# Comparative Benchmarking: Tiny ViT vs. FOVI-ViT with LoRA

A reproduction and benchmarking pipeline comparing a **Standard Pretrained Tiny Vision Transformer** (`vit_tiny_patch16_224` / DeiT-Tiny) vs. **FOVI-ViT** using **LoRA (Low-Rank Adaptation)** on a 100-class dataset (50,000 images).

Methodology follows the ICML 2026 paper:
> **FOVI: A biologically-inspired foveated interface for deep vision models**  
> *Nicholas M. Blauch, George A. Alvarez, Talia Konkle*  
> [GitHub](https://github.com/nblauch/fovi) | [arXiv:2602.03766](https://arxiv.org/abs/2602.03766)

---

## Key Features

1. **Exact Paper-Aligned Architecture & LoRA Strategy**:
   - **Early-Layer LoRA**: LoRA ($r=8, \alpha=8.0$) applied to the first half of transformer blocks (layers 0–5).
   - **Frozen Late Layers**: Layers 6–11 are frozen to retain pre-trained representations.
   - **Foveated Patchification**: kNN-convolution on retinal sensor manifold ($a=2.79, 64$ patches, $\sim 3,976$ retinal pixels).
   - **Saccadic Multi-Fixation**: 4 random fixations in central disk ($r=0.25$) with mean logit averaging during training.
2. **Reproducible 3-Way Stratified Data Split**:
   - Train: 40,000 images (80%)
   - Val: 5,000 images (10%)
   - Test: 5,000 images (10%) — strictly held out.
3. **Complexity & Efficiency Profiling**:
   - Theoretical GFLOPs via `fovi.estimate_flops` and `fvcore`.
   - Latency (ms/image), Throughput (FPS), and peak VRAM.
4. **Complete Progress & Logging**:
   - Dynamic `tqdm` progress bars.
   - Dual console and file logging (`train_baseline.log`, `train_fovi.log`, `evaluation.log`).
   - Checkpointing for best, final, and LoRA adapter weights.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Profile Model FLOPs & Complexity
```bash
python profile_flops.py
```

### 2. Train Models (Baseline & FOVI)
```bash
# Train both models sequentially (e.g. 50 epochs)
python train.py --model all --epochs 50 --batch_size 64 --lr 5e-4

# Or train individually:
python train.py --model baseline --epochs 50 --batch_size 64 --lr 5e-4
python train.py --model fovi --epochs 50 --batch_size 64 --lr 5e-4
```

### 3. Evaluate on Held-Out Test Set (Table 1 Protocol)
```bash
python evaluate_and_compare.py
```

---

## Repository Structure

```
d:\Research\
├── dataset.py                # Dataset loader & stratified train/val/test split
├── models.py                 # Baseline ViT + LoRA & FOVI-ViT + LoRA architectures
├── profile_flops.py          # FLOPs, latency, and memory profiling module
├── train.py                  # Training pipeline with AMP FP16, tqdm, and logging
├── evaluate_and_compare.py   # Final test evaluation & comparison table generator
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules for checkpoints and cache
└── README.md                 # Project overview and instructions
```
