# TransGAN-style CIFAR-10

This project implements a convolution-free Transformer GAN for CIFAR-10,
primarily following **TransGAN (Jiang, Chang, Wang, NeurIPS 2021)**.

## Architecture

### Generator

```text
z (512)
   ↓
Linear
   ↓
8 × 8 × 1024
   ↓
5 × Transformer Blocks
   ↓
2× Token PixelShuffle
   ↓
16 × 16 × 256
   ↓
4 × Transformer Blocks
   ↓
2× Token PixelShuffle
   ↓
32 × 32 × 64
   ↓
2 × Transformer Blocks
   ↓
Linear
   ↓
32 × 32 × 3
```

### Discriminator

```text
32 × 32 × 3
   ↓
2 × 2 patch embedding
   ↓
16 × 16 × 192 tokens
   ↓
3 × Transformer Blocks
   ↓
2× token average pooling
   ↓
8 × 8 × 192
   ↓
multi-scale concatenation
   ↓
8 × 8 × 384
   ↓
3 × Transformer Blocks
   ↓
CLS token + Transformer Block
   ↓
real/fake scalar
```

There are no convolution layers in either network.

## Training

The implementation uses WGAN-GP:

- discriminator loss: `E[D(fake)] - E[D(real)] + λ GP`
- generator loss: `-E[D(fake)]`
- `lambda_gp = 10`
- Adam with `(beta1, beta2) = (0.0, 0.99)`
- differentiable color/cutout augmentation
- horizontal flip on CIFAR-10

## Install

```bash
pip install -r requirements.txt
```

## Train

For a GPU:

```bash
python transgan_cifar10.py --epochs 100 --batch_size 64
```

For CPU testing:

```bash
python transgan_cifar10.py --epochs 1 --batch_size 8 --workers 0 --cpu
```

Generated sample grids and checkpoints are written to:

```text
runs/transgan_cifar10/
```

## Important note

This is an educational/experimental implementation based on the architecture
configuration in the TransGAN paper. It is intentionally self-contained and
readable rather than reproducing every engineering optimization in the
authors' official repository (distributed training, mixed precision,
checkpointed gradients, etc.).
