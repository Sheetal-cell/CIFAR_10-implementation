"""
TransGAN-style CIFAR-10 implementation
Based primarily on:
  Jiang, Chang, Wang. "TransGAN: Two Pure Transformers Can Make One Strong GAN,
  and That Can Scale Up." NeurIPS 2021.
The architecture follows the CIFAR-10 configuration from the paper appendix,
while keeping the implementation self-contained and easier to study/modify.

Important: this is a faithful educational re-implementation, not the authors'
exact training code/checkpoint implementation.
"""

import argparse
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


# -----------------------------
# Reproducibility / utilities
# -----------------------------

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.LayerNorm)):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# ---------------------------------------
# Relative-position self-attention
# ---------------------------------------

class RelativeSelfAttention(nn.Module):
    """
    Multi-head self-attention with a learnable relative-position bias.

    x: [B, N, C]
    """
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Created lazily because sequence sizes change between stages.
        self.relative_bias = nn.ParameterDict()

    def _get_bias(self, h, w, device):
        key = f"{h}x{w}"
        n = h * w

        if key not in self.relative_bias:
            table_size = (2 * h - 1) * (2 * w - 1)
            p = nn.Parameter(torch.zeros(table_size, self.heads, device=device))
            nn.init.trunc_normal_(p, std=0.02)
            self.relative_bias[key] = p

        table = self.relative_bias[key]

        coords_h = torch.arange(h, device=device)
        coords_w = torch.arange(w, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords = coords.flatten(1)                         # [2, N]
        rel = coords[:, :, None] - coords[:, None, :]      # [2, N, N]
        rel[0] += h - 1
        rel[1] += w - 1
        rel[0] *= 2 * w - 1
        index = rel[0] + rel[1]                             # [N, N]

        bias = table[index.reshape(-1)].reshape(n, n, self.heads)
        return bias.permute(2, 0, 1).unsqueeze(0)           # [1, heads, N, N]

    def forward(self, x, hw=None):
        b, n, c = x.shape
        if hw is None:
            s = int(math.sqrt(n))
            assert s * s == n, "hw is required for non-square sequence lengths"
            h, w = s, s
        else:
            h, w = hw
            assert h * w == n

        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self._get_bias(h, w, x.device)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj_drop(self.proj(out))
        return out


# ---------------------------------------
# Grid self-attention
# ---------------------------------------

class GridSelfAttention(nn.Module):
    """
    Grid attention from TransGAN.

    Instead of attending over all H*W tokens at once, tokens are rearranged
    into local grids. This reduces the quadratic attention cost at larger
    spatial resolutions.

    For CIFAR-10 this is mostly a faithful structural component; at 16x16
    and below standard attention is already affordable.
    """
    def __init__(self, dim, heads=4, grid_size=4, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.grid_size = grid_size
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, hw):
        b, n, c = x.shape
        h, w = hw
        assert n == h * w

        # [B,H,W,C] -> [B,gh,gw,gH,gW,C]
        gh = h // self.grid_size
        gw = w // self.grid_size
        assert h % self.grid_size == 0 and w % self.grid_size == 0

        x2 = x.reshape(b, h, w, c)
        x2 = x2.reshape(
            b, gh, self.grid_size, gw, self.grid_size, c
        ).permute(0, 1, 3, 2, 4, 5)
        x2 = x2.reshape(b * gh * gw, self.grid_size * self.grid_size, c)

        qkv = self.qkv(x2).reshape(
            x2.shape[0], x2.shape[1], 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(x2.shape[0], x2.shape[1], c)
        out = self.drop(self.proj(out))

        # Reverse grid partition.
        out = out.reshape(b, gh, gw, self.grid_size, self.grid_size, c)
        out = out.permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(b, h, w, c)
        return out.reshape(b, n, c)


# ---------------------------------------
# Transformer block
# ---------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        grid=False,
        grid_size=4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if grid:
            self.attn = GridSelfAttention(
                dim, heads=heads, grid_size=grid_size, dropout=dropout
            )
        else:
            self.attn = RelativeSelfAttention(
                dim, heads=heads, dropout=dropout
            )

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, hw):
        x = x + self.attn(self.norm1(x), hw)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------
# Convolution-free token upsampling
# ---------------------------------------

class TokenPixelShuffle(nn.Module):
    """
    2x spatial upsampling with channel reduction.

    [B, H*W, C] -> [B, (2H)*(2W), C/4]
    This mirrors the CIFAR-10 TransGAN appendix's PixelShuffle stages.
    """
    def forward(self, x, hw):
        b, n, c = x.shape
        h, w = hw
        assert n == h * w
        assert c % 4 == 0

        x = x.reshape(b, h, w, c)
        x = x.reshape(b, h, w, 2, 2, c // 4)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, h * 2, w * 2, c // 4)
        return x.reshape(b, (h * 2) * (w * 2), c // 4), (h * 2, w * 2)


# ---------------------------------------
# TransGAN Generator - CIFAR-10
# ---------------------------------------

class TransGANGenerator(nn.Module):
    """
    CIFAR-10 configuration from the TransGAN appendix:

        z: 512
        -> MLP
        -> 8x8 x 1024
        -> 5 Transformer blocks
        -> PixelShuffle
        -> 16x16 x 256
        -> 4 Transformer blocks
        -> PixelShuffle
        -> 32x32 x 64
        -> 2 Transformer blocks
        -> Linear
        -> 32x32 x 3
    """
    def __init__(self, z_dim=512, heads=4):
        super().__init__()
        self.z_dim = z_dim

        self.fc = nn.Linear(z_dim, 8 * 8 * 1024)

        self.blocks1 = nn.ModuleList([
            TransformerBlock(1024, heads=heads, mlp_ratio=2.0)
            for _ in range(5)
        ])

        self.up1 = TokenPixelShuffle()

        self.blocks2 = nn.ModuleList([
            TransformerBlock(256, heads=heads, mlp_ratio=4.0)
            for _ in range(4)
        ])

        self.up2 = TokenPixelShuffle()

        self.blocks3 = nn.ModuleList([
            TransformerBlock(64, heads=heads, mlp_ratio=4.0)
            for _ in range(2)
        ])

        self.to_rgb = nn.Linear(64, 3)
        self.apply(init_weights)

    def forward(self, z):
        x = self.fc(z).reshape(z.shape[0], 64, 1024)
        hw = (8, 8)

        for block in self.blocks1:
            x = block(x, hw)

        x, hw = self.up1(x, hw)

        for block in self.blocks2:
            x = block(x, hw)

        x, hw = self.up2(x, hw)

        for block in self.blocks3:
            x = block(x, hw)

        x = self.to_rgb(x)
        x = x.reshape(z.shape[0], 32, 32, 3)
        x = x.permute(0, 3, 1, 2)
        return torch.tanh(x)


# ---------------------------------------
# Patchify / token pooling
# ---------------------------------------

class PatchEmbed(nn.Module):
    """
    Convolution-free patch embedding.

    CIFAR-10:
      32x32x3 -> 16x16 tokens
      patch size = 2
      token dimension = 192
    """
    def __init__(self, in_ch=3, patch_size=2, dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_ch * patch_size * patch_size, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0

        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(b, (h // p) * (w // p), p * p * c)
        return self.proj(x), (h // p, w // p)


class TokenAvgPool2x(nn.Module):
    """Average-pool a square token map by 2 without convolution."""
    def forward(self, x, hw):
        b, n, c = x.shape
        h, w = hw
        assert h % 2 == 0 and w % 2 == 0
        x = x.reshape(b, h, w, c)
        x = x.reshape(b, h // 2, 2, w // 2, 2, c)
        x = x.mean(dim=(2, 4))
        return x.reshape(b, (h // 2) * (w // 2), c), (h // 2, w // 2)


# ---------------------------------------
# TransGAN Discriminator - CIFAR-10
# ---------------------------------------

class TransGANDiscriminator(nn.Module):
    """
    Multi-scale, convolution-free discriminator.

    CIFAR-10 appendix configuration:
      32x32x3
      -> 2x2 patch embedding -> 16x16 x 192
      -> 3 transformer blocks
      -> average pool -> 8x8 x 192
      -> concatenate a second projected scale -> 8x8 x 384
      -> 3 transformer blocks
      -> CLS token
      -> transformer block
      -> scalar real/fake score

    WGAN-GP uses the raw scalar output, so there is no sigmoid.
    """
    def __init__(self, heads=4):
        super().__init__()

        self.patch = PatchEmbed(3, patch_size=2, dim=192)

        self.stage1 = nn.ModuleList([
            TransformerBlock(192, heads=heads, mlp_ratio=2.0)
            for _ in range(3)
        ])

        self.pool = TokenAvgPool2x()

        # The second branch provides another representation at the
        # downsampled scale before concatenation.
        self.scale2_proj = nn.Linear(192, 192)

        self.merge = nn.Linear(192 * 2, 384)

        self.stage2 = nn.ModuleList([
            TransformerBlock(384, heads=heads, mlp_ratio=2.0)
            for _ in range(3)
        ])

        self.cls = nn.Parameter(torch.zeros(1, 1, 384))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.final_block = TransformerBlock(384, heads=heads, mlp_ratio=2.0)
        self.head = nn.Linear(384, 1)

        self.apply(init_weights)

    def forward(self, x):
        x, hw = self.patch(x)

        for block in self.stage1:
            x = block(x, hw)

        pooled, hw2 = self.pool(x, hw)

        # Multi-scale representation:
        # branch A = pooled stage-1 features
        # branch B = independently projected pooled features
        branch_a = pooled
        branch_b = self.scale2_proj(pooled)
        x = torch.cat([branch_a, branch_b], dim=-1)
        x = self.merge(x)

        for block in self.stage2:
            x = block(x, hw2)

        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        # The final block sees the CLS token and image tokens together.
        x = self.final_block(x, (1, x.shape[1]))
        return self.head(x[:, 0]).squeeze(-1)


# ---------------------------------------
# Differentiable augmentation
# ---------------------------------------

def diffaug(x, prob_color=1.0, prob_cutout=0.3):
    """
    Lightweight differentiable augmentation inspired by the TransGAN recipe.
    The survey/paper reports Color and Cutout as useful augmentation choices.
    """
    if prob_color > 0 and torch.rand((), device=x.device) < prob_color:
        # Brightness
        x = x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.2

        # Saturation
        mean = x.mean(dim=1, keepdim=True)
        sat = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.4
        x = mean + sat * (x - mean)

        # Contrast
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        contrast = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.4
        x = mean + contrast * (x - mean)

    if prob_cutout > 0 and torch.rand((), device=x.device) < prob_cutout:
        b, _, h, w = x.shape
        cut = max(1, h // 4)
        cy = torch.randint(0, h, (b,), device=x.device)
        cx = torch.randint(0, w, (b,), device=x.device)

        yy = torch.arange(h, device=x.device).view(1, h, 1)
        xx = torch.arange(w, device=x.device).view(1, 1, w)

        for i in range(b):
            mask = (
                (yy >= cy[i] - cut // 2)
                & (yy < cy[i] + cut // 2)
                & (xx >= cx[i] - cut // 2)
                & (xx < cx[i] + cut // 2)
            )
            x[i] = x[i].masked_fill(mask.expand_as(x[i]), 0)

    return x.clamp(-1, 1)


# ---------------------------------------
# WGAN-GP
# ---------------------------------------

def gradient_penalty(discriminator, real, fake):
    b = real.size(0)
    alpha = torch.rand(b, 1, 1, 1, device=real.device)

    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    score = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(b, -1)
    norm = gradients.norm(2, dim=1)
    return ((norm - 1.0) ** 2).mean()


# ---------------------------------------
# Dataset
# ---------------------------------------

def get_dataloader(data_dir, batch_size, workers, subset_size=128):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Use only a small subset for CPU testing
    if subset_size is not None:
        subset_size = min(subset_size, len(dataset))
        dataset = torch.utils.data.Subset(
            dataset,
            range(subset_size)
        )

    print(f"Using {len(dataset)} training images")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        drop_last=True,
    )


# ---------------------------------------
# Checkpoint / sample helpers
# ---------------------------------------

@torch.no_grad()
def save_samples(generator, fixed_z, path):
    generator.eval()
    fake = generator(fixed_z)
    fake = (fake + 1) / 2
    save_image(
        fake,
        path,
        nrow=8,
        normalize=False,
    )
    generator.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------
# Training
# ---------------------------------------

def train(args):
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print("Device:", device)

    loader = get_dataloader(
    args.data_dir,
    args.batch_size,
    args.workers,
    args.subset_size
)
    G = TransGANGenerator(z_dim=args.z_dim).to(device)
    D = TransGANDiscriminator().to(device)

    print(f"Generator parameters:     {count_parameters(G):,}")
    print(f"Discriminator parameters: {count_parameters(D):,}")

    # WGAN-GP recipe. The paper reports Adam beta1=0 and beta2=0.99.
    opt_g = torch.optim.Adam(
        G.parameters(),
        lr=args.lr,
        betas=(0.0, 0.99),
    )
    opt_d = torch.optim.Adam(
        D.parameters(),
        lr=args.lr,
        betas=(0.0, 0.99),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    fixed_z = torch.randn(64, args.z_dim, device=device)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for real, _ in loader:
            real = real.to(device, non_blocking=True)
            b = real.size(0)

            # -----------------
            # D update
            # -----------------
            for _ in range(args.d_steps):
                z = torch.randn(b, args.z_dim, device=device)

                with torch.no_grad():
                    fake = G(z)

                # Apply the same augmentation family to both distributions.
                real_aug = diffaug(real)
                fake_aug = diffaug(fake)

                d_real = D(real_aug)
                d_fake = D(fake_aug)

                gp = gradient_penalty(D, real_aug, fake_aug)

                d_loss = d_fake.mean() - d_real.mean() + args.gp_lambda * gp

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            # -----------------
            # G update
            # -----------------
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z)
            fake_aug = diffaug(fake)

            g_loss = -D(fake_aug).mean()

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            global_step += 1

            if global_step % args.log_every == 0:
                print(
                    f"Epoch {epoch:03d}/{args.epochs} "
                    f"Step {global_step:06d} "
                    f"D: {d_loss.item():+.4f} "
                    f"G: {g_loss.item():+.4f} "
                    f"GP: {gp.item():.4f}"
                )

        # Save samples after every epoch.
        sample_path = os.path.join(
            args.out_dir, f"samples_epoch_{epoch:04d}.png"
        )
        save_samples(G, fixed_z, sample_path)

        ckpt_path = os.path.join(
            args.out_dir, f"checkpoint_epoch_{epoch:04d}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "optimizer_g": opt_g.state_dict(),
                "optimizer_d": opt_d.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )

    print("Training finished.")


# ---------------------------------------
# CLI
# ---------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TransGAN-style pure Transformer GAN for CIFAR-10"
    )

    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/transgan_cifar10")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--subset_size", type=int, default=128)

    p.add_argument("--z_dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gp_lambda", type=float, default=10.0)
    p.add_argument("--d_steps", type=int, default=1)

    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
