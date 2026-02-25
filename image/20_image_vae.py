"""
T2I Chapter 1: Convolutional Variational Autoencoder (VAE)
===========================================================
The VAE is the first stage of a Latent Diffusion Model (LDM). It compresses
high-resolution images into a compact latent space where diffusion is cheaper.

Architecture (following Stable Diffusion's KL-regularized autoencoder):

  Encoder:
    Input  (B, 3, H, W)
      ↓  Conv stem 3→C
      ↓  DownBlock x3  (stride-2 conv, ResBlocks, GroupNorm + SiLU)
      ↓  Mid (ResBlock + Attention + ResBlock)
      ↓  Conv 1x1  → (B, 2*latent_dim, H/8, W/8)
        Split → mu, log_var

  Reparameterize:
    z = mu + eps * exp(0.5 * log_var),  eps ~ N(0,I)

  Decoder:
    Input  (B, latent_dim, H/8, W/8)
      ↓  Conv 1x1  → C
      ↓  Mid (ResBlock + Attention + ResBlock)
      ↓  UpBlock x3  (nearest upsample, ResBlocks)
      ↓  GroupNorm + SiLU + Conv → (B, 3, H, W)

Loss:
  L = L_recon + β * L_KL
  L_recon = MSE(recon, target)
  L_KL    = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

Design choices:
  - GroupNorm (not BatchNorm) for stability with small batch sizes
  - SiLU (Swish) activation for smooth gradients
  - ResBlocks allow deep networks without vanishing gradients
  - Beta-VAE weighting (β<1) for better reconstruction quality
  - Spatial attention at bottleneck for global coherence
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────
# 1.  Basic building blocks
# ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU.
    Optionally changes channel count via 1x1 skip projection.

    GroupNorm groups: we use min(32, channels) to handle small channel counts.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        groups_in  = min(32, in_ch)
        groups_out = min(32, out_ch)

        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups_in,  in_ch)
        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.act   = nn.SiLU()
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.skip  = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    """
    Single-head self-attention on spatial feature maps.
    Flattens H*W into sequence, applies attention, reshapes back.
    Used at the bottleneck for global context.
    """

    def __init__(self, channels: int):
        super().__init__()
        groups = min(32, channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv  = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)                               # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)                   # (B, C, H, W) each
        # Flatten spatial
        q = q.reshape(B, C, -1)                         # (B, C, HW)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1)
        # Attention
        attn = torch.einsum("bci,bcj->bij", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)                  # (B, HW, HW)
        out  = torch.einsum("bij,bcj->bci", attn, v)   # (B, C, HW)
        out  = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    """Downsample block: stride-2 conv + N ResBlocks."""

    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.down     = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res_blks = nn.ModuleList([
            ResBlock(out_ch, out_ch) for _ in range(n_res)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        for blk in self.res_blks:
            x = blk(x)
        return x


class UpBlock(nn.Module):
    """Upsample block: nearest 2x + conv + N ResBlocks."""

    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.up       = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.res_blks = nn.ModuleList([
            ResBlock(out_ch, out_ch) for _ in range(n_res)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        for blk in self.res_blks:
            x = blk(x)
        return x


# ─────────────────────────────────────────────────────────────
# 2.  Encoder
# ─────────────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """
    Convolutional encoder:
      (B, 3, H, W) → (B, latent_dim, H/8, W/8) [mu]
                    → (B, latent_dim, H/8, W/8) [log_var]

    Returns (mu, log_var) separately so the caller can reparameterize.
    """

    def __init__(
        self,
        in_channels: int  = 3,
        base_ch:     int  = 64,
        ch_mult:     Tuple[int, ...] = (1, 2, 4),
        latent_dim:  int  = 4,
        n_res:       int  = 2,
    ):
        super().__init__()
        channels = [base_ch * m for m in ch_mult]

        # Stem
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Downsampling blocks
        downs = []
        for i in range(len(channels) - 1):
            downs.append(DownBlock(channels[i], channels[i + 1], n_res))
        self.downs = nn.ModuleList(downs)

        # Bottleneck
        C = channels[-1]
        self.mid = nn.Sequential(
            ResBlock(C, C),
            SpatialAttention(C),
            ResBlock(C, C),
        )

        # Output: project to 2*latent_dim (mu + log_var)
        groups = min(32, C)
        self.out_norm = nn.GroupNorm(groups, C)
        self.out_conv = nn.Conv2d(C, 2 * latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 3, H, W)
        Returns: mu (B, L, H/8, W/8), log_var (B, L, H/8, W/8)
        """
        h = self.stem(x)
        for down in self.downs:
            h = down(h)
        h = self.mid(h)
        h = F.silu(self.out_norm(h))
        h = self.out_conv(h)
        mu, log_var = h.chunk(2, dim=1)
        return mu, log_var


# ─────────────────────────────────────────────────────────────
# 3.  Reparameterization trick
# ─────────────────────────────────────────────────────────────

def reparameterize(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    """
    Sample z ~ N(mu, sigma^2) using reparameterization trick.
    z = mu + eps * exp(0.5 * log_var),  eps ~ N(0, I)

    At eval time (training=False), returns mu directly (deterministic).
    """
    if not training:
        return mu
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# ─────────────────────────────────────────────────────────────
# 4.  Decoder
# ─────────────────────────────────────────────────────────────

class ConvDecoder(nn.Module):
    """
    Convolutional decoder:
      (B, latent_dim, H/8, W/8) → (B, 3, H, W)

    Mirror of the encoder, with UpBlocks replacing DownBlocks.
    """

    def __init__(
        self,
        out_channels: int  = 3,
        base_ch:      int  = 64,
        ch_mult:      Tuple[int, ...] = (1, 2, 4),
        latent_dim:   int  = 4,
        n_res:        int  = 2,
    ):
        super().__init__()
        # Channel sizes in reverse order
        channels = [base_ch * m for m in reversed(ch_mult)]

        # Input projection
        self.in_conv = nn.Conv2d(latent_dim, channels[0], 1)

        # Bottleneck
        C = channels[0]
        self.mid = nn.Sequential(
            ResBlock(C, C),
            SpatialAttention(C),
            ResBlock(C, C),
        )

        # Upsampling blocks
        ups = []
        for i in range(len(channels) - 1):
            ups.append(UpBlock(channels[i], channels[i + 1], n_res))
        self.ups = nn.ModuleList(ups)

        # Output
        C_out = channels[-1]
        groups = min(32, C_out)
        self.out_norm = nn.GroupNorm(groups, C_out)
        self.out_conv = nn.Conv2d(C_out, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim, H/8, W/8)
        Returns: (B, 3, H, W) in range [-1, 1] (tanh output)
        """
        h = self.in_conv(z)
        h = self.mid(h)
        for up in self.ups:
            h = up(h)
        h = F.silu(self.out_norm(h))
        h = self.out_conv(h)
        return torch.tanh(h)


# ─────────────────────────────────────────────────────────────
# 5.  Full VAE
# ─────────────────────────────────────────────────────────────

class ConvVAE(nn.Module):
    """
    Complete Convolutional VAE combining encoder, reparameterization, and decoder.

    Latent space: spatial (not flattened) — preserves 2D structure.
    This is essential for LDMs which do spatial diffusion in latent space.

    The encode() method returns (mu, log_var) for use during training.
    The decode() method maps latents back to images.
    The forward() method returns (recon, mu, log_var) for loss computation.
    """

    def __init__(
        self,
        in_channels: int  = 3,
        base_ch:     int  = 64,
        ch_mult:     Tuple[int, ...] = (1, 2, 4),
        latent_dim:  int  = 4,
        n_res:       int  = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(in_channels, base_ch, ch_mult, latent_dim, n_res)
        self.decoder = ConvDecoder(in_channels, base_ch, ch_mult, latent_dim, n_res)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B,3,H,W) → (mu, log_var) each (B, L, H/8, W/8)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, L, H/8, W/8) → (B, 3, H, W)"""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        Returns: (reconstruction, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var, self.training)
        recon = self.decode(z)
        return recon, mu, log_var

    def sample(
        self,
        n: int,
        latent_h: int,
        latent_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate n random images from prior N(0,I)."""
        z = torch.randn(n, self.latent_dim, latent_h, latent_w, device=device)
        return self.decode(z)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────
# 6.  VAE Loss
# ─────────────────────────────────────────────────────────────

def vae_loss(
    recon:   torch.Tensor,
    target:  torch.Tensor,
    mu:      torch.Tensor,
    log_var: torch.Tensor,
    beta:    float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Beta-VAE loss = reconstruction loss + beta * KL divergence.

    Args:
        recon   : reconstructed images (B, 3, H, W) in [-1, 1]
        target  : original images (B, 3, H, W) in [-1, 1]
        mu      : encoder mean (B, L, h, w)
        log_var : encoder log variance (B, L, h, w)
        beta    : KL weight (<<1 gives better reconstruction quality)

    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction loss (MSE per pixel, averaged)
    recon_loss = F.mse_loss(recon, target, reduction="mean")

    # KL divergence:  -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(
        1 + log_var - mu.pow(2) - log_var.exp()
    )

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ─────────────────────────────────────────────────────────────
# 7.  Smoke-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CONVOLUTIONAL VAE — Smoke Tests")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    B = 2
    H = W = 32   # tiny 32x32 images for smoke test

    # Tiny config for fast testing
    vae = ConvVAE(
        in_channels=3,
        base_ch=16,
        ch_mult=(1, 2, 4),
        latent_dim=4,
        n_res=1,
    ).to(device)
    print(f"\n[1] VAE params: {vae.num_parameters():,}")

    # Forward pass
    x = torch.randn(B, 3, H, W, device=device)
    recon, mu, log_var = vae(x)

    print(f"\n[2] Forward pass")
    print(f"  input:   {x.shape}")
    print(f"  recon:   {recon.shape}")
    print(f"  mu:      {mu.shape}")
    print(f"  log_var: {log_var.shape}")
    assert recon.shape == x.shape, f"recon shape mismatch: {recon.shape}"
    assert mu.shape == (B, 4, H // 8, W // 8), f"mu shape: {mu.shape}"

    # Loss
    total, recon_l, kl_l = vae_loss(recon, x, mu, log_var)
    print(f"\n[3] Loss")
    print(f"  total={total.item():.4f}  recon={recon_l.item():.4f}  kl={kl_l.item():.4f}")

    # Sampling
    samples = vae.sample(4, H // 8, W // 8, device)
    print(f"\n[4] Sample: {samples.shape}  range=[{samples.min():.2f}, {samples.max():.2f}]")

    # Encode / decode roundtrip
    mu2, lv2 = vae.encode(x)
    recon2 = vae.decode(mu2)
    print(f"\n[5] Encode/decode roundtrip: {x.shape} → z{mu2.shape} → {recon2.shape}")

    # Backward
    total.backward()
    print(f"\n[6] Backward pass OK")

    print("\n[OK] All VAE tests passed")
