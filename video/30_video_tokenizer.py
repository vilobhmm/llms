"""
Video Tokenizer and 3D VAE for Text-to-Video Generation
========================================================

This module implements a 3D Variational Autoencoder (VAE) for compressing video
into a compact latent representation, following the approach used in VideoLDM,
CogVideoX, and similar video generation systems.

Architecture Overview:
----------------------
The 3D VAE operates on video tensors of shape (B, C, T, H, W) where:
  - B = batch size
  - C = channels (3 for RGB)
  - T = number of frames
  - H = height
  - W = width

Encoder: (B, 3, T, H, W) → (B, latent_dim, T//2, H//8, W//8)
  - Spatial downsampling: 8x (3 conv stages with stride 2)
  - Temporal downsampling: 2x (1 stage with temporal stride 2)
  - 3D ResBlocks with GroupNorm + SiLU activations

Decoder: (B, latent_dim, T//2, H//8, W//8) → (B, 3, T, H, W)
  - Spatial upsampling via transposed convolutions
  - Temporal upsampling via transposed convolution

The latent space uses reparameterization trick (mu + eps*sigma) during training.
For inference, the mean is used directly (deterministic encoding).

Key Design Choices:
-------------------
- 3D convolutions for joint spatial-temporal processing
- GroupNorm (groups=8) for stable training with small batch sizes
- SiLU activation (smooth gradient flow)
- Residual connections for gradient flow in deep networks
- Separate mu/logvar head for VAE reparameterization
- KL divergence loss term for regularizing latent space

Usage:
------
    tokenizer = VideoTokenizer(latent_dim=4)
    video = torch.randn(2, 3, 8, 64, 64)  # (B, C, T, H, W)
    mu, logvar, z = tokenizer.encode(video)
    recon = tokenizer.decode(z)
    loss = tokenizer.loss(video, recon, mu, logvar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock3D(nn.Module):
    """
    3D Residual Block with GroupNorm and SiLU activation.

    Applies two 3D convolutions with a residual connection.
    If in_channels != out_channels, a 1x1x1 projection is used for the skip connection.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        groups: Number of groups for GroupNorm (default: 8)
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        # Clamp groups to valid range
        groups = min(groups, in_channels)
        while in_channels % groups != 0 and groups > 1:
            groups -= 1

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        out_groups = min(groups, out_channels)
        while out_channels % out_groups != 0 and out_groups > 1:
            out_groups -= 1
        self.norm2 = nn.GroupNorm(out_groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection projection if channels differ
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            (B, out_channels, T, H, W)
        """
        residual = self.skip(x)

        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + residual


class DownBlock3D(nn.Module):
    """
    Downsampling block combining ResBlock3D with spatial or temporal stride.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        spatial_downsample: If True, apply 2x spatial downsampling (H, W)
        temporal_downsample: If True, apply 2x temporal downsampling (T)
        groups: GroupNorm groups
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_downsample: bool = True,
        temporal_downsample: bool = False,
        groups: int = 8,
    ):
        super().__init__()
        self.res = ResBlock3D(in_channels, out_channels, groups=groups)

        if spatial_downsample and temporal_downsample:
            stride = (2, 2, 2)
        elif spatial_downsample:
            stride = (1, 2, 2)
        elif temporal_downsample:
            stride = (2, 1, 1)
        else:
            stride = (1, 1, 1)

        self.downsample = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        x = self.downsample(x)
        return x


class UpBlock3D(nn.Module):
    """
    Upsampling block combining ResBlock3D with transposed convolution.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        spatial_upsample: If True, apply 2x spatial upsampling
        temporal_upsample: If True, apply 2x temporal upsampling
        groups: GroupNorm groups
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        groups: int = 8,
    ):
        super().__init__()
        self.res = ResBlock3D(in_channels, out_channels, groups=groups)

        if spatial_upsample and temporal_upsample:
            stride = (2, 2, 2)
        elif spatial_upsample:
            stride = (1, 2, 2)
        elif temporal_upsample:
            stride = (2, 1, 1)
        else:
            stride = (1, 1, 1)

        if spatial_upsample or temporal_upsample:
            self.upsample = nn.ConvTranspose3d(
                out_channels, out_channels,
                kernel_size=stride, stride=stride
            )
        else:
            self.upsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        x = self.upsample(x)
        return x


class VideoEncoder(nn.Module):
    """
    3D Convolutional Encoder for videos.

    Compresses (B, 3, T, H, W) → (B, latent_dim*2, T//2, H//8, W//8)
    The *2 is for producing both mu and logvar for VAE.

    Architecture:
        - Initial projection: 3 → base_channels
        - 3 spatial downsampling stages (×8 spatial compression)
        - 1 temporal downsampling stage (×2 temporal compression)
        - Final conv to 2*latent_dim (mu + logvar)

    Args:
        in_channels: Input video channels (default: 3 for RGB)
        latent_dim: Latent space dimensionality
        base_channels: Base feature channels (doubled at each stage)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        C = base_channels

        # Initial projection
        self.input_proj = nn.Conv3d(in_channels, C, kernel_size=3, padding=1)

        # Spatial downsampling: H,W → H/2,W/2 → H/4,W/4 → H/8,W/8
        self.down1 = DownBlock3D(C, C*2, spatial_downsample=True)      # /2 spatial
        self.down2 = DownBlock3D(C*2, C*4, spatial_downsample=True)    # /4 spatial
        self.down3 = DownBlock3D(C*4, C*8, spatial_downsample=True)    # /8 spatial

        # Temporal downsampling: T → T/2
        self.down_t = DownBlock3D(C*8, C*8, spatial_downsample=False, temporal_downsample=True)

        # Middle ResBlock
        self.mid = ResBlock3D(C*8, C*8)

        # Final projection to latent (mu + logvar)
        groups = min(8, C*8)
        while (C*8) % groups != 0 and groups > 1:
            groups -= 1
        self.out_norm = nn.GroupNorm(groups, C*8)
        self.out_proj = nn.Conv3d(C*8, latent_dim * 2, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) video tensor
        Returns:
            (B, 2*latent_dim, T//2, H//8, W//8) — concatenated mu and logvar
        """
        h = self.input_proj(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down_t(h)
        h = self.mid(h)
        h = self.act(self.out_norm(h))
        h = self.out_proj(h)
        return h


class VideoDecoder(nn.Module):
    """
    3D Convolutional Decoder for videos.

    Reconstructs (B, latent_dim, T//2, H//8, W//8) → (B, 3, T, H, W)

    Architecture (mirror of encoder):
        - Initial projection from latent_dim → base_channels*8
        - Middle ResBlock
        - Temporal upsampling: T//2 → T
        - 3 spatial upsampling stages
        - Final conv to RGB

    Args:
        out_channels: Output video channels (default: 3 for RGB)
        latent_dim: Latent space dimensionality
        base_channels: Base feature channels
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_dim: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        C = base_channels

        # Initial projection from latent
        self.input_proj = nn.Conv3d(latent_dim, C*8, kernel_size=3, padding=1)

        # Middle ResBlock
        self.mid = ResBlock3D(C*8, C*8)

        # Temporal upsampling: T//2 → T
        self.up_t = UpBlock3D(C*8, C*8, spatial_upsample=False, temporal_upsample=True)

        # Spatial upsampling: H//8,W//8 → H//4,W//4 → H//2,W//2 → H,W
        self.up3 = UpBlock3D(C*8, C*4, spatial_upsample=True)
        self.up2 = UpBlock3D(C*4, C*2, spatial_upsample=True)
        self.up1 = UpBlock3D(C*2, C, spatial_upsample=True)

        # Final projection to RGB
        groups = min(8, C)
        while C % groups != 0 and groups > 1:
            groups -= 1
        self.out_norm = nn.GroupNorm(groups, C)
        self.out_proj = nn.Conv3d(C, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim, T//2, H//8, W//8) latent tensor
        Returns:
            (B, 3, T, H, W) reconstructed video
        """
        h = self.input_proj(z)
        h = self.mid(h)
        h = self.up_t(h)
        h = self.up3(h)
        h = self.up2(h)
        h = self.up1(h)
        h = self.act(self.out_norm(h))
        h = self.out_proj(h)
        return torch.tanh(h)  # Output in [-1, 1]


class VideoTokenizer(nn.Module):
    """
    Complete 3D VAE for video compression and reconstruction.

    Combines VideoEncoder and VideoDecoder with reparameterization trick
    for variational autoencoding of video sequences.

    The KL loss regularizes the latent space toward N(0,1).

    Compression ratios:
        - Spatial: 8x (H/8, W/8)
        - Temporal: 2x (T/2)
        - Channel: 3 → latent_dim (e.g., 3 → 4)

    Args:
        in_channels: Input video channels (default: 3)
        latent_dim: Latent dimensionality (default: 4)
        base_channels: Base feature map channels (default: 32)
        kl_weight: Weight for KL divergence loss (default: 1e-4)

    Example:
        >>> tokenizer = VideoTokenizer(latent_dim=4, base_channels=32)
        >>> video = torch.randn(1, 3, 4, 32, 32)
        >>> mu, logvar, z = tokenizer.encode(video)
        >>> recon = tokenizer.decode(z)
        >>> print(recon.shape)  # (1, 3, 4, 32, 32)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 4,
        base_channels: int = 32,
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.encoder = VideoEncoder(in_channels, latent_dim, base_channels)
        self.decoder = VideoDecoder(in_channels, latent_dim, base_channels)

    def encode(self, x: torch.Tensor):
        """
        Encode video to latent distribution parameters.

        Args:
            x: (B, C, T, H, W) video tensor
        Returns:
            mu: (B, latent_dim, T//2, H//8, W//8) mean
            logvar: (B, latent_dim, T//2, H//8, W//8) log variance
            z: (B, latent_dim, T//2, H//8, W//8) sampled latent
        """
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, -30.0, 20.0)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * sigma

        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        Returns:
            z: Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Deterministic during inference

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to video.

        Args:
            z: (B, latent_dim, T//2, H//8, W//8)
        Returns:
            (B, C, T, H, W) reconstructed video in [-1, 1]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Full VAE forward pass: encode + decode.

        Args:
            x: (B, C, T, H, W) input video
        Returns:
            recon: (B, C, T, H, W) reconstruction
            mu: Latent mean
            logvar: Latent log variance
            z: Sampled latent
        """
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict:
        """
        Compute VAE loss = reconstruction loss + KL divergence.

        Args:
            x: Original video (B, C, T, H, W)
            recon: Reconstructed video (B, C, T, H, W)
            mu: Latent mean
            logvar: Latent log variance
        Returns:
            Dict with 'total', 'recon', 'kl' losses
        """
        # L2 reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total = recon_loss + self.kl_weight * kl_loss

        return {
            'total': total,
            'recon': recon_loss,
            'kl': kl_loss,
        }


class VideoPatchTokenizer(nn.Module):
    """
    3D Patch-based Tokenizer for Video DiT.

    Divides video into 3D patches and projects them to token embeddings.
    This is used as the input stage of the Video DiT (Diffusion Transformer).

    Patch sizes:
        - Temporal: pt (e.g., 1 or 2 frames per patch)
        - Spatial: ph x pw (e.g., 2x2 or 4x4 pixels per patch)

    For a video of shape (B, C, T, H, W) with patch (pt, ph, pw):
        N_patches = (T//pt) * (H//ph) * (W//pw)

    Args:
        in_channels: Input channels
        embed_dim: Output embedding dimension
        patch_size: (temporal_patch, height_patch, width_patch)

    Example:
        >>> tok = VideoPatchTokenizer(3, 256, patch_size=(1, 4, 4))
        >>> x = torch.randn(2, 3, 4, 16, 16)
        >>> tokens, (nt, nh, nw) = tok(x)
        >>> print(tokens.shape)  # (2, 16, 256)
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        patch_size: tuple = (1, 4, 4),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        pt, ph, pw = patch_size

        # 3D conv with patch_size kernel and stride to extract non-overlapping patches
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            tokens: (B, N_patches, embed_dim)
            grid_shape: (nt, nh, nw) number of patches in each dimension
        """
        B, C, T, H, W = x.shape
        pt, ph, pw = self.patch_size

        # Project to patches: (B, D, nt, nh, nw)
        tokens = self.proj(x)  # (B, embed_dim, T//pt, H//ph, W//pw)
        nt, nh, nw = tokens.shape[2], tokens.shape[3], tokens.shape[4]

        # Flatten spatial-temporal patch dimensions: (B, N_patches, embed_dim)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, D)

        return tokens, (nt, nh, nw)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing 3D Video Tokenizer (VAE)")
    print("=" * 60)

    device = torch.device("cpu")

    # Tiny test: 2 frames, 16x16
    B, C, T, H, W = 1, 3, 4, 16, 16
    video = torch.randn(B, C, T, H, W, device=device)

    print(f"\nInput video shape: {video.shape}")

    # Test VideoEncoder
    print("\n[1] Testing VideoEncoder...")
    encoder = VideoEncoder(in_channels=3, latent_dim=4, base_channels=16)
    with torch.no_grad():
        enc_out = encoder(video)
    print(f"  Encoder output: {enc_out.shape}")
    print(f"  Expected: ({B}, 8, {T//2}, {H//8}, {W//8}) = ({B}, 8, {T//2}, {H//8}, {W//8})")
    assert enc_out.shape == (B, 8, T//2, H//8, W//8), f"Shape mismatch: {enc_out.shape}"
    print("  PASSED")

    # Test VideoDecoder
    print("\n[2] Testing VideoDecoder...")
    decoder = VideoDecoder(out_channels=3, latent_dim=4, base_channels=16)
    z = torch.randn(B, 4, T//2, H//8, W//8, device=device)
    with torch.no_grad():
        dec_out = decoder(z)
    print(f"  Decoder output: {dec_out.shape}")
    assert dec_out.shape == (B, C, T, H, W), f"Shape mismatch: {dec_out.shape}"
    print("  PASSED")

    # Test full VideoTokenizer (VAE)
    print("\n[3] Testing VideoTokenizer (full VAE)...")
    tokenizer = VideoTokenizer(in_channels=3, latent_dim=4, base_channels=16)
    tokenizer.train()
    recon, mu, logvar, z = tokenizer(video)
    print(f"  Reconstruction: {recon.shape}")
    print(f"  Mu: {mu.shape}")
    print(f"  Logvar: {logvar.shape}")
    print(f"  z: {z.shape}")
    assert recon.shape == video.shape, f"Recon shape mismatch: {recon.shape}"

    losses = tokenizer.loss(video, recon, mu, logvar)
    print(f"  Losses: recon={losses['recon']:.4f}, kl={losses['kl']:.4f}, total={losses['total']:.4f}")
    losses['total'].backward()
    print("  Backward pass OK")
    print("  PASSED")

    # Test VideoPatchTokenizer
    print("\n[4] Testing VideoPatchTokenizer...")
    patch_tok = VideoPatchTokenizer(in_channels=3, embed_dim=64, patch_size=(1, 4, 4))
    with torch.no_grad():
        tokens, (nt, nh, nw) = patch_tok(video)
    print(f"  Tokens: {tokens.shape}")
    print(f"  Grid: nt={nt}, nh={nh}, nw={nw}")
    expected_n = (T//1) * (H//4) * (W//4)
    assert tokens.shape == (B, expected_n, 64), f"Shape mismatch: {tokens.shape}"
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All VideoTokenizer tests PASSED")
    print("=" * 60)
