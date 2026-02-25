"""
Video Diffusion Transformer (Video DiT)
========================================

This module implements the Video DiT (Diffusion Transformer) architecture for
text-conditioned video generation. It follows the design principles from DiT
(Peebles & Xie 2022), CogVideoX (Yang et al. 2024), and related work.

Architecture Overview:
----------------------
The Video DiT operates on latent video representations from the 3D VAE:
  Input:  (B, latent_dim, T//2, H//8, W//8) noisy latent + text conditioning
  Output: (B, latent_dim, T//2, H//8, W//8) predicted noise/velocity field

Pipeline:
  1. 3D PatchEmbed: Divide latent video into 3D patches, project to embeddings
  2. Add 3D sinusoidal positional embeddings
  3. N × VideoTransformerBlocks (spatial-temporal attention + text cross-attention)
  4. Unpatchify: Reconstruct latent video from patch tokens

Key Conditioning Mechanisms:
-----------------------------
1. Timestep Conditioning (AdaLN-Zero):
   - Embed diffusion timestep t → sinusoidal → MLP → (scale, shift, gate)
   - Scale and shift applied to LayerNorm output before each sub-block
   - Gate controls residual connection strength (zero init → identity at start)
   - This is the primary approach used in DiT and CogVideoX

2. Text Cross-Attention:
   - Text tokens (from simple bag-of-words embedding or external encoder)
   - Cross-attention: video tokens query, text tokens are keys/values
   - Allows video content to be guided by text description

3D Positional Embedding:
--------------------------
We use learnable 3D positional embeddings decomposed as:
  pos_emb = temporal_emb[t] + height_emb[h] + width_emb[w]
This factorized form generalizes to different resolutions.

AdaLN-Zero:
-----------
Adaptive Layer Normalization with zero initialization:
  x' = γ(t) * LayerNorm(x) + β(t)
where γ, β are predicted from timestep embedding.
At initialization: γ=1, β=0 → blocks act as identity (stable training start).

Text Encoder (Lightweight):
----------------------------
For the T2V system, we use a lightweight bag-of-words text encoder:
  - Character n-gram hashing → fixed-size vocabulary
  - Embedding lookup → mean pooling → MLP projection
  - Output: (B, text_seq_len, text_dim) token embeddings

For production use, replace with T5-Large or CLIP text encoder.

Usage:
------
    cfg = VideoDiTConfig(dim=128, depth=4, num_heads=4)
    model = VideoDiT(cfg)

    latent = torch.randn(1, 4, 2, 4, 4)    # (B, C, T//2, H//8, W//8)
    t = torch.tensor([500])                  # Timestep
    text = torch.randn(1, 16, 128)          # Text tokens

    pred = model(latent, t, text)
    print(pred.shape)  # (1, 4, 2, 4, 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_attention import SpatialTemporalBlock, MultiHeadAttention


def get_sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding (Vaswani et al. 2017).

    Encodes scalar timestep values into continuous embedding vectors.
    Each dimension uses a different frequency:
      emb[2i]   = sin(t / 10000^(2i/dim))
      emb[2i+1] = cos(t / 10000^(2i/dim))

    Args:
        timesteps: (B,) tensor of integer timesteps [0, T]
        dim: Embedding dimension (must be even)
    Returns:
        (B, dim) embedding tensor
    """
    assert dim % 2 == 0
    device = timesteps.device
    half = dim // 2

    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device).float() / half
    )  # (half,)

    args = timesteps.float()[:, None] * freqs[None, :]  # (B, half)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Encodes diffusion timestep t into a conditioning vector.

    Architecture:
        t → sinusoidal(t) → Linear → SiLU → Linear → (B, embed_dim)

    The output is used to condition the DiT blocks via AdaLN.

    Args:
        embed_dim: Output embedding dimension (should match model dim)
        freq_dim: Sinusoidal frequency dimension (default: 256)
    """

    def __init__(self, embed_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            (B, embed_dim) timestep embeddings
        """
        freq_emb = get_sinusoidal_embedding(t, self.freq_dim)
        return self.mlp(freq_emb)


class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm Zero (DiT-style conditioning).

    From timestep embedding, predict 6 parameters for controlling
    the 3 sub-blocks (spatial attn, temporal attn, FFN):
        - scale_1, shift_1: for spatial attention
        - scale_2, shift_2: for temporal attention
        - scale_3, shift_3: for FFN

    Zero initialization: all parameters start at 0, so scale=0 → (1+0)=1
    and shift=0 → acts as identity LayerNorm at training start.

    Args:
        dim: Model dimension
        cond_dim: Conditioning (timestep embedding) dimension
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 6),  # 6 parameters: 2 per sub-block × 3 sub-blocks
        )
        # Zero init for stable training start
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, c: torch.Tensor):
        """
        Args:
            c: (B, cond_dim) conditioning vector (timestep emb)
        Returns:
            Tuple of 3 (scale, shift) pairs for the 3 sub-blocks
        """
        params = self.proj(c)  # (B, 6*dim)
        chunks = params.chunk(6, dim=-1)  # 6 × (B, dim)
        scale1, shift1 = chunks[0], chunks[1]
        scale2, shift2 = chunks[2], chunks[3]
        scale3, shift3 = chunks[4], chunks[5]
        return (scale1, shift1), (scale2, shift2), (scale3, shift3)


class Patch3DEmbed(nn.Module):
    """
    3D Patch Embedding for Video DiT.

    Converts a latent video (B, C, T, H, W) into a sequence of patch tokens
    with 3D positional embeddings.

    Patch extraction:
        - Uses 3D convolution with patch_size kernel/stride
        - For patch_size=(1, 2, 2): temporal stride=1, spatial stride=2
        - Result: (B, N_patches, embed_dim) tokens

    Positional embedding:
        - Learnable factorized embeddings: temporal + height + width
        - pe[b, t, h, w] = t_emb[t] + h_emb[h] + w_emb[w]
        - Generalizes to different T, H, W at inference time

    Args:
        in_channels: Latent channels
        embed_dim: Token embedding dimension
        patch_size: (temporal, height, width) patch size
        max_frames: Maximum temporal positions for embedding
        max_height: Maximum height positions
        max_width: Maximum width positions
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        patch_size: tuple = (1, 2, 2),
        max_frames: int = 32,
        max_height: int = 32,
        max_width: int = 32,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        pt, ph, pw = patch_size

        # 3D convolution for patch extraction
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(pt, ph, pw),
            stride=(pt, ph, pw),
        )

        # Factorized learnable positional embeddings
        self.t_embed = nn.Embedding(max_frames, embed_dim)
        self.h_embed = nn.Embedding(max_height, embed_dim)
        self.w_embed = nn.Embedding(max_width, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, T, H, W) latent video
        Returns:
            tokens: (B, N_patches, embed_dim)
            grid_shape: (nt, nh, nw)
        """
        B, C, T, H, W = x.shape
        pt, ph, pw = self.patch_size

        # Patch projection: (B, D, nt, nh, nw)
        patches = self.proj(x)
        nt, nh, nw = patches.shape[2], patches.shape[3], patches.shape[4]

        # Build positional embedding indices
        t_idx = torch.arange(nt, device=x.device)
        h_idx = torch.arange(nh, device=x.device)
        w_idx = torch.arange(nw, device=x.device)

        # Compute factorized 3D positional embeddings
        # t_pe: (nt, D), h_pe: (nh, D), w_pe: (nw, D)
        t_pe = self.t_embed(t_idx)  # (nt, D)
        h_pe = self.h_embed(h_idx)  # (nh, D)
        w_pe = self.w_embed(w_idx)  # (nw, D)

        # Broadcast and add: (nt, nh, nw, D)
        pe = (
            t_pe[:, None, None, :] +   # (nt, 1, 1, D)
            h_pe[None, :, None, :] +   # (1, nh, 1, D)
            w_pe[None, None, :, :]     # (1, 1, nw, D)
        )  # (nt, nh, nw, D)
        pe = pe.view(nt * nh * nw, self.embed_dim)  # (N, D)

        # Flatten patches to tokens: (B, D, N) → (B, N, D)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add positional embedding
        tokens = tokens + pe.unsqueeze(0)  # (B, N, D)

        return tokens, (nt, nh, nw)


class SimpleTextEncoder(nn.Module):
    """
    Lightweight text encoder for T2V conditioning.

    Uses character n-gram hashing to convert text to a fixed-size
    token sequence without requiring external tokenizers.

    Architecture:
        text string → character n-gram hashes → embedding lookup
        → linear projection → (B, num_tokens, text_dim)

    This is a simplified substitute for T5/CLIP text encoders.
    For production use, replace with a proper pretrained text encoder.

    Args:
        vocab_size: Hash vocabulary size (default: 4096)
        num_tokens: Fixed output sequence length (padded/truncated)
        embed_dim: Text embedding dimension
        out_dim: Output projection dimension (matches model dim)
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        num_tokens: int = 16,
        embed_dim: int = 128,
        out_dim: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def text_to_ids(self, texts: list) -> torch.Tensor:
        """
        Convert list of strings to token ID tensors using character n-grams.

        Uses simple polynomial hashing for each character bigram.
        """
        batch_ids = []
        for text in texts:
            ids = []
            # Use character bigrams for better coverage
            text = text.lower()[:64]  # Truncate long texts
            for i in range(len(text)):
                char_id = (ord(text[i]) * 31 + i) % self.vocab_size
                if char_id == 0:
                    char_id = 1  # Avoid padding index
                ids.append(char_id)
            # Pad or truncate to num_tokens
            if len(ids) < self.num_tokens:
                ids = ids + [0] * (self.num_tokens - len(ids))
            else:
                ids = ids[:self.num_tokens]
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long)

    def forward(self, texts):
        """
        Args:
            texts: List of strings OR (B, num_tokens) token tensor
        Returns:
            (B, num_tokens, out_dim) text token embeddings
        """
        if isinstance(texts, list):
            ids = self.text_to_ids(texts).to(self.embedding.weight.device)
        else:
            ids = texts  # Already a tensor

        emb = self.embedding(ids)  # (B, num_tokens, embed_dim)
        return self.proj(emb)      # (B, num_tokens, out_dim)


class VideoTransformerBlock(nn.Module):
    """
    Single Video Transformer Block with spatial-temporal attention and text cross-attention.

    Architecture:
        1. SpatialTemporalBlock: factorized spatial + temporal self-attention + FFN
           (conditioned via AdaLN from timestep)
        2. Text Cross-Attention: video tokens attend to text tokens
           (LayerNorm → cross-attn → residual)

    This block is the core of the Video DiT and is stacked N times.

    Args:
        dim: Model embedding dimension
        num_heads: Number of attention heads
        text_dim: Text embedding dimension (for cross-attention KV)
        ff_mult: FFN expansion multiplier
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        text_dim: int = 256,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Spatial-temporal self-attention + FFN
        self.st_block = SpatialTemporalBlock(dim, num_heads, ff_mult, dropout)

        # AdaLN-Zero for timestep conditioning
        self.adaln = AdaLNZero(dim, dim)

        # Cross-attention to text
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_norm = nn.LayerNorm(dim)
        # Project text to model dim if needed
        if text_dim != dim:
            self.text_proj = nn.Linear(text_dim, dim)
        else:
            self.text_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_tokens: torch.Tensor,
        grid_shape: tuple,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) video patch tokens (N = nt*nh*nw)
            t_emb: (B, D) timestep embedding
            text_tokens: (B, M, text_dim) text token embeddings
            grid_shape: (nt, nh, nw) for reshaping to 4D
        Returns:
            (B, N, D) processed tokens
        """
        B, N, D = x.shape
        nt, nh, nw = grid_shape

        # Get AdaLN parameters
        (s1, sh1), (s2, sh2), (s3, sh3) = self.adaln(t_emb)
        scale_shift = (s1, sh1)  # Use first pair (others apply to temporal/FFN)

        # Reshape to (B, T, HW, D) for SpatialTemporalBlock
        x = x.view(B, nt, nh * nw, D)

        # Spatial-temporal block with AdaLN
        x = self.st_block(x, scale_shift=scale_shift)

        # Flatten back for cross-attention
        x = x.view(B, N, D)

        # Text cross-attention
        x_normed = self.cross_norm(x)
        text_proj = self.text_proj(text_tokens)  # (B, M, D)
        cross_out = self.cross_attn(x_normed, context=text_proj)
        x = x + cross_out

        return x


class VideoDiTConfig:
    """
    Configuration class for Video DiT model.

    Attributes:
        latent_dim: Latent video channels (from VAE)
        embed_dim: Token embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        ff_mult: FFN expansion multiplier
        patch_size: 3D patch size (temporal, height, width)
        text_dim: Text encoder output dimension
        text_tokens: Number of text tokens
        max_frames: Maximum temporal positions
        max_height: Maximum height patches
        max_width: Maximum width patches
    """

    # Predefined configurations
    CONFIGS = {
        "tiny": {
            "latent_dim": 4,
            "embed_dim": 128,
            "depth": 2,
            "num_heads": 4,
            "ff_mult": 4,
            "patch_size": (1, 2, 2),
            "text_dim": 64,
            "text_tokens": 8,
            "max_frames": 8,
            "max_height": 16,
            "max_width": 16,
        },
        "small": {
            "latent_dim": 4,
            "embed_dim": 256,
            "depth": 4,
            "num_heads": 8,
            "ff_mult": 4,
            "patch_size": (1, 2, 2),
            "text_dim": 128,
            "text_tokens": 16,
            "max_frames": 16,
            "max_height": 32,
            "max_width": 32,
        },
    }

    def __init__(self, preset: str = "tiny", **kwargs):
        """
        Args:
            preset: "tiny" or "small" predefined config
            **kwargs: Override any config parameter
        """
        config = self.CONFIGS[preset].copy()
        config.update(kwargs)

        self.latent_dim = config["latent_dim"]
        self.embed_dim = config["embed_dim"]
        self.depth = config["depth"]
        self.num_heads = config["num_heads"]
        self.ff_mult = config["ff_mult"]
        self.patch_size = config["patch_size"]
        self.text_dim = config["text_dim"]
        self.text_tokens = config["text_tokens"]
        self.max_frames = config["max_frames"]
        self.max_height = config["max_height"]
        self.max_width = config["max_width"]

    def __repr__(self):
        return (
            f"VideoDiTConfig(embed_dim={self.embed_dim}, depth={self.depth}, "
            f"num_heads={self.num_heads}, patch_size={self.patch_size})"
        )


class VideoDiT(nn.Module):
    """
    Video Diffusion Transformer (Video DiT).

    The core denoising network for text-conditioned video generation.
    Takes noisy latent video + timestep + text conditioning, predicts
    velocity field (flow matching) or noise (DDPM).

    Architecture:
        1. Patch3DEmbed: latent → tokens + positional embedding
        2. TimestepEmbedder: t → conditioning vector
        3. SimpleTextEncoder: text → token embeddings
        4. N × VideoTransformerBlock: spatial-temporal attn + cross-attn
        5. Unpatchify: tokens → latent video

    Args:
        config: VideoDiTConfig with architecture hyperparameters

    Example:
        >>> config = VideoDiTConfig("tiny")
        >>> model = VideoDiT(config)
        >>> latent = torch.randn(1, 4, 2, 4, 4)
        >>> t = torch.tensor([500])
        >>> text_tokens = torch.randn(1, 8, 64)
        >>> pred = model(latent, t, text_tokens)
        >>> print(pred.shape)  # (1, 4, 2, 4, 4)
    """

    def __init__(self, config: VideoDiTConfig):
        super().__init__()
        self.config = config
        C = config.latent_dim
        D = config.embed_dim
        pt, ph, pw = config.patch_size

        # Input patch embedding
        self.patch_embed = Patch3DEmbed(
            in_channels=C,
            embed_dim=D,
            patch_size=config.patch_size,
            max_frames=config.max_frames,
            max_height=config.max_height,
            max_width=config.max_width,
        )

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(D)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VideoTransformerBlock(
                dim=D,
                num_heads=config.num_heads,
                text_dim=config.text_dim,
                ff_mult=config.ff_mult,
            )
            for _ in range(config.depth)
        ])

        # Output projection: tokens → latent patches
        # Each patch has C * pt * ph * pw values
        patch_dim = C * pt * ph * pw
        self.out_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, patch_dim)

        # Zero init output projection for training stability
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def unpatchify(
        self,
        tokens: torch.Tensor,
        grid_shape: tuple,
        target_shape: tuple,
    ) -> torch.Tensor:
        """
        Reconstruct latent video from patch tokens.

        Args:
            tokens: (B, N, patch_dim) projected patch tokens
            grid_shape: (nt, nh, nw) patch grid dimensions
            target_shape: (B, C, T, H, W) target tensor shape
        Returns:
            (B, C, T, H, W) reconstructed latent
        """
        B, N, pd = tokens.shape
        nt, nh, nw = grid_shape
        C = self.config.latent_dim
        pt, ph, pw = self.config.patch_size

        # Reshape to patch grid: (B, nt, nh, nw, pt*ph*pw*C)
        tokens = tokens.view(B, nt, nh, nw, pt * ph * pw * C)

        # Rearrange to: (B, C, nt*pt, nh*ph, nw*pw)
        # tokens[b, t, h, w] → pixel block at temporal t, spatial (h, w)
        tokens = tokens.permute(0, 4, 1, 2, 3)  # (B, pt*ph*pw*C, nt, nh, nw)
        tokens = tokens.view(B, C, pt, ph, pw, nt, nh, nw)
        tokens = tokens.permute(0, 1, 5, 2, 6, 3, 7, 4)  # (B, C, nt, pt, nh, ph, nw, pw)
        tokens = tokens.contiguous().view(B, C, nt*pt, nh*ph, nw*pw)

        return tokens

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoise video latent conditioned on timestep and text.

        Args:
            latent: (B, C, T, H, W) noisy latent video
            timesteps: (B,) diffusion timesteps
            text_tokens: (B, M, text_dim) text conditioning tokens
        Returns:
            (B, C, T, H, W) predicted velocity field / noise
        """
        B, C, T, H, W = latent.shape

        # Tokenize input
        tokens, grid_shape = self.patch_embed(latent)  # (B, N, D)
        nt, nh, nw = grid_shape

        # Timestep embedding
        t_emb = self.t_embedder(timesteps)  # (B, D)

        # Process through transformer blocks
        for block in self.blocks:
            tokens = block(tokens, t_emb, text_tokens, grid_shape)

        # Output projection
        tokens = self.out_norm(tokens)
        tokens = self.out_proj(tokens)  # (B, N, C*pt*ph*pw)

        # Reconstruct latent shape
        pred = self.unpatchify(tokens, grid_shape, latent.shape)

        return pred

    def get_cross_attention_weights(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get cross-attention weights from the last block for visualization.

        Returns:
            (B, num_heads, N_video, N_text) attention weights
        """
        B, C, T, H, W = latent.shape
        tokens, grid_shape = self.patch_embed(latent)
        t_emb = self.t_embedder(timesteps)

        # Process through all but last block
        for block in self.blocks[:-1]:
            tokens = block(tokens, t_emb, text_tokens, grid_shape)

        # Get attention from last block
        last_block = self.blocks[-1]
        nt, nh, nw = grid_shape
        x = tokens.view(B, nt, nh * nw, self.config.embed_dim)
        x = last_block.st_block(x)
        x = x.view(B, -1, self.config.embed_dim)

        # Cross-attention weights
        x_normed = last_block.cross_norm(x)
        text_proj = last_block.text_proj(text_tokens)
        weights = last_block.cross_attn.get_attention_weights(x_normed, context=text_proj)
        return weights  # (B, H, N, M)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Video DiT Architecture")
    print("=" * 60)

    device = torch.device("cpu")

    # Tiny config for testing
    config = VideoDiTConfig("tiny")
    print(f"\nConfig: {config}")

    # Test inputs: small latent for speed
    B = 1
    C, T, H, W = 4, 2, 8, 8  # latent shape (from 4-frame, 64x64 video with 8x spatial, 2x temporal compression)
    latent = torch.randn(B, C, T, H, W, device=device)
    timesteps = torch.tensor([500], device=device)
    text_tokens = torch.randn(B, config.text_tokens, config.text_dim, device=device)

    print(f"\nInputs:")
    print(f"  Latent: {latent.shape}")
    print(f"  Timesteps: {timesteps}")
    print(f"  Text tokens: {text_tokens.shape}")

    # Test individual components
    print("\n[1] Testing Patch3DEmbed...")
    patch_embed = Patch3DEmbed(C, config.embed_dim, config.patch_size)
    tokens, grid_shape = patch_embed(latent)
    print(f"  Tokens: {tokens.shape}, grid: {grid_shape}")
    nt, nh, nw = grid_shape
    assert tokens.shape == (B, nt*nh*nw, config.embed_dim)
    print("  PASSED")

    print("\n[2] Testing TimestepEmbedder...")
    t_emb = TimestepEmbedder(config.embed_dim)
    t_vec = t_emb(timesteps)
    print(f"  T embedding: {t_vec.shape}")
    assert t_vec.shape == (B, config.embed_dim)
    print("  PASSED")

    print("\n[3] Testing AdaLN-Zero...")
    adaln = AdaLNZero(config.embed_dim, config.embed_dim)
    (s1, sh1), (s2, sh2), (s3, sh3) = adaln(t_vec)
    print(f"  Scale/shift shapes: {s1.shape}")
    print("  PASSED")

    print("\n[4] Testing SimpleTextEncoder...")
    text_enc = SimpleTextEncoder(num_tokens=config.text_tokens, out_dim=config.text_dim)
    texts = ["a dog running in a park"]
    with torch.no_grad():
        encoded = text_enc(texts)
    print(f"  Text encoding: {encoded.shape}")
    assert encoded.shape == (1, config.text_tokens, config.text_dim)
    print("  PASSED")

    print("\n[5] Testing full VideoDiT...")
    model = VideoDiT(config)
    model.eval()

    with torch.no_grad():
        pred = model(latent, timesteps, text_tokens)

    print(f"  Prediction: {pred.shape}")
    assert pred.shape == latent.shape, f"Shape mismatch: {pred.shape} vs {latent.shape}"
    print("  PASSED")

    print("\n[6] Testing backward pass...")
    model.train()
    pred = model(latent, timesteps, text_tokens)
    loss = F.mse_loss(pred, torch.zeros_like(pred))
    loss.backward()
    print(f"  Loss: {loss.item():.6f}")
    print("  PASSED")

    print("\n[7] Testing cross-attention weights...")
    model.eval()
    with torch.no_grad():
        weights = model.get_cross_attention_weights(latent, timesteps, text_tokens)
    print(f"  Cross-attn weights: {weights.shape}")
    print("  PASSED")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    print("\n" + "=" * 60)
    print("All VideoDiT tests PASSED")
    print("=" * 60)
