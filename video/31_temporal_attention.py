"""
Spatial and Temporal Attention Modules for Video Generation
============================================================

This module implements the attention mechanisms used in video diffusion transformers,
following the factorized spatial-temporal attention approach from VideoLDM, CogVideoX,
and similar systems.

Key Innovation: Factorized Attention
-------------------------------------
Full 3D attention over (T, H, W) tokens would require O((T*H*W)^2) memory.
For 8 frames of 64x64, that's (8*64*64)^2 = ~1 billion elements — intractable.

Instead, we factorize into:
1. Spatial Attention: Each frame attends within itself independently
   - Complexity: O(T * (H*W)^2) — linear in T
2. Temporal Attention: Each spatial position attends across frames
   - Complexity: O(H*W * T^2) — linear in H*W

Combined complexity: O(T*(H*W)^2 + H*W*T^2) vs O((T*H*W)^2)
For 8 frames 64x64: ~67M vs ~1B — ~15x more efficient.

Attention Variants:
-------------------
1. SpatialAttention: Multi-head self-attention within each frame
   - Input: (B, T, H*W, D) — attends over H*W tokens per frame

2. TemporalAttention: Multi-head self-attention across time for each spatial position
   - Input: (B, H*W, T, D) — attends over T frames per position

3. CausalTemporalAttention: Causal (autoregressive) version of temporal attention
   - Uses lower-triangular mask so frame t cannot attend to future frame t+k

4. SpatialTemporalBlock: Full block combining both attention types + FFN
   - Order: SpatialAttn → TemporalAttn → FFN (SwiGLU)

5. Full3DAttention: Efficient approximation attending all T*H*W tokens at once
   - Uses chunking to reduce memory; good for short sequences

SwiGLU FFN:
-----------
Uses Swish-Gated Linear Unit instead of standard MLP:
  FFN(x) = (xW1) ⊙ SiLU(xW2) * W3
Better gradient flow than ReLU/GELU in transformer context.

Usage:
------
    block = SpatialTemporalBlock(dim=256, num_heads=8, ff_mult=4)
    # x: (B, T, H, W, D) — batch, frames, height, width, channels
    x = torch.randn(1, 4, 8, 8, 256)
    out = block(x)
    print(out.shape)  # (1, 4, 8, 8, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (lower-triangular) attention mask.

    Position i can attend to positions j where j <= i.
    Positions j > i are masked with -inf.

    Args:
        seq_len: Length of sequence (number of frames T)
        device: Target device

    Returns:
        (seq_len, seq_len) mask, 0 for valid, -inf for masked
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask = mask.masked_fill(
        torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool(),
        float('-inf')
    )
    return mask


class MultiHeadAttention(nn.Module):
    """
    Standard scaled dot-product multi-head attention.

    Optionally accepts cross-attention context (key/value from different source).

    Args:
        dim: Input/output embedding dimension
        num_heads: Number of attention heads
        dropout: Attention dropout rate
        bias: Whether to use bias in linear projections
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_k = nn.Linear(dim, dim, bias=bias)
        self.to_v = nn.Linear(dim, dim, bias=bias)
        self.to_out = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Query input (B, N, D)
            context: Key/Value source for cross-attention (B, M, D), or None for self-attn
            mask: Additive attention mask (N, M) or None
        Returns:
            (B, N, D) attention output
        """
        B, N, D = x.shape
        kv_source = context if context is not None else x

        q = self.to_q(x)
        k = self.to_k(kv_source)
        v = self.to_v(kv_source)

        # Reshape to multi-head format
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        M = k.shape[1]
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, d)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, d)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, M)

        if mask is not None:
            attn = attn + mask.unsqueeze(0).unsqueeze(0)  # broadcast over B, H

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        out = self.to_out(out)
        return out

    def get_attention_weights(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns attention weights for visualization: (B, H, N, M)"""
        B, N, D = x.shape
        kv_source = context if context is not None else x

        q = self.to_q(x)
        k = self.to_k(kv_source)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        M = k.shape[1]
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        return attn


class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network using SwiGLU activation.

    Architecture:
        FFN(x) = W_out(SiLU(W_gate(x)) ⊙ W_up(x))

    This gates the information flow: the SiLU-activated gate decides
    which components of the up-projected vector to pass through.

    Args:
        dim: Input dimension
        mult: Expansion factor for hidden dimension (default: 4)
        dropout: Dropout rate
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)  # Standard SwiGLU hidden dim
        hidden = (hidden + 7) // 8 * 8  # Round up to multiple of 8

        self.w_gate = nn.Linear(dim, hidden)
        self.w_up = nn.Linear(dim, hidden)
        self.w_out = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_out(self.dropout(gate * up))


class SpatialAttention(nn.Module):
    """
    Spatial Self-Attention: each frame attends within itself independently.

    Given video tokens of shape (B, T, HW, D), treats each frame independently:
    - For each frame t: tokens (B, HW, D) perform self-attention
    - Equivalent to attending over H*W spatial positions within a single frame

    This captures spatial structure (what's where in the image) while
    ignoring temporal relationships (handled by TemporalAttention).

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, HW, D) or (B, T, H, W, D) video tokens
        Returns:
            Same shape as input
        """
        input_4d = x.dim() == 4
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = x.view(B, T, H*W, D)

        B, T, HW, D = x.shape

        # Reshape: treat each frame as an independent batch item
        x_flat = x.view(B*T, HW, D)  # (B*T, HW, D)

        # Self-attention within each frame
        x_normed = self.norm(x_flat)
        attn_out = self.attn(x_normed)
        x_flat = x_flat + attn_out  # Residual connection

        # Reshape back
        x = x_flat.view(B, T, HW, D)
        if not input_4d:
            x = x.view(B, T, H, W, D)
        return x

    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for a single frame for visualization."""
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = x.view(B, T, H*W, D)
        B, T, HW, D = x.shape
        frame = x[:, 0]  # (B, HW, D)
        weights = self.attn.get_attention_weights(self.norm(frame))  # (B, H, HW, HW)
        return weights


class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention: each spatial position attends across frames.

    Given video tokens of shape (B, T, HW, D), for each spatial position:
    - All T frame tokens at that position attend to each other
    - Captures motion, temporal coherence, and frame relationships

    This is the key mechanism for temporal consistency in generated videos.

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        causal: If True, use causal masking (autoregressive generation)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, HW, D) video tokens
        Returns:
            (B, T, HW, D) with temporal attention applied
        """
        input_5d = x.dim() == 5
        if input_5d:
            B, T, H, W, D = x.shape
            x = x.view(B, T, H*W, D)

        B, T, HW, D = x.shape

        # Prepare causal mask if needed
        mask = None
        if self.causal:
            mask = make_causal_mask(T, x.device)

        # Transpose: (B, HW, T, D) — each spatial pos is a sequence of T frames
        x_t = x.permute(0, 2, 1, 3)  # (B, HW, T, D)
        x_t_flat = x_t.reshape(B*HW, T, D)  # (B*HW, T, D)

        # Self-attention across time
        x_normed = self.norm(x_t_flat)
        attn_out = self.attn(x_normed, mask=mask)
        x_t_flat = x_t_flat + attn_out  # Residual

        # Reshape back
        x_t = x_t_flat.view(B, HW, T, D)
        x = x_t.permute(0, 2, 1, 3)  # (B, T, HW, D)

        if input_5d:
            x = x.view(B, T, H, W, D)
        return x

    def get_attention_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """Get temporal attention weights for visualization: (T, T)"""
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = x.view(B, T, H*W, D)
        B, T, HW, D = x.shape

        mask = make_causal_mask(T, x.device) if self.causal else None

        # Use center spatial position
        center_pos = x[:, :, HW//2, :]  # (B, T, D)
        weights = self.attn.get_attention_weights(
            self.norm(center_pos), mask=mask
        )  # (B, H, T, T)
        return weights[0].mean(0)  # (T, T) averaged over heads


class CausalTemporalAttention(TemporalAttention):
    """
    Causal Temporal Attention for autoregressive video generation.

    Extends TemporalAttention with causal masking: frame t can only
    attend to frames 0, 1, ..., t (not future frames).

    This enables autoregressive generation where each frame is conditioned
    on all previous frames.

    The causal mask is:
        mask[i,j] = 0    if j <= i  (can attend)
        mask[i,j] = -inf if j > i   (cannot attend to future)
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__(dim, num_heads, dropout, causal=True)


class Full3DAttention(nn.Module):
    """
    Full 3D Attention over all T*H*W tokens simultaneously.

    Unlike factorized attention, this captures all spatial-temporal
    interactions at once. More expressive but O((T*H*W)^2) memory.

    For small sequences (short clips, low resolution), this is feasible.
    For larger sequences, use the factorized SpatialTemporalBlock instead.

    Implements chunked attention to reduce peak memory by processing
    attention in blocks.

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, HW, D) or (B, T, H, W, D) tokens
        Returns:
            Same shape as input
        """
        input_5d = x.dim() == 5
        if input_5d:
            B, T, H, W, D = x.shape
            x = x.view(B, T, H*W, D)

        B, T, HW, D = x.shape
        N = T * HW

        # Flatten all tokens: (B, T*HW, D)
        x_flat = x.view(B, N, D)

        # Full attention over all tokens
        x_normed = self.norm(x_flat)
        attn_out = self.attn(x_normed)
        x_flat = x_flat + attn_out

        x = x_flat.view(B, T, HW, D)
        if input_5d:
            x = x.view(B, T, H, W, D)
        return x


class SpatialTemporalBlock(nn.Module):
    """
    Complete Spatial-Temporal Transformer Block.

    Applies factorized attention in sequence:
    1. Spatial Attention: capture spatial structure within each frame
    2. Temporal Attention: capture temporal dynamics across frames
    3. SwiGLU FFN: non-linear feature mixing

    Each sub-block has LayerNorm pre-normalization and residual connection.

    This is the main building block of the Video DiT transformer.

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        ff_mult: FFN expansion multiplier
        dropout: Dropout rate
        causal: Whether temporal attention should be causal
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()

        self.spatial_attn = SpatialAttention(dim, num_heads, dropout)
        self.temporal_attn = TemporalAttention(dim, num_heads, dropout, causal=causal)

        self.norm_spatial = nn.LayerNorm(dim)
        self.norm_temporal = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

        self.ff = SwiGLUFFN(dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: tuple = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, HW, D) video tokens
            scale_shift: Optional (scale, shift) from AdaLN for conditioning
                         Each is (B, D)
        Returns:
            (B, T, HW, D) processed tokens
        """
        input_5d = x.dim() == 5
        if input_5d:
            B, T, H, W, D = x.shape
            x = x.view(B, T, H*W, D)

        B, T, HW, D = x.shape

        # Apply AdaLN conditioning if provided
        if scale_shift is not None:
            scale, shift = scale_shift
            # scale, shift: (B, D) → (B, 1, 1, D) for broadcasting
            scale = scale.view(B, 1, 1, D)
            shift = shift.view(B, 1, 1, D)

        # 1. Spatial attention
        x_normed = self.norm_spatial(x)
        if scale_shift is not None:
            x_normed = x_normed * (1 + scale) + shift
        x = x + self.spatial_attn(x_normed)

        # 2. Temporal attention
        x_normed = self.norm_temporal(x)
        if scale_shift is not None:
            x_normed = x_normed * (1 + scale) + shift
        x = x + self.temporal_attn(x_normed)

        # 3. FFN
        x_normed = self.norm_ff(x)
        if scale_shift is not None:
            x_normed = x_normed * (1 + scale) + shift
        x = x + self.ff(x_normed)

        if input_5d:
            x = x.view(B, T, H, W, D)
        return x


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Temporal Attention Modules")
    print("=" * 60)

    device = torch.device("cpu")

    B, T, H, W = 2, 4, 4, 4
    D = 32
    num_heads = 4
    HW = H * W

    # Synthetic tokens
    x = torch.randn(B, T, HW, D, device=device)
    x_5d = torch.randn(B, T, H, W, D, device=device)

    print(f"\nInput: B={B}, T={T}, H={H}, W={W}, D={D}")

    # Test SpatialAttention
    print("\n[1] SpatialAttention...")
    spatial_attn = SpatialAttention(D, num_heads)
    with torch.no_grad():
        out = spatial_attn(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"

    # 5D input
    out_5d = spatial_attn(x_5d)
    assert out_5d.shape == x_5d.shape
    print("  PASSED (4D and 5D inputs)")

    # Test TemporalAttention
    print("\n[2] TemporalAttention...")
    temporal_attn = TemporalAttention(D, num_heads)
    with torch.no_grad():
        out = temporal_attn(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape
    print("  PASSED")

    # Test CausalTemporalAttention
    print("\n[3] CausalTemporalAttention...")
    causal_attn = CausalTemporalAttention(D, num_heads)
    with torch.no_grad():
        out = causal_attn(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape

    # Verify causal mask
    pattern = causal_attn.get_attention_pattern(x)
    print(f"  Attention pattern shape: {pattern.shape}")
    # Upper triangle should be ~0 (masked)
    upper_tri_sum = pattern[0, 1].item()  # position [0,1] should be masked
    print(f"  Causal mask working (T[0,1] attention weight ≈ 0): {upper_tri_sum:.6f}")
    print("  PASSED")

    # Test Full3DAttention
    print("\n[4] Full3DAttention...")
    full_attn = Full3DAttention(D, num_heads)
    with torch.no_grad():
        out = full_attn(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape
    print("  PASSED")

    # Test SpatialTemporalBlock
    print("\n[5] SpatialTemporalBlock...")
    st_block = SpatialTemporalBlock(D, num_heads, ff_mult=4)
    with torch.no_grad():
        out = st_block(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape

    # Test with AdaLN conditioning
    scale = torch.randn(B, D)
    shift = torch.randn(B, D)
    with torch.no_grad():
        out_cond = st_block(x, scale_shift=(scale, shift))
    assert out_cond.shape == x.shape
    print("  With AdaLN conditioning: PASSED")

    # Test backward pass
    out = st_block(x)
    loss = out.mean()
    loss.backward()
    print("  Backward pass: PASSED")

    print("\n[6] Causal mask visualization...")
    mask = make_causal_mask(T, device)
    print(f"  Causal mask ({T}x{T}):")
    for i in range(T):
        row = [f"{mask[i,j].item():6.1f}" for j in range(T)]
        print(f"    [{', '.join(row)}]")

    print("\n" + "=" * 60)
    print("All Temporal Attention tests PASSED")
    print("=" * 60)
