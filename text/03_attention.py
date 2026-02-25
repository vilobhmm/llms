"""
Chapter 3: Attention Mechanisms
================================
Builds attention from the ground up in 4 steps:

  Step 1 — Simplified self-attention (no parameters)
  Step 2 — Scaled dot-product self-attention with Q,K,V projections
  Step 3 — Causal (masked) self-attention  ← used in GPT decoders
  Step 4 — Multi-head causal self-attention ← full production version

Each class is self-contained so you can study them independently.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────
# Step 1 — Simplified Self-Attention (no params)
# ──────────────────────────────────────────────

class SimpleSelfAttention(nn.Module):
    """
    Pedagogical version: attention weights are just
    softmax( x @ x.T / sqrt(d) ) applied to x itself.

    No learnable Q/K/V projections — helps understand the core idea.

    Attention(X) = softmax( X Xᵀ / √d ) X
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        d = x.size(-1)
        scores  = torch.bmm(x, x.transpose(1, 2)) / math.sqrt(d)   # (B, T, T)
        weights = F.softmax(scores, dim=-1)                          # (B, T, T)
        return torch.bmm(weights, x)                                  # (B, T, d)


# ──────────────────────────────────────────────
# Step 2 — Scaled Dot-Product Attention (with QKV)
# ──────────────────────────────────────────────

class SelfAttentionV2(nn.Module):
    """
    Adds learnable weight matrices:
        Q = X Wq,  K = X Wk,  V = X Wv
        Attention(Q,K,V) = softmax( Q Kᵀ / √d_k ) V

    This is the core formula from "Attention is All You Need".
    """

    def __init__(self, d_model: int, d_k: int, bias: bool = False):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_k, bias=bias)
        self.Wk = nn.Linear(d_model, d_k, bias=bias)
        self.Wv = nn.Linear(d_model, d_k, bias=bias)
        self.scale = math.sqrt(d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)                                         # (B, T, d_k)
        K = self.Wk(x)                                         # (B, T, d_k)
        V = self.Wv(x)                                         # (B, T, d_k)
        scores  = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, T, T)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, V)                            # (B, T, d_k)


# ──────────────────────────────────────────────
# Step 3 — Causal Self-Attention
# ──────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Adds a **causal mask** so token at position t can only attend to
    positions 0 … t (not the future).  Essential for auto-regressive LMs.

    The mask is a lower-triangular matrix filled with -inf above the diagonal:

        ┌  0   -∞  -∞  -∞ ┐
        │  0    0  -∞  -∞ │
        │  0    0   0  -∞ │
        └  0    0   0   0 ┘

    After softmax the -inf entries become 0, effectively blocking future tokens.
    """

    def __init__(
        self,
        d_model: int,
        d_k: int,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.Wq    = nn.Linear(d_model, d_k, bias=bias)
        self.Wk    = nn.Linear(d_model, d_k, bias=bias)
        self.Wv    = nn.Linear(d_model, d_k, bias=bias)
        self.drop  = nn.Dropout(dropout)
        self.scale = math.sqrt(d_k)

        # Pre-allocate causal mask (upper triangle = -inf)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        B, T, _ = x.shape
        Q = self.Wq(x)                                              # (B, T, d_k)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale      # (B, T, T)

        # Apply causal mask: fill future positions with -inf
        causal_mask = self.mask[:T, :T].bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        return torch.bmm(weights, V)                                 # (B, T, d_k)


# ──────────────────────────────────────────────
# Step 4 — Multi-Head Causal Self-Attention
# ──────────────────────────────────────────────

class MultiHeadCausalAttention(nn.Module):
    """
    Runs h attention heads **in parallel**, each with its own Q,K,V projection
    of dimension head_dim = d_model // n_heads.

    The heads are concatenated and projected back to d_model via Wo.

    Architecture:
        d_model  = total model dimension (e.g. 768)
        n_heads  = number of attention heads (e.g. 12)
        head_dim = d_model // n_heads  (e.g. 64)

    Implementation note:
        We fuse all Q, K, V projections into a single matrix
        (3 * d_model output) for efficiency, then split them.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        # Fused QKV projection + output projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj  = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
    ):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        Optionally also returns attention weights for visualisation.
        """
        B, T, C = x.shape
        H       = self.n_heads
        Dh      = self.head_dim

        # ── Project to Q, K, V ──────────────────────────────────────────
        qkv = self.qkv_proj(x)                       # (B, T, 3*C)
        q, k, v = qkv.split(self.d_model, dim=-1)    # each: (B, T, C)

        # Reshape to (B, H, T, Dh) for per-head computation
        def split_heads(t):
            return t.view(B, T, H, Dh).transpose(1, 2)   # (B, H, T, Dh)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # ── Scaled dot-product scores ────────────────────────────────────
        scale  = math.sqrt(Dh)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale    # (B, H, T, T)

        # ── Causal mask ─────────────────────────────────────────────────
        causal = self.mask[:T, :T].bool()
        scores = scores.masked_fill(causal, float("-inf"))

        # ── Softmax + dropout ────────────────────────────────────────────
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)             # (B, H, T, T)

        # ── Weighted sum of values ───────────────────────────────────────
        out = torch.matmul(attn_weights, v)                     # (B, H, T, Dh)

        # ── Merge heads ─────────────────────────────────────────────────
        out = out.transpose(1, 2).contiguous().view(B, T, C)    # (B, T, C)
        out = self.resid_drop(self.out_proj(out))

        if return_attn_weights:
            return out, attn_weights
        return out


# ──────────────────────────────────────────────
# Optional: Flash-Attention style  (uses F.scaled_dot_product_attention)
# ──────────────────────────────────────────────

class FlashMultiHeadAttention(nn.Module):
    """
    Leverages PyTorch 2.0's fused scaled_dot_product_attention which
    implements the Flash Attention kernel (memory-efficient, no explicit O(T²) matrix).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj  = nn.Linear(d_model, d_model, bias=bias)
        self.dropout   = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, Dh   = self.n_heads, self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        def split_h(t):
            return t.view(B, T, H, Dh).transpose(1, 2)

        q, k, v = split_h(q), split_h(k), split_h(v)

        # is_causal=True enables the causal mask inside the fused kernel
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )                                                      # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ──────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    B, T, D = 2, 16, 128
    x = torch.randn(B, T, D)

    print("=" * 60)
    print("ATTENTION MODULE TESTS")
    print("=" * 60)

    # Step 1
    out = SimpleSelfAttention()(x)
    print(f"[SimpleSelfAttention]       {x.shape} → {out.shape}")

    # Step 2
    out = SelfAttentionV2(D, d_k=64)(x)
    print(f"[SelfAttentionV2]           {x.shape} → {out.shape}")

    # Step 3
    out = CausalSelfAttention(D, d_k=64)(x)
    print(f"[CausalSelfAttention]       {x.shape} → {out.shape}")

    # Step 4
    mha = MultiHeadCausalAttention(D, n_heads=8, max_seq_len=T)
    out, attn_w = mha(x, return_attn_weights=True)
    print(f"[MultiHeadCausalAttention]  {x.shape} → {out.shape}")
    print(f"  Attention weights shape:  {attn_w.shape}  (B, H, T, T)")

    # Causal mask check: upper triangle should be zero
    upper_tri_sum = attn_w[0, 0].triu(1).sum().item()
    print(f"  Upper-triangle weight sum (should be ~0): {upper_tri_sum:.6f}")

    # Flash attention
    flash = FlashMultiHeadAttention(D, n_heads=8)
    out_flash = flash(x)
    print(f"[FlashMultiHeadAttention]   {x.shape} → {out_flash.shape}")

    print("\nAll attention tests passed!")
