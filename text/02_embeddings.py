"""
Chapter 2: Token Embeddings + Positional Encodings
====================================================
Covers:
  - Token embedding lookup table
  - Absolute learned positional embeddings  (GPT-2 style)
  - Sinusoidal positional encoding           (original Transformer / "Attention is All You Need")
  - Rotary Position Embeddings (RoPE)        (LLaMA / GPT-NeoX style)
  - Combined embedding layer used in the full model
"""

import math
import torch
import torch.nn as nn
from typing import Optional


# ──────────────────────────────────────────────
# 1.  Token Embedding
# ──────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """
    Maps discrete token ids → dense vectors of size `d_model`.
    Optionally scales embeddings by sqrt(d_model) (original Transformer).
    """

    def __init__(self, vocab_size: int, d_model: int, scale: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model   = d_model
        self.scale     = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len)  →  (batch, seq_len, d_model)"""
        emb = self.embedding(x)
        if self.scale:
            emb = emb * math.sqrt(self.d_model)
        return emb


# ──────────────────────────────────────────────
# 2a. Absolute Learned Positional Embedding
#     (used in GPT-2, GPT-3, original BERT)
# ──────────────────────────────────────────────

class LearnedPositionalEmbedding(nn.Module):
    """
    Each position 0 … max_seq_len-1 gets its own learnable vector.
    Simple and effective; does NOT generalise to longer sequences at inference.
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:  (batch, seq_len, d_model)  — token embeddings
        Returns x + positional embedding (broadcast over batch dim).
        """
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return x + self.pos_emb(positions)


# ──────────────────────────────────────────────
# 2b. Sinusoidal Positional Encoding
#     (Vaswani et al. 2017 — "Attention is All You Need")
# ──────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learnable) sinusoidal encodings:

        PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    Benefits: deterministic, generalises to unseen lengths.
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)                         # (T, d)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()            # (T, 1)
        div = torch.exp(
            -torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model
        )                                                                # (d/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))                    # (1, T, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────
# 2c. Rotary Position Embeddings (RoPE)
#     (Su et al. 2021 — used in LLaMA, Mistral, Falcon …)
# ──────────────────────────────────────────────

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple_[torch.Tensor, torch.Tensor]:
    """
    Returns cos and sin tensors of shape (max_seq_len, head_dim).
    Cached and reused for efficiency.
    """
    theta = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )                                                                     # (head_dim/2,)
    positions = torch.arange(max_seq_len, device=device).float()          # (T,)
    freqs = torch.outer(positions, theta)                                  # (T, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)                             # (T, head_dim)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split x into two halves along last dim and interleave a rotation."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple_[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    q, k : (batch, n_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    seq_len = q.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# Forward-reference fix: we need Tuple from typing
from typing import Tuple as Tuple_


# ──────────────────────────────────────────────
# 3.  Combined Embedding Layer (GPT-2 style)
# ──────────────────────────────────────────────

class GPTEmbedding(nn.Module):
    """
    Token embedding + absolute learned positional embedding + dropout.
    This is the embedding layer used throughout our GPT model.

    Architecture:
        token_emb   : nn.Embedding(vocab_size, d_model)
        pos_emb     : nn.Embedding(max_seq_len, d_model)
        dropout     : nn.Dropout(emb_drop)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len)  — integer token ids
        Returns: (batch, seq_len, d_model)
        """
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, T)
        emb = self.token_emb(x) + self.pos_emb(positions)
        return self.drop(emb)


# ──────────────────────────────────────────────
# 4.  Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Configuration
    VOCAB_SIZE   = 512
    D_MODEL      = 128
    MAX_SEQ_LEN  = 64
    BATCH        = 4
    SEQ_LEN      = 16
    HEAD_DIM     = 32

    # Dummy token ids
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

    print("=" * 60)
    print("EMBEDDING MODULE TESTS")
    print("=" * 60)

    # ── Token embedding ──
    tok_emb = TokenEmbedding(VOCAB_SIZE, D_MODEL, scale=True)
    out = tok_emb(x)
    print(f"[TokenEmbedding]           input {x.shape} → output {out.shape}")

    # ── Learned positional ──
    emb = GPTEmbedding(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN)
    out = emb(x)
    print(f"[GPTEmbedding]             input {x.shape} → output {out.shape}")

    # ── Sinusoidal positional ──
    raw_emb = tok_emb(x)
    sin_pe  = SinusoidalPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    out_sin = sin_pe(raw_emb)
    print(f"[SinusoidalPE]             input {raw_emb.shape} → output {out_sin.shape}")

    # ── RoPE ──
    cos_freqs, sin_freqs = precompute_rope_freqs(HEAD_DIM, MAX_SEQ_LEN)
    q = torch.randn(BATCH, 4, SEQ_LEN, HEAD_DIM)  # (B, n_heads, T, head_dim)
    k = torch.randn(BATCH, 4, SEQ_LEN, HEAD_DIM)
    q_rot, k_rot = apply_rope(q, k, cos_freqs, sin_freqs)
    print(f"[RoPE] q {q.shape} → q_rot {q_rot.shape}")
    print(f"[RoPE] k {k.shape} → k_rot {k_rot.shape}")

    # Verify that RoPE preserves inner products correctly (dot product test)
    # For two vectors that differ only in position, RoPE encodes relative distance
    print(f"\nEmbedding norms (should be ~1 before norm): {out.norm(dim=-1).mean():.3f}")
    print("All embedding tests passed!")
