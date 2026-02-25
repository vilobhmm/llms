"""
TTS Chapter 3: Text Encoder
============================
Encodes phoneme / character token sequences into rich contextual
representations that the acoustic model conditions on.

Architecture: Transformer encoder (same as BERT / the encoder half of T5)

  Input tokens  →  Embedding + Positional Encoding
                        ↓
               N × TransformerEncoderLayer
               (Multi-Head Self-Attention + FFN + LayerNorm)
                        ↓
               Projection  →  hidden states  (B, T_text, H)

Design choices used by Qwen3-TTS / FastSpeech2:
  • Pre-LN (LayerNorm before attention) for stable training
  • Relative position encoding for better long-sequence generalisation
  • FFN with GELU activation
  • Dropout for regularisation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ──────────────────────────────────────────────
# Reuse phoneme vocab from text processing
# ──────────────────────────────────────────────
import sys, os

# Inline the constants so this file runs standalone
_PHONEMES = [
    "AA","AE","AH","AO","AW","AX","AY","EH","ER","EY",
    "IH","IY","OW","OY","UH","UW",
    "B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","P",
    "R","S","SH","T","TH","V","W","Y","Z","ZH","SIL","SP",
]
_PHONEME2ID = {"<PAD>":0,"<BOS>":1,"<EOS>":2,"<UNK>":3,
               **{p:i+4 for i,p in enumerate(_PHONEMES)}}
PHONEME_VOCAB_SIZE = len(_PHONEME2ID)


# ──────────────────────────────────────────────
# 1.  Positional Encoding (sinusoidal + learned)
# ──────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic Vaswani et al. sinusoidal encoding — deterministic, no params.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learnable position embeddings — simpler, works well for short seqs."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.embed   = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T   = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return self.dropout(x + self.embed(pos))


# ──────────────────────────────────────────────
# 2.  Multi-Head Self-Attention  (Pre-LN variant)
# ──────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product attention with optional causal mask.

    Supports:
      • Bidirectional (encoder, causal=False)
      • Causal / autoregressive (decoder, causal=True)
      • Key-padding mask for variable-length batches
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:           torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal:      bool = False,
    ) -> torch.Tensor:
        """
        x    : (B, T, D)
        mask : (B, T)  — True for positions to ignore (padding)
        """
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)            # (B, H, T, Dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, T)

        # Causal mask (upper-triangle = -inf)
        if causal:
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
            attn = attn + causal_mask

        # Key-padding mask (pad positions = -inf)
        if key_padding_mask is not None:
            # mask: (B, T) → (B, 1, 1, T)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = self.drop(F.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


# ──────────────────────────────────────────────
# 3.  Feed-Forward Network
# ──────────────────────────────────────────────

class PositionwiseFFN(nn.Module):
    """Two-layer FFN with GELU.  d_ff typically = 4 × d_model."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# 4.  Transformer Encoder Layer  (Pre-LN)
# ──────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    """
    Pre-LN layout (more stable than post-LN):

        x → LayerNorm → MHSA → + → LayerNorm → FFN → +
        └───────────────────────┘  └──────────────────┘
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = PositionwiseFFN(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=src_key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# 5.  Full Text Encoder
# ──────────────────────────────────────────────

class TextEncoder(nn.Module):
    """
    Phoneme (or character) sequence → contextualised hidden states.

    Input:  token IDs   (B, T_text)
    Output: hidden      (B, T_text, d_model)
    """

    def __init__(
        self,
        vocab_size:  int   = PHONEME_VOCAB_SIZE,
        d_model:     int   = 256,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        d_ff:        int   = 1024,
        max_len:     int   = 512,
        dropout:     float = 0.1,
        pad_id:      int   = 0,
        pos_type:    str   = "sinusoidal",   # or "learned"
    ):
        super().__init__()
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model)     # standard Transformer embedding scale

        if pos_type == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_enc = LearnedPositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,           # (B, T)
        mask:   Optional[torch.Tensor] = None,  # (B, T) True = pad
    ) -> torch.Tensor:
        if mask is None:
            mask = (tokens == self.pad_id)

        x = self.embed(tokens) * self.scale  # (B, T, D)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)

        return self.norm(x)                  # (B, T, D)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# 6.  Cross-Attention (text → audio alignment)
# ──────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Used in the decoder: audio frames attend to text encoder states.

    Q from audio frames,  K/V from text encoder.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out     = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        query:    torch.Tensor,                   # (B, T_q, D)
        context:  torch.Tensor,                   # (B, T_k, D)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T_k)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, attn_weights)."""
        B, Tq, D = query.shape
        Tk       = context.shape[1]
        H, Dh    = self.n_heads, self.d_head

        q  = self.q_proj(query).reshape(B, Tq, H, Dh).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, Tk, 2, H, Dh).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)                       # (B, H, Tk, Dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, Tq, Tk)

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_w = F.softmax(attn, dim=-1)
        out    = (self.drop(attn_w) @ v).transpose(1, 2).reshape(B, Tq, D)
        return self.out(out), attn_w.mean(1)       # mean over heads


# ──────────────────────────────────────────────
# 7.  Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TEXT ENCODER MODULE — Tests")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Positional Encoding ──────────────────────────────────
    print("\n[1] Positional Encodings")
    for name, cls in [("Sinusoidal", SinusoidalPositionalEncoding),
                       ("Learned",   LearnedPositionalEncoding)]:
        pe = cls(d_model=256, max_len=512).to(device)
        x  = torch.randn(2, 30, 256, device=device)
        y  = pe(x)
        print(f"  {name}: {x.shape} → {y.shape}")

    # ── Multi-Head Attention ─────────────────────────────────
    print("\n[2] Multi-Head Self-Attention")
    mhsa  = MultiHeadSelfAttention(256, 4).to(device)
    x     = torch.randn(2, 20, 256, device=device)
    pad_m = torch.zeros(2, 20, dtype=torch.bool, device=device)
    pad_m[1, 15:] = True
    out   = mhsa(x, key_padding_mask=pad_m)
    print(f"  input: {x.shape}  →  output: {out.shape}")

    # ── TransformerEncoderLayer ───────────────────────────────
    print("\n[3] TransformerEncoderLayer")
    layer = TransformerEncoderLayer(256, 4, 1024).to(device)
    out   = layer(x, src_key_padding_mask=pad_m)
    print(f"  output: {out.shape}")

    # ── Full TextEncoder ──────────────────────────────────────
    print("\n[4] TextEncoder")
    enc  = TextEncoder(
        vocab_size=PHONEME_VOCAB_SIZE,
        d_model=256, n_heads=4, n_layers=4, d_ff=1024
    ).to(device)
    toks = torch.randint(4, PHONEME_VOCAB_SIZE, (2, 30), device=device)
    toks[1, 25:] = 0   # padding
    h    = enc(toks)
    print(f"  tokens: {toks.shape}  →  hidden: {h.shape}")
    print(f"  params: {enc.num_parameters():,}")

    # ── CrossAttention ───────────────────────────────────────
    print("\n[5] CrossAttention (audio → text)")
    xattn  = CrossAttention(256, 4).to(device)
    audio  = torch.randn(2, 100, 256, device=device)  # audio frames
    text_h = torch.randn(2, 30, 256, device=device)   # text states
    out, w = xattn(audio, text_h)
    print(f"  query: {audio.shape}  context: {text_h.shape}  →  {out.shape}")
    print(f"  attn weights: {w.shape}  (should be B,Tq,Tk = 2,100,30)")
