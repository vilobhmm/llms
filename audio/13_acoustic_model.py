"""
TTS Chapter 4: Acoustic Model — VALL-E / Qwen3-TTS Style
==========================================================
Qwen3-TTS follows the VALL-E paradigm:

  "Given text tokens + (optional) acoustic prompt,
   predict the RVQ codec tokens that represent the target speech."

Two-stage generation:

  ┌─────────────────────────────────────────────────────┐
  │  Stage 1 — AR (AutoRegressive)                      │
  │  Predict codebook-1 tokens left-to-right            │
  │  Like a language model over audio tokens            │
  │                                                     │
  │  [text tokens] [c1_1, c1_2, …, c1_T] → next c1     │
  │         ↑ conditional on text encoding              │
  └─────────────────────────────────────────────────────┘
               ↓  (c1 tokens provided)
  ┌─────────────────────────────────────────────────────┐
  │  Stage 2 — NAR (Non-AutoRegressive)                 │
  │  Predict codebooks 2…N in parallel                  │
  │  Each stage conditions on all previous codebooks    │
  │                                                     │
  │  [text tokens] [c1_1…c1_T] → [c2_1…c2_T] (parallel)│
  │  [c1,c2 tokens] → [c3_1…c3_T]                      │
  │  …                                                  │
  └─────────────────────────────────────────────────────┘

This design gives:
  • High prosody quality (AR captures long-range structure)
  • Fast inference (NAR runs in O(1) parallel passes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ──────────────────────────────────────────────
# Shared constants (inline for standalone use)
# ──────────────────────────────────────────────
_PHONEMES = [
    "AA","AE","AH","AO","AW","AX","AY","EH","ER","EY",
    "IH","IY","OW","OY","UH","UW",
    "B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","P",
    "R","S","SH","T","TH","V","W","Y","Z","ZH","SIL","SP",
]
PHONEME_VOCAB_SIZE = len(_PHONEMES) + 4   # +4 special tokens


# ──────────────────────────────────────────────
# 1.  Building blocks (lightweight versions of
#     what's in 12_text_encoder.py)
# ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (used in LLaMA, Qwen, etc.)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.g   = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — relative PE used in Qwen/LLaMA.
    Encodes relative position directly into attention weights.
    """
    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t   = torch.arange(max_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)               # (max_len, dim)
        self.register_buffer("cos_cache", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cache", emb.sin().unsqueeze(0).unsqueeze(0))

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T  = q.shape[2]
        cos = self.cos_cache[:, :, offset:offset + T]
        sin = self.sin_cache[:, :, offset:offset + T]
        q   = q * cos + self._rotate_half(q) * sin
        k   = k * cos + self._rotate_half(k) * sin
        return q, k


class CausalSelfAttention(nn.Module):
    """Causal multi-head attention with RoPE (for AR decoder)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.rope    = RotaryEmbedding(self.d_head)

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:     torch.Tensor,
        offset: int = 0,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.rope(q, k, offset=offset)

        attn  = (q @ k.transpose(-2, -1)) * self.scale
        # Causal mask
        if mask is None:
            causal = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
            attn = attn + causal
        else:
            attn = attn.masked_fill(mask, float("-inf"))

        attn = self.drop(F.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class FullAttention(nn.Module):
    """Bidirectional (non-causal) attention for NAR layers."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn = self.drop(F.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class SwiGLU(nn.Module):
    """SwiGLU FFN (used in Qwen/LLaMA instead of GELU FFN)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff,    d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ──────────────────────────────────────────────
# 2.  AR Transformer Block
# ──────────────────────────────────────────────

class ARBlock(nn.Module):
    """Causal transformer block (for AR stage)."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), offset=offset)
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# 3.  NAR Transformer Block
# ──────────────────────────────────────────────

class NARBlock(nn.Module):
    """Bidirectional transformer block (for NAR stage)."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = FullAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# 4.  AR Model — predicts first codebook tokens
# ──────────────────────────────────────────────

class ARModel(nn.Module):
    """
    Autoregressive model for codebook-1 prediction.

    Input sequence format (concatenated):
        [text tokens …]  [<sep>]  [audio codebook-1 tokens …]

    At each step the model predicts the next audio token.
    The text prefix provides the conditioning context.
    """

    def __init__(
        self,
        text_vocab:     int   = PHONEME_VOCAB_SIZE,
        audio_vocab:    int   = 1024,      # codebook size
        d_model:        int   = 512,
        n_heads:        int   = 8,
        n_layers:       int   = 6,
        d_ff:           int   = 2048,
        max_text_len:   int   = 256,
        max_audio_len:  int   = 2048,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.d_model     = d_model
        self.audio_vocab = audio_vocab

        # Separate embeddings for text vs audio tokens
        self.text_embed  = nn.Embedding(text_vocab,  d_model)
        self.audio_embed = nn.Embedding(audio_vocab + 1, d_model)  # +1 for BOS_audio
        self.sep_embed   = nn.Parameter(torch.randn(1, 1, d_model))
        self.scale       = math.sqrt(d_model)

        self.blocks = nn.ModuleList([
            ARBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm  = RMSNorm(d_model)
        self.head  = nn.Linear(d_model, audio_vocab, bias=False)

        # Weight tying
        self.audio_embed.weight = nn.Parameter(
            self.audio_embed.weight.clone()
        )

    def forward(
        self,
        text_ids:  torch.Tensor,            # (B, T_text)
        audio_ids: torch.Tensor,            # (B, T_audio)   — teacher-forced
    ) -> torch.Tensor:
        """Returns logits (B, T_audio, audio_vocab)."""
        B = text_ids.shape[0]

        t_emb = self.text_embed(text_ids) * self.scale      # (B, T_text, D)
        a_emb = self.audio_embed(audio_ids) * self.scale    # (B, T_audio, D)
        sep   = self.sep_embed.expand(B, -1, -1)            # (B, 1, D)

        # Concatenate: text | sep | audio (shifted right by 1)
        x     = torch.cat([t_emb, sep, a_emb], dim=1)       # (B, T_text+1+T_audio, D)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Only predict at audio positions
        T_text  = text_ids.shape[1]
        audio_h = x[:, T_text + 1:]                         # (B, T_audio, D)
        return self.head(audio_h)                            # (B, T_audio, V_audio)

    @torch.no_grad()
    def generate(
        self,
        text_ids:    torch.Tensor,          # (B, T_text)
        max_len:     int   = 500,
        temperature: float = 1.0,
        top_k:       int   = 50,
        stop_token:  int   = 0,
    ) -> torch.Tensor:
        """
        Autoregressively sample codebook-1 tokens.
        Returns (B, T_audio) int tensor.
        """
        B      = text_ids.shape[0]
        device = text_ids.device

        # Start with BOS_audio
        audio_ids = torch.full((B, 1), self.audio_vocab, device=device).long()
        t_emb     = self.text_embed(text_ids) * self.scale
        sep       = self.sep_embed.expand(B, -1, -1)

        for step in range(max_len):
            a_emb  = self.audio_embed(audio_ids) * self.scale
            x      = torch.cat([t_emb, sep, a_emb], dim=1)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            logits = self.head(x[:, -1])               # (B, V_audio)
            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                v, _ = logits.topk(top_k, dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))

            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)        # (B, 1)
            audio_ids = torch.cat([audio_ids, next_t], dim=1)

            if (next_t == stop_token).all():
                break

        return audio_ids[:, 1:]   # strip BOS_audio

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# 5.  NAR Model — fills in codebooks 2..N
# ──────────────────────────────────────────────

class NARModel(nn.Module):
    """
    Non-autoregressive model for codebooks 2..N.

    Given:
      - text hidden states
      - all already-predicted codebook tokens (stacked)
    Predicts the next codebook tokens in parallel (all T frames at once).
    """

    def __init__(
        self,
        text_vocab:     int   = PHONEME_VOCAB_SIZE,
        audio_vocab:    int   = 1024,
        num_quantizers: int   = 8,
        d_model:        int   = 512,
        n_heads:        int   = 8,
        n_layers:       int   = 4,
        d_ff:           int   = 2048,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.audio_vocab    = audio_vocab

        self.text_embed  = nn.Embedding(text_vocab, d_model)
        # One embedding table per codebook level
        self.audio_embeds = nn.ModuleList([
            nn.Embedding(audio_vocab, d_model)
            for _ in range(num_quantizers)
        ])
        self.scale = math.sqrt(d_model)

        # Stage embedding (tells model which codebook it's predicting)
        self.stage_embed = nn.Embedding(num_quantizers, d_model)

        self.blocks = nn.ModuleList([
            NARBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm  = RMSNorm(d_model)
        self.head  = nn.Linear(d_model, audio_vocab, bias=False)

    def forward(
        self,
        text_ids:         torch.Tensor,    # (B, T_text)
        audio_codes:      torch.Tensor,    # (B, T_audio, stage)  — codes[0..stage-1]
        target_stage:     int,             # which codebook we're predicting (1-indexed)
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns logits (B, T_audio, audio_vocab).
        target_stage: int in [1, N-1]  (predicting stage+1 from stages 0..stage)
        """
        B, T_text = text_ids.shape
        T_audio   = audio_codes.shape[1]

        # Text embeddings
        t_emb = self.text_embed(text_ids) * self.scale  # (B, T_text, D)

        # Sum embeddings of already-generated codebooks
        a_emb = torch.zeros(B, T_audio, t_emb.shape[-1], device=text_ids.device)
        for i in range(min(target_stage, audio_codes.shape[2])):
            a_emb = a_emb + self.audio_embeds[i](audio_codes[:, :, i]) * self.scale

        # Add stage embedding (broadcast over T_audio)
        stage_t = torch.full((B, T_audio), target_stage,
                             dtype=torch.long, device=text_ids.device)
        a_emb   = a_emb + self.stage_embed(stage_t)

        # Concatenate text + audio, run bidirectional transformer
        x = torch.cat([t_emb, a_emb], dim=1)    # (B, T_text+T_audio, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Return logits for the audio portion only
        audio_h = x[:, T_text:]                 # (B, T_audio, D)
        return self.head(audio_h)

    @torch.no_grad()
    def infer_stage(
        self,
        text_ids:    torch.Tensor,
        audio_codes: torch.Tensor,             # (B, T_audio, ≥target_stage)
        target_stage: int,
        temperature:  float = 1.0,
    ) -> torch.Tensor:
        """Predict one codebook stage → (B, T_audio) token ids."""
        logits = self.forward(text_ids, audio_codes, target_stage)
        logits = logits / max(temperature, 1e-8)
        probs  = F.softmax(logits, dim=-1)      # (B, T_audio, V)
        # Sample independently at each frame position
        B, T, V = probs.shape
        tokens  = torch.multinomial(probs.view(B * T, V), 1).view(B, T)
        return tokens

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# 6.  Combined VALLE Model
# ──────────────────────────────────────────────

class VALLEModel(nn.Module):
    """
    Full VALL-E / Qwen3-TTS acoustic model.

    AR stage  : text → codebook-1 tokens  (autoregressive)
    NAR stages: text + codebook-1..k → codebook-k+1 (parallel)
    """

    def __init__(
        self,
        text_vocab:     int   = PHONEME_VOCAB_SIZE,
        audio_vocab:    int   = 1024,
        num_quantizers: int   = 8,
        # AR config
        ar_d_model:     int   = 512,
        ar_n_heads:     int   = 8,
        ar_n_layers:    int   = 6,
        ar_d_ff:        int   = 2048,
        # NAR config
        nar_d_model:    int   = 512,
        nar_n_heads:    int   = 8,
        nar_n_layers:   int   = 4,
        nar_d_ff:       int   = 2048,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers

        self.ar_model = ARModel(
            text_vocab=text_vocab,
            audio_vocab=audio_vocab,
            d_model=ar_d_model, n_heads=ar_n_heads,
            n_layers=ar_n_layers, d_ff=ar_d_ff, dropout=dropout,
        )
        self.nar_model = NARModel(
            text_vocab=text_vocab,
            audio_vocab=audio_vocab,
            num_quantizers=num_quantizers,
            d_model=nar_d_model, n_heads=nar_n_heads,
            n_layers=nar_n_layers, d_ff=nar_d_ff, dropout=dropout,
        )

    def forward_ar(
        self,
        text_ids:  torch.Tensor,
        audio_ids: torch.Tensor,
    ) -> torch.Tensor:
        """AR forward — returns logits over codebook-1."""
        return self.ar_model(text_ids, audio_ids)

    def forward_nar(
        self,
        text_ids:     torch.Tensor,
        audio_codes:  torch.Tensor,
        target_stage: int,
    ) -> torch.Tensor:
        """NAR forward — returns logits for target_stage codebook."""
        return self.nar_model(text_ids, audio_codes, target_stage)

    @torch.no_grad()
    def generate(
        self,
        text_ids:    torch.Tensor,
        max_len:     int   = 500,
        temperature: float = 1.0,
        top_k:       int   = 50,
    ) -> torch.Tensor:
        """
        Full inference: text → all RVQ codes.
        Returns (B, T_audio, N_quantizers).
        """
        # Stage 1: AR generates codebook-1
        c1 = self.ar_model.generate(
            text_ids, max_len=max_len,
            temperature=temperature, top_k=top_k
        )                                      # (B, T_audio)

        T_audio = c1.shape[1]
        codes   = c1.unsqueeze(-1)             # (B, T_audio, 1)

        # Stages 2..N: NAR
        for stage in range(1, self.num_quantizers):
            next_c = self.nar_model.infer_stage(
                text_ids, codes, target_stage=stage,
                temperature=max(0.1, temperature * 0.5),
            )                                  # (B, T_audio)
            codes = torch.cat(
                [codes, next_c.unsqueeze(-1)], dim=-1
            )                                  # (B, T_audio, stage+1)

        return codes                           # (B, T_audio, N_quantizers)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# 7.  Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ACOUSTIC MODEL (VALL-E) — Tests")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, T_text, T_audio = 2, 20, 30
    N_Q, V_A = 4, 256   # small for testing

    # ── AR Model ─────────────────────────────────────────────
    print("\n[1] AR Model — forward pass (teacher-forced)")
    ar = ARModel(
        text_vocab=PHONEME_VOCAB_SIZE, audio_vocab=V_A,
        d_model=128, n_heads=4, n_layers=2, d_ff=512
    ).to(device)
    text  = torch.randint(4, PHONEME_VOCAB_SIZE, (B, T_text), device=device)
    audio = torch.randint(0, V_A, (B, T_audio), device=device)
    logits = ar(text, audio)
    print(f"  text: {text.shape}  audio: {audio.shape}  → logits: {logits.shape}")
    print(f"  AR params: {ar.num_parameters():,}")

    # ── AR Inference ─────────────────────────────────────────
    print("\n[2] AR Model — generation")
    c1 = ar.generate(text, max_len=15, temperature=1.0, top_k=20)
    print(f"  generated codebook-1: {c1.shape}  values ∈ [{c1.min()},{c1.max()}]")

    # ── NAR Model ────────────────────────────────────────────
    print("\n[3] NAR Model — forward pass")
    nar = NARModel(
        text_vocab=PHONEME_VOCAB_SIZE, audio_vocab=V_A,
        num_quantizers=N_Q,
        d_model=128, n_heads=4, n_layers=2, d_ff=512
    ).to(device)
    T_a  = c1.shape[1]
    codes_so_far = c1.unsqueeze(-1)          # (B, T_a, 1)
    nar_logits   = nar(text, codes_so_far, target_stage=1)
    print(f"  codes_so_far: {codes_so_far.shape}  →  logits: {nar_logits.shape}")
    print(f"  NAR params: {nar.num_parameters():,}")

    # ── Full VALL-E ───────────────────────────────────────────
    print("\n[4] VALL-E full generate")
    valle = VALLEModel(
        text_vocab=PHONEME_VOCAB_SIZE, audio_vocab=V_A,
        num_quantizers=N_Q,
        ar_d_model=128, ar_n_heads=4, ar_n_layers=2, ar_d_ff=512,
        nar_d_model=128, nar_n_heads=4, nar_n_layers=2, nar_d_ff=512,
    ).to(device)
    all_codes = valle.generate(text, max_len=12, temperature=0.8, top_k=20)
    print(f"  all_codes: {all_codes.shape}  (B, T_audio, N_quantizers)")
    print(f"  Total params: {valle.num_parameters():,}")

    # ── Loss computation ──────────────────────────────────────
    print("\n[5] Training loss (AR + NAR combined)")
    # AR loss
    ar_logits = valle.forward_ar(text, audio[:, :-1])   # predict next token
    ar_loss   = F.cross_entropy(
        ar_logits.reshape(-1, V_A),
        audio[:, 1:].reshape(-1)
    )
    # NAR loss (one stage)
    fake_codes = torch.randint(0, V_A, (B, T_audio, 1), device=device)
    nar_logits = valle.forward_nar(text, fake_codes, target_stage=1)
    nar_target = torch.randint(0, V_A, (B, T_audio), device=device)
    nar_loss   = F.cross_entropy(nar_logits.reshape(-1, V_A), nar_target.reshape(-1))
    total_loss = ar_loss + nar_loss
    print(f"  AR loss: {ar_loss.item():.4f}   NAR loss: {nar_loss.item():.4f}")
    print(f"  Total  : {total_loss.item():.4f}")
