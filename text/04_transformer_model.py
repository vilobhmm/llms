"""
Chapter 4: GPT-Style Transformer Model
========================================
Assembles all prior components into a complete auto-regressive language model.

Architecture (GPT-2 / GPT-3 decoder-only):

    Input IDs
        ↓
    [TokenEmb + PosEmb]
        ↓
    [TransformerBlock × N]   ← each block:
    │   ├─ LayerNorm
    │   ├─ MultiHeadCausalAttention
    │   ├─ Residual connection
    │   ├─ LayerNorm
    │   ├─ FeedForward  (MLP)
    │   └─ Residual connection
        ↓
    [LayerNorm]
        ↓
    [Linear head]  →  logits over vocab
        ↓
    Cross-Entropy Loss  (next-token prediction)

Key design choices aligned with Raschka's book:
  - Pre-LayerNorm  (norm before attention/FFN, not after)
  - GELU activation in FFN
  - Dropout on embeddings, attention, residual paths
  - Weight tying: input embedding = output projection  (saves params)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# ──────────────────────────────────────────────
# 1.  Model Configuration
# ──────────────────────────────────────────────

@dataclass
class GPTConfig:
    """Hyper-parameters for the GPT model.

    Preset sizes:
        GPT-2 small  : n_layers=12, n_heads=12, d_model=768,  ctx=1024
        GPT-2 medium : n_layers=24, n_heads=16, d_model=1024, ctx=1024
        GPT-2 large  : n_layers=36, n_heads=20, d_model=1280, ctx=1024
        GPT-2 XL     : n_layers=48, n_heads=25, d_model=1600, ctx=1024

    We default to a tiny "debug" size so the notebook runs on CPU.
    """
    vocab_size:   int   = 50257    # GPT-2 / tiktoken default
    context_len:  int   = 256
    d_model:      int   = 256      # embedding dimension
    n_heads:      int   = 8
    n_layers:     int   = 6
    ffn_mult:     int   = 4        # FFN hidden dim = ffn_mult * d_model
    dropout:      float = 0.1
    bias:         bool  = False    # biases in Linear layers (False = slightly better)
    weight_tying: bool  = True     # tie input/output embeddings


# ──────────────────────────────────────────────
# 2.  Feed-Forward Network (MLP)
# ──────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation:
        Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model) → Dropout

    GELU (Gaussian Error Linear Unit) is smoother than ReLU and
    outperforms it on language modelling tasks (Hendrycks & Gimpel, 2016).
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, hidden, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# 3.  Multi-Head Causal Self-Attention (inline)
# ──────────────────────────────────────────────

class MultiHeadCausalAttention(nn.Module):
    """Compact version wired to GPTConfig."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model  = cfg.d_model

        self.qkv_proj  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Causal mask registered as non-parameter buffer
        mask = torch.triu(
            torch.ones(cfg.context_len, cfg.context_len), diagonal=1
        )
        self.register_buffer("mask", mask)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        H, Dh   = self.n_heads, self.head_dim

        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        def split_h(t):
            return t.view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)

        q, k, v = split_h(q), split_h(k), split_h(v)

        scores  = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        scores  = scores.masked_fill(self.mask[:T, :T].bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)                   # (B,H,T,Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))

        return (out, weights) if return_weights else (out, None)


# ──────────────────────────────────────────────
# 4.  Transformer Block
# ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block (used in GPT-2 and later):

        x = x + Attention( LayerNorm(x) )
        x = x + FFN( LayerNorm(x) )

    Pre-norm (norm before the sub-layer) trains more stably than
    post-norm and enables deeper networks without warm-up tricks.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadCausalAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = FeedForward(cfg)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ):
        # Attention sub-layer with residual
        normed_x      = self.ln1(x)
        attn_out, w   = self.attn(normed_x, return_weights=return_attn)
        x             = x + attn_out

        # FFN sub-layer with residual
        x = x + self.ffn(self.ln2(x))

        return (x, w) if return_attn else x


# ──────────────────────────────────────────────
# 5.  Full GPT Model
# ──────────────────────────────────────────────

class GPT(nn.Module):
    """
    Complete GPT-style decoder-only Transformer.

    Forward pass returns:
        logits  : (batch, seq_len, vocab_size)   — raw next-token scores
        loss    : scalar cross-entropy (only when targets are provided)

    Weight tying:
        The output head (Linear → vocab) shares weights with the token
        embedding matrix.  This halves the largest parameter block and
        slightly improves perplexity.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embedding layers ───────────────────────────────────────────
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.context_len, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        # ── Transformer blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        # ── Final layer-norm + language-model head ─────────────────────
        self.ln_f  = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        if cfg.weight_tying:
            self.lm_head.weight = self.token_emb.weight

        # Initialise weights (GPT-2 style)
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2*n_layers) (GPT-2 paper)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ── Parameter count ───────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        targets:   Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ):
        """
        input_ids : (batch, seq_len)
        targets   : (batch, seq_len)  — optional, for loss computation
        """
        B, T = input_ids.shape
        assert T <= self.cfg.context_len, (
            f"Sequence length {T} > context_len {self.cfg.context_len}"
        )

        # ── Embeddings ──────────────────────────────────────────────────
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(positions))

        # ── Transformer blocks ──────────────────────────────────────────
        all_attn_weights = [] if return_attn_weights else None
        for block in self.blocks:
            if return_attn_weights:
                x, w = block(x, return_attn=True)
                all_attn_weights.append(w)
            else:
                x = block(x)

        # ── Head ─────────────────────────────────────────────────────────
        x      = self.ln_f(x)
        logits = self.lm_head(x)                                   # (B, T, V)

        # ── Loss (if targets provided) ───────────────────────────────────
        loss = None
        if targets is not None:
            # Flatten to (B*T, V) and (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        if return_attn_weights:
            return logits, loss, all_attn_weights
        return logits, loss

    # ── Greedy / temperature sampling ────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Auto-regressive text generation.

        temperature : controls randomness (< 1 = sharper, > 1 = softer)
        top_k       : keep only top-k logits before sampling
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context length
            ids_ctx = input_ids[:, -self.cfg.context_len:]
            logits, _ = self(ids_ctx)
            logits = logits[:, -1, :] / temperature              # (B, V)

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            probs     = F.softmax(logits, dim=-1)
            next_id   = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids


# ──────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cfg = GPTConfig(
        vocab_size   = 512,
        context_len  = 32,
        d_model      = 128,
        n_heads      = 4,
        n_layers     = 2,
        dropout      = 0.1,
        weight_tying = True,
    )
    model = GPT(cfg)
    total_params = model.num_parameters()
    print(f"GPT Model — {total_params:,} parameters")
    print(f"Config: {cfg}\n")

    # Forward pass
    B, T = 2, 16
    ids     = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(ids, targets)
    print(f"Input  : {ids.shape}")
    print(f"Logits : {logits.shape}")
    print(f"Loss   : {loss.item():.4f}  (expected ~ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f})")

    # Generation
    prompt  = torch.randint(0, cfg.vocab_size, (1, 4))
    gen     = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=40)
    print(f"\nGenerate: prompt {prompt.shape} → output {gen.shape}")

    # Attention weights
    logits, loss, attn_ws = model(ids, targets, return_attn_weights=True)
    print(f"\nAttention weights per layer: {[w.shape for w in attn_ws]}")
    print("Transformer model test passed!")
