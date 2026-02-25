"""
Full-Duplex Dialogue Language Model — Moshi-Style Two-Stream Architecture
=========================================================================
This module implements the core language model of an Audio-to-Audio (A2A)
dialogue system, following the design principles of Moshi (Défossez et al.,
2024) and the GPT-4o speech-to-speech approach.

Design Philosophy:
------------------
Traditional turn-based dialogue requires one speaker to finish before the
other responds. Full-duplex dialogue — like natural human conversation — has
both speakers active simultaneously, with real-time interruptions, backchannels,
and overlapping speech.

The key architectural insight from Moshi is the TWO-STREAM model:
  - Stream 1 (User):      Encodes incoming user speech tokens at every frame
  - Stream 2 (Assistant): Generates outgoing assistant speech tokens at every frame

Both streams are processed jointly by a causal Transformer, enabling the model
to generate responses even while the user is still speaking.

Architecture Overview:
----------------------

  User tokens    → [Embed] ─────────────────┐
                                             ▼
                                     [Joint Causal Transformer]
                                             │
  Asst tokens    → [Embed] ─────────────────┘
                                             │
                                  ┌──────────┴──────────┐
                                  ▼                     ▼
                         [Main Stream Head]    [Inner Monologue Head]
                         (speech tokens)       (latent "thought" tokens)
                                  │
                         [Turn Detector]
                         (end-of-turn probability)

Inner Monologue:
  Inspired by chain-of-thought reasoning, the assistant generates "thought"
  tokens at each frame that are never vocalized. These capture reasoning about
  what to say next, enabling coherent multi-sentence responses without
  committing to audio too early.

Turn-Taking:
  The TurnDetector predicts the probability that the user has finished speaking
  and the assistant should begin (or continue) its response. This is trained
  with binary labels derived from the VAD signal.

Multi-Codebook Prediction:
  Speech tokens are organized in N_q codebook levels (from RVQ).
  The model predicts all N_q levels jointly using a "depth transformer"
  — a small per-frame transformer that generates codebook level i from
  levels 0..i-1 (like byte-pair encoding but for audio codes).

References:
  - Moshi: https://arxiv.org/abs/2410.00037
  - AudioLM: https://arxiv.org/abs/2209.03143
  - VALL-E: https://arxiv.org/abs/2301.02111
  - GPT-4o system card: https://openai.com/index/hello-gpt-4o/
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Multi-Codebook Token Embedding
# ─────────────────────────────────────────────────────────────────────────────

class MultiCodebookEmbedding(nn.Module):
    """
    Embeds multi-level speech tokens (B, T, N_q) → (B, T, hidden_dim).

    For N_q codebook levels, each with vocabulary size vocab_size:
      1. Embed each level separately: (B, T) → (B, T, D) × N_q
      2. Sum (or concatenate + project) all level embeddings
      3. Result: a single (B, T, hidden_dim) tensor per frame

    Summation is used rather than concatenation to keep the dimensionality
    constant regardless of N_q (same as how positional encodings are summed
    with token embeddings in Transformer).
    """

    def __init__(
        self,
        vocab_size:   int,
        hidden_dim:   int,
        num_quantizers: int,
        padding_idx:  Optional[int] = None,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
            for _ in range(num_quantizers)
        ])
        self.scale = math.sqrt(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T, N_q)  — integer codes
        Returns: (B, T, hidden_dim)
        """
        # Sum embeddings across codebook levels
        out = self.embeddings[0](tokens[..., 0])
        for i in range(1, self.num_quantizers):
            out = out + self.embeddings[i](tokens[..., i])
        return out * self.scale


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Rotary Positional Encoding (RoPE)
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (Su et al., 2021) used in LLaMA, GPT-NeoX.

    Unlike sinusoidal PE added to tokens, RoPE rotates Q and K vectors
    by position-dependent angles. This provides:
      - Relative position awareness (attention depends on relative distance)
      - Better extrapolation to longer sequences
      - No learned parameters

    Used by Moshi and most modern LLMs for causal language modeling.
    """

    def __init__(self, dim: int, max_len: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

        # Pre-compute cos/sin for positions 0..max_len
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t   = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs  = torch.outer(t, self.inv_freq)        # (seq_len, dim/2)
        emb    = torch.cat([freqs, freqs], dim=-1)    # (seq_len, dim)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (B, n_heads, T, head_dim)
        Returns rotated q, k
        """
        if seq_len > self.max_len:
            self._build_cache(seq_len)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Causal Transformer Block (with RoPE)
# ─────────────────────────────────────────────────────────────────────────────

class CausalTransformerBlock(nn.Module):
    """
    Single transformer block with:
      - Multi-head causal self-attention (with RoPE)
      - Position-wise feed-forward network (SwiGLU activation)
      - Pre-LayerNorm (more stable than post-norm)

    SwiGLU (Shazeer, 2020):
      FFN(x) = (xW₁ ⊙ σ(xW₃)) W₂
    This gated variant is used in PaLM, LLaMA, and Moshi for better capacity.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads:    int,
        ffn_dim:    int,
        dropout:    float = 0.1,
        causal:     bool  = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.head_dim   = hidden_dim // n_heads
        self.causal     = causal

        # Pre-norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # QKV projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim)

        # SwiGLU FFN
        self.ffn_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.ffn_up   = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x:    (B, T, hidden_dim)
        mask: (T, T) causal attention mask (or None)
        Returns: (B, T, hidden_dim), updated kv_cache
        """
        B, T, D = x.shape
        # Self-attention with pre-norm
        h = self.norm1(x)
        qkv = self.qkv_proj(h)                               # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                      # each (B, T, D)

        # Reshape to multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, T)

        # KV cache for inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_cache = (k, v)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn  = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        if mask is not None:
            # mask: (T, T) or (B, 1, T, T)
            if mask.dim() == 2:
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            else:
                attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                           # (B, H, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        x   = x + self.dropout(out)

        # SwiGLU FFN with pre-norm
        h   = self.norm2(x)
        gate = F.silu(self.ffn_gate(h))
        up   = self.ffn_up(h)
        x   = x + self.dropout(self.ffn_down(gate * up))

        return x, new_cache


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Causal Transformer (stack of blocks)
# ─────────────────────────────────────────────────────────────────────────────

class CausalTransformer(nn.Module):
    """
    Stack of CausalTransformerBlocks with a final LayerNorm.

    This is the backbone of the dialogue LM — processes the joint
    user+assistant token sequence causally (each position sees only past).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads:    int,
        n_layers:   int,
        ffn_dim:    int,
        dropout:    float = 0.1,
        causal:     bool  = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(hidden_dim, n_heads, ffn_dim, dropout, causal)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular boolean mask for causal attention."""
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(
        self,
        x:     torch.Tensor,
        causal: bool = True,
        kv_caches: Optional[List] = None,
    ) -> Tuple[torch.Tensor, List]:
        """
        x: (B, T, hidden_dim)
        Returns: hidden states (B, T, hidden_dim), updated kv_caches list
        """
        mask = self._causal_mask(x.size(1), x.device) if causal else None
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, mask=mask, kv_cache=cache)
            new_caches.append(new_cache)
        x = self.norm(x)
        return x, new_caches


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Depth Transformer (per-frame multi-codebook prediction)
# ─────────────────────────────────────────────────────────────────────────────

class DepthTransformer(nn.Module):
    """
    Small transformer that predicts codebook tokens sequentially within a frame.

    For frame t, the depth transformer predicts:
      code[t, 0], code[t, 1], ..., code[t, N_q-1]
    each conditioned on the main transformer's hidden state h[t] and all
    previous codebook levels within the same frame.

    This "depth-wise" prediction (introduced in Moshi) ensures that:
      1. Coarser codebook levels (0) are predicted first
      2. Finer levels use coarser ones as context
      3. The model learns the hierarchical structure of RVQ codes

    Architecture: 2-layer transformer over a sequence of length N_q.
    """

    def __init__(
        self,
        hidden_dim:     int,
        num_quantizers: int,
        vocab_size:     int,
        n_heads:        int = 4,
        n_layers:       int = 2,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.vocab_size     = vocab_size

        # Project main transformer output to depth transformer dim
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Embed already-predicted codes (as context)
        self.code_embed = nn.Embedding(vocab_size, hidden_dim)

        # Depth transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Prediction heads for each codebook level
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size, bias=False)
            for _ in range(num_quantizers)
        ])

    def forward(
        self,
        h: torch.Tensor,
        target_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h: (B, T, hidden_dim) — main transformer hidden states
        target_codes: (B, T, N_q) — ground truth codes (for teacher forcing)
        Returns: logits (B, T, N_q, vocab_size)
        """
        B, T, D = h.shape
        all_logits = []

        # Project main hidden state
        ctx = self.input_proj(h)  # (B, T, D)

        for q_idx in range(self.num_quantizers):
            # Context: main hidden state + previous level embeddings
            if q_idx == 0:
                x = ctx
            else:
                if target_codes is not None:
                    # Teacher forcing: use ground truth previous codes
                    prev = self.code_embed(target_codes[..., q_idx - 1])  # (B, T, D)
                else:
                    # Autoregressive: use predicted codes
                    prev_idx = all_logits[-1].argmax(dim=-1)              # (B, T)
                    prev = self.code_embed(prev_idx)
                x = ctx + prev

            # Run through depth transformer blocks
            x_seq = x  # (B, T, D)
            for block in self.blocks:
                x_seq = block(x_seq)
            x_seq = self.norm(x_seq)

            logits = self.heads[q_idx](x_seq)  # (B, T, vocab_size)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=2)  # (B, T, N_q, vocab_size)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Inner Monologue Stream
# ─────────────────────────────────────────────────────────────────────────────

class InnerMonologue(nn.Module):
    """
    The "inner monologue" is the assistant's latent thought process.

    Before generating speech tokens, the assistant generates hidden
    "thought tokens" that capture:
      - What it wants to say next
      - Its understanding of the conversation context
      - Planning for multi-turn responses

    These thought tokens are never vocalized (never decoded to audio).
    They serve as a bridge between understanding the user and generating
    appropriate speech — analogous to chain-of-thought in text LLMs.

    The inner monologue predicts tokens from a separate (private) vocabulary
    of size monologue_vocab_size.

    Architecture:
      hidden_state → Linear → [LayerNorm + GELU] → monologue_vocab_size logits
    """

    def __init__(
        self,
        hidden_dim:          int,
        monologue_vocab_size: int = 256,
        dropout:             float = 0.1,
    ):
        super().__init__()
        self.vocab_size = monologue_vocab_size
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, monologue_vocab_size),
        )
        self.embed = nn.Embedding(monologue_vocab_size, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, T, hidden_dim)
        Returns: logits (B, T, monologue_vocab_size)
        """
        return self.head(h)

    def get_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, T) → embeddings (B, T, hidden_dim)"""
        return self.embed(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Turn Detector
# ─────────────────────────────────────────────────────────────────────────────

class TurnDetector(nn.Module):
    """
    Predicts the probability of an end-of-turn event at each frame.

    End-of-turn (EoT) detection is crucial for full-duplex dialogue:
      - EoT = 1: user has stopped speaking → assistant should respond
      - EoT = 0: user is still speaking → assistant waits (or backchannel)

    The detector uses:
      - User stream hidden states (current speech)
      - Assistant stream hidden states (current response state)
      - Their difference (surprise signal)

    Output: per-frame scalar ∈ [0, 1] via sigmoid.
    Trained with BCE loss against VAD-derived turn labels.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        user_h: torch.Tensor,
        asst_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        user_h: (B, T, hidden_dim)
        asst_h: (B, T, hidden_dim)
        Returns: (B, T) turn probability logits
        """
        combined = torch.cat([user_h, asst_h], dim=-1)  # (B, T, 2D)
        return self.classifier(combined).squeeze(-1)     # (B, T)

    def compute_loss(
        self,
        user_h:      torch.Tensor,
        asst_h:      torch.Tensor,
        turn_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        turn_labels: (B, T) float — 1.0 at end-of-turn frames
        Returns: (loss, probs)
        """
        logits = self.forward(user_h, asst_h)
        loss   = self.loss_fn(logits, turn_labels)
        probs  = torch.sigmoid(logits)
        return loss, probs


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Two-Stream Dialogue Language Model
# ─────────────────────────────────────────────────────────────────────────────

class TwoStreamModel(nn.Module):
    """
    The core two-stream dialogue model.

    Processes user and assistant speech token streams simultaneously.
    Following Moshi's architecture:

    1. Embed user tokens: (B, T, N_q) → (B, T, D)
    2. Embed assistant tokens: (B, T, N_q) → (B, T, D)
    3. Combine: interleave or sum user + assistant embeddings
    4. Causal Transformer: joint sequence → hidden states
    5. Split hidden states → user_h, asst_h
    6. Inner Monologue head: asst_h → monologue logits
    7. Depth Transformer: asst_h → speech token logits
    8. Turn Detector: user_h + asst_h → turn probability

    The "joint" processing (step 4) allows the assistant to see the user's
    speech while generating its own response — true full-duplex operation.
    """

    def __init__(
        self,
        hidden_dim:           int = 256,
        n_heads:              int = 4,
        n_layers:             int = 4,
        ffn_dim:              int = 1024,
        vocab_size:           int = 512,
        num_quantizers:       int = 4,
        monologue_vocab_size: int = 256,
        dropout:              float = 0.1,
    ):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.num_quantizers  = num_quantizers

        # Embeddings for both streams
        self.user_embed = MultiCodebookEmbedding(
            vocab_size, hidden_dim, num_quantizers
        )
        self.asst_embed = MultiCodebookEmbedding(
            vocab_size, hidden_dim, num_quantizers
        )

        # Stream combination (learned weighting)
        self.stream_combine = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        # Main joint causal transformer
        self.transformer = CausalTransformer(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            causal=True,
        )

        # Inner monologue (hidden thought tokens)
        self.inner_monologue = InnerMonologue(
            hidden_dim=hidden_dim,
            monologue_vocab_size=monologue_vocab_size,
            dropout=dropout,
        )

        # Depth transformer for multi-codebook speech prediction
        self.depth_transformer = DepthTransformer(
            hidden_dim=hidden_dim,
            num_quantizers=num_quantizers,
            vocab_size=vocab_size,
            n_heads=max(1, n_heads // 2),
            n_layers=2,
            dropout=dropout,
        )

        # Turn detector
        self.turn_detector = TurnDetector(hidden_dim=hidden_dim, dropout=dropout)

        # Separate projection heads for user and assistant hidden states
        self.user_proj = nn.Linear(hidden_dim, hidden_dim)
        self.asst_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        user_tokens:  torch.Tensor,
        asst_tokens:  torch.Tensor,
        turn_labels:  Optional[torch.Tensor] = None,
        teacher_force: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        user_tokens:  (B, T, N_q) — incoming user speech codes
        asst_tokens:  (B, T, N_q) — outgoing assistant speech codes (teacher forcing)
        turn_labels:  (B, T) float — end-of-turn ground truth
        teacher_force: use ground truth codes in depth transformer

        Returns dict with:
          speech_logits    : (B, T, N_q, vocab_size)
          monologue_logits : (B, T, monologue_vocab_size)
          turn_logits      : (B, T)
          losses           : dict of individual loss terms
        """
        # 1. Embed both streams
        user_emb = self.user_embed(user_tokens)   # (B, T, D)
        asst_emb = self.asst_embed(asst_tokens)   # (B, T, D)

        # 2. Combine streams (concatenate → project)
        combined = self.stream_combine(
            torch.cat([user_emb, asst_emb], dim=-1)
        )                                          # (B, T, D)

        # 3. Joint causal transformer
        hidden, _ = self.transformer(combined, causal=True)

        # 4. Split into user-focused and assistant-focused projections
        user_h = self.user_proj(hidden)            # (B, T, D)
        asst_h = self.asst_proj(hidden)            # (B, T, D)

        # 5. Inner monologue prediction (on assistant hidden states)
        monologue_logits = self.inner_monologue(asst_h)  # (B, T, M)

        # 6. Speech token prediction (depth transformer)
        target_codes = asst_tokens if teacher_force else None
        speech_logits = self.depth_transformer(asst_h, target_codes)
        # shape: (B, T, N_q, vocab_size)

        # 7. Turn detection
        turn_logits = self.turn_detector(user_h, asst_h)  # (B, T)

        # 8. Compute losses if labels provided
        losses: Dict[str, torch.Tensor] = {}

        # Speech token prediction loss (cross-entropy, shifted by 1)
        B, T, N_q, V = speech_logits.shape
        if T > 1:
            # Predict t+1 from t
            pred_logits  = speech_logits[:, :-1]          # (B, T-1, N_q, V)
            target_codes_shifted = asst_tokens[:, 1:]     # (B, T-1, N_q)
            speech_loss = F.cross_entropy(
                pred_logits.reshape(-1, V),
                target_codes_shifted.reshape(-1),
                ignore_index=-1,
            )
            losses["speech"] = speech_loss

        # Turn detection loss
        if turn_labels is not None:
            turn_loss, _ = self.turn_detector.compute_loss(user_h, asst_h, turn_labels)
            losses["turn"] = turn_loss

        # Inner monologue loss (predict uniform distribution as surrogate)
        # In real training, these would be latent codes from a teacher model
        monologue_targets = torch.zeros(
            B, T, dtype=torch.long, device=user_tokens.device
        )
        monologue_loss = F.cross_entropy(
            monologue_logits.reshape(-1, self.inner_monologue.vocab_size),
            monologue_targets.reshape(-1),
        )
        losses["monologue"] = monologue_loss

        return {
            "speech_logits":    speech_logits,
            "monologue_logits": monologue_logits,
            "turn_logits":      turn_logits,
            "user_hidden":      user_h,
            "asst_hidden":      asst_h,
            "losses":           losses,
        }

    @torch.no_grad()
    def generate(
        self,
        user_tokens:  torch.Tensor,
        max_new_frames: int = 50,
        temperature:  float = 1.0,
        top_k:        int   = 50,
    ) -> torch.Tensor:
        """
        Autoregressively generate assistant speech tokens.

        user_tokens: (B, T_cond, N_q) — user speech context
        Returns: (B, max_new_frames, N_q) — generated assistant tokens
        """
        B, T_cond, N_q = user_tokens.shape
        device = user_tokens.device

        # Start with silence token (index 0) for assistant
        asst_tokens = torch.zeros(B, 1, N_q, dtype=torch.long, device=device)

        generated: List[torch.Tensor] = []

        for step in range(max_new_frames):
            # Extend user tokens to current length
            T_cur = asst_tokens.shape[1]
            if T_cur <= T_cond:
                user_ctx = user_tokens[:, :T_cur]
            else:
                # Pad user stream with last frame (silence) if needed
                pad = user_tokens[:, -1:].expand(B, T_cur - T_cond, N_q)
                user_ctx = torch.cat([user_tokens, pad], dim=1)

            # Forward pass
            output = self.forward(user_ctx, asst_tokens, teacher_force=False)
            speech_logits = output["speech_logits"]  # (B, T, N_q, V)

            # Take logits for the last frame
            last_logits = speech_logits[:, -1]       # (B, N_q, V)

            # Sample new codes for each codebook level
            new_codes: List[torch.Tensor] = []
            for q_idx in range(N_q):
                logits_q = last_logits[:, q_idx] / max(temperature, 1e-6)  # (B, V)
                # Top-k sampling
                if top_k > 0:
                    top_v, _ = torch.topk(logits_q, min(top_k, logits_q.size(-1)))
                    threshold = top_v[:, -1].unsqueeze(-1)
                    logits_q  = logits_q.masked_fill(logits_q < threshold, float("-inf"))
                probs = F.softmax(logits_q, dim=-1)
                code  = torch.multinomial(probs, 1)   # (B, 1)
                new_codes.append(code)

            new_frame = torch.cat(new_codes, dim=-1).unsqueeze(1)  # (B, 1, N_q)
            asst_tokens = torch.cat([asst_tokens, new_frame], dim=1)
            generated.append(new_frame)

        return torch.cat(generated, dim=1)  # (B, max_new_frames, N_q)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# __main__ smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DIALOGUE LM MODULE — Smoke Tests")
    print("=" * 60)

    B, T, N_q, V = 2, 16, 4, 128

    user_tokens = torch.randint(0, V, (B, T, N_q))
    asst_tokens = torch.randint(0, V, (B, T, N_q))
    turn_labels = torch.zeros(B, T).float()
    turn_labels[:, -1] = 1.0   # end of turn at last frame

    # ── MultiCodebookEmbedding ───────────────────────────────────
    print("\n[1] MultiCodebookEmbedding")
    emb = MultiCodebookEmbedding(V, 64, N_q)
    out = emb(user_tokens)
    print(f"  tokens : {user_tokens.shape}  →  embed : {out.shape}")

    # ── CausalTransformerBlock ───────────────────────────────────
    print("\n[2] CausalTransformerBlock")
    block = CausalTransformerBlock(hidden_dim=64, n_heads=4, ffn_dim=128)
    x = torch.randn(B, T, 64)
    out, cache = block(x)
    print(f"  x : {x.shape}  →  out : {out.shape}")

    # ── DepthTransformer ─────────────────────────────────────────
    print("\n[3] DepthTransformer")
    depth_tf = DepthTransformer(
        hidden_dim=64, num_quantizers=N_q, vocab_size=V,
        n_heads=2, n_layers=2,
    )
    h = torch.randn(B, T, 64)
    logits = depth_tf(h, asst_tokens)
    print(f"  h : {h.shape}  →  logits : {logits.shape}  (B, T, N_q, V)")

    # ── InnerMonologue ───────────────────────────────────────────
    print("\n[4] InnerMonologue")
    mono = InnerMonologue(hidden_dim=64, monologue_vocab_size=64)
    mono_logits = mono(h)
    print(f"  h : {h.shape}  →  monologue : {mono_logits.shape}")

    # ── TwoStreamModel ───────────────────────────────────────────
    print("\n[5] TwoStreamModel (full forward)")
    model = TwoStreamModel(
        hidden_dim=64, n_heads=4, n_layers=2, ffn_dim=128,
        vocab_size=V, num_quantizers=N_q,
        monologue_vocab_size=64,
    )
    output = model(user_tokens, asst_tokens, turn_labels)
    print(f"  speech_logits    : {output['speech_logits'].shape}")
    print(f"  monologue_logits : {output['monologue_logits'].shape}")
    print(f"  turn_logits      : {output['turn_logits'].shape}")
    print(f"  losses           : { {k: v.item():.4f} for k, v in output['losses'].items() }")
    print(f"  params           : {model.num_parameters():,}")

    # ── Generate ─────────────────────────────────────────────────
    print("\n[6] Autoregressive generation (5 frames)")
    model.eval()
    generated = model.generate(user_tokens, max_new_frames=5)
    print(f"  user_tokens : {user_tokens.shape}")
    print(f"  generated   : {generated.shape}  (B, new_frames, N_q)")

    print("\nAll smoke tests passed.")
