"""
T2I Chapter 3: Diffusion Transformer (DiT) Denoiser
=====================================================
The DiT (Peebles & Xie, 2023) replaces the U-Net backbone in diffusion models
with a pure Transformer architecture operating on image patches.

Following Stable Diffusion 3 / FLUX, we implement MM-DiT (Multi-Modal DiT)
which treats image patches and text tokens as two interleaved token sequences.

Architecture:

  Input latent z_t:   (B, C, H/8, W/8)  — noisy latent from VAE
  Text conditioning:  (B, T_text, D)     — from text encoder
  Timestep t:         (B,)               — normalized to [0, 1]

  1. PatchEmbed: latent → (B, N_patches, D)
  2. TimestepEmbed: t → (B, D)  — sinusoidal MLP
  3. For each MM-DiT block:
       a. AdaLN-Zero: modulate with timestep embedding
       b. Self-attention on image patches
       c. Cross-attention from image patches to text tokens
       d. FFN with SiLU
  4. Unpatch: (B, N_patches, D) → (B, C, H/8, W/8)  — predicted noise/velocity

Key design choices:
  - AdaLN-Zero: initialize output gates to zero → identity at start of training
  - Classifier-Free Guidance (CFG): train with null text conditioning
  - Sinusoidal timestep embedding followed by MLP (same as DiT / DDPM)
  - Text cross-attention: image patches attend to text encoder states
  - Final layer: separate projection per output channel pair (mu/sigma in pred_v)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 1.  Timestep Embedding
# ─────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """
    Embed scalar timestep t ∈ [0, 1000] (or [0,1]) into a vector.

    Sinusoidal encoding → two-layer MLP → conditioning vector c_t.
    Identical in spirit to DDPM / DiT timestep embedding.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half).float() / half
        )
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) in [0, 1]  →  (B, dim)"""
        t   = t.float().unsqueeze(1)              # (B, 1)
        arg = t * self.freqs.unsqueeze(0)         # (B, half)
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return self.mlp(emb)                      # (B, dim)


# ─────────────────────────────────────────────────────────────
# 2.  Patch Embedding
# ─────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """
    Embed a spatial latent map (or image) into a sequence of patch tokens.

    Input:  (B, C, H, W)
    Output: (B, N_patches, D)   where N_patches = (H/P) * (W/P)

    Also provides an unpatch() method (requires knowing the original H, W, P).
    """

    def __init__(
        self,
        patch_size: int,
        in_ch: int,
        d_model: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch      = in_ch
        self.d_model    = d_model
        # Projection via strided convolution (linear per patch)
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        x: (B, C, H, W)
        Returns: (tokens, h_patches, w_patches)
          tokens: (B, N, D)
          h_patches, w_patches: grid dimensions
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x)                        # (B, D, H/P, W/P)
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        x   = x.reshape(B, self.d_model, h_p * w_p).transpose(1, 2)  # (B, N, D)
        return x, h_p, w_p

    def unpatch(
        self,
        tokens: torch.Tensor,
        h_p: int,
        w_p: int,
        out_ch: int,
    ) -> torch.Tensor:
        """
        Inverse of forward: (B, N, D) → (B, out_ch, H, W).

        Uses a learned linear projection from D → out_ch * patch_size^2,
        then pixel-shuffle to restore spatial layout.
        """
        raise NotImplementedError("Use UnpatchLayer instead")


class UnpatchLayer(nn.Module):
    """
    Project patch tokens back to spatial feature map.

    (B, N, D) → (B, out_ch, H, W)
    where H = h_p * patch_size, W = w_p * patch_size.
    """

    def __init__(self, d_model: int, patch_size: int, out_ch: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_ch     = out_ch
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, out_ch * patch_size * patch_size, bias=True)
        # Zero-init for stable start
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, h_p: int, w_p: int) -> torch.Tensor:
        """
        tokens: (B, N, D)   where N = h_p * w_p
        Returns: (B, out_ch, h_p*P, w_p*P)
        """
        B, N, D = tokens.shape
        P = self.patch_size
        x = self.proj(self.norm(tokens))          # (B, N, out_ch*P*P)
        x = x.reshape(B, h_p, w_p, self.out_ch, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5)          # (B, out_ch, h_p, P, w_p, P)
        x = x.reshape(B, self.out_ch, h_p * P, w_p * P)
        return x


# ─────────────────────────────────────────────────────────────
# 3.  AdaLN-Zero (Adaptive Layer Norm with Zero Initialization)
# ─────────────────────────────────────────────────────────────

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with zero-initialized gates.

    Following DiT (Peebles & Xie, 2023):
      Given condition c (timestep + text summary):
        [α1, β1, γ1, α2, β2, γ2] = Linear(SiLU(c))
        x = γ1 * LN(x) + β1
        x = x + α1 * attention(x)
        x = γ2 * LN(x) + β2
        x = x + α2 * FFN(x)

      Zero-init of Linear → α_i=0 at t=0 → identity residual → stable training start.

    For simplicity we combine modulation in one AdaLN-Zero module that returns
    6 scalars (α_attn, β_attn, γ_attn, α_ffn, β_ffn, γ_ffn).
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN   = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model, bias=True),
        )
        # Zero-init → all gates start at 0 → identity at start
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,       # (B, T, D)
        c: torch.Tensor,       # (B, cond_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          x_norm1, gate_attn, shift_attn,   for attention branch
          x_norm2, gate_ffn,  shift_ffn,    for FFN branch
        """
        params = self.adaLN(c)                 # (B, 6D)
        (shift1, scale1, gate1,
         shift2, scale2, gate2) = params.chunk(6, dim=-1)

        # Modulate LN for attention branch
        x_n1 = self.norm(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        # Modulate LN for FFN branch
        x_n2 = self.norm(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)

        return x_n1, gate1.unsqueeze(1), x_n2, gate2.unsqueeze(1)


# ─────────────────────────────────────────────────────────────
# 4.  Multi-Head Cross-Attention (image → text)
# ─────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Image patch tokens attend to text encoder states.

    Q from image patches, K/V from text tokens.
    Returns attended features + attention weights (for visualization).
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
        query:   torch.Tensor,              # (B, N_img, D)
        context: torch.Tensor,              # (B, T_text, D)
        key_mask: Optional[torch.Tensor] = None,  # (B, T_text) True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, attn_weights)  shapes (B,N,D) and (B,H,N,T)."""
        B, N, D = query.shape
        Tk       = context.shape[1]
        H, Dh    = self.n_heads, self.d_head

        q  = self.q_proj(query).reshape(B, N, H, Dh).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, Tk, 2, H, Dh).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if key_mask is not None:
            attn = attn.masked_fill(
                key_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_w = F.softmax(attn, dim=-1)
        out    = (self.drop(attn_w) @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out), attn_w


# ─────────────────────────────────────────────────────────────
# 5.  Self-Attention on Image Patches
# ─────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, attn_weights)."""
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_w = self.drop(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        out = (attn_w @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out), attn_w


# ─────────────────────────────────────────────────────────────
# 6.  MM-DiT Block
# ─────────────────────────────────────────────────────────────

class MMDiTBlock(nn.Module):
    """
    Multi-Modal DiT block (image patches + text cross-attention).

    Operations per block:
      1. AdaLN-Zero modulate image features with timestep condition
      2. Self-attention on image patches (spatial reasoning)
      3. Cross-attention: image patches → text tokens (text conditioning)
      4. FFN with AdaLN-Zero modulation
      5. Residual connections with gated (learned) scaling

    Text context is injected via cross-attention (not concatenation),
    following DiT-XL/2 and FLUX architecture.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        d_ff:     int,
        cond_dim: int,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.ada_ln    = AdaLNZero(d_model, cond_dim)
        self.self_attn = SelfAttention(d_model, n_heads, dropout)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)

        self.norm_ca   = nn.LayerNorm(d_model)
        self.norm_ctx  = nn.LayerNorm(d_model)

        self.ffn       = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        # Zero-init FFN output for AdaLN-Zero stability
        nn.init.zeros_(self.ffn[-2].weight)

    def forward(
        self,
        x:        torch.Tensor,           # (B, N_img, D) — image patch tokens
        c:        torch.Tensor,           # (B, cond_dim) — timestep embedding
        ctx:      torch.Tensor,           # (B, T_text, D) — text context
        ctx_mask: Optional[torch.Tensor] = None,  # (B, T_text)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (updated_x, attn_weights) where attn_weights is cross-attn map.
        """
        # ── Self-attention with AdaLN-Zero ────────────────────
        x_n1, gate1, x_n2, gate2 = self.ada_ln(x, c)
        sa_out, _  = self.self_attn(x_n1)
        x = x + gate1 * sa_out

        # ── Cross-attention to text ───────────────────────────
        ca_out, attn_w = self.cross_attn(
            self.norm_ca(x),
            self.norm_ctx(ctx),
            key_mask=ctx_mask,
        )
        x = x + ca_out

        # ── FFN with AdaLN-Zero ───────────────────────────────
        x = x + gate2 * self.ffn(x_n2)

        return x, attn_w


# ─────────────────────────────────────────────────────────────
# 7.  Full DiT Denoiser
# ─────────────────────────────────────────────────────────────

class DiTDenoiser(nn.Module):
    """
    Full Diffusion Transformer denoiser for latent diffusion.

    Input:
      z_t     : (B, C, H, W)  — noisy latent (from VAE)
      t       : (B,)          — timestep in [0, 1]
      ctx     : (B, T, D_ctx) — text encoder hidden states
      ctx_mask: (B, T)        — True for pad positions (optional)

    Output:
      v_pred  : (B, C, H, W)  — predicted noise or velocity

    Architecture:
      PatchEmbed(z_t) → [TimestepEmbed + text pool → cond]
      → N × MMDiTBlock(tokens, cond, ctx)
      → UnpatchLayer → (B, C, H, W)

    Supports Classifier-Free Guidance (CFG) at inference:
      output = uncond + guidance_scale * (cond - uncond)
    """

    def __init__(
        self,
        # Latent (VAE output) config
        in_ch:       int   = 4,        # VAE latent channels
        latent_h:    int   = 4,        # latent height (H/8)
        latent_w:    int   = 4,        # latent width  (W/8)
        patch_size:  int   = 1,        # patch size in latent space (1 = per-pixel)
        # Model config
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        d_ff:        int   = 512,
        # Text conditioning config
        ctx_dim:     int   = 128,      # text encoder output dim
        # Misc
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.in_ch      = in_ch
        self.patch_size = patch_size
        self.latent_h   = latent_h
        self.latent_w   = latent_w

        # Patch embedding: latent → tokens
        self.patch_embed = PatchEmbed(patch_size, in_ch, d_model)
        n_patches = (latent_h // patch_size) * (latent_w // patch_size)
        self.n_patches = n_patches

        # Learnable position embeddings for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(d_model)

        # Project text context to d_model if sizes differ
        self.ctx_proj = (
            nn.Linear(ctx_dim, d_model, bias=False)
            if ctx_dim != d_model else nn.Identity()
        )

        # Text pooling for conditioning signal (summary of text)
        self.text_pool_proj = nn.Linear(d_model, d_model, bias=False)

        # Condition: timestep + text pool → cond_dim
        cond_dim = d_model
        self.cond_norm = nn.LayerNorm(d_model)

        # DiT blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(d_model, n_heads, d_ff, cond_dim, dropout)
            for _ in range(n_layers)
        ])

        # Unpatch: tokens → latent
        self.unpatch = UnpatchLayer(d_model, patch_size, in_ch)

    def forward(
        self,
        z_t:      torch.Tensor,                      # (B, C, H, W)
        t:        torch.Tensor,                      # (B,)
        ctx:      torch.Tensor,                      # (B, T_text, D_ctx)
        ctx_mask: Optional[torch.Tensor] = None,     # (B, T_text)
    ) -> torch.Tensor:
        """
        Returns predicted velocity/noise: (B, C, H, W).
        """
        B = z_t.shape[0]

        # 1. Embed patches
        tokens, h_p, w_p = self.patch_embed(z_t)    # (B, N, D)
        tokens = tokens + self.pos_embed             # add position

        # 2. Project text context
        ctx_proj = self.ctx_proj(ctx)                # (B, T, D)

        # 3. Build conditioning vector: timestep + text summary
        t_emb = self.time_embed(t)                   # (B, D)

        # Pool text to a single vector (mean over non-pad positions)
        if ctx_mask is not None:
            # ctx_mask: (B, T) True=pad → invert for mean-pooling
            valid = (~ctx_mask).float().unsqueeze(-1)   # (B, T, 1)
            n_valid = valid.sum(1).clamp(min=1)
            txt_pool = (ctx_proj * valid).sum(1) / n_valid  # (B, D)
        else:
            txt_pool = ctx_proj.mean(dim=1)          # (B, D)

        txt_pool = self.text_pool_proj(txt_pool)     # (B, D)
        cond     = self.cond_norm(t_emb + txt_pool)  # (B, D)

        # 4. Pass through MM-DiT blocks
        attn_weights = None
        for block in self.blocks:
            tokens, attn_w = block(tokens, cond, ctx_proj, ctx_mask)
            attn_weights = attn_w   # save last layer for visualization

        # 5. Unpatch: tokens → spatial latent
        out = self.unpatch(tokens, h_p, w_p)         # (B, C, H, W)
        return out

    def forward_with_cfg(
        self,
        z_t:           torch.Tensor,          # (B, C, H, W)
        t:             torch.Tensor,          # (B,)
        ctx:           torch.Tensor,          # (B, T, D)
        null_ctx:      torch.Tensor,          # (B, T, D) — null/empty text
        guidance_scale: float = 7.5,
        ctx_mask:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Classifier-Free Guidance (CFG) inference step.

        Combines conditional and unconditional predictions:
          v = v_uncond + guidance_scale * (v_cond - v_uncond)

        guidance_scale = 1.0 → no guidance
        guidance_scale = 7.5 → typical FLUX/SD value
        """
        # Concatenate for single forward pass (2B batch)
        z_cat  = torch.cat([z_t, z_t], dim=0)
        t_cat  = torch.cat([t, t], dim=0)
        ctx_cat = torch.cat([ctx, null_ctx], dim=0)

        mask_cat = None
        if ctx_mask is not None:
            mask_cat = torch.cat([ctx_mask, ctx_mask], dim=0)

        out_cat = self.forward(z_cat, t_cat, ctx_cat, mask_cat)
        v_cond, v_uncond = out_cat.chunk(2, dim=0)

        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def get_last_attn_weights(
        self,
        z_t:      torch.Tensor,
        t:        torch.Tensor,
        ctx:      torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (output, cross_attn_weights).
        attn_weights: (B, n_heads, N_patches, T_text) from last block.
        """
        B = z_t.shape[0]
        tokens, h_p, w_p = self.patch_embed(z_t)
        tokens = tokens + self.pos_embed
        ctx_p  = self.ctx_proj(ctx)
        t_emb  = self.time_embed(t)

        if ctx_mask is not None:
            valid    = (~ctx_mask).float().unsqueeze(-1)
            n_valid  = valid.sum(1).clamp(min=1)
            txt_pool = (ctx_p * valid).sum(1) / n_valid
        else:
            txt_pool = ctx_p.mean(dim=1)

        txt_pool = self.text_pool_proj(txt_pool)
        cond     = self.cond_norm(t_emb + txt_pool)

        attn_weights = None
        for block in self.blocks:
            tokens, attn_w = block(tokens, cond, ctx_p, ctx_mask)
            attn_weights   = attn_w

        out = self.unpatch(tokens, h_p, w_p)
        return out, attn_weights

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────
# 8.  Smoke-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DiT DENOISER — Smoke Tests")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    # Tiny config: 32x32 images → 4x4 latents (scale=8) → patch_size=1 → 16 patches
    B        = 2
    C        = 4    # latent channels
    LAT_H    = 4    # 32/8 = 4
    LAT_W    = 4
    T_TEXT   = 10
    D_MODEL  = 64
    CTX_DIM  = 64

    # ── TimestepEmbedding ─────────────────────────────────────
    print("\n[1] TimestepEmbedding")
    te  = TimestepEmbedding(D_MODEL).to(device)
    t   = torch.rand(B, device=device)
    emb = te(t)
    print(f"  t{t.shape} → emb{emb.shape}")

    # ── PatchEmbed ────────────────────────────────────────────
    print("\n[2] PatchEmbed (patch_size=1 on 4x4 latent → 16 tokens)")
    pe    = PatchEmbed(patch_size=1, in_ch=C, d_model=D_MODEL).to(device)
    z     = torch.randn(B, C, LAT_H, LAT_W, device=device)
    toks, h_p, w_p = pe(z)
    print(f"  z{z.shape} → tokens{toks.shape}  grid=({h_p},{w_p})")

    # ── UnpatchLayer ──────────────────────────────────────────
    print("\n[3] UnpatchLayer")
    up   = UnpatchLayer(D_MODEL, patch_size=1, out_ch=C).to(device)
    recon = up(toks, h_p, w_p)
    print(f"  tokens{toks.shape} → latent{recon.shape}")
    assert recon.shape == (B, C, LAT_H, LAT_W), f"Shape mismatch: {recon.shape}"

    # ── MM-DiT Block ──────────────────────────────────────────
    print("\n[4] MMDiTBlock")
    block = MMDiTBlock(D_MODEL, n_heads=4, d_ff=256, cond_dim=D_MODEL).to(device)
    cond  = torch.randn(B, D_MODEL, device=device)
    ctx   = torch.randn(B, T_TEXT, CTX_DIM, device=device)
    out_b, attn_w = block(toks, cond, ctx)
    print(f"  tokens{toks.shape} cond{cond.shape} ctx{ctx.shape}")
    print(f"  → out{out_b.shape}  attn{attn_w.shape}")

    # ── Full DiT Denoiser ─────────────────────────────────────
    print("\n[5] DiTDenoiser (full forward)")
    dit = DiTDenoiser(
        in_ch=C, latent_h=LAT_H, latent_w=LAT_W,
        patch_size=1, d_model=D_MODEL, n_heads=4,
        n_layers=2, d_ff=256, ctx_dim=CTX_DIM,
    ).to(device)

    z_t    = torch.randn(B, C, LAT_H, LAT_W, device=device)
    t_vec  = torch.rand(B, device=device)
    ctx_in = torch.randn(B, T_TEXT, CTX_DIM, device=device)

    v_pred = dit(z_t, t_vec, ctx_in)
    print(f"  z_t{z_t.shape} t{t_vec.shape} ctx{ctx_in.shape}")
    print(f"  → v_pred{v_pred.shape}  params={dit.num_parameters():,}")
    assert v_pred.shape == z_t.shape, f"Shape mismatch: {v_pred.shape}"

    # ── CFG forward ───────────────────────────────────────────
    print("\n[6] CFG forward")
    null_ctx = torch.zeros_like(ctx_in)
    v_cfg    = dit.forward_with_cfg(z_t, t_vec, ctx_in, null_ctx, guidance_scale=7.5)
    print(f"  CFG output{v_cfg.shape}")

    # ── Attention weights ─────────────────────────────────────
    print("\n[7] Cross-attention weights")
    _, attn = dit.get_last_attn_weights(z_t, t_vec, ctx_in)
    print(f"  attn weights{attn.shape}  (B, heads, N_patches, T_text)")

    # ── Backward ──────────────────────────────────────────────
    print("\n[8] Backward pass")
    loss = v_pred.pow(2).mean()
    loss.backward()
    print(f"  loss={loss.item():.4f} ✓")

    print("\n[OK] All DiT tests passed")
