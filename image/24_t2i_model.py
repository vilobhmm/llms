"""
T2I Chapter 5: Full Latent Diffusion Model (LDM)
=================================================
Wires together all components into a complete Text-to-Image pipeline,
following the FLUX / Stable Diffusion 3 approach:

  Text prompt
    │
    ▼
  CLIPTextEncoder / CharTokenizer
    │ text embeddings (B, T, D)
    ▼
  ┌──────────── Training: image → latent ───────────┐
  │  Image (B, 3, H, W)                             │
  │    ↓ ConvVAE.encode()                           │
  │  Latent z (B, C, H/8, W/8)                      │
  │    ↓ DDPMScheduler.q_sample() or FM path        │
  │  Noisy latent z_t                               │
  └─────────────────────────────────────────────────┘
    │
    ▼
  DiTDenoiser(z_t, t, text_emb)
    │ predicted noise / velocity
    ▼
  DDPMScheduler.p_sample() or FM.euler_sample()
    │ denoised latent z_hat
    ▼
  ConvVAE.decode()
    │
    ▼
  Image (B, 3, H, W)

The synthesize() method runs the full inference pipeline:
  text → tokens → encode → DiT denoising loop → decode → image

Config sizes: "tiny" (testing), "small", "medium"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import os
import sys
import importlib.util


# ─────────────────────────────────────────────────────────────
# Module loader (same pattern as TTS codebase)
# ─────────────────────────────────────────────────────────────

def _load(name: str, fname: str):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(here, fname)
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_HERE = os.path.dirname(os.path.abspath(__file__))

_vae  = _load("vae",  "20_image_vae.py")
_clip = _load("clip", "21_clip_encoder.py")
_dit  = _load("dit",  "22_dit_denoiser.py")
_sch  = _load("sch",  "23_ddpm_scheduler.py")

ConvVAE              = _vae.ConvVAE
vae_loss             = _vae.vae_loss
CLIPTextEncoder      = _clip.CLIPTextEncoder
CLIPModel            = _clip.CLIPModel
CharTokenizer        = _clip.CharTokenizer
clip_loss            = _clip.clip_loss
DiTDenoiser          = _dit.DiTDenoiser
DDPMScheduler        = _sch.DDPMScheduler
DDIMSampler          = _sch.DDIMSampler
FlowMatchingScheduler = _sch.FlowMatchingScheduler


# ─────────────────────────────────────────────────────────────
# 1.  T2I Config Dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class T2IConfig:
    """
    Configuration dataclass for the full T2I pipeline.

    Sizes:
      tiny   — for smoke-tests and CI (very fast, tiny tensors)
      small  — for demo training on CPU (fits in memory, trains in minutes)
      medium — larger capacity (reasonable quality, needs decent hardware)
    """
    size: str = "tiny"

    # Image dimensions
    img_size:    int = 32      # input/output image size (H=W)
    in_channels: int = 3      # RGB

    # VAE
    vae_base_ch:   int   = 16
    vae_ch_mult:   tuple = (1, 2, 4)
    vae_latent_dim: int  = 4   # C in (B, C, H/8, W/8)
    vae_n_res:     int   = 1
    vae_beta:      float = 0.001  # KL weight

    # Derived: latent spatial size
    @property
    def latent_h(self) -> int:
        return self.img_size // 8

    @property
    def latent_w(self) -> int:
        return self.img_size // 8

    # CLIP / Text encoder
    text_vocab:     int = 99       # CharTokenizer vocab
    text_max_len:   int = 77
    text_d_model:   int = 64
    text_n_heads:   int = 4
    text_n_layers:  int = 2
    text_d_ff:      int = 256
    text_embed_dim: int = 64

    # Image encoder (for CLIP training)
    img_patch_size:    int = 4
    img_enc_d_model:   int = 64
    img_enc_n_heads:   int = 4
    img_enc_n_layers:  int = 2
    img_enc_d_ff:      int = 256
    clip_embed_dim:    int = 64

    # DiT denoiser
    dit_d_model:    int = 64
    dit_n_heads:    int = 4
    dit_n_layers:   int = 2
    dit_d_ff:       int = 256
    dit_patch_size: int = 1    # patch size in latent space

    # Diffusion
    ddpm_T:          int   = 100    # total diffusion steps (small for demo)
    ddpm_schedule:   str   = "cosine"
    ddim_n_steps:    int   = 10     # DDIM fast sampling steps
    fm_n_steps:      int   = 20     # flow-matching inference steps
    guidance_scale:  float = 7.5
    p_uncond:        float = 0.1    # CFG dropout probability during training

    # Misc
    dropout: float = 0.1

    def __post_init__(self):
        """Override defaults based on size."""
        if self.size == "tiny":
            # Absolute minimum for smoke-tests
            self.vae_base_ch      = 8
            self.vae_ch_mult      = (1, 2, 4)
            self.vae_n_res        = 1
            self.text_d_model     = 32
            self.text_n_layers    = 1
            self.text_d_ff        = 128
            self.text_embed_dim   = 32
            self.img_enc_d_model  = 32
            self.img_enc_n_layers = 1
            self.img_enc_d_ff     = 128
            self.clip_embed_dim   = 32
            self.dit_d_model      = 32
            self.dit_n_heads      = 2
            self.dit_n_layers     = 1
            self.dit_d_ff         = 128
            self.ddpm_T           = 20
            self.ddim_n_steps     = 5
            self.fm_n_steps       = 5

        elif self.size == "small":
            self.vae_base_ch      = 16
            self.vae_ch_mult      = (1, 2, 4)
            self.vae_n_res        = 1
            self.text_d_model     = 64
            self.text_n_layers    = 2
            self.text_d_ff        = 256
            self.text_embed_dim   = 64
            self.img_enc_d_model  = 64
            self.img_enc_n_layers = 2
            self.img_enc_d_ff     = 256
            self.clip_embed_dim   = 64
            self.dit_d_model      = 64
            self.dit_n_heads      = 4
            self.dit_n_layers     = 2
            self.dit_d_ff         = 256
            self.ddpm_T           = 100
            self.ddim_n_steps     = 10
            self.fm_n_steps       = 20

        elif self.size == "medium":
            self.vae_base_ch      = 32
            self.vae_ch_mult      = (1, 2, 4)
            self.vae_n_res        = 2
            self.text_d_model     = 128
            self.text_n_layers    = 4
            self.text_d_ff        = 512
            self.text_embed_dim   = 128
            self.img_enc_d_model  = 128
            self.img_enc_n_layers = 4
            self.img_enc_d_ff     = 512
            self.clip_embed_dim   = 128
            self.dit_d_model      = 128
            self.dit_n_heads      = 4
            self.dit_n_layers     = 4
            self.dit_d_ff         = 512
            self.ddpm_T           = 1000
            self.ddim_n_steps     = 50
            self.fm_n_steps       = 50


# ─────────────────────────────────────────────────────────────
# 2.  Full Latent Diffusion Model
# ─────────────────────────────────────────────────────────────

class LatentDiffusionModel(nn.Module):
    """
    Full Text-to-Image pipeline:

    Components:
      tokenizer    : CharTokenizer
      vae          : ConvVAE          (image ↔ latent)
      text_encoder : CLIPTextEncoder  (text → embeddings)
      clip_model   : CLIPModel        (for CLIP training phase)
      dit          : DiTDenoiser      (latent denoising)
      ddpm_sch     : DDPMScheduler    (forward/reverse process)
      fm_sch       : FlowMatchingScheduler (flow-matching variant)
      ddim_sampler : DDIMSampler      (fast inference)

    Training phases:
      1. VAE:  train encode/decode
      2. CLIP: train text-image contrastive alignment
      3. LDM:  train DiT with frozen VAE + text encoder

    Inference (synthesize):
      text → tokens → encode → FM/DDIM loop → decode → image
    """

    def __init__(self, cfg: T2IConfig):
        super().__init__()
        self.cfg = cfg

        # Tokenizer (not a nn.Module)
        self.tokenizer = CharTokenizer(max_len=cfg.text_max_len)

        # VAE
        self.vae = ConvVAE(
            in_channels=cfg.in_channels,
            base_ch=cfg.vae_base_ch,
            ch_mult=cfg.vae_ch_mult,
            latent_dim=cfg.vae_latent_dim,
            n_res=cfg.vae_n_res,
        )

        # CLIP text + image encoder
        self.clip_model = CLIPModel(
            text_vocab=cfg.text_vocab,
            text_d_model=cfg.text_d_model,
            text_n_heads=cfg.text_n_heads,
            text_n_layers=cfg.text_n_layers,
            text_d_ff=cfg.text_d_ff,
            img_size=cfg.img_size,
            patch_size=cfg.img_patch_size,
            img_d_model=cfg.img_enc_d_model,
            img_n_heads=cfg.img_enc_n_heads,
            img_n_layers=cfg.img_enc_n_layers,
            img_d_ff=cfg.img_enc_d_ff,
            embed_dim=cfg.clip_embed_dim,
            dropout=cfg.dropout,
            max_text_len=cfg.text_max_len,
        )

        # Text-only encoder (used during LDM diffusion conditioning)
        # Shares the same text encoder as CLIP
        self.text_encoder = self.clip_model.text_encoder

        # Null text embedding for CFG (learned, one per T position)
        self.null_ctx = nn.Parameter(
            torch.zeros(1, 1, cfg.text_d_model)
        )

        # DiT denoiser
        self.dit = DiTDenoiser(
            in_ch=cfg.vae_latent_dim,
            latent_h=cfg.latent_h,
            latent_w=cfg.latent_w,
            patch_size=cfg.dit_patch_size,
            d_model=cfg.dit_d_model,
            n_heads=cfg.dit_n_heads,
            n_layers=cfg.dit_n_layers,
            d_ff=cfg.dit_d_ff,
            ctx_dim=cfg.text_d_model,   # DiT needs hidden states, not pooled
            dropout=cfg.dropout,
        )

        # Schedulers (not nn.Modules)
        self.ddpm_sch = DDPMScheduler(
            T=cfg.ddpm_T,
            schedule=cfg.ddpm_schedule,
        )
        self.ddim_sampler = DDIMSampler(
            self.ddpm_sch,
            n_steps=cfg.ddim_n_steps,
        )
        self.fm_sch = FlowMatchingScheduler()

    # ──────────────────────────────────────────────────────────
    # Forward passes for each training phase
    # ──────────────────────────────────────────────────────────

    def forward_vae(
        self,
        images: torch.Tensor,    # (B, 3, H, W) in [-1, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE training forward pass.
        Returns (recon, mu, log_var, loss).
        """
        recon, mu, log_var = self.vae(images)
        loss, r_loss, kl_loss = vae_loss(recon, images, mu, log_var, self.cfg.vae_beta)
        return recon, mu, log_var, loss

    def forward_clip(
        self,
        images: torch.Tensor,    # (B, 3, H, W)
        tokens: torch.Tensor,    # (B, T_text)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CLIP training forward pass.
        Returns (img_emb, txt_emb, contrastive_loss).
        """
        img_emb, txt_emb, logits = self.clip_model(images, tokens)
        loss = clip_loss(logits)
        return img_emb, txt_emb, loss

    def _encode_text(
        self,
        tokens:   torch.Tensor,        # (B, T)
        use_cfg:  bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode text tokens to hidden states.
        Returns (ctx_hidden, null_ctx) where null_ctx is used for CFG.

        ctx_hidden: (B, T, text_d_model) — full hidden sequence for cross-attn
        """
        pad_mask = (tokens == self.tokenizer.PAD_ID)  # (B, T)

        # We need hidden states (before pooling), not the pooled+projected CLIP emb.
        # Re-run the transformer layers to get sequence output:
        text_enc = self.text_encoder
        B, T_txt = tokens.shape

        mask = (tokens == text_enc.pad_id)
        x    = text_enc.tok_embed(tokens) * text_enc.scale
        x    = text_enc.pos_enc(x)
        for layer in text_enc.layers:
            x = layer(x, key_padding_mask=mask)
        ctx = text_enc.norm(x)   # (B, T, d_model) — raw hidden states

        # Build null context for CFG
        B_ctx = ctx.shape[0]
        null  = self.null_ctx.expand(B_ctx, ctx.shape[1], ctx.shape[2])

        return ctx, null, pad_mask

    def forward_ldm(
        self,
        images:  torch.Tensor,    # (B, 3, H, W)
        tokens:  torch.Tensor,    # (B, T_text)
        use_flow: bool = True,
    ) -> torch.Tensor:
        """
        LDM training loss.
        Encodes image → latent, adds noise, predicts noise/velocity, computes MSE.

        Uses flow matching (if use_flow=True) or DDPM.
        Applies CFG dropout: randomly replaces text ctx with null ctx.
        """
        B = images.shape[0]
        device = images.device

        # 1. Encode image → latent
        with torch.no_grad():
            mu, log_var = self.vae.encode(images)
        z_0 = mu  # use mean for training stability (like SD3/FLUX)

        # 2. Encode text
        ctx, null_ctx, ctx_mask = self._encode_text(tokens)

        # 3. CFG dropout: randomly replace ctx with null for some samples
        cfg_mask = torch.rand(B, device=device) < self.cfg.p_uncond
        ctx_train = torch.where(
            cfg_mask.reshape(B, 1, 1),
            null_ctx,
            ctx,
        )

        # 4. Forward diffusion / flow path + prediction
        if use_flow:
            def model_fn(x_t, t, cx, mask=None):
                return self.dit(x_t, t, cx, mask)
            loss = self.fm_sch.training_loss(model_fn, z_0, ctx_train, None)
        else:
            def model_fn(x_t, t, cx, mask=None):
                return self.dit(x_t, t, cx, mask)
            loss = self.ddpm_sch.training_loss(model_fn, z_0, ctx_train, None)

        return loss

    # ──────────────────────────────────────────────────────────
    # Inference: synthesize(text) → image
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def synthesize(
        self,
        text:           str,
        n_steps:        int   = 20,
        guidance_scale: float = 7.5,
        use_flow:       bool  = True,
        seed:           Optional[int] = None,
        device:         Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        text → (1, 3, H, W) image tensor in [-1, 1].

        Args:
          text            : text prompt (str)
          n_steps         : number of denoising steps
          guidance_scale  : CFG scale (1.0 = no guidance)
          use_flow        : use flow-matching (True) or DDIM (False)
          seed            : optional random seed for reproducibility
          device          : target device

        Returns:
          image: (1, 3, H, W) in range [-1, 1]
        """
        if device is None:
            device = next(self.parameters()).device

        if seed is not None:
            torch.manual_seed(seed)

        # 1. Tokenize text
        ids  = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        toks = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        # Pad to a fixed length for batching convenience
        T_max = min(len(ids), self.cfg.text_max_len)
        toks_padded = torch.zeros(1, T_max, dtype=torch.long, device=device)
        toks_padded[0, :min(len(ids), T_max)] = toks[0, :T_max]
        toks = toks_padded

        # 2. Encode text → hidden states
        ctx, null_ctx, ctx_mask = self._encode_text(toks)

        # 3. Define model function (with CFG)
        def model_fn_cfg(x_t, t_vec, ctx_in, mask=None):
            # Single forward: conditional
            v_cond   = self.dit(x_t, t_vec, ctx_in, mask)
            v_uncond = self.dit(x_t, t_vec, null_ctx, mask)
            return v_uncond + guidance_scale * (v_cond - v_uncond)

        # 4. Denoise in latent space
        latent_shape = (
            1, self.cfg.vae_latent_dim,
            self.cfg.latent_h, self.cfg.latent_w
        )

        if use_flow:
            traj = self.fm_sch.euler_sample(
                model_fn_cfg, latent_shape, ctx, device,
                n_steps=n_steps, ctx_mask=ctx_mask
            )
            z_denoised = traj[-1]
        else:
            traj = self.ddim_sampler.sample(
                lambda x, t, c, m: model_fn_cfg(x, t, c, m),
                latent_shape, ctx, device, ctx_mask=ctx_mask
            )
            z_denoised = traj[-1]

        # 5. Decode latent → image
        image = self.vae.decode(z_denoised)   # (1, 3, H, W) in [-1, 1]
        return image

    @torch.no_grad()
    def synthesize_batch(
        self,
        texts:          List[str],
        n_steps:        int   = 20,
        guidance_scale: float = 7.5,
        use_flow:       bool  = True,
        device:         Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Batch synthesize: texts → (B, 3, H, W)."""
        images = []
        for text in texts:
            img = self.synthesize(text, n_steps, guidance_scale, use_flow, device=device)
            images.append(img)
        return torch.cat(images, dim=0)

    # ──────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────

    def num_parameters(self) -> dict:
        """Parameter counts per component."""
        return {
            "vae":          sum(p.numel() for p in self.vae.parameters()),
            "clip_text":    sum(p.numel() for p in self.clip_model.text_encoder.parameters()),
            "clip_image":   sum(p.numel() for p in self.clip_model.image_encoder.parameters()),
            "dit":          sum(p.numel() for p in self.dit.parameters()),
            "total":        sum(p.numel() for p in self.parameters()),
        }

    def freeze_vae(self):
        """Freeze VAE for LDM training phase."""
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def freeze_text_encoder(self):
        """Freeze text encoder for LDM training phase."""
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)


# ─────────────────────────────────────────────────────────────
# 3.  Model Summary Utility
# ─────────────────────────────────────────────────────────────

def model_summary(model: LatentDiffusionModel) -> str:
    counts = model.num_parameters()
    lines  = ["=" * 55, "T2I Latent Diffusion Model Summary", "=" * 55]
    for name, n in counts.items():
        if name == "total":
            lines.append("-" * 55)
        lines.append(f"  {name:<18} {n:>12,} params")
    lines.append("=" * 55)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 4.  Smoke-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("FULL T2I LATENT DIFFUSION MODEL — Smoke Tests")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # Use tiny config for speed
    cfg   = T2IConfig(size="tiny")
    model = LatentDiffusionModel(cfg).to(device)
    print(model_summary(model))

    B = 2
    H = W = cfg.img_size
    T_text = 15

    # ── Tokenizer ─────────────────────────────────────────────
    print("\n[1] Tokenizer")
    tok  = model.tokenizer
    ids  = tok.encode("a photo of a cat", add_bos=True, add_eos=True)
    text = tok.decode(ids)
    print(f"  'a photo of a cat' → {len(ids)} tokens → '{text}'")

    # ── VAE Forward ───────────────────────────────────────────
    print("\n[2] VAE training forward")
    images = torch.randn(B, 3, H, W, device=device)
    recon, mu, log_var, vae_l = model.forward_vae(images)
    print(f"  images{images.shape} → recon{recon.shape}  mu{mu.shape}")
    print(f"  VAE loss: {vae_l.item():.4f}")

    # ── CLIP Forward ──────────────────────────────────────────
    print("\n[3] CLIP training forward")
    tokens = torch.randint(4, cfg.text_vocab, (B, T_text), device=device)
    img_emb, txt_emb, clip_l = model.forward_clip(images, tokens)
    print(f"  img_emb{img_emb.shape}  txt_emb{txt_emb.shape}")
    print(f"  CLIP loss: {clip_l.item():.4f}")

    # ── LDM Forward (Flow Matching) ───────────────────────────
    print("\n[4] LDM training forward (flow matching)")
    ldm_loss = model.forward_ldm(images, tokens, use_flow=True)
    print(f"  LDM flow-matching loss: {ldm_loss.item():.4f}")

    # ── LDM Forward (DDPM) ────────────────────────────────────
    print("\n[5] LDM training forward (DDPM)")
    ldm_loss_ddpm = model.forward_ldm(images, tokens, use_flow=False)
    print(f"  LDM DDPM loss: {ldm_loss_ddpm.item():.4f}")

    # ── Text encoding ─────────────────────────────────────────
    print("\n[6] Text encoding (hidden states)")
    ctx, null_ctx, ctx_mask = model._encode_text(tokens)
    print(f"  tokens{tokens.shape} → ctx{ctx.shape}  null_ctx{null_ctx.shape}")

    # ── Synthesis ─────────────────────────────────────────────
    print("\n[7] Synthesis (flow matching, 3 steps)")
    model.eval()
    img_out = model.synthesize(
        "a photo of a cat",
        n_steps=3,
        guidance_scale=7.5,
        use_flow=True,
        seed=42,
        device=device,
    )
    print(f"  output image: {img_out.shape}  range=[{img_out.min():.2f},{img_out.max():.2f}]")

    # ── Synthesis with DDIM ───────────────────────────────────
    print("\n[8] Synthesis (DDIM, 3 steps)")
    img_out2 = model.synthesize(
        "a landscape with mountains",
        n_steps=3,
        guidance_scale=3.0,
        use_flow=False,
        seed=42,
        device=device,
    )
    print(f"  output image: {img_out2.shape}")

    # ── Backward ──────────────────────────────────────────────
    print("\n[9] Backward pass (LDM loss)")
    model.train()
    loss = model.forward_ldm(images, tokens)
    loss.backward()
    print(f"  backward OK, loss={loss.item():.4f}")

    print("\n[OK] All T2I model tests passed")
