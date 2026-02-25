"""
TTS Chapter 5: Flow Matching Vocoder
======================================
Qwen3-TTS uses *Conditional Flow Matching* (CFM) to convert the discrete
codec representation / mel-spectrogram into a high-quality waveform.

Flow Matching overview:
  • Learn a time-dependent vector field  v_θ(x_t, t, cond)
    that transports a simple prior (Gaussian noise) to data.

  • Probability path:  x_t = (1 - t)·x_1 + t·x_0
      where  x_0 ~ N(0, I)  (noise)
             x_1  = data   (clean mel / waveform)
             t    ∈ [0, 1]

  • Conditional flow:  u_t(x_t | x_1) = (x_1 - x_t) / (1 - t)
    → optimal straight-line trajectory

  • Training objective:  E_t,x_0,x_1 [ ‖v_θ(x_t, t, cond) − u_t‖² ]

  • Inference: Euler ODE solver  x_{t+Δt} = x_t + Δt · v_θ(x_t, t, cond)
    starting from x_0 ~ N(0, I), integrate to t=1.

Architecture of v_θ:
  Mel / codec condition  ──┐
                           ├── [concat] → U-Net style transformer → v
  Noisy target x_t      ──┘
  Time embedding t      ──┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple


# ──────────────────────────────────────────────
# 1.  Time-step Sinusoidal Embedding
# ──────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """
    Embed scalar timestep t ∈ [0,1] into a vector.
    Uses sinusoidal encoding followed by a 2-layer MLP.
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
        """t: (B,) scalar in [0,1]  →  (B, dim)"""
        t   = t.float().unsqueeze(1)                          # (B, 1)
        arg = t * self.freqs.unsqueeze(0)                     # (B, half)
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        return self.mlp(emb)


# ──────────────────────────────────────────────
# 2.  Adaptive Layer Norm (with time/condition)
# ──────────────────────────────────────────────

class AdaLayerNorm(nn.Module):
    """
    AdaLN: modulate LayerNorm scale + shift with a condition vector.
    Used in DiT (Diffusion Transformers) and flow-matching models.
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm  = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj  = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, T, dim)
        cond : (B, cond_dim) — time embedding (+ optional other condition)
        """
        gamma, beta = self.proj(cond).chunk(2, dim=-1)   # (B, dim) each
        return self.norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# ──────────────────────────────────────────────
# 3.  Flow Matching Transformer Block
# ──────────────────────────────────────────────

class FlowBlock(nn.Module):
    """
    Transformer block conditioned on time + optional acoustic context.
    AdaLN-Zero initialization (γ=1, β=0 at start → identity residual).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 cond_dim: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.norm1   = AdaLayerNorm(d_model, cond_dim)
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2   = AdaLayerNorm(d_model, cond_dim)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Zero-init output projections (AdaLN-Zero)
        nn.init.zeros_(self.attn_out.weight)
        nn.init.zeros_(self.ffn[-1].weight)

    def forward(
        self,
        x:    torch.Tensor,         # (B, T, D)
        cond: torch.Tensor,         # (B, cond_dim)
        ctx:  Optional[torch.Tensor] = None,  # (B, T_ctx, D) cross-attn context
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        # Self-attention with AdaLN
        h = self.norm1(x, cond)
        qkv = self.qkv(h).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop(
            F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        )
        h = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.attn_out(h)

        # FFN with AdaLN
        x = x + self.ffn(self.norm2(x, cond))
        return x


# ──────────────────────────────────────────────
# 4.  Conditional Flow Matching Network (v_θ)
# ──────────────────────────────────────────────

class FlowMatchingNet(nn.Module):
    """
    Vector field estimator for Conditional Flow Matching.

    Takes:
      x_t   : noisy target   (B, n_mels, T)   — mel or waveform frame
      t     : timestep       (B,)  ∈ [0,1]
      cond  : acoustic cond  (B, n_mels, T)   — codec-decoded mel (speaker/content)

    Outputs:
      v_θ   : predicted velocity  same shape as x_t

    Architecture: lightweight DiT-style U-Net with skip connections
    """

    def __init__(
        self,
        n_mels:   int   = 80,
        d_model:  int   = 256,
        n_heads:  int   = 4,
        n_layers: int   = 6,
        d_ff:     int   = 1024,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        cond_dim = d_model

        # Time embedding
        self.time_embed  = TimestepEmbedding(cond_dim)

        # Input projections (noisy target + condition → D)
        self.x_proj   = nn.Conv1d(n_mels, d_model, kernel_size=1)
        self.ctx_proj = nn.Conv1d(n_mels, d_model, kernel_size=1)

        # Combine x and condition
        self.in_proj = nn.Conv1d(d_model * 2, d_model, kernel_size=1)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FlowBlock(d_model, n_heads, d_ff, cond_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output projection → velocity
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_mels),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(
        self,
        x_t:  torch.Tensor,        # (B, n_mels, T)
        t:    torch.Tensor,        # (B,)
        cond: torch.Tensor,        # (B, n_mels, T)
    ) -> torch.Tensor:
        # Time condition
        t_emb = self.time_embed(t)              # (B, D)

        # Project inputs
        x = self.x_proj(x_t)                   # (B, D, T)
        c = self.ctx_proj(cond)                 # (B, D, T)
        x = self.in_proj(torch.cat([x, c], dim=1))  # (B, D, T)
        x = x.permute(0, 2, 1)                 # (B, T, D)

        for blk in self.blocks:
            x = blk(x, t_emb)

        v = self.out_proj(x)                    # (B, T, n_mels)
        return v.permute(0, 2, 1)              # (B, n_mels, T)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# 5.  Flow Matching Training Loss
# ──────────────────────────────────────────────

def flow_matching_loss(
    model: FlowMatchingNet,
    x_1:  torch.Tensor,           # clean mel (B, n_mels, T)
    cond: torch.Tensor,           # conditioning mel (B, n_mels, T)
) -> torch.Tensor:
    """
    Compute CFM loss:
      1. Sample t ~ Uniform(0, 1)
      2. Sample x_0 ~ N(0, I)
      3. Interpolate: x_t = (1-t)*x_1 + t*x_0  (linear path)
      4. Target velocity: u_t = x_0 - x_1  (derivative of linear path)
      5. Loss = MSE(v_θ(x_t, t, cond), u_t)
    """
    B = x_1.shape[0]
    device = x_1.device

    # Sample noise and time
    x_0 = torch.randn_like(x_1)
    t   = torch.rand(B, device=device)

    # Interpolate (optimal transport path)
    t_broad = t.view(B, 1, 1)
    x_t     = (1 - t_broad) * x_1 + t_broad * x_0

    # Target: velocity of linear path = dx_t/dt = x_0 - x_1
    u_t = x_0 - x_1

    # Predict velocity
    v_pred = model(x_t, t, cond)

    return F.mse_loss(v_pred, u_t)


# ──────────────────────────────────────────────
# 6.  ODE Solvers for Inference
# ──────────────────────────────────────────────

@torch.no_grad()
def euler_solve(
    model:   FlowMatchingNet,
    cond:    torch.Tensor,           # (B, n_mels, T)
    n_steps: int = 50,
    device:  Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Simple Euler ODE solver.
    Integrates  dx/dt = v_θ(x, t, cond)  from t=0 to t=1.

    Returns clean mel (B, n_mels, T).
    """
    if device is None:
        device = cond.device

    # Start from noise
    x = torch.randn_like(cond)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((cond.shape[0],), i / n_steps,
                       dtype=torch.float32, device=device)
        v = model(x, t, cond)
        x = x + dt * v

    return x


@torch.no_grad()
def midpoint_solve(
    model:   FlowMatchingNet,
    cond:    torch.Tensor,
    n_steps: int = 20,
) -> torch.Tensor:
    """
    2nd-order midpoint / Heun solver — 2× fewer steps for same quality.
    """
    x  = torch.randn_like(cond)
    dt = 1.0 / n_steps
    B  = cond.shape[0]

    for i in range(n_steps):
        t0 = torch.full((B,), i / n_steps,
                        dtype=torch.float32, device=cond.device)
        t1 = torch.full((B,), (i + 1) / n_steps,
                        dtype=torch.float32, device=cond.device)

        v0  = model(x, t0, cond)
        x_m = x + 0.5 * dt * v0                # midpoint
        v_m = model(x_m, (t0 + t1) / 2, cond)
        x   = x + dt * v_m

    return x


# ──────────────────────────────────────────────
# 7.  Griffin-Lim Vocoder (fast baseline, no NN)
# ──────────────────────────────────────────────

class GriffinLim(nn.Module):
    """
    Classic Griffin-Lim algorithm: estimate waveform from magnitude spectrogram.
    No parameters — used as a fast baseline for mel → waveform.

    Note: requires n_fft / hop_length consistent with the MelSpectrogram extractor.
    """

    def __init__(
        self,
        n_fft:      int = 1024,
        hop_length: int = 256,
        n_mels:     int = 80,
        sample_rate: int = 24_000,
        n_iter:     int = 60,
    ):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.n_mels     = n_mels
        self.n_iter     = n_iter
        self.register_buffer("window", torch.hann_window(n_fft))

        # Inverse mel filterbank (pseudo-inverse)
        import math
        fb = self._mel_filterbank(sample_rate, n_fft, n_mels, 0.0, 8000.0).float()
        self.register_buffer("inv_fb", fb.T)   # (n_fft//2+1, n_mels)

    @staticmethod
    def _hz_to_mel(f): return 2595 * math.log10(1 + f / 700)
    @staticmethod
    def _mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)

    def _mel_filterbank(self, sr, n_fft, n_mels, f_min, f_max):
        import math
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts  = torch.tensor([self._mel_to_hz(m.item()) for m in mel_pts])
        bins    = torch.floor((n_fft + 1) * hz_pts / sr).long()
        fb      = torch.zeros(n_mels, n_fft // 2 + 1)
        for m in range(1, n_mels + 1):
            for k in range(bins[m-1], bins[m]):
                if bins[m] > bins[m-1]:
                    fb[m-1, k] = (k - bins[m-1]) / (bins[m] - bins[m-1])
            for k in range(bins[m], bins[m+1]):
                if bins[m+1] > bins[m]:
                    fb[m-1, k] = (bins[m+1] - k) / (bins[m+1] - bins[m])
        return fb

    def mel_to_linear(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, n_mels, T) → (B, n_fft//2+1, T) linear magnitude."""
        inv = self.inv_fb.to(mel.dtype)
        return torch.clamp(torch.einsum("fm,bmt->bft", inv, mel), min=0)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, n_mels, T)  (log-mel)
        Returns waveform (B, T_samples)
        """
        # Convert log-mel back to linear magnitude
        linear = self.mel_to_linear(mel.exp())   # (B, n_fft//2+1, T)

        # Run Griffin-Lim on each sample
        wavs = []
        for i in range(linear.shape[0]):
            mag = linear[i]                      # (F, T)
            # Random initial phase
            angles = torch.rand_like(mag) * 2 * math.pi
            spec   = mag * torch.exp(1j * angles)

            for _ in range(self.n_iter):
                wav = torch.istft(
                    spec, self.n_fft, self.hop_length,
                    window=self.window
                )
                stft = torch.stft(
                    wav, self.n_fft, self.hop_length,
                    window=self.window, return_complex=True
                )
                # Impose magnitude
                phase = stft / (stft.abs() + 1e-8)
                spec  = mag * phase

            wav = torch.istft(spec, self.n_fft, self.hop_length, window=self.window)
            wavs.append(wav)
        return torch.stack(wavs)


# ──────────────────────────────────────────────
# 8.  Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("FLOW MATCHING VOCODER — Tests")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import math

    B, n_mels, T = 2, 80, 50

    # ── Timestep Embedding ───────────────────────────────────
    print("\n[1] TimestepEmbedding")
    te  = TimestepEmbedding(dim=128).to(device)
    t   = torch.rand(B, device=device)
    emb = te(t)
    print(f"  t: {t.shape}  →  emb: {emb.shape}")

    # ── FlowMatchingNet ──────────────────────────────────────
    print("\n[2] FlowMatchingNet forward")
    fm = FlowMatchingNet(
        n_mels=n_mels, d_model=64, n_heads=4, n_layers=2, d_ff=256
    ).to(device)
    x_t  = torch.randn(B, n_mels, T, device=device)
    cond = torch.randn(B, n_mels, T, device=device)
    v    = fm(x_t, t, cond)
    print(f"  x_t: {x_t.shape}  cond: {cond.shape}  →  v: {v.shape}")
    print(f"  params: {fm.num_parameters():,}")

    # ── Training loss ────────────────────────────────────────
    print("\n[3] Flow Matching loss")
    clean = torch.randn(B, n_mels, T, device=device)
    loss  = flow_matching_loss(fm, clean, cond)
    print(f"  loss: {loss.item():.4f}")

    # ── Euler ODE solve ──────────────────────────────────────
    print("\n[4] Euler ODE solve (10 steps)")
    fm.eval()
    generated = euler_solve(fm, cond, n_steps=10)
    print(f"  generated mel: {generated.shape}")

    # ── Midpoint solver ─────────────────────────────────────
    print("\n[5] Midpoint solve (5 steps)")
    gen2 = midpoint_solve(fm, cond, n_steps=5)
    print(f"  generated mel: {gen2.shape}")

    # ── Griffin-Lim ─────────────────────────────────────────
    print("\n[6] Griffin-Lim vocoder")
    gl  = GriffinLim(n_fft=512, hop_length=128, n_mels=n_mels, n_iter=5)
    log_mel = torch.randn(2, n_mels, T)                # fake log-mel
    wav     = gl(log_mel)
    print(f"  mel: {log_mel.shape}  →  waveform: {wav.shape}")
    print(f"  Duration at 24kHz: {wav.shape[1]/24000:.3f}s")
