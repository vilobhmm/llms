"""
T2I Chapter 4: Noise Scheduler (DDPM + DDIM + Flow Matching)
=============================================================
The noise scheduler defines how clean data is corrupted during training (forward
process) and how it is gradually denoised during inference (reverse process).

We implement three variants:

1. DDPM (Ho et al., 2020) — Denoising Diffusion Probabilistic Models
   • Linear or cosine beta schedule
   • Forward: q(x_t | x_0) = N(sqrt(ᾱ_t)*x_0, (1-ᾱ_t)*I)
   • Reverse: DDPM sampler (stochastic)

2. DDIM (Song et al., 2021) — Denoising Diffusion Implicit Models
   • Same training as DDPM, deterministic inference
   • Allows fast sampling with fewer steps (50 → 10)
   • Deterministic: η=0, Stochastic: η=1 (recovers DDPM)

3. Flow Matching (Lipman et al., 2022; Liu et al., 2022)
   • Straight-line (OT) path between noise and data
   • x_t = (1-t)*x_0 + t*x_1  where x_0=noise, x_1=data
   • Target velocity: v = x_1 - x_0
   • Simpler, easier to train, faster inference (used in FLUX)

Classifier-Free Guidance (CFG):
   • During training: randomly drop text conditioning with p_uncond
   • At inference: v_cfg = v_uncond + s*(v_cond - v_uncond)
   • Larger guidance scale → more text-adherent but less diverse outputs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable


# ─────────────────────────────────────────────────────────────
# 1.  Beta Schedules
# ─────────────────────────────────────────────────────────────

def linear_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end:   float = 2e-2,
) -> torch.Tensor:
    """
    Linear beta schedule (Ho et al., 2020).
    beta_t increases linearly from beta_start to beta_end.
    T=1000 steps is standard; use fewer steps for fast testing.
    """
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(
    T: int,
    s: float = 8e-3,
) -> torch.Tensor:
    """
    Cosine beta schedule (Nichol & Dhariwal, 2021).
    Avoids too-fast noise injection early on.

    α̅_t = cos((t/T + s) / (1 + s) * π/2)^2 / α̅_0
    β_t = 1 - α̅_t / α̅_{t-1}   clipped to [0, 0.999]
    """
    t     = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((t / T) + s) / (1 + s) * math.pi / 2
    ).pow(2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999).float()


# ─────────────────────────────────────────────────────────────
# 2.  DDPM Scheduler
# ─────────────────────────────────────────────────────────────

class DDPMScheduler:
    """
    DDPM forward and reverse process.

    Pre-computes all schedule quantities:
      β_t, α_t = 1-β_t, ᾱ_t = Π α_t, √ᾱ_t, √(1-ᾱ_t), etc.

    Forward process (training):
      q(x_t | x_0) = N(√ᾱ_t * x_0,  (1-ᾱ_t) * I)
      x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε,  ε ~ N(0,I)

    Reverse process (inference, DDPM):
      p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), σ_t^2 * I)
      μ_θ = (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * ε_θ)
      σ_t^2 = β_t  (or  β̃_t = β_t*(1-ᾱ_{t-1})/(1-ᾱ_t))
    """

    def __init__(
        self,
        T:           int   = 1000,
        schedule:    str   = "cosine",
        beta_start:  float = 1e-4,
        beta_end:    float = 2e-2,
        clip_denoised: bool = True,
    ):
        self.T           = T
        self.clip_denoised = clip_denoised

        # Beta schedule
        if schedule == "linear":
            betas = linear_beta_schedule(T, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas                           # (T,)

        # Derived quantities (all as tensors, same device as inputs at call time)
        alphas          = 1.0 - betas
        alphas_cumprod  = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register("alphas",              alphas)
        self.register("alphas_cumprod",      alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register("sqrt_alphas_cumprod",      alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register("log_one_minus_alphas_cumprod",  (1 - alphas_cumprod).log())
        self.register("sqrt_recip_alphas_cumprod",  (1.0 / alphas_cumprod).sqrt())
        self.register("sqrt_recipm1_alphas_cumprod", (1.0 / alphas_cumprod - 1).sqrt())

        # Posterior variance β̃_t
        posterior_var = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register("posterior_var",    posterior_var.clamp(min=1e-20))
        self.register("posterior_log_var_clipped",
                      posterior_var.clamp(min=1e-20).log())

    def register(self, name: str, val: torch.Tensor):
        setattr(self, name, val)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Index array `arr` at timestep `t` and broadcast to `shape`."""
        arr  = arr.to(t.device)
        vals = arr[t]                                        # (B,)
        return vals.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    # ── Forward process ──────────────────────────────────────

    def q_sample(
        self,
        x_0: torch.Tensor,    # (B, C, H, W) — clean latent
        t:   torch.Tensor,    # (B,) — integer timesteps in [0, T-1]
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0):
          x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε

        Returns (x_t, noise).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_a   = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_1ma = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t = sqrt_a * x_0 + sqrt_1ma * noise
        return x_t, noise

    # ── Reverse process (DDPM) ───────────────────────────────

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t:   torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct x_0 from x_t and predicted noise."""
        coef1 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        coef2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return coef1 * x_t - coef2 * noise_pred

    def q_posterior_mean(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t:   torch.Tensor,
    ) -> torch.Tensor:
        """Mean of q(x_{t-1} | x_t, x_0)."""
        coef1 = self._extract(
            self.betas * self.alphas_cumprod_prev.sqrt() / (1 - self.alphas_cumprod),
            t, x_0.shape
        )
        coef2 = self._extract(
            self.alphas.sqrt() * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod),
            t, x_t.shape
        )
        return coef1 * x_0 + coef2 * x_t

    @torch.no_grad()
    def p_sample(
        self,
        model_fn: Callable,       # (x_t, t, ctx) → noise_pred
        x_t:      torch.Tensor,
        t:        torch.Tensor,   # integer scalar or (B,) tensor
        ctx:      torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Single DDPM reverse step: sample x_{t-1} given x_t.
        """
        B = x_t.shape[0]
        if t.ndim == 0:
            t = t.expand(B)

        noise_pred = model_fn(x_t, t.float() / self.T, ctx, ctx_mask)
        x_0_pred   = self.predict_x0_from_noise(x_t, t, noise_pred)

        if clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)

        mu    = self.q_posterior_mean(x_0_pred, x_t, t)
        var   = self._extract(self.posterior_var, t, x_t.shape)
        noise = torch.randn_like(x_t) if (t > 0).any() else torch.zeros_like(x_t)
        return mu + var.sqrt() * noise

    @torch.no_grad()
    def ddpm_sample(
        self,
        model_fn: Callable,
        shape:    tuple,
        ctx:      torch.Tensor,
        device:   torch.device,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full DDPM reverse chain: sample x_0 from noise."""
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.tensor([i], device=device)
            x = self.p_sample(model_fn, x, t, ctx, ctx_mask)
        return x

    # ── Training loss ────────────────────────────────────────

    def training_loss(
        self,
        model_fn: Callable,      # (x_t, t_frac, ctx) → noise_pred
        x_0:      torch.Tensor,  # (B, C, H, W) — clean latent
        ctx:      torch.Tensor,  # text context
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDPM training loss:
          1. Sample t ~ Uniform(0, T-1)
          2. Sample noise ε ~ N(0, I)
          3. Compute x_t via forward process
          4. Predict noise ε_θ(x_t, t)
          5. Loss = MSE(ε_θ, ε)
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x_0.device)
        x_t, noise = self.q_sample(x_0, t)
        t_frac     = t.float() / self.T
        noise_pred = model_fn(x_t, t_frac, ctx, ctx_mask)
        return F.mse_loss(noise_pred, noise)


# ─────────────────────────────────────────────────────────────
# 3.  DDIM Sampler (fast deterministic inference)
# ─────────────────────────────────────────────────────────────

class DDIMSampler:
    """
    DDIM sampler (Song et al., 2021).
    Uses the same pre-trained DDPM model but samples in far fewer steps.

    η = 0: fully deterministic (ODE)
    η = 1: stochastic (recovers DDPM)

    With N_steps = 10-50 instead of 1000, quality is nearly identical.
    """

    def __init__(
        self,
        scheduler: DDPMScheduler,
        n_steps:   int   = 50,
        eta:       float = 0.0,
    ):
        self.scheduler = scheduler
        self.n_steps   = n_steps
        self.eta       = eta

        # Sub-sample timesteps (evenly spaced)
        T = scheduler.T
        step_size  = T // n_steps
        self.timesteps = list(reversed(range(0, T, step_size)))[:n_steps]

    @torch.no_grad()
    def sample(
        self,
        model_fn:  Callable,     # (x_t, t_frac, ctx) → noise_pred
        shape:     tuple,
        ctx:       torch.Tensor,
        device:    torch.device,
        ctx_mask:  Optional[torch.Tensor] = None,
        verbose:   bool = False,
    ) -> List[torch.Tensor]:
        """
        DDIM reverse chain. Returns list of intermediate states
        (for trajectory visualization).
        """
        sch = self.scheduler
        x = torch.randn(shape, device=device)
        trajectory = [x.clone()]

        for i, t_val in enumerate(self.timesteps):
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else -1
            B = x.shape[0]
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Noise prediction
            t_frac     = t_tensor.float() / sch.T
            noise_pred = model_fn(x, t_frac, ctx, ctx_mask)

            # Reconstruct x_0
            x_0_pred = sch.predict_x0_from_noise(x, t_tensor, noise_pred)
            x_0_pred = x_0_pred.clamp(-1, 1)

            # Get schedule values
            at  = sch.alphas_cumprod[t_val].to(device)
            at_ = sch.alphas_cumprod[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0, device=device)

            sigma = self.eta * ((1 - at_) / (1 - at)).sqrt() * (1 - at / at_).sqrt()

            # DDIM update
            c1 = at_.sqrt()
            c2 = (1 - at_ - sigma ** 2).clamp(min=0).sqrt()
            x  = c1 * x_0_pred + c2 * noise_pred
            if self.eta > 0 and t_prev >= 0:
                x = x + sigma * torch.randn_like(x)

            if verbose and i % 10 == 0:
                print(f"  DDIM step {i}/{self.n_steps}")

            trajectory.append(x.clone())

        return trajectory


# ─────────────────────────────────────────────────────────────
# 4.  Flow Matching Scheduler
# ─────────────────────────────────────────────────────────────

class FlowMatchingScheduler:
    """
    Conditional Flow Matching (CFM) scheduler.

    Used in FLUX and many modern T2I models as a simpler alternative to DDPM.

    Probability path (optimal transport / straight-line):
      x_t = (1 - t)*x_0 + t*x_1      where t ∈ [0, 1]
      x_0 ~ N(0, I)   (noise)
      x_1 = data      (clean latent)

    Target velocity at time t:
      v*(x_t | x_0, x_1) = x_1 - x_0   (constant along path)

    Training:
      v_θ(x_t, t, cond) ≈ v*  →  MSE loss

    Inference (Euler ODE):
      x_{t+Δt} = x_t + Δt * v_θ(x_t, t, cond)
      Integrate from t=0 to t=1.
    """

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def sample_path(
        self,
        x_1:  torch.Tensor,    # (B, C, H, W) — clean data
        t:    torch.Tensor,    # (B,) in [0, 1]
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t on the flow path and compute target velocity.

        Returns (x_t, v_target)
        """
        if noise is None:
            noise = torch.randn_like(x_1)

        t_broad  = t.reshape(t.shape[0], *([1] * (x_1.ndim - 1)))
        x_t      = (1 - t_broad) * noise + t_broad * x_1
        v_target = x_1 - noise         # constant velocity along straight path

        return x_t, v_target

    def training_loss(
        self,
        model_fn: Callable,     # (x_t, t, ctx, ctx_mask) → v_pred
        x_1:      torch.Tensor,
        ctx:      torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flow matching training loss.
          1. Sample t ~ Uniform(0, 1)
          2. Sample x_0 ~ N(0, I)
          3. Compute x_t = (1-t)*x_0 + t*x_1
          4. Predict v_θ(x_t, t)
          5. Loss = MSE(v_θ, x_1 - x_0)
        """
        B = x_1.shape[0]
        t = torch.rand(B, device=x_1.device)
        x_t, v_target = self.sample_path(x_1, t)
        v_pred = model_fn(x_t, t, ctx, ctx_mask)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def euler_sample(
        self,
        model_fn: Callable,
        shape:    tuple,
        ctx:      torch.Tensor,
        device:   torch.device,
        n_steps:  int = 50,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Euler ODE solver: integrate from t=0 (noise) to t=1 (data).
        Returns list of trajectory snapshots.
        """
        x = torch.randn(shape, device=device)
        dt = 1.0 / n_steps
        trajectory = [x.clone()]

        for i in range(n_steps):
            t_val = torch.full((x.shape[0],), i / n_steps,
                               dtype=torch.float32, device=device)
            v = model_fn(x, t_val, ctx, ctx_mask)
            x = x + dt * v
            trajectory.append(x.clone())

        return trajectory

    @torch.no_grad()
    def heun_sample(
        self,
        model_fn: Callable,
        shape:    tuple,
        ctx:      torch.Tensor,
        device:   torch.device,
        n_steps:  int = 20,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Heun (2nd-order) ODE solver — better accuracy with fewer steps.
        """
        x  = torch.randn(shape, device=device)
        dt = 1.0 / n_steps
        B  = x.shape[0]

        for i in range(n_steps):
            t0 = torch.full((B,), i / n_steps, dtype=torch.float32, device=device)
            t1 = torch.full((B,), (i + 1) / n_steps, dtype=torch.float32, device=device)

            v0  = model_fn(x, t0, ctx, ctx_mask)
            x1_ = x + dt * v0                        # Euler step
            v1  = model_fn(x1_, t1, ctx, ctx_mask)
            x   = x + 0.5 * dt * (v0 + v1)          # corrector

        return x


# ─────────────────────────────────────────────────────────────
# 5.  Utility: schedule visualization data
# ─────────────────────────────────────────────────────────────

def get_schedule_curves(
    T: int = 1000,
    schedule: str = "cosine",
) -> dict:
    """
    Return alpha_t and sigma_t curves for plotting.

    alpha_t = sqrt(alphas_cumprod)    — signal coefficient
    sigma_t = sqrt(1-alphas_cumprod)  — noise coefficient
    """
    sch = DDPMScheduler(T=T, schedule=schedule)
    return {
        "t":       torch.arange(T).float() / T,
        "alpha_t": sch.sqrt_alphas_cumprod,
        "sigma_t": sch.sqrt_one_minus_alphas_cumprod,
        "beta_t":  sch.betas,
        "snr_t":   (sch.sqrt_alphas_cumprod / sch.sqrt_one_minus_alphas_cumprod).log(),
    }


# ─────────────────────────────────────────────────────────────
# 6.  Smoke-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DDPM / DDIM / FLOW MATCHING SCHEDULERS — Smoke Tests")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    B = 2
    C = 4
    H = W = 4    # tiny latent for speed

    # ── Beta Schedules ────────────────────────────────────────
    print("\n[1] Beta schedules")
    b_lin = linear_beta_schedule(1000)
    b_cos = cosine_beta_schedule(1000)
    print(f"  linear betas:  min={b_lin.min():.5f}  max={b_lin.max():.5f}")
    print(f"  cosine betas:  min={b_cos.min():.5f}  max={b_cos.max():.5f}")

    # ── DDPM Scheduler ────────────────────────────────────────
    print("\n[2] DDPMScheduler")
    sch = DDPMScheduler(T=100, schedule="cosine")
    x_0 = torch.randn(B, C, H, W, device=device)
    t   = torch.randint(0, 100, (B,), device=device)
    x_t, noise = sch.q_sample(x_0, t)
    print(f"  forward: x_0{x_0.shape} → x_t{x_t.shape}  noise{noise.shape}")
    assert x_t.shape == x_0.shape

    # Dummy model for testing
    def dummy_model(x, t_frac, ctx, mask=None):
        return torch.zeros_like(x)

    ctx = torch.randn(B, 5, 64, device=device)
    loss = sch.training_loss(dummy_model, x_0, ctx)
    print(f"  DDPM training loss: {loss.item():.4f}")

    # ── DDIM Sampler ──────────────────────────────────────────
    print("\n[3] DDIM Sampler (10 steps)")
    ddim = DDIMSampler(sch, n_steps=10, eta=0.0)
    traj = ddim.sample(dummy_model, (B, C, H, W), ctx, device)
    print(f"  trajectory length: {len(traj)}  final shape: {traj[-1].shape}")

    # ── Flow Matching Scheduler ───────────────────────────────
    print("\n[4] FlowMatchingScheduler")
    fm = FlowMatchingScheduler()

    t_rand = torch.rand(B, device=device)
    x_t_fm, v_tgt = fm.sample_path(x_0, t_rand)
    print(f"  flow path: x_0{x_0.shape} → x_t{x_t_fm.shape}  v*{v_tgt.shape}")

    def dummy_velocity(x, t, ctx, mask=None):
        return torch.zeros_like(x)

    fm_loss = fm.training_loss(dummy_velocity, x_0, ctx)
    print(f"  flow matching training loss: {fm_loss.item():.4f}")

    # Euler sampling
    traj_fm = fm.euler_sample(dummy_velocity, (B, C, H, W), ctx, device, n_steps=10)
    print(f"  euler trajectory: {len(traj_fm)} steps, final{traj_fm[-1].shape}")

    # Heun sampling
    x_heun = fm.heun_sample(dummy_velocity, (B, C, H, W), ctx, device, n_steps=5)
    print(f"  heun sample: {x_heun.shape}")

    # ── Schedule curves ───────────────────────────────────────
    print("\n[5] Schedule visualization data")
    curves = get_schedule_curves(T=100, schedule="cosine")
    for k, v in curves.items():
        print(f"  {k}: shape={v.shape}  min={v.min():.4f}  max={v.max():.4f}")

    print("\n[OK] All scheduler tests passed")
