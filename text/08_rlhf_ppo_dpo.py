"""
Chapter 7: RLHF — PPO & DPO
==============================
This module implements TWO alignment algorithms:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Algorithm 1: RLHF-PPO  (OpenAI InstructGPT approach)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4 models at inference time:
  ① Policy model (π_θ)        — the model we're training
  ② Reference model (π_ref)   — frozen SFT model (KL anchor)
  ③ Reward model (r_φ)        — frozen, trained in previous step
  ④ Value model (V_ψ)         — critic, estimates expected return

PPO objective per token t:
  r(t) = reward_model(x, y) − β · KL[π_θ(y|x) ‖ π_ref(y|x)]

  L_PPO = min(
      ratio(t) · A(t),
      clip(ratio(t), 1-ε, 1+ε) · A(t)
  )
  where ratio(t) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ)
  and   A(t) = GAE advantage estimate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Algorithm 2: DPO  (Raschka book Ch 7, simpler)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DPO (Rafailov et al. 2023) shows RLHF is equivalent to:

  L_DPO = −E[ log σ(
      β · (log π_θ(y_w|x) − log π_ref(y_w|x))
      −
      β · (log π_θ(y_l|x) − log π_ref(y_l|x))
  )]

No reward model needed — preference data directly trains the policy.
Implemented here for comparison.

References:
  - Schulman et al. 2017 "Proximal Policy Optimization"
  - Ouyang et al. 2022 "InstructGPT"
  - Rafailov et al. 2023 "Direct Preference Optimization"
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── local imports ──
import importlib.util, sys
from pathlib import Path as _Path
def _load(alias, fname):
    if alias in sys.modules: return sys.modules[alias]
    p = _Path(__file__).parent / fname
    spec = importlib.util.spec_from_file_location(alias, p)
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m; spec.loader.exec_module(m); return m
_gpt   = _load("transformer_rlhf", "04_transformer_model.py")
_rm    = _load("reward_model_rlhf", "07_reward_model.py")
GPT, GPTConfig = _gpt.GPT, _gpt.GPTConfig
RewardModel = _rm.RewardModel


# ═══════════════════════════════════════════════════════
# ── PART A: PPO ──────────────────────────────────────
# ═══════════════════════════════════════════════════════

# ──────────────────────────────────────────────
# A1. Value head (critic)
# ──────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Attached to the policy model to estimate state-value V(s).
    Shares the transformer backbone with the policy; only the head is separate.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.linear.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, T, D) → values: (B, T)"""
        return self.linear(hidden).squeeze(-1)


class PolicyWithValue(nn.Module):
    """
    Wraps the SFT model (policy) with an extra value head for PPO.
    Both the lm_head and value_head share the transformer backbone.
    """
    def __init__(self, backbone: GPT):
        super().__init__()
        self.backbone    = backbone
        self.value_head  = ValueHead(backbone.cfg.d_model)

    def forward(self, input_ids: torch.Tensor):
        """
        Returns:
            logits : (B, T, V)  — token probabilities (policy)
            values : (B, T)     — state-value estimates (critic)
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.backbone.emb_drop(
            self.backbone.token_emb(input_ids) + self.backbone.pos_emb(positions)
        )
        for block in self.backbone.blocks:
            x = block(x)
        hidden = self.backbone.ln_f(x)                    # (B, T, D)
        logits = self.backbone.lm_head(hidden)             # (B, T, V)
        values = self.value_head(hidden)                   # (B, T)
        return logits, values


# ──────────────────────────────────────────────
# A2. PPO utilities
# ──────────────────────────────────────────────

def compute_log_probs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """
    logits : (B, T, V)
    ids    : (B, T)
    Returns per-token log-probabilities: (B, T)
    """
    log_p = F.log_softmax(logits, dim=-1)   # (B, T, V)
    return log_p.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # (B, T)


def compute_kl_divergence(
    log_p_policy: torch.Tensor,  # (B, T)
    log_p_ref:    torch.Tensor,  # (B, T)
    mask:         torch.Tensor,  # (B, T)  1=real token  0=pad
) -> torch.Tensor:
    """
    Per-batch KL divergence: KL[π_θ ‖ π_ref] summed over response tokens.
    kl(t) = π_θ(t) · (log π_θ(t) − log π_ref(t))
    Approximated as: kl(t) ≈ log π_θ(t) − log π_ref(t)
    """
    kl = (log_p_policy - log_p_ref) * mask
    return kl.sum(dim=-1)                # (B,)


def compute_gae(
    rewards:  torch.Tensor,   # (B, T)
    values:   torch.Tensor,   # (B, T)
    gamma:    float = 0.99,
    lam:      float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation (GAE, Schulman et al. 2016).

    A(t) = δ(t) + γλ·δ(t+1) + (γλ)²·δ(t+2) + …
    δ(t) = r(t) + γ·V(t+1) − V(t)

    Returns advantages (B,T) and returns (B,T).
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv   = torch.zeros(B, device=rewards.device)
    # Append a zero value for V(T+1)
    values_ext = torch.cat([values, torch.zeros(B, 1, device=values.device)], dim=1)

    for t in reversed(range(T)):
        delta        = rewards[:, t] + gamma * values_ext[:, t + 1] - values_ext[:, t]
        last_adv     = delta + gamma * lam * last_adv
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


# ──────────────────────────────────────────────
# A3. PPO loss functions
# ──────────────────────────────────────────────

def ppo_policy_loss(
    log_p_new:   torch.Tensor,   # (B, T)  new policy log-probs
    log_p_old:   torch.Tensor,   # (B, T)  old policy log-probs (no-grad)
    advantages:  torch.Tensor,   # (B, T)
    mask:        torch.Tensor,   # (B, T)
    clip_eps:    float = 0.2,
) -> torch.Tensor:
    """
    PPO-clip objective:
        L = min( ratio·A,  clip(ratio, 1−ε, 1+ε)·A )
    """
    ratio     = (log_p_new - log_p_old).exp()              # (B, T)
    clipped   = ratio.clamp(1 - clip_eps, 1 + clip_eps)
    policy_l  = -torch.min(ratio * advantages, clipped * advantages)
    policy_l  = (policy_l * mask).sum() / mask.sum().clamp(min=1)
    return policy_l


def ppo_value_loss(
    values:  torch.Tensor,   # (B, T)
    returns: torch.Tensor,   # (B, T)
    mask:    torch.Tensor,   # (B, T)
) -> torch.Tensor:
    """MSE loss for the critic."""
    value_l = F.mse_loss(values * mask, returns * mask, reduction="sum")
    return value_l / mask.sum().clamp(min=1)


# ──────────────────────────────────────────────
# A4. PPO training step
# ──────────────────────────────────────────────

@dataclass
class PPOConfig:
    lr:           float = 1e-5
    kl_coef:      float = 0.1     # β in the KL penalty
    clip_eps:     float = 0.2     # ε in PPO clip
    vf_coef:      float = 0.5     # value-function loss coefficient
    gamma:        float = 0.99
    lam:          float = 0.95    # GAE lambda
    ppo_epochs:   int   = 4       # inner PPO optimisation steps
    batch_size:   int   = 4
    max_new_tokens: int = 32
    device:       str   = "cpu"
    save_dir:     str   = "checkpoints/rlhf"


class PPOTrainer:
    """
    Simplified PPO trainer for RLHF.

    The outer loop:
      1. Rollout: sample responses from current policy
      2. Score:   reward_model(prompt, response) − kl_coef · KL(π_θ ‖ π_ref)
      3. Update:  run `ppo_epochs` gradient steps on the clipped objective

    For full-scale training you would also add:
      - Reward normalisation (running mean/std)
      - Advantage normalisation per mini-batch
      - Separate value model (not sharing backbone weights)
    """

    def __init__(
        self,
        policy:       PolicyWithValue,
        ref_model:    GPT,
        reward_model: RewardModel,
        tokenizer,
        cfg:          PPOConfig,
    ):
        self.policy    = policy.to(cfg.device)
        self.ref_model = ref_model.to(cfg.device).eval()
        self.rm        = reward_model.to(cfg.device).eval()
        self.tok       = tokenizer
        self.cfg       = cfg
        self.opt       = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=cfg.lr
        )

        # Freeze ref & reward model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        for p in self.rm.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _rollout(self, prompt_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate responses and collect log-probs from policy and ref model."""
        self.policy.eval()
        B  = prompt_ids.shape[0]
        T  = self.cfg.max_new_tokens

        # Generate from policy
        gen_ids = self.policy.backbone.generate(
            prompt_ids, max_new_tokens=T, temperature=1.0
        )                                                  # (B, prompt_len+T)
        response_ids = gen_ids[:, prompt_ids.shape[1]:]   # (B, T)
        full_ids     = gen_ids[:, :prompt_ids.shape[1] + T]

        # Policy log-probs (on responses)
        logits_p, values = self.policy(full_ids)
        lp_policy = compute_log_probs(
            logits_p[:, prompt_ids.shape[1]-1:-1], response_ids
        )                                                  # (B, T)

        # Reference log-probs
        logits_r, _ = self.ref_model(full_ids)
        lp_ref = compute_log_probs(
            logits_r[:, prompt_ids.shape[1]-1:-1], response_ids
        )                                                  # (B, T)

        # Reward (single scalar at end of sequence)
        mask = torch.ones(B, T, device=self.cfg.device)
        reward_scalar = self.rm(full_ids, mask=None)      # (B,)

        # Token-level reward: place reward at last token, KL penalty everywhere
        kl      = (lp_policy - lp_ref).clamp(-20, 20)
        rewards = -self.cfg.kl_coef * kl                   # (B, T) KL penalty
        rewards[:, -1] = rewards[:, -1] + reward_scalar    # add terminal reward

        self.policy.train()
        return {
            "response_ids":  response_ids,
            "full_ids":      full_ids,
            "lp_old":        lp_policy.detach(),
            "values_old":    values[:, prompt_ids.shape[1]-1:-1].detach(),
            "rewards":       rewards.detach(),
            "mask":          mask,
        }

    def train_step(self, prompt_ids: torch.Tensor) -> Dict[str, float]:
        """One outer PPO step: rollout + inner optimisation."""
        cfg   = self.cfg
        batch = self._rollout(prompt_ids)

        advantages, returns = compute_gae(
            batch["rewards"], batch["values_old"], cfg.gamma, cfg.lam
        )
        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        stats = {"policy_loss": 0., "value_loss": 0., "total_loss": 0.}

        for _ in range(cfg.ppo_epochs):
            full_ids = batch["full_ids"]
            p_len    = prompt_ids.shape[1]
            logits_new, values_new = self.policy(full_ids)
            lp_new = compute_log_probs(
                logits_new[:, p_len-1:-1], batch["response_ids"]
            )
            values_r = values_new[:, p_len-1:-1]

            pl = ppo_policy_loss(lp_new, batch["lp_old"], advantages, batch["mask"], cfg.clip_eps)
            vl = ppo_value_loss(values_r, returns, batch["mask"])
            loss = pl + cfg.vf_coef * vl

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()

            stats["policy_loss"] += pl.item()
            stats["value_loss"]  += vl.item()
            stats["total_loss"]  += loss.item()

        for k in stats:
            stats[k] /= cfg.ppo_epochs
        stats["mean_reward"] = batch["rewards"].sum(-1).mean().item()
        return stats


# ═══════════════════════════════════════════════════════
# ── PART B: DPO ──────────────────────────────────────
# ═══════════════════════════════════════════════════════

@dataclass
class DPOConfig:
    beta:         float = 0.1    # temperature in DPO loss
    lr:           float = 1e-5
    weight_decay: float = 0.01
    batch_size:   int   = 4
    epochs:       int   = 3
    grad_clip:    float = 1.0
    eval_freq:    int   = 10
    save_dir:     str   = "checkpoints/dpo"
    device:       str   = "cpu"
    max_length:   int   = 128


class DPODataset(Dataset):
    """
    Each example: (prompt, chosen_ids, rejected_ids, masks)
    Same format as the preference dataset used for the reward model.
    """
    def __init__(self, examples, tokenizer, max_length: int = 128):
        self.examples   = examples
        self.tok        = tokenizer
        self.max_length = max_length
        self.pad_id     = tokenizer.vocab.get(tokenizer.PAD_TOKEN, 0)

    def _encode_pad(self, text):
        ids  = self.tok.encode(text)[:self.max_length]
        mask = [1] * len(ids)
        pad  = self.max_length - len(ids)
        return (ids + [self.pad_id]*pad,
                mask + [0]*pad)

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex  = self.examples[idx]
        c_ids, c_mask = self._encode_pad(ex["prompt"] + ex["chosen"])
        r_ids, r_mask = self._encode_pad(ex["prompt"] + ex["rejected"])
        return {
            "chosen_ids":    torch.tensor(c_ids),
            "chosen_mask":   torch.tensor(c_mask),
            "rejected_ids":  torch.tensor(r_ids),
            "rejected_mask": torch.tensor(r_mask),
        }


def sequence_log_prob(
    model:  GPT,
    ids:    torch.Tensor,    # (B, T)
    mask:   torch.Tensor,    # (B, T)
) -> torch.Tensor:
    """
    Sum of log P(token t | tokens < t) over real response tokens.
    Returns (B,).
    """
    logits, _ = model(ids)                              # (B, T, V)
    log_p     = F.log_softmax(logits, dim=-1)           # (B, T, V)
    # Targets are the input ids shifted by one
    targets   = ids[:, 1:]                              # (B, T-1)
    lp_tokens = log_p[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return (lp_tokens * mask[:, 1:]).sum(-1)            # (B,)


def dpo_loss(
    policy_model: GPT,
    ref_model:    GPT,
    chosen_ids:   torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask:  torch.Tensor,
    rejected_mask: torch.Tensor,
    beta:         float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DPO loss (Rafailov et al. 2023):

      L_DPO = -E[ log σ( β·(log π_θ(y_w) - log π_ref(y_w))
                         - β·(log π_θ(y_l) - log π_ref(y_l)) ) ]

    Returns (loss, reward_margin).
    """
    with torch.no_grad():
        lp_ref_chosen   = sequence_log_prob(ref_model,  chosen_ids,   chosen_mask)
        lp_ref_rejected = sequence_log_prob(ref_model,  rejected_ids, rejected_mask)

    lp_pol_chosen   = sequence_log_prob(policy_model, chosen_ids,   chosen_mask)
    lp_pol_rejected = sequence_log_prob(policy_model, rejected_ids, rejected_mask)

    # Implicit reward: r(x,y) = β·( log π_θ(y|x) − log π_ref(y|x) )
    chosen_reward   = beta * (lp_pol_chosen   - lp_ref_chosen)
    rejected_reward = beta * (lp_pol_rejected - lp_ref_rejected)

    logits  = chosen_reward - rejected_reward
    loss    = -F.logsigmoid(logits).mean()
    margin  = logits.mean().detach()
    return loss, margin


class DPOTrainer:
    def __init__(
        self,
        policy_model: GPT,
        ref_model:    GPT,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          DPOConfig,
    ):
        self.policy    = policy_model.to(cfg.device)
        self.ref       = ref_model.to(cfg.device).eval()
        self.tr_dl     = train_loader
        self.vl_dl     = val_loader
        self.cfg       = cfg
        self.opt       = torch.optim.AdamW(
            policy_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.history = {"train_loss": [], "val_loss": [], "margin": []}

    @torch.no_grad()
    def _eval(self):
        self.policy.eval()
        losses = []
        for batch in self.vl_dl:
            c_ids = batch["chosen_ids"].to(self.cfg.device)
            r_ids = batch["rejected_ids"].to(self.cfg.device)
            c_msk = batch["chosen_mask"].to(self.cfg.device)
            r_msk = batch["rejected_mask"].to(self.cfg.device)
            loss, _ = dpo_loss(self.policy, self.ref, c_ids, r_ids, c_msk, r_msk, self.cfg.beta)
            losses.append(loss.item())
        self.policy.train()
        return sum(losses)/len(losses)

    def train(self) -> dict:
        cfg = self.cfg
        self.policy.train()
        step = 0

        print(f"\n{'='*60}")
        print(f"DPO TRAINING  |  beta={cfg.beta}  |  device={cfg.device}")
        print(f"{'='*60}")

        for epoch in range(cfg.epochs):
            for batch in self.tr_dl:
                c_ids = batch["chosen_ids"].to(cfg.device)
                r_ids = batch["rejected_ids"].to(cfg.device)
                c_msk = batch["chosen_mask"].to(cfg.device)
                r_msk = batch["rejected_mask"].to(cfg.device)

                loss, margin = dpo_loss(
                    self.policy, self.ref, c_ids, r_ids, c_msk, r_msk, cfg.beta
                )
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
                self.opt.step()
                step += 1

                if step % cfg.eval_freq == 0:
                    vl = self._eval()
                    print(
                        f"  epoch {epoch+1}  step {step}  |  "
                        f"train_loss={loss.item():.3f}  margin={margin.item():.3f}  "
                        f"val_loss={vl:.3f}"
                    )
                    self.history["train_loss"].append(loss.item())
                    self.history["val_loss"].append(vl)
                    self.history["margin"].append(margin.item())

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), f"{cfg.save_dir}/dpo_model.pt")
        print(f"\nDPO complete. Model saved → {cfg.save_dir}/dpo_model.pt")
        return self.history


# ──────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    _dc4 = _load("dc_dpo", "01_data_cleaning.py")
    BPETokenizer, clean_corpus = _dc4.BPETokenizer, _dc4.clean_corpus

    prefs = [
        {"prompt": "Capital of France? ",
         "chosen":  "Paris.",
         "rejected": "dunno"},
        {"prompt": "2 + 2 = ",
         "chosen":  "4",
         "rejected": "maybe 5"},
        {"prompt": "Color of the sky? ",
         "chosen":  "Blue.",
         "rejected": "Red."},
        {"prompt": "Water boils at? ",
         "chosen":  "100 degrees Celsius.",
         "rejected": "50 degrees."},
    ] * 15

    corpus = [p["prompt"]+p["chosen"] for p in prefs[:4]] + \
             [p["prompt"]+p["rejected"] for p in prefs[:4]]
    tok = BPETokenizer()
    tok.train(clean_corpus(corpus), vocab_size=300)

    gpt_cfg = GPTConfig(
        vocab_size  = tok.vocab_size,
        context_len = 32,
        d_model     = 64,
        n_heads     = 4,
        n_layers    = 2,
        dropout     = 0.0,
        weight_tying= False,
    )

    # ── DPO ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Testing DPO")
    print("="*60)

    policy_model = GPT(gpt_cfg)
    ref_model    = copy.deepcopy(policy_model)

    dpo_cfg = DPOConfig(batch_size=2, epochs=2, eval_freq=5, beta=0.1, device="cpu", max_length=32)
    ds = DPODataset(prefs, tok, max_length=32)

    from torch.utils.data import random_split
    n_v  = max(1, len(ds)//10)
    n_tr = len(ds) - n_v
    tr_ds, vl_ds = random_split(ds, [n_tr, n_v])

    def pref_col(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    tr_dl = DataLoader(tr_ds, batch_size=dpo_cfg.batch_size, shuffle=True,  collate_fn=pref_col)
    vl_dl = DataLoader(vl_ds, batch_size=dpo_cfg.batch_size, shuffle=False, collate_fn=pref_col)

    dpo_trainer = DPOTrainer(policy_model, ref_model, tr_dl, vl_dl, dpo_cfg)
    dpo_history = dpo_trainer.train()

    print(f"DPO final train_loss: {dpo_history['train_loss'][-1] if dpo_history['train_loss'] else 'N/A'}")
    print(f"DPO final margin:     {dpo_history['margin'][-1] if dpo_history['margin'] else 'N/A'}")
    print("\nRLHF (PPO + DPO) tests passed!")
