"""
Chapter 5: Pre-Training on Unlabeled Data
==========================================
Implements the full pre-training loop for a GPT model.

Pre-training teaches the model the statistical patterns of language via
**next-token prediction** (causal language modeling):

    Given  [t₁, t₂, ..., tₙ]
    Predict [t₂, t₃, ..., tₙ₊₁]

Loss = mean CrossEntropy over all (input, target) pairs in the batch.

Included:
  - AdamW optimiser with weight decay
  - Cosine annealing learning-rate schedule with linear warmup
  - Gradient clipping
  - Periodic train/val loss logging + perplexity
  - Checkpoint save / load
  - Simple text generation callback
"""

import math
import time
import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── local imports (via importlib for digit-prefixed filenames) ──
import importlib.util, sys
from pathlib import Path as _Path

def _load(alias, fname):
    if alias in sys.modules: return sys.modules[alias]
    p = _Path(__file__).parent / fname
    spec = importlib.util.spec_from_file_location(alias, p)
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m; spec.loader.exec_module(m); return m

_gpt = _load("transformer_m", "04_transformer_model.py")
_dc  = _load("data_cleaning_m", "01_data_cleaning.py")
GPT, GPTConfig = _gpt.GPT, _gpt.GPTConfig
BPETokenizer, SlidingWindowDataset, clean_corpus = _dc.BPETokenizer, _dc.SlidingWindowDataset, _dc.clean_corpus


# ──────────────────────────────────────────────
# 1.  Training configuration
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    # data
    context_len:  int   = 256
    stride:       int   = 128
    batch_size:   int   = 8

    # optimisation
    lr:           float = 4e-4
    weight_decay: float = 0.1
    beta1:        float = 0.9
    beta2:        float = 0.95
    grad_clip:    float = 1.0

    # schedule
    warmup_steps: int   = 100
    max_steps:    int   = 2000

    # logging / checkpointing
    eval_freq:    int   = 100
    eval_iters:   int   = 20
    save_dir:     str   = "checkpoints"
    device:       str   = "cpu"


# ──────────────────────────────────────────────
# 2.  Learning-rate schedule (warmup + cosine)
# ──────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """
    Linear warmup followed by cosine annealing to 10% of peak LR.

         lr
          │ /‾‾╲
          │/    ╲___
          └────────── step
          warmup  max_steps
    """
    min_lr = cfg.lr * 0.1
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    if step >= cfg.max_steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────
# 3.  Loss evaluation  (no gradient)
# ──────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: GPT,
    loader: DataLoader,
    n_iters: int,
    device: str,
) -> Tuple[float, float]:
    """
    Returns (mean_loss, perplexity) over n_iters batches.
    Perplexity = e^loss — lower is better; a random model gives vocab_size.
    """
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses) if losses else float("inf")
    return mean_loss, math.exp(mean_loss)


# ──────────────────────────────────────────────
# 4.  Checkpoint helpers
# ──────────────────────────────────────────────

def save_checkpoint(model: GPT, optimizer, step: int, loss: float, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":        step,
        "loss":        loss,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "config":      model.cfg,
    }, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path: str, device: str) -> dict:
    return torch.load(path, map_location=device)


# ──────────────────────────────────────────────
# 5.  Main training loop
# ──────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the pre-training loop.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, train_cfg)
        history = trainer.train()
    """

    def __init__(
        self,
        model:        GPT,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          TrainConfig,
        tokenizer:    Optional[BPETokenizer] = None,
    ):
        self.model        = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.tokenizer    = tokenizer

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = cfg.lr,
            betas        = (cfg.beta1, cfg.beta2),
            weight_decay = cfg.weight_decay,
        )

        self.history = {
            "step":       [],
            "train_loss": [],
            "val_loss":   [],
            "train_ppl":  [],
            "val_ppl":    [],
            "lr":         [],
        }
        self._step       = 0
        self._data_iter  = iter(train_loader)

    def _get_batch(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.train_loader)
            return next(self._data_iter)

    def train(self) -> dict:
        cfg   = self.cfg
        model = self.model
        model.train()

        print(f"\n{'='*60}")
        print(f"PRE-TRAINING  |  {model.num_parameters():,} params  |  device={cfg.device}")
        print(f"{'='*60}")
        t0 = time.time()

        for step in range(cfg.max_steps):
            self._step = step

            # ── Learning-rate update ──────────────────────────────────
            lr = get_lr(step, cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # ── Forward + backward ───────────────────────────────────
            x, y = self._get_batch()
            x, y = x.to(cfg.device), y.to(cfg.device)

            _, loss = model(x, y)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            self.optimizer.step()

            # ── Periodic evaluation ──────────────────────────────────
            if step % cfg.eval_freq == 0 or step == cfg.max_steps - 1:
                tr_loss, tr_ppl = estimate_loss(
                    model, self.train_loader, cfg.eval_iters, cfg.device
                )
                vl_loss, vl_ppl = estimate_loss(
                    model, self.val_loader, cfg.eval_iters, cfg.device
                )
                elapsed = time.time() - t0

                print(
                    f"  step {step:5d}/{cfg.max_steps} | "
                    f"lr={lr:.2e} | "
                    f"train_loss={tr_loss:.3f}  ppl={tr_ppl:.1f} | "
                    f"val_loss={vl_loss:.3f}  ppl={vl_ppl:.1f} | "
                    f"t={elapsed:.1f}s"
                )

                self.history["step"].append(step)
                self.history["train_loss"].append(tr_loss)
                self.history["val_loss"].append(vl_loss)
                self.history["train_ppl"].append(tr_ppl)
                self.history["val_ppl"].append(vl_ppl)
                self.history["lr"].append(lr)

                # Sample generation (if tokenizer provided)
                if self.tokenizer is not None:
                    self._generate_sample()

        # Final checkpoint
        save_checkpoint(
            model, self.optimizer, self._step,
            self.history["val_loss"][-1],
            f"{cfg.save_dir}/pretrain_final.pt",
        )
        print(f"\nPre-training complete. Best val_loss={min(self.history['val_loss']):.3f}")
        return self.history

    def _generate_sample(self, prompt: str = "the cat", n: int = 20) -> None:
        """Quick generation preview during training."""
        if self.tokenizer is None:
            return
        self.model.eval()
        ids    = self.tokenizer.encode(prompt)
        tensor = torch.tensor([ids], device=self.cfg.device)
        out    = self.model.generate(tensor, max_new_tokens=n, temperature=0.8, top_k=40)
        text   = self.tokenizer.decode(out[0].tolist())
        print(f"    SAMPLE: {text!r}")
        self.model.train()


# ──────────────────────────────────────────────
# 6.  Smoke-test with tiny synthetic data
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # ── Tiny tokenizer ─────────────────────────────────────────────
    corpus = [
        "the transformer architecture uses self attention mechanisms",
        "language models predict the next token given the context",
        "pre training on large corpora gives general representations",
        "byte pair encoding splits words into subword units",
        "gradient descent minimizes the cross entropy loss",
        "residual connections allow gradients to flow through deep networks",
        "layer normalization stabilizes training in deep transformers",
        "the feed forward network expands and contracts the representation",
        "attention heads capture different syntactic and semantic patterns",
        "positional embeddings encode the order of tokens in a sequence",
    ] * 20  # repeat so dataset is large enough
    tok = BPETokenizer()
    tok.train(clean_corpus(corpus), vocab_size=300)

    all_ids = tok.encode(" ".join(corpus))
    ctx     = 32
    ds      = SlidingWindowDataset(all_ids, context_length=ctx, stride=16)

    n_val   = max(1, len(ds) // 10)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=4, shuffle=False, drop_last=True)

    # ── Small GPT ──────────────────────────────────────────────────
    gpt_cfg = GPTConfig(
        vocab_size  = tok.vocab_size,
        context_len = ctx,
        d_model     = 64,
        n_heads     = 4,
        n_layers    = 2,
        dropout     = 0.1,
    )
    model = GPT(gpt_cfg)

    # ── Train ──────────────────────────────────────────────────────
    train_cfg = TrainConfig(
        context_len  = ctx,
        batch_size   = 4,
        lr           = 3e-3,
        warmup_steps = 20,
        max_steps    = 200,
        eval_freq    = 40,
        eval_iters   = 5,
        save_dir     = "checkpoints",
        device       = "cpu",
    )

    trainer = Trainer(model, train_dl, val_dl, train_cfg, tokenizer=tok)
    history = trainer.train()

    print(f"\nFinal train loss : {history['train_loss'][-1]:.3f}")
    print(f"Final val   loss : {history['val_loss'][-1]:.3f}")
    print(f"Loss improvement : {history['train_loss'][0] - history['train_loss'][-1]:.3f}")
