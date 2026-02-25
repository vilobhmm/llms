"""
Chapter 7: Reward Model for RLHF
==================================
The reward model (RM) is the lynchpin of RLHF.

Training pipeline:
  1. Collect *preference pairs*: (prompt, chosen_response, rejected_response)
     where a human labeller marked chosen > rejected.
  2. Train the RM to assign a higher scalar score to chosen vs rejected.
  3. Use the Bradley-Terry probability model as the loss:

         P(chosen > rejected) = σ( r(x, y_w) − r(x, y_l) )

         Loss = −log P = −log σ( r_w − r_l )

     where r_w = reward for chosen (winner), r_l = reward for rejected (loser).

Architecture:
  We take the SFT model and replace the language-model head (vocab-sized)
  with a scalar regression head (→ 1 value per sequence).
  The score for the whole sequence comes from the *last non-padding token*,
  matching the InstructGPT / Anthropic HH approach.

References:
  - Ouyang et al. 2022 "Training language models to follow instructions…" (InstructGPT)
  - Bai et al. 2022 "Training a Helpful and Harmless Assistant…" (Anthropic HH)
  - Bradley & Terry 1952 (pairwise comparison model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── local import ──
import importlib.util, sys
from pathlib import Path as _Path
def _load(alias, fname):
    if alias in sys.modules: return sys.modules[alias]
    p = _Path(__file__).parent / fname
    spec = importlib.util.spec_from_file_location(alias, p)
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m; spec.loader.exec_module(m); return m
_gpt = _load("transformer_rm", "04_transformer_model.py")
GPT, GPTConfig = _gpt.GPT, _gpt.GPTConfig


# ──────────────────────────────────────────────
# 1.  Reward Model Architecture
# ──────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    GPT backbone + scalar head.

    The backbone weights are initialised from the SFT checkpoint.
    The scalar head projects the final hidden state to a single number.

    Score extraction:
        We use the hidden state at the *last real token* position
        (i.e. just before padding) rather than a fixed position,
        because responses vary in length.
    """

    def __init__(self, backbone: GPT):
        super().__init__()
        # Keep all transformer layers
        self.token_emb = backbone.token_emb
        self.pos_emb   = backbone.pos_emb
        self.emb_drop  = backbone.emb_drop
        self.blocks    = backbone.blocks
        self.ln_f      = backbone.ln_f

        d_model = backbone.cfg.d_model

        # Replace LM head with scalar reward head
        self.reward_head = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.reward_head.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) 1=real 0=pad
    ) -> torch.Tensor:
        """
        Returns reward scores of shape (B,) — one scalar per example.
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)              # (B, T, D)

        # Extract the hidden state at the LAST real token
        if attention_mask is not None:
            # last real position = sum of mask - 1
            last_pos = attention_mask.sum(dim=1) - 1          # (B,)
            last_pos = last_pos.clamp(0, T - 1)
        else:
            last_pos = torch.full((B,), T - 1, device=x.device, dtype=torch.long)

        # Gather hidden state at last_pos for each example in the batch
        idx = last_pos.view(B, 1, 1).expand(B, 1, x.size(-1))
        last_hidden = x.gather(1, idx).squeeze(1)             # (B, D)

        scores = self.reward_head(last_hidden).squeeze(-1)    # (B,)
        return scores


# ──────────────────────────────────────────────
# 2.  Preference Dataset
# ──────────────────────────────────────────────

class PreferenceDataset(Dataset):
    """
    Each example is a dict with:
        "prompt"   : str
        "chosen"   : str  (human-preferred response)
        "rejected" : str  (less preferred response)

    Returns tokenised (chosen_ids, rejected_ids, chosen_mask, rejected_mask).
    """

    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer,
        max_length: int = 256,
    ):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.pad_id     = tokenizer.vocab.get(tokenizer.PAD_TOKEN, 0)

    def _encode_and_pad(self, text: str) -> Tuple[List[int], List[int]]:
        ids = self.tokenizer.encode(text)[:self.max_length]
        mask = [1] * len(ids)
        pad_len = self.max_length - len(ids)
        ids  = ids  + [self.pad_id] * pad_len
        mask = mask + [0]           * pad_len
        return ids, mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        full_chosen   = ex["prompt"] + ex["chosen"]
        full_rejected = ex["prompt"] + ex["rejected"]

        c_ids, c_mask = self._encode_and_pad(full_chosen)
        r_ids, r_mask = self._encode_and_pad(full_rejected)

        return {
            "chosen_ids":    torch.tensor(c_ids,  dtype=torch.long),
            "rejected_ids":  torch.tensor(r_ids,  dtype=torch.long),
            "chosen_mask":   torch.tensor(c_mask, dtype=torch.long),
            "rejected_mask": torch.tensor(r_mask, dtype=torch.long),
        }


# ──────────────────────────────────────────────
# 3.  Bradley-Terry Preference Loss
# ──────────────────────────────────────────────

def preference_loss(
    chosen_rewards:   torch.Tensor,   # (B,)
    rejected_rewards: torch.Tensor,   # (B,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bradley-Terry pairwise ranking loss:

        loss = −mean[ log σ(r_chosen − r_rejected) ]

    Equivalent to binary cross-entropy where the label is always 1
    (we always want chosen > rejected).

    Also returns `margin` = mean(r_chosen − r_rejected), a useful metric.
    Higher margin = reward model is more confident in its rankings.
    """
    logits = chosen_rewards - rejected_rewards       # (B,)
    loss   = -F.logsigmoid(logits).mean()
    margin = logits.mean().detach()
    return loss, margin


# ──────────────────────────────────────────────
# 4.  Reward Model Trainer
# ──────────────────────────────────────────────

@dataclass
class RMConfig:
    lr:           float = 1e-5
    weight_decay: float = 0.01
    batch_size:   int   = 4
    epochs:       int   = 3
    grad_clip:    float = 1.0
    eval_freq:    int   = 10
    save_dir:     str   = "checkpoints/rm"
    device:       str   = "cpu"


class RewardModelTrainer:
    def __init__(
        self,
        reward_model: RewardModel,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          RMConfig,
    ):
        self.rm   = reward_model.to(cfg.device)
        self.tr_dl = train_loader
        self.vl_dl = val_loader
        self.cfg   = cfg
        self.opt   = torch.optim.AdamW(
            self.rm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.history = {"train_loss": [], "val_loss": [], "train_margin": [], "val_margin": []}

    @torch.no_grad()
    def _eval(self):
        self.rm.eval()
        losses, margins = [], []
        for batch in self.vl_dl:
            c_ids  = batch["chosen_ids"].to(self.cfg.device)
            r_ids  = batch["rejected_ids"].to(self.cfg.device)
            c_mask = batch["chosen_mask"].to(self.cfg.device)
            r_mask = batch["rejected_mask"].to(self.cfg.device)
            r_c    = self.rm(c_ids, c_mask)
            r_r    = self.rm(r_ids, r_mask)
            loss, margin = preference_loss(r_c, r_r)
            losses.append(loss.item())
            margins.append(margin.item())
        self.rm.train()
        return sum(losses)/len(losses), sum(margins)/len(margins)

    def train(self) -> dict:
        cfg  = self.cfg
        rm   = self.rm
        rm.train()
        step = 0

        print(f"\n{'='*60}")
        print(f"REWARD MODEL TRAINING  |  device={cfg.device}")
        print(f"{'='*60}")

        for epoch in range(cfg.epochs):
            for batch in self.tr_dl:
                c_ids  = batch["chosen_ids"].to(cfg.device)
                r_ids  = batch["rejected_ids"].to(cfg.device)
                c_mask = batch["chosen_mask"].to(cfg.device)
                r_mask = batch["rejected_mask"].to(cfg.device)

                r_c  = rm(c_ids, c_mask)
                r_r  = rm(r_ids, r_mask)
                loss, margin = preference_loss(r_c, r_r)

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(rm.parameters(), cfg.grad_clip)
                self.opt.step()
                step += 1

                if step % cfg.eval_freq == 0:
                    vl_loss, vl_margin = self._eval()
                    print(
                        f"  epoch {epoch+1}  step {step}  |  "
                        f"train_loss={loss.item():.3f}  margin={margin.item():.3f}  |  "
                        f"val_loss={vl_loss:.3f}  val_margin={vl_margin:.3f}"
                    )
                    self.history["train_loss"].append(loss.item())
                    self.history["val_loss"].append(vl_loss)
                    self.history["train_margin"].append(margin.item())
                    self.history["val_margin"].append(vl_margin)

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(rm.state_dict(), f"{cfg.save_dir}/reward_model.pt")
        print(f"\nReward model saved → {cfg.save_dir}/reward_model.pt")
        return self.history


# ──────────────────────────────────────────────
# 5.  Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    _dc3 = _load("dc_rm", "01_data_cleaning.py")
    BPETokenizer, clean_corpus = _dc3.BPETokenizer, _dc3.clean_corpus

    # ── Synthetic preference pairs ───────────────────────────────
    prefs = [
        {"prompt": "What is 2+2? Answer:",
         "chosen":   " 4",
         "rejected": " I don't know"},
        {"prompt": "Describe the sky. Answer:",
         "chosen":   " The sky is blue during the day.",
         "rejected": " sky"},
        {"prompt": "Who wrote Hamlet? Answer:",
         "chosen":   " William Shakespeare wrote Hamlet.",
         "rejected": " someone"},
        {"prompt": "Capital of Japan? Answer:",
         "chosen":   " Tokyo is the capital of Japan.",
         "rejected": " Japan"},
    ] * 15

    all_text = [p["prompt"] + p["chosen"] for p in prefs[:4]] + \
               [p["prompt"] + p["rejected"] for p in prefs[:4]]
    tok = BPETokenizer()
    tok.train(clean_corpus(all_text), vocab_size=300)

    rm_cfg = RMConfig(batch_size=2, epochs=2, eval_freq=5, device="cpu")

    ds = PreferenceDataset(prefs, tok, max_length=32)
    from torch.utils.data import random_split
    n_v  = max(1, len(ds)//10)
    n_tr = len(ds) - n_v
    tr_ds, vl_ds = random_split(ds, [n_tr, n_v])

    def pref_collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    tr_dl = DataLoader(tr_ds, batch_size=rm_cfg.batch_size, shuffle=True,  collate_fn=pref_collate)
    vl_dl = DataLoader(vl_ds, batch_size=rm_cfg.batch_size, shuffle=False, collate_fn=pref_collate)

    # Build backbone GPT (in practice load from SFT checkpoint)
    gpt_cfg = GPTConfig(
        vocab_size  = tok.vocab_size,
        context_len = 32,
        d_model     = 64,
        n_heads     = 4,
        n_layers    = 2,
        dropout     = 0.0,
        weight_tying= False,
    )
    backbone = GPT(gpt_cfg)
    rm       = RewardModel(backbone)

    print(f"Reward model params: {sum(p.numel() for p in rm.parameters()):,}")

    trainer  = RewardModelTrainer(rm, tr_dl, vl_dl, rm_cfg)
    history  = trainer.train()

    # Quick score check
    rm.eval()
    with torch.no_grad():
        ex = prefs[0]
        good_ids, good_mask = ds._encode_and_pad(ex["prompt"] + ex["chosen"])
        bad_ids,  bad_mask  = ds._encode_and_pad(ex["prompt"] + ex["rejected"])
        g_t = torch.tensor([good_ids])
        b_t = torch.tensor([bad_ids])
        gm  = torch.tensor([good_mask])
        bm  = torch.tensor([bad_mask])
        r_good = rm(g_t, gm).item()
        r_bad  = rm(b_t, bm).item()
        print(f"\nChosen reward  : {r_good:.4f}")
        print(f"Rejected reward: {r_bad:.4f}")
        print(f"Margin (chosen - rejected): {r_good - r_bad:.4f}  (should be > 0)")
