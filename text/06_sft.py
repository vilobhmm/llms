"""
Chapter 6 & 7: Supervised Fine-Tuning (SFT)
=============================================
SFT adapts a pre-trained language model to follow instructions.

Pipeline:
  Pre-trained weights
      ↓
  Format examples as Alpaca-style prompts
      ↓
  Compute loss ONLY on the response tokens (not on the instruction)
      ↓
  Fine-tune with a lower learning rate

Alpaca instruction template (Raschka book, Ch 7):
  ╔══════════════════════════════════════════════════════╗
  ║ Below is an instruction that describes a task...    ║
  ║                                                      ║
  ║ ### Instruction:                                     ║
  ║ <instruction>                                        ║
  ║                                                      ║
  ║ ### Input:                                           ║
  ║ <optional input>                                     ║
  ║                                                      ║
  ║ ### Response:                                        ║
  ║ <model response>                                     ║
  ╚══════════════════════════════════════════════════════╝

Loss masking:
  Tokens in the instruction/input portion are set to -100 so
  nn.CrossEntropyLoss ignores them.  The model only learns to
  generate the response.
"""

import copy
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── local import ──
import importlib.util, sys
from pathlib import Path as _Path
def _load(alias, fname):
    if alias in sys.modules: return sys.modules[alias]
    p = _Path(__file__).parent / fname
    spec = importlib.util.spec_from_file_location(alias, p)
    m = importlib.util.module_from_spec(spec); sys.modules[alias] = m; spec.loader.exec_module(m); return m
_gpt = _load("transformer_sft", "04_transformer_model.py")
GPT, GPTConfig = _gpt.GPT, _gpt.GPTConfig


# ──────────────────────────────────────────────
# 1.  Alpaca-style prompt formatter
# ──────────────────────────────────────────────

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_block}"
    "### Response:\n"
)


def format_alpaca_prompt(
    instruction: str,
    input_text: str = "",
    response: str = "",
) -> str:
    """Return full Alpaca-formatted string (prompt + response)."""
    input_block = f"### Input:\n{input_text}\n\n" if input_text.strip() else ""
    prompt = ALPACA_TEMPLATE.format(
        instruction=instruction,
        input_block=input_block,
    )
    return prompt + response


def get_response_start(prompt: str, full_text: str) -> int:
    """Return the character index where the response starts in full_text."""
    return len(prompt)


# ──────────────────────────────────────────────
# 2.  SFT Dataset
# ──────────────────────────────────────────────

class InstructionDataset(Dataset):
    """
    Each item is an (instruction, optional_input, response) triple.

    Tokenises the full prompt+response and creates:
        input_ids  : full token sequence (instruction + response)
        labels     : same as input_ids BUT instruction tokens replaced with -100
                     so loss is only computed on response tokens.

    Example:
        input_ids = [tok("Below is...Instruction:\n...Response:\nParis"), EOS]
        labels    = [-100, -100, ..., tok("Paris"), EOS]
    """

    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
    ):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_id    = tokenizer.vocab.get(tokenizer.EOS_TOKEN, 0)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex          = self.examples[idx]
        instruction = ex.get("instruction", "")
        input_text  = ex.get("input", "")
        response    = ex.get("response", ex.get("output", ""))

        # Build prompt (without response) and full text
        prompt    = format_alpaca_prompt(instruction, input_text)
        full_text = prompt + response

        # Tokenise
        prompt_ids   = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)
        full_ids     = prompt_ids + response_ids + [self.eos_id]

        # Truncate
        full_ids = full_ids[:self.max_length]

        # Create labels: mask prompt tokens with -100
        labels = [-100] * len(prompt_ids) + response_ids + [self.eos_id]
        labels = labels[:self.max_length]

        # Pad to max_length
        pad_len  = self.max_length - len(full_ids)
        full_ids = full_ids + [0] * pad_len
        labels   = labels   + [-100] * pad_len

        return (
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(labels,   dtype=torch.long),
        )


def collate_fn(batch):
    """Dynamic padding to the longest example in the batch."""
    xs, ys = zip(*batch)
    # Already padded to max_length in dataset; just stack
    return torch.stack(xs), torch.stack(ys)


# ──────────────────────────────────────────────
# 3.  SFT Trainer
# ──────────────────────────────────────────────

@dataclass
class SFTConfig:
    lr:           float = 5e-5
    weight_decay: float = 0.1
    batch_size:   int   = 4
    epochs:       int   = 3
    grad_clip:    float = 1.0
    eval_freq:    int   = 20       # evaluate every N steps
    save_dir:     str   = "checkpoints/sft"
    device:       str   = "cpu"
    max_length:   int   = 256


class SFTTrainer:
    """
    Fine-tunes a pre-trained GPT on instruction-following data.

    Key differences from pre-training:
      1. Loss only on response tokens (instruction tokens masked)
      2. Much lower learning rate (5e-5 vs 4e-4)
      3. Fewer epochs (2-3 vs many)
      4. No warmup required (model already has good representations)
    """

    def __init__(
        self,
        model:        GPT,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          SFTConfig,
    ):
        self.model        = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.optimizer    = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.history      = {"train_loss": [], "val_loss": [], "step": []}

    @torch.no_grad()
    def _eval(self) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
            logits, _ = self.model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
            )
            total += loss.item()
            count += 1
        self.model.train()
        return total / max(count, 1)

    def train(self) -> dict:
        cfg   = self.cfg
        model = self.model
        model.train()
        global_step = 0

        print(f"\n{'='*60}")
        print(f"SFT  |  device={cfg.device}  |  epochs={cfg.epochs}")
        print(f"{'='*60}")

        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(cfg.device), y.to(cfg.device)

                # Forward
                logits, _ = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                self.optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % cfg.eval_freq == 0:
                    val_loss = self._eval()
                    tr_avg   = epoch_loss / (batch_idx + 1)
                    print(
                        f"  epoch {epoch+1}/{cfg.epochs}  "
                        f"step {global_step}  |  "
                        f"train_loss={tr_avg:.3f}  val_loss={val_loss:.3f}"
                    )
                    self.history["step"].append(global_step)
                    self.history["train_loss"].append(tr_avg)
                    self.history["val_loss"].append(val_loss)

            print(f"  ── Epoch {epoch+1} avg loss: {epoch_loss/len(self.train_loader):.3f}")

        # Save
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{cfg.save_dir}/sft_model.pt")
        print(f"\nSFT complete. Model saved → {cfg.save_dir}/sft_model.pt")
        return self.history


# ──────────────────────────────────────────────
# 4.  Response generation utility
# ──────────────────────────────────────────────

@torch.no_grad()
def generate_response(
    model: GPT,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cpu",
) -> str:
    """Generate a response for a given instruction."""
    model.eval()
    prompt  = format_alpaca_prompt(instruction, input_text)
    ids     = tokenizer.encode(prompt)
    tensor  = torch.tensor([ids], device=device)
    output  = model.generate(tensor, max_new_tokens=max_new_tokens,
                              temperature=temperature, top_k=top_k)
    full    = tokenizer.decode(output[0].tolist())
    # Extract only the response part
    response_start = full.find("### Response:\n")
    if response_start != -1:
        return full[response_start + len("### Response:\n"):].strip()
    return full[len(prompt):].strip()


# ──────────────────────────────────────────────
# 5.  Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    _dc2 = _load("dc_sft", "01_data_cleaning.py")
    BPETokenizer, clean_corpus = _dc2.BPETokenizer, _dc2.clean_corpus

    # ── Tiny instruction dataset ──────────────────────────────────
    examples = [
        {"instruction": "What is the capital of France?",
         "input": "", "response": "The capital of France is Paris."},
        {"instruction": "Translate to Spanish: Hello",
         "input": "", "response": "Hola"},
        {"instruction": "Write a haiku about the ocean",
         "input": "", "response": "Waves crash on the shore\nSalt and spray fill the cool air\nDeep blue endless sea"},
        {"instruction": "What is 2 + 2?",
         "input": "", "response": "2 + 2 equals 4."},
        {"instruction": "Summarize in one sentence",
         "input": "Transformers use self-attention to process sequences in parallel.",
         "response": "Transformers process sequences in parallel using self-attention."},
        {"instruction": "Name three primary colors",
         "input": "", "response": "The three primary colors are red, blue, and yellow."},
    ] * 10  # repeat for demonstration

    # ── Tokenizer on the SFT corpus ───────────────────────────────
    all_texts = []
    for ex in examples[:6]:
        all_texts.append(format_alpaca_prompt(
            ex["instruction"], ex.get("input",""), ex["response"]
        ))

    tok = BPETokenizer()
    tok.train(clean_corpus(all_texts), vocab_size=400)

    cfg_sft = SFTConfig(max_length=64, batch_size=2, epochs=2, eval_freq=10, device="cpu")

    ds = InstructionDataset(examples, tok, max_length=cfg_sft.max_length)
    n_val  = max(1, len(ds) // 10)
    n_tr   = len(ds) - n_val
    from torch.utils.data import random_split
    tr_ds, vl_ds = random_split(ds, [n_tr, n_val])
    tr_dl = DataLoader(tr_ds, batch_size=cfg_sft.batch_size, shuffle=True,  collate_fn=collate_fn)
    vl_dl = DataLoader(vl_ds, batch_size=cfg_sft.batch_size, shuffle=False, collate_fn=collate_fn)

    # ── GPT model ─────────────────────────────────────────────────
    gpt_cfg = GPTConfig(
        vocab_size  = tok.vocab_size,
        context_len = cfg_sft.max_length,
        d_model     = 64,
        n_heads     = 4,
        n_layers    = 2,
        dropout     = 0.1,
    )
    model = GPT(gpt_cfg)

    trainer = SFTTrainer(model, tr_dl, vl_dl, cfg_sft)
    history = trainer.train()

    # ── Generation test ───────────────────────────────────────────
    response = generate_response(
        model, tok,
        instruction="What is the capital of France?",
        max_new_tokens=20, device="cpu"
    )
    print(f"\nGenerated response: {response!r}")
    print("SFT test passed!")
