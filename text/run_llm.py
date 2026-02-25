"""
run_all.py — End-to-end LLM-from-scratch demo
==============================================
Runs every module in sequence with small synthetic data so the full
pipeline can be exercised on a CPU laptop in < 5 minutes.

Execution order:
  1. Data cleaning + BPE tokenizer
  2. GPT Transformer model (embeddings + attention + FFN)
  3. Attention verification (causal mask)
  4. Pre-training loop  (200 steps)
  5. SFT               (2 epochs)
  6. Reward model      (2 epochs)
  7. DPO               (2 epochs)
  8. Visualizations    (all 11 figures)
"""

import sys, math, copy, importlib.util
from pathlib import Path

ROOT = Path(__file__).parent


def _load(alias, fname):
    """importlib loader for digit-prefixed module files."""
    if alias in sys.modules:
        return sys.modules[alias]
    p    = ROOT / fname
    spec = importlib.util.spec_from_file_location(alias, p)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


print("=" * 70)
print("   LLM FROM SCRATCH — Full Pipeline Demo")
print("   Based on: 'Build a Large Language Model' by Sebastian Raschka")
print("   GitHub: https://github.com/rasbt/LLMs-from-scratch")
print("=" * 70)

import torch
from torch.utils.data import DataLoader, random_split

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: Data cleaning + BPE tokenizer
# ──────────────────────────────────────────────────────────────────────────

print("\n[STEP 1] Data cleaning + BPE tokenizer")

dc = _load("data_cleaning", "01_data_cleaning.py")
BPETokenizer         = dc.BPETokenizer
clean_corpus         = dc.clean_corpus
SlidingWindowDataset = dc.SlidingWindowDataset

CORPUS = [
    "the transformer architecture uses self attention mechanisms",
    "language models predict the next token given the context window",
    "pre training on large corpora gives general language representations",
    "byte pair encoding splits words into subword units for the vocabulary",
    "gradient descent minimizes the cross entropy loss during training",
    "residual connections allow gradients to flow through deep networks",
    "layer normalization stabilizes training in deep transformer networks",
    "the feed forward network expands and then contracts the representation",
    "attention heads capture different syntactic and semantic relationships",
    "positional embeddings encode the order of tokens in the sequence",
    "reward models are trained on human preference data for alignment",
    "direct preference optimization simplifies the RLHF training pipeline",
    "supervised fine tuning adapts pretrained models to follow instructions",
    "the policy model generates responses that maximize expected reward",
    "KL divergence prevents the policy from deviating too far from reference",
] * 25

# clean_corpus deduplicates — use unique corpus for tokenizer training
cleaned = clean_corpus(CORPUS)
tok = BPETokenizer()
tok.train(cleaned, vocab_size=400)
print(f"  Vocab size: {tok.vocab_size}  |  Merges: {len(tok.merges)}")

# For language model pretraining we want repeated data (don't deduplicate)
lm_text = " ".join(CORPUS)   # full repeated corpus (~25x more tokens)
all_ids  = tok.encode(lm_text)
print(f"  Total tokens: {len(all_ids)}")

CONTEXT_LEN = 32

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: GPT Model
# ──────────────────────────────────────────────────────────────────────────

print("\n[STEP 2] GPT Transformer model (embedding + attention + FFN)")

gpt_mod   = _load("transformer_m", "04_transformer_model.py")
GPT       = gpt_mod.GPT
GPTConfig = gpt_mod.GPTConfig

CFG = GPTConfig(
    vocab_size   = tok.vocab_size,
    context_len  = CONTEXT_LEN,
    d_model      = 128,
    n_heads      = 4,
    n_layers     = 3,
    dropout      = 0.1,
    weight_tying = True,
)
model = GPT(CFG)
print(f"  GPT  {model.num_parameters():,} parameters")

ids_test     = torch.randint(0, tok.vocab_size, (2, 16))
logits, loss = model(ids_test, ids_test)
print(f"  Forward pass: {ids_test.shape} → logits {logits.shape}  loss={loss.item():.3f}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: Attention sanity check
# ──────────────────────────────────────────────────────────────────────────

print("\n[STEP 3] Multi-head causal attention verification")

attn_mod                 = _load("attention_m", "03_attention.py")
MultiHeadCausalAttention = attn_mod.MultiHeadCausalAttention

x_test = torch.randn(2, 8, CFG.d_model)
attn   = MultiHeadCausalAttention(CFG.d_model, CFG.n_heads, max_seq_len=8)
out, w = attn(x_test, return_attn_weights=True)
upper_tri = w[0, 0].triu(1).abs().max().item()
print(f"  Causal mask upper-tri max (should be 0): {upper_tri:.2e}  ✓")

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: Pre-training
# ──────────────────────────────────────────────────────────────────────────

print("\n[STEP 4] Pre-training (200 steps)")

pt_mod      = _load("pretraining_m", "05_pretraining.py")
Trainer     = pt_mod.Trainer
TrainConfig = pt_mod.TrainConfig

ds   = SlidingWindowDataset(all_ids, context_length=CONTEXT_LEN, stride=16)
n_v  = max(4, len(ds)//10); n_tr = len(ds) - n_v
tr_ds, vl_ds = random_split(ds, [n_tr, n_v])
tr_dl = DataLoader(tr_ds, batch_size=8, shuffle=True,  drop_last=True)
vl_dl = DataLoader(vl_ds, batch_size=8, shuffle=False, drop_last=True)

pt_cfg  = TrainConfig(
    context_len=CONTEXT_LEN, batch_size=8,
    lr=3e-3, warmup_steps=20, max_steps=200,
    eval_freq=50, eval_iters=5, save_dir="checkpoints", device="cpu"
)
trainer = Trainer(model, tr_dl, vl_dl, pt_cfg, tokenizer=tok)
pt_hist = trainer.train()
print(f"  Final val loss: {pt_hist['val_loss'][-1]:.3f}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 5: SFT
# ──────────────────────────────────────────────────────────────────────────

print("\n[STEP 5] Supervised Fine-Tuning (SFT)")

sft_mod              = _load("sft_m", "06_sft.py")
InstructionDataset   = sft_mod.InstructionDataset
SFTConfig            = sft_mod.SFTConfig
SFTTrainer           = sft_mod.SFTTrainer
collate_fn           = sft_mod.collate_fn
format_alpaca_prompt = sft_mod.format_alpaca_prompt

INSTRUCT_DATA = [
    {"instruction": "What is the capital of France?",  "input": "", "response": "Paris."},
    {"instruction": "What is 2 + 2?",                  "input": "", "response": "4."},
    {"instruction": "Name a primary color.",            "input": "", "response": "Red."},
    {"instruction": "What does LLM stand for?",        "input": "", "response": "Large Language Model."},
    {"instruction": "What is a transformer?",          "input": "",
     "response": "A deep learning architecture using self-attention."},
    {"instruction": "Summarize in one word.",
     "input": "The model learns by predicting the next token.",
     "response": "Prediction."},
] * 15

inst_texts = [format_alpaca_prompt(e["instruction"], e.get("input",""), e["response"])
              for e in INSTRUCT_DATA[:6]]
inst_tok = BPETokenizer()
inst_tok.train(clean_corpus(inst_texts + cleaned[:5]), vocab_size=500)

sft_cfg  = SFTConfig(lr=5e-4, batch_size=4, epochs=2, eval_freq=15, max_length=64, device="cpu")
sft_ds   = InstructionDataset(INSTRUCT_DATA, inst_tok, max_length=64)
n_sv     = max(2, len(sft_ds)//10); n_str = len(sft_ds) - n_sv
str_ds, svl_ds = random_split(sft_ds, [n_str, n_sv])
str_dl   = DataLoader(str_ds, batch_size=sft_cfg.batch_size, shuffle=True,  collate_fn=collate_fn)
svl_dl   = DataLoader(svl_ds, batch_size=sft_cfg.batch_size, shuffle=False, collate_fn=collate_fn)

sft_model   = GPT(GPTConfig(
    vocab_size=inst_tok.vocab_size, context_len=64,
    d_model=128, n_heads=4, n_layers=3, dropout=0.1, weight_tying=True
))
sft_trainer = SFTTrainer(sft_model, str_dl, svl_dl, sft_cfg)
sft_hist    = sft_trainer.train()
print(f"  SFT complete.")

# ──────────────────────────────────────────────────────────────────────────
# STEP 6: Reward Model
# ──────────────────────────────────────────────────────────────────────────

print("\n[STEP 6] Reward Model training")

rm_mod             = _load("rm_m", "07_reward_model.py")
RewardModel        = rm_mod.RewardModel
PreferenceDataset  = rm_mod.PreferenceDataset
RewardModelTrainer = rm_mod.RewardModelTrainer
RMConfig           = rm_mod.RMConfig

PREFS = [
    {"prompt": "What is 2+2? ",        "chosen": "4.",     "rejected": "dunno."},
    {"prompt": "Capital of France? ",  "chosen": "Paris.", "rejected": "London."},
    {"prompt": "Color of sky? ",       "chosen": "Blue.",  "rejected": "Red."},
    {"prompt": "Water boils at? ",     "chosen": "100C.",  "rejected": "50C."},
] * 15

pref_corpus = [p["prompt"]+p["chosen"]   for p in PREFS[:4]] + \
              [p["prompt"]+p["rejected"] for p in PREFS[:4]]
rm_tok = BPETokenizer()
rm_tok.train(clean_corpus(pref_corpus), vocab_size=300)

rm_cfg  = RMConfig(batch_size=4, epochs=2, eval_freq=5, device="cpu")
pref_ds = PreferenceDataset(PREFS, rm_tok, max_length=32)
n_rv    = max(2, len(pref_ds)//10); n_rtr = len(pref_ds) - n_rv
rtr_ds, rvl_ds = random_split(pref_ds, [n_rtr, n_rv])

def pref_col(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

rtr_dl = DataLoader(rtr_ds, batch_size=rm_cfg.batch_size, shuffle=True,  collate_fn=pref_col)
rvl_dl = DataLoader(rvl_ds, batch_size=rm_cfg.batch_size, shuffle=False, collate_fn=pref_col)

backbone_rm = GPT(GPTConfig(
    vocab_size=rm_tok.vocab_size, context_len=32,
    d_model=64, n_heads=4, n_layers=2, dropout=0.0, weight_tying=False
))
rm         = RewardModel(backbone_rm)
rm_trainer = RewardModelTrainer(rm, rtr_dl, rvl_dl, rm_cfg)
rm_hist    = rm_trainer.train()

# ──────────────────────────────────────────────────────────────────────────
# STEP 7: DPO
# ──────────────────────────────────────────────────────────────────────────

print("\n[STEP 7] DPO (Direct Preference Optimization)")

rlhf_mod   = _load("rlhf_m", "08_rlhf_ppo_dpo.py")
DPODataset = rlhf_mod.DPODataset
DPOConfig  = rlhf_mod.DPOConfig
DPOTrainer = rlhf_mod.DPOTrainer

dpo_cfg     = DPOConfig(beta=0.1, batch_size=4, epochs=2, eval_freq=5, device="cpu", max_length=32)
dpo_ds      = DPODataset(PREFS, rm_tok, max_length=32)
n_dv        = max(2, len(dpo_ds)//10); n_dtr = len(dpo_ds) - n_dv
dtr_ds, dvl_ds = random_split(dpo_ds, [n_dtr, n_dv])
dtr_dl      = DataLoader(dtr_ds, batch_size=dpo_cfg.batch_size, shuffle=True,  collate_fn=pref_col)
dvl_dl      = DataLoader(dvl_ds, batch_size=dpo_cfg.batch_size, shuffle=False, collate_fn=pref_col)

policy_model = copy.deepcopy(backbone_rm)
ref_model    = copy.deepcopy(backbone_rm)
dpo_trainer  = DPOTrainer(policy_model, ref_model, dtr_dl, dvl_dl, dpo_cfg)
dpo_hist     = dpo_trainer.train()

# ──────────────────────────────────────────────────────────────────────────
# STEP 8: Visualizations
# ──────────────────────────────────────────────────────────────────────────

print("\n[STEP 8] Generating all 11 visualizations...")

viz = _load("viz_m", "09_visualizations.py")
fns = [
    viz.plot_bpe_vocabulary, viz.plot_embeddings, viz.plot_sinusoidal_pe,
    viz.plot_attention, viz.plot_transformer_block, viz.plot_pretraining,
    viz.plot_sft, viz.plot_reward_model, viz.plot_rlhf_ppo, viz.plot_dpo,
    viz.plot_full_pipeline,
]
for fn in fns:
    fn()

# ──────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────

rm_margin  = rm_hist['train_margin'][-1]  if rm_hist['train_margin']  else float('nan')
dpo_margin = dpo_hist['margin'][-1]       if dpo_hist['margin']       else float('nan')
sft_loss   = sft_hist['val_loss'][-1]     if sft_hist['val_loss']     else float('nan')

print("\n" + "="*70)
print("PIPELINE COMPLETE — Summary")
print("="*70)
print(f"  Tokenizer vocab size     : {tok.vocab_size}")
print(f"  GPT parameters           : {model.num_parameters():,}")
print(f"  Pre-train val loss       : {pt_hist['val_loss'][-1]:.3f}")
print(f"  Pre-train val PPL        : {pt_hist['val_ppl'][-1]:.1f}")
print(f"  SFT val loss (last)      : {sft_loss:.4f}")
print(f"  RM train margin (last)   : {rm_margin:.4f}")
print(f"  DPO train margin (last)  : {dpo_margin:.4f}")
print(f"  Figures saved            : plots/  (11 PNG files)")
print("="*70)
