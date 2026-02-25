"""
Comprehensive Visualizations for LLM Concepts
===============================================
This module generates 10 publication-quality plots that cover every
component built in this series:

  Fig 1 — BPE Tokenizer: vocabulary growth curve
  Fig 2 — Token + Positional Embeddings: heatmaps
  Fig 3 — Sinusoidal PE: wave patterns across positions
  Fig 4 — Attention Patterns: causal mask + per-head heatmaps
  Fig 5 — Transformer Block: architecture diagram (text-art + signal)
  Fig 6 — Pre-training: loss curves + perplexity + LR schedule
  Fig 7 — SFT: train vs val loss, instruction-masking illustration
  Fig 8 — Reward Model: score distributions + Bradley-Terry curve
  Fig 9 — RLHF Pipeline: PPO components breakdown
  Fig 10 — DPO: preference margin over training
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

BLUE   = "#2E86AB"
GREEN  = "#28A745"
RED    = "#E63946"
ORANGE = "#F4A261"
PURPLE = "#7B2D8B"
GRAY   = "#6C757D"


# ──────────────────────────────────────────────────────────────
# Helper: save figure
# ──────────────────────────────────────────────────────────────

def savefig(name: str) -> None:
    path = f"{OUT_DIR}/{name}"
    plt.savefig(path, bbox_inches="tight")
    plt.close("all")
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════
# Fig 1 — BPE Vocabulary Growth
# ══════════════════════════════════════════════════════════════

def plot_bpe_vocabulary():
    """Shows how vocabulary size grows as BPE merge operations proceed."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 1 — BPE Tokenizer", fontweight="bold", fontsize=12)

    # ── Left: vocab size vs merges ──────────────────────────────
    ax = axes[0]
    n_chars = 60          # initial character vocab
    n_merges = np.arange(0, 500, 5)
    vocab_sizes = n_chars + n_merges
    ax.plot(n_merges, vocab_sizes, color=BLUE, lw=2)
    ax.axhline(y=256,  color=GREEN,  ls="--", lw=1.2, label="Small model (256)")
    ax.axhline(y=50257, color=RED,   ls="--", lw=1.2, label="GPT-2 (50 257)")
    ax.set_xlabel("Number of BPE Merges")
    ax.set_ylabel("Vocabulary Size")
    ax.set_title("Vocabulary Growth with BPE Merges")
    ax.legend()
    ax.fill_between(n_merges, vocab_sizes, alpha=0.1, color=BLUE)

    # ── Right: token length distribution ───────────────────────
    ax = axes[1]
    # Simulated distribution: most tokens are 3-6 chars (subword units)
    np.random.seed(42)
    char_lens = np.concatenate([
        np.random.randint(1, 3, 100),    # short tokens (very common)
        np.random.randint(3, 7, 250),    # medium tokens (most common)
        np.random.randint(7, 12, 50),    # longer tokens
    ])
    ax.hist(char_lens, bins=range(1, 14), color=PURPLE, alpha=0.7, edgecolor="white")
    ax.set_xlabel("Token Length (characters)")
    ax.set_ylabel("Count in Vocabulary")
    ax.set_title("Token Length Distribution After BPE")

    # Annotation
    ax.annotate("Subword\nsweet spot", xy=(4, 80), fontsize=8,
                xytext=(7, 100), arrowprops=dict(arrowstyle="->", color=GRAY),
                color=GRAY)

    plt.tight_layout()
    savefig("fig01_bpe_tokenizer.png")


# ══════════════════════════════════════════════════════════════
# Fig 2 — Token + Positional Embedding Heatmaps
# ══════════════════════════════════════════════════════════════

def plot_embeddings():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Fig 2 — Token & Positional Embeddings", fontweight="bold", fontsize=12)

    np.random.seed(0)
    d, T = 64, 24

    # ── Token embedding (random) ────────────────────────────────
    tok_emb = np.random.randn(T, d) * 0.5
    im = axes[0].imshow(tok_emb, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    axes[0].set_title("Token Embeddings\n(24 tokens × 64 dims)")
    axes[0].set_xlabel("Embedding Dimension")
    axes[0].set_ylabel("Token Position")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # ── Sinusoidal PE ───────────────────────────────────────────
    pe = np.zeros((T, d))
    positions = np.arange(T)[:, None]
    div = np.exp(-np.arange(0, d, 2) * math.log(10000.0) / d)
    pe[:, 0::2] = np.sin(positions * div)
    pe[:, 1::2] = np.cos(positions * div)
    im2 = axes[1].imshow(pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("Sinusoidal Positional Encoding\n(24 pos × 64 dims)")
    axes[1].set_xlabel("Embedding Dimension")
    axes[1].set_ylabel("Position")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # ── Combined (tok + pos) ────────────────────────────────────
    combined = tok_emb + pe
    im3 = axes[2].imshow(combined, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axes[2].set_title("Token + Positional (Combined)\nInput to Transformer")
    axes[2].set_xlabel("Embedding Dimension")
    axes[2].set_ylabel("Token Position")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    savefig("fig02_embeddings.png")


# ══════════════════════════════════════════════════════════════
# Fig 3 — Sinusoidal PE: Wave Patterns
# ══════════════════════════════════════════════════════════════

def plot_sinusoidal_pe():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Fig 3 — Sinusoidal Positional Encoding Detail", fontweight="bold", fontsize=12)

    T, d = 100, 128
    pe = np.zeros((T, d))
    positions = np.arange(T)[:, None]
    div = np.exp(-np.arange(0, d, 2) * math.log(10000.0) / d)
    pe[:, 0::2] = np.sin(positions * div)
    pe[:, 1::2] = np.cos(positions * div)

    # ── Full heatmap ──────────────────────────────────────────
    im = axes[0, 0].imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_title("Full PE Matrix (dims × positions)")
    axes[0, 0].set_xlabel("Position")
    axes[0, 0].set_ylabel("Dimension")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

    # ── Low-freq dims (long-range patterns) ──────────────────
    ax = axes[0, 1]
    for i, color in zip([0, 2, 4], [BLUE, GREEN, RED]):
        ax.plot(pe[:, i], label=f"dim {i}", color=color, lw=1.5)
    ax.set_title("Low-Frequency Dimensions\n(capture long-range positions)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Value")
    ax.legend()

    # ── High-freq dims (short-range) ─────────────────────────
    ax = axes[1, 0]
    for i, color in zip([60, 62, 64], [ORANGE, PURPLE, GRAY]):
        ax.plot(pe[:, i], label=f"dim {i}", color=color, lw=1.5)
    ax.set_title("High-Frequency Dimensions\n(capture local/fine-grained positions)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Value")
    ax.legend()

    # ── Dot-product similarity between positions ─────────────
    ax = axes[1, 1]
    # Two positions should have higher similarity when close together
    sims = pe @ pe.T                                        # (T, T)
    sims /= (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-8)
    sims /= (np.linalg.norm(pe, axis=1, keepdims=True).T + 1e-8)
    im2 = ax.imshow(sims, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Positional Similarity\n(close positions ≈ similar encoding)")
    ax.set_xlabel("Position i")
    ax.set_ylabel("Position j")
    plt.colorbar(im2, ax=ax, shrink=0.8)

    plt.tight_layout()
    savefig("fig03_sinusoidal_pe.png")


# ══════════════════════════════════════════════════════════════
# Fig 4 — Attention Patterns
# ══════════════════════════════════════════════════════════════

def plot_attention():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle("Fig 4 — Causal Attention Patterns (Multi-Head)", fontweight="bold", fontsize=12)

    T = 12
    np.random.seed(42)
    tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "slept", ".", "It", "was", "warm"]

    # ── Causal mask ──────────────────────────────────────────
    ax = axes[0, 0]
    mask = np.tril(np.ones((T, T)))
    ax.imshow(mask, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Causal Mask\n(lower triangular)")
    ax.set_xticks(range(T)); ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(T)); ax.set_yticklabels(tokens, fontsize=7)

    # ── 6 simulated attention heads with different patterns ──
    patterns = {
        "Head 1\n(local)":      _local_attention(T, 3),
        "Head 2\n(prev token)": _prev_token_attention(T),
        "Head 3\n(broad)":      _broad_attention(T),
        "Head 4\n(noun-verb)":  _diagonal_attention(T, 4),
        "Head 5\n(end focus)":  _end_focus_attention(T),
        "Head 6\n(uniform)":    _uniform_attention(T),
    }
    for (title, attn), ax in zip(patterns.items(), axes.flatten()[1:7]):
        im = ax.imshow(attn * np.tril(np.ones((T, T))), cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(0, T, 3))
        ax.set_yticks(range(0, T, 3))
        plt.colorbar(im, ax=ax, shrink=0.7)

    # ── Average attention (all heads) ─────────────────────────
    ax = axes[1, 3]
    all_heads = np.stack(list(patterns.values()))
    avg_attn  = all_heads.mean(0) * np.tril(np.ones((T, T)))
    im = ax.imshow(avg_attn, cmap="Greens", vmin=0, vmax=0.5)
    ax.set_title("Average Across\nAll Heads", fontsize=8)
    ax.set_xticks(range(0, T, 3))
    ax.set_yticks(range(0, T, 3))
    plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    savefig("fig04_attention_patterns.png")


def _local_attention(T, window):
    a = np.zeros((T, T))
    for i in range(T):
        start = max(0, i - window + 1)
        a[i, start:i+1] = np.random.dirichlet(np.ones(i - start + 1))
    return a

def _prev_token_attention(T):
    a = np.zeros((T, T))
    a[0, 0] = 1
    for i in range(1, T):
        a[i, i-1] = 0.8
        a[i, i]   = 0.2
    return a

def _broad_attention(T):
    a = np.zeros((T, T))
    for i in range(T):
        weights = np.exp(-np.arange(i+1) * 0.3)[::-1]
        a[i, :i+1] = weights / weights.sum()
    return a

def _diagonal_attention(T, offset):
    a = np.zeros((T, T))
    for i in range(T):
        j = max(0, i - offset)
        a[i, j] = 0.7
        a[i, i] = 0.3
    return a

def _end_focus_attention(T):
    a = np.zeros((T, T))
    for i in range(T):
        a[i, :i+1] = 0.1
        a[i, i]    = 0.9 - 0.1 * i / T
        if i > 0:
            a[i, :i+1] /= a[i, :i+1].sum()
    return a

def _uniform_attention(T):
    a = np.zeros((T, T))
    for i in range(T):
        a[i, :i+1] = 1.0 / (i + 1)
    return a


# ══════════════════════════════════════════════════════════════
# Fig 5 — Transformer Block Architecture
# ══════════════════════════════════════════════════════════════

def plot_transformer_block():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")
    fig.suptitle("Fig 5 — Transformer Block (Pre-LayerNorm, GPT-style)",
                 fontweight="bold", fontsize=12)

    def box(ax, xy, w, h, label, color, fontsize=10):
        x, y = xy
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.9, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)

    def arrow(ax, x, y1, y2, color="#555"):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=2)

    def plus(ax, x, y, size=0.6):
        circle = plt.Circle((x, y), size/2, color=GREEN, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, "+", ha="center", va="center", fontsize=14,
                fontweight="bold", color="white", zorder=4)

    cx = 3.0  # center x

    # Layers bottom-to-top
    box(ax, (cx, 0.3),  4, 1.0, "Input x", BLUE)
    arrow(ax, cx+2, 1.3, 2.2)
    box(ax, (cx, 2.2),  4, 0.9, "LayerNorm 1", "#4A4E69")
    arrow(ax, cx+2, 3.1, 4.0)
    box(ax, (cx, 4.0),  4, 1.4, "Multi-Head\nCausal Attention", PURPLE)
    arrow(ax, cx+2, 5.4, 6.5)
    plus(ax, cx+2, 7.1)
    ax.text(cx+2+0.9, 7.1, "Residual", fontsize=8, va="center", color=GRAY)
    # Residual bypass
    ax.annotate("", xy=(cx+3.8, 7.1), xytext=(cx+3.8, 0.8),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5,
                                connectionstyle="arc3,rad=-0.3"), zorder=2)
    arrow(ax, cx+2, 7.6, 8.5)
    box(ax, (cx, 8.5),  4, 0.9, "LayerNorm 2", "#4A4E69")
    arrow(ax, cx+2, 9.4, 10.3)
    box(ax, (cx, 10.3), 4, 2.0,
        "Feed-Forward MLP\nLinear→GELU→Linear", ORANGE)
    arrow(ax, cx+2, 12.3, 13.3)
    plus(ax, cx+2, 13.9)
    ax.text(cx+2+0.9, 13.9, "Residual", fontsize=8, va="center", color=GRAY)
    ax.annotate("", xy=(cx+3.8, 13.9), xytext=(cx+3.8, 8.2),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5,
                                connectionstyle="arc3,rad=-0.3"), zorder=2)
    arrow(ax, cx+2, 14.4, 15.5)
    box(ax, (cx, 15.5), 4, 1.0, "Output x'", BLUE)

    # Equations on the right
    eqs = [
        (14.5, r"$x' = x + \mathrm{FFN}(\mathrm{LN}(x''))$"),
        (12.0, r"$x'' = x + \mathrm{MHA}(\mathrm{LN}(x))$"),
        ( 5.0, r"$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$"),
        ( 2.0, r"$\mathrm{LN}(x) = \frac{x-\mu}{\sigma+\epsilon}\odot\gamma+\beta$"),
    ]
    for y, eq in eqs:
        ax.text(8.5, y, eq, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="#F8F9FA", alpha=0.7))

    savefig("fig05_transformer_block.png")


# ══════════════════════════════════════════════════════════════
# Fig 6 — Pre-Training Curves
# ══════════════════════════════════════════════════════════════

def plot_pretraining():
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Fig 6 — Pre-Training Dynamics", fontweight="bold", fontsize=12)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    np.random.seed(7)
    steps = np.arange(0, 2000, 40)
    N     = len(steps)

    # Simulate training curves with decay + noise
    train_loss = 6.5 * np.exp(-steps / 800) + 2.2 + np.random.randn(N) * 0.1
    val_loss   = 6.5 * np.exp(-steps / 800) + 2.5 + np.random.randn(N) * 0.12
    train_ppl  = np.exp(train_loss)
    val_ppl    = np.exp(val_loss)

    # LR schedule (warmup + cosine)
    warmup = 100
    lr_peak = 4e-4
    lr = np.where(
        steps < warmup,
        lr_peak * steps / warmup,
        lr_peak * 0.1 + 0.5 * (lr_peak - lr_peak * 0.1) * (1 + np.cos(np.pi * (steps - warmup) / (2000 - warmup)))
    )

    # ── Loss ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(steps, train_loss, color=BLUE, lw=2, label="Train Loss")
    ax.plot(steps, val_loss,   color=RED,  lw=2, label="Val Loss", ls="--")
    ax.fill_between(steps, train_loss, val_loss, alpha=0.15, color=ORANGE, label="Generalisation gap")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()

    # ── Perplexity ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.semilogy(steps, train_ppl, color=BLUE, lw=2, label="Train PPL")
    ax2.semilogy(steps, val_ppl,   color=RED,  lw=2, label="Val PPL", ls="--")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Perplexity (log scale)")
    ax2.set_title("Perplexity = exp(loss)")
    ax2.legend()
    ax2.axhline(y=math.e**2.2, color=GRAY, ls=":", lw=1, label=f"Final ≈ {math.e**2.2:.0f}")

    # ── LR Schedule ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, lr * 1e4, color=GREEN, lw=2)
    ax3.axvline(x=warmup, color=GRAY, ls="--", lw=1, label=f"Warmup={warmup}")
    ax3.fill_between(steps[:np.searchsorted(steps, warmup)],
                     lr[:np.searchsorted(steps, warmup)] * 1e4, alpha=0.3, color=GREEN)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("LR × 10⁴")
    ax3.set_title("LR: Warmup + Cosine\nAnnealing")
    ax3.legend(fontsize=7)

    # ── Gradient norm ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    grad_norms = 2.5 * np.exp(-steps / 500) + 0.3 + np.abs(np.random.randn(N) * 0.2)
    clipped    = np.clip(grad_norms, 0, 1.0)
    ax4.plot(steps, grad_norms, color=RED, lw=1.5, alpha=0.7, label="Before clip")
    ax4.plot(steps, clipped,    color=BLUE, lw=1.5, label="After clip (=1.0)")
    ax4.axhline(y=1.0, color=GRAY, ls="--", lw=1)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Gradient Norm")
    ax4.set_title("Gradient Clipping")
    ax4.legend(fontsize=7)

    plt.savefig(f"{OUT_DIR}/fig06_pretraining.png", bbox_inches="tight")
    plt.close("all")
    print(f"  saved → {OUT_DIR}/fig06_pretraining.png")


# ══════════════════════════════════════════════════════════════
# Fig 7 — SFT: Loss + Instruction Masking
# ══════════════════════════════════════════════════════════════

def plot_sft():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fig 7 — Supervised Fine-Tuning (SFT)", fontweight="bold", fontsize=12)

    np.random.seed(3)
    steps = np.arange(0, 300, 15)
    N = len(steps)
    tr = 3.5 * np.exp(-steps / 120) + 0.6 + np.random.randn(N) * 0.08
    vl = 3.5 * np.exp(-steps / 120) + 0.9 + np.random.randn(N) * 0.1

    # ── SFT Loss ─────────────────────────────────────────────
    axes[0].plot(steps, tr, color=BLUE, lw=2, label="Train")
    axes[0].plot(steps, vl, color=RED,  lw=2, label="Val", ls="--")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("SFT Loss (lower LR\nthan pretraining)")
    axes[0].legend()

    # ── Instruction masking illustration ──────────────────────
    ax = axes[1]
    token_labels = ["Below", "is", "an", "inst.", "###", "Inst:", "What", "is", "2+2?",
                    "###", "Resp:", "The", "ans.", "is", "4", "."]
    n_tok = len(token_labels)
    n_prompt = 11  # first 11 tokens are instruction/prompt

    colors = [GRAY if i < n_prompt else GREEN for i in range(n_tok)]
    labels_fmt = ["-100" if i < n_prompt else "active" for i in range(n_tok)]

    ax.barh(range(n_tok), [1]*n_tok, color=colors, alpha=0.7, edgecolor="white")
    for i, (lbl, fmt) in enumerate(zip(token_labels, labels_fmt)):
        ax.text(0.5, i, f'"{lbl}"', ha="center", va="center", fontsize=7,
                color="white" if fmt == "-100" else "black", fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Loss Masking:\nGray=masked(-100), Green=active")
    ax.invert_yaxis()
    # Legend patches
    gray_p  = mpatches.Patch(color=GRAY,  alpha=0.7, label="Instruction (ignored)")
    green_p = mpatches.Patch(color=GREEN, alpha=0.7, label="Response (trained)")
    ax.legend(handles=[gray_p, green_p], fontsize=8, loc="lower right")

    # ── Pre-training vs SFT comparison ───────────────────────
    ax = axes[2]
    cats  = ["Pre-training\nLR", "SFT LR", "Pre-train\nepochs", "SFT\nepochs", "Data\nsize"]
    pt_v  = [4e-4, 0, 10, 0, 100]
    sft_v = [0, 5e-5, 0, 3, 1]
    x     = np.arange(len(cats))
    w     = 0.35
    ax.bar(x - w/2, pt_v,  w, label="Pre-training", color=BLUE,   alpha=0.8)
    ax.bar(x + w/2, sft_v, w, label="SFT",          color=GREEN,  alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
    ax.set_title("Pre-training vs SFT\n(relative magnitudes)")
    ax.legend()
    ax.set_ylabel("Relative Value")

    plt.tight_layout()
    savefig("fig07_sft.png")


# ══════════════════════════════════════════════════════════════
# Fig 8 — Reward Model
# ══════════════════════════════════════════════════════════════

def plot_reward_model():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Fig 8 — Reward Model (Bradley-Terry)", fontweight="bold", fontsize=12)

    np.random.seed(11)

    # ── Score distributions ───────────────────────────────────
    chosen_scores   = np.random.normal(1.5,  0.6, 300)
    rejected_scores = np.random.normal(-0.5, 0.6, 300)
    axes[0].hist(chosen_scores,   bins=30, alpha=0.7, color=GREEN, label="Chosen")
    axes[0].hist(rejected_scores, bins=30, alpha=0.7, color=RED,   label="Rejected")
    axes[0].axvline(x=chosen_scores.mean(),   color=GREEN, ls="--", lw=2)
    axes[0].axvline(x=rejected_scores.mean(), color=RED,   ls="--", lw=2)
    axes[0].set_xlabel("Reward Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Reward Score Distribution\n(after training)")
    axes[0].legend()

    # ── Bradley-Terry sigmoid curve ───────────────────────────
    ax = axes[1]
    margin = np.linspace(-4, 4, 200)
    prob   = 1 / (1 + np.exp(-margin))
    ax.plot(margin, prob, color=BLUE, lw=2)
    ax.fill_between(margin, prob, 0.5, where=prob > 0.5, alpha=0.2, color=GREEN)
    ax.fill_between(margin, prob, 0.5, where=prob < 0.5, alpha=0.2, color=RED)
    ax.axhline(y=0.5, color=GRAY, ls="--", lw=1)
    ax.axvline(x=0.0, color=GRAY, ls="--", lw=1)
    ax.set_xlabel("r(chosen) − r(rejected)")
    ax.set_ylabel("P(chosen > rejected)")
    ax.set_title("Bradley-Terry Model:\nσ(r_w − r_l)")
    ax.text( 1.5, 0.3, "Chosen wins", color=GREEN, fontsize=9)
    ax.text(-3.5, 0.7, "Rejected wins", color=RED,   fontsize=9)

    # ── Training loss + margin ────────────────────────────────
    ax = axes[2]
    steps = np.arange(100)
    rm_loss   = 0.7 * np.exp(-steps / 35) + 0.05 + np.random.randn(100) * 0.02
    rm_margin = 2.0 * (1 - np.exp(-steps / 25)) + np.random.randn(100) * 0.1
    ax2 = ax.twinx()
    ax.plot(steps,  rm_loss,   color=RED,   lw=2, label="Loss")
    ax2.plot(steps, rm_margin, color=GREEN, lw=2, label="Margin", ls="--")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss", color=RED)
    ax2.set_ylabel("Margin r_w − r_l", color=GREEN)
    ax.set_title("Reward Model Training:\nLoss ↓  Margin ↑")
    lines  = [plt.Line2D([0],[0], color=RED,   lw=2, label="Loss"),
              plt.Line2D([0],[0], color=GREEN, lw=2, ls="--", label="Margin")]
    ax.legend(handles=lines)

    plt.tight_layout()
    savefig("fig08_reward_model.png")


# ══════════════════════════════════════════════════════════════
# Fig 9 — PPO RLHF Pipeline
# ══════════════════════════════════════════════════════════════

def plot_rlhf_ppo():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fig 9 — RLHF with PPO", fontweight="bold", fontsize=12)

    # ── Pipeline diagram ─────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def draw_box(ax, x, y, w, h, label, sub, color):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.2", facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h*0.65, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        ax.text(x+w/2, y+h*0.25, sub,   ha="center", va="center",
                fontsize=7, color="white", alpha=0.9)

    draw_box(ax, 0.5, 9.5, 3.5, 1.5, "Policy π_θ",       "SFT model (trainable)", BLUE)
    draw_box(ax, 0.5, 6.5, 3.5, 1.5, "Reference π_ref",  "SFT model (frozen)",    GRAY)
    draw_box(ax, 5.5, 6.5, 3.5, 1.5, "Reward Model r_φ", "Pref model (frozen)",   GREEN)
    draw_box(ax, 5.5, 9.5, 3.5, 1.5, "Value Model V_ψ",  "Critic (trainable)",    ORANGE)
    draw_box(ax, 2.5, 2.0, 5.0, 2.0, "PPO Update",
             "L = min(ratio·A, clip·A)\n− β·KL[π_θ‖π_ref]", PURPLE)

    # Arrows
    for (x1, y1, x2, y2) in [
        (2.25, 9.5, 2.25, 4.0),    # policy → PPO
        (2.25, 6.5, 2.25, 4.0),    # ref → PPO
        (7.25, 6.5, 7.25, 4.0),    # reward → PPO
        (7.25, 9.5, 7.25, 4.0),    # value → PPO
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))

    ax.text(5.0, 0.5, "Prompt → Response → Optimized Policy",
            ha="center", fontsize=9, style="italic", color=GRAY)
    ax.set_title("PPO RLHF Architecture\n(4 model components)")

    # ── PPO training curves ────────────────────────────────────
    ax2 = axes[1]
    np.random.seed(5)
    steps = np.arange(200)
    reward   = 1.8 * (1 - np.exp(-steps / 60)) + np.random.randn(200) * 0.15
    kl       = 0.05 * np.exp(steps / 200)       + np.random.randn(200) * 0.01
    pol_loss = 0.4  * np.exp(-steps / 100)       + np.random.randn(200) * 0.03

    ax2b = ax2.twinx()
    ax2.plot(steps,  reward,   color=GREEN, lw=2, label="Mean Reward")
    ax2.plot(steps,  pol_loss, color=BLUE,  lw=2, label="Policy Loss", ls="--")
    ax2b.plot(steps, kl,       color=RED,   lw=2, label="KL Divergence", ls=":")
    ax2.set_xlabel("PPO Update Step")
    ax2.set_ylabel("Reward / Policy Loss")
    ax2b.set_ylabel("KL(π_θ ‖ π_ref)", color=RED)
    ax2.set_title("PPO Training: Reward ↑ KL ↓")
    lines = [
        plt.Line2D([0],[0], color=GREEN, lw=2,          label="Reward"),
        plt.Line2D([0],[0], color=BLUE,  lw=2, ls="--", label="Policy Loss"),
        plt.Line2D([0],[0], color=RED,   lw=2, ls=":",  label="KL Divergence"),
    ]
    ax2.legend(handles=lines)

    plt.tight_layout()
    savefig("fig09_rlhf_ppo.png")


# ══════════════════════════════════════════════════════════════
# Fig 10 — DPO: Direct Preference Optimization
# ══════════════════════════════════════════════════════════════

def plot_dpo():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Fig 10 — DPO: Direct Preference Optimization", fontweight="bold", fontsize=12)

    np.random.seed(99)
    steps = np.arange(0, 300, 10)
    N = len(steps)

    dpo_loss   = 0.7 * np.exp(-steps/80)  + 0.05 + np.random.randn(N)*0.02
    dpo_margin = 2.5 * (1-np.exp(-steps/70)) + np.random.randn(N)*0.1

    # ── DPO vs RLHF comparison ────────────────────────────────
    ax = axes[0]
    categories = ["Num\nModels", "Need\nReward\nModel", "Stability", "Compute"]
    rlhf_v = [4, 5, 3, 4]
    dpo_v  = [2, 1, 4, 2]
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x-w/2, rlhf_v, w, label="RLHF-PPO", color=PURPLE, alpha=0.8)
    ax.bar(x+w/2, dpo_v,  w, label="DPO",       color=BLUE,   alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Relative complexity / need (1-5)")
    ax.set_title("RLHF-PPO vs DPO\n(lower = simpler / less needed)")
    ax.legend()

    # ── DPO loss + margin ─────────────────────────────────────
    ax = axes[1]
    ax.plot(steps, dpo_loss,   color=RED,   lw=2, label="DPO Loss")
    ax.plot(steps, dpo_margin, color=GREEN, lw=2, label="Reward Margin")
    ax.axhline(y=0, color=GRAY, ls="--", lw=1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Value")
    ax.set_title("DPO Training Curves\nβ=0.1")
    ax.legend()

    # ── Implicit reward visualization ─────────────────────────
    ax = axes[2]
    beta = 0.1
    # Simulate lp_policy and lp_ref for two responses
    policy_chosen   = np.linspace(-5, 0, 100)
    ref_chosen      = -3.0 * np.ones(100)
    policy_rejected = np.linspace(-8, -3, 100)
    ref_rejected    = -5.0 * np.ones(100)

    r_chosen   = beta * (policy_chosen   - ref_chosen)
    r_rejected = beta * (policy_rejected - ref_rejected)
    margin     = r_chosen - r_rejected
    ax.plot(np.linspace(0,1,100), r_chosen,   color=GREEN, lw=2, label="Implicit r(chosen)")
    ax.plot(np.linspace(0,1,100), r_rejected, color=RED,   lw=2, label="Implicit r(rejected)")
    ax.fill_between(np.linspace(0,1,100), r_chosen, r_rejected, alpha=0.2, color=BLUE, label="Margin")
    ax.axhline(y=0, color=GRAY, ls="--", lw=1)
    ax.set_xlabel("Training Progress")
    ax.set_ylabel("Implicit Reward β·(log π_θ − log π_ref)")
    ax.set_title("DPO Implicit Rewards:\nr_chosen > r_rejected")
    ax.legend(fontsize=8)

    plt.tight_layout()
    savefig("fig10_dpo.png")


# ══════════════════════════════════════════════════════════════
# Fig 11 — Full LLM Training Pipeline Summary
# ══════════════════════════════════════════════════════════════

def plot_full_pipeline():
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle("Fig 11 — Full LLM Training Pipeline", fontweight="bold", fontsize=13)
    ax.set_xlim(0, 20)
    ax.set_ylim(-1, 5)
    ax.axis("off")

    stages = [
        ("Raw Text\nData",        GRAY,   0.5,  "trillion\ntokens"),
        ("BPE\nTokenizer",        BLUE,   2.5,  "vocab=50K"),
        ("Pre-Train\n(GPT)",      PURPLE, 5.0,  "next-token\nprediction"),
        ("SFT",                   GREEN,  8.5,  "instruction\nfollowing"),
        ("Reward\nModel",         ORANGE, 11.5, "pref pairs\nBradley-Terry"),
        ("RLHF / DPO",            RED,    14.5, "align with\nhuman prefs"),
        ("Aligned\nLLM",          BLUE,   17.5, "ChatGPT\nClaude etc."),
    ]

    for label, color, x, sub in stages:
        # Stage box
        rect = mpatches.FancyBboxPatch((x, 1.0), 2.5, 2.0,
            boxstyle="round,pad=0.2", facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x+1.25, 2.3, label, ha="center", va="center",
                fontsize=9,  fontweight="bold", color="white")
        ax.text(x+1.25, 1.5, sub,   ha="center", va="center",
                fontsize=7.5, color="white", alpha=0.9)

    # Arrows between stages
    for x in [3.0, 5.5, 8.0, 11.0, 14.0, 17.0]:
        ax.annotate("", xy=(x, 2.0), xytext=(x-0.1, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.5))

    # Labels below
    ax.text(10.0, 0.1, "Increasing alignment with human intent →",
            ha="center", fontsize=10, color=GRAY, style="italic")

    savefig("fig11_full_pipeline.png")


# ══════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING ALL VISUALIZATIONS")
    print(f"Output directory: {OUT_DIR}/")
    print("=" * 60)

    plots = [
        ("BPE Tokenizer",              plot_bpe_vocabulary),
        ("Embeddings",                 plot_embeddings),
        ("Sinusoidal PE",              plot_sinusoidal_pe),
        ("Attention Patterns",         plot_attention),
        ("Transformer Block",          plot_transformer_block),
        ("Pre-training Curves",        plot_pretraining),
        ("SFT",                        plot_sft),
        ("Reward Model",               plot_reward_model),
        ("RLHF / PPO",                 plot_rlhf_ppo),
        ("DPO",                        plot_dpo),
        ("Full Pipeline",              plot_full_pipeline),
    ]

    for name, fn in plots:
        print(f"\n[{name}]")
        fn()

    print("\n" + "=" * 60)
    print(f"All {len(plots)} figures generated in '{OUT_DIR}/'")
    print("=" * 60)

    # Print index
    print("\nFigure index:")
    for i, (name, _) in enumerate(plots, 1):
        print(f"  fig{i:02d} — {name}")
