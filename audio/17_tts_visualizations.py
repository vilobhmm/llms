"""
TTS Chapter 8: Visualizations
================================
12 figures illustrating every stage of the Qwen3-TTS pipeline:

  Fig 1  — Text Normalization & Phoneme Tokenization
  Fig 2  — Phoneme Vocabulary (ARPAbet distribution)
  Fig 3  — Mel Spectrogram (synthetic speech pattern)
  Fig 4  — RVQ Codebook Usage (per quantizer level)
  Fig 5  — Audio Codec Reconstruction (input vs reconstructed mel)
  Fig 6  — AR Model: Attention Heatmap
  Fig 7  — Token Probability Distribution (AR sampling)
  Fig 8  — NAR Model: Multi-codebook Prediction
  Fig 9  — Flow Matching Trajectory (x_0 → x_1 in 5 steps)
  Fig 10 — Waveform: Griffin-Lim output
  Fig 11 — Training Curves (codec / VALLE / flow matching losses)
  Fig 12 — TTS Pipeline Architecture Diagram
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os, sys, importlib.util, math, random

# ── Load TTS modules ─────────────────────────────────────────
def _load(name, fname):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(here, fname)
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_tp  = _load("tp",  "10_text_processing.py")
_ac  = _load("ac",  "11_audio_codec.py")
_te  = _load("te",  "12_text_encoder.py")
_am  = _load("am",  "13_acoustic_model.py")
_fm  = _load("fm",  "14_flow_matching.py")
_tts = _load("tts", "15_tts_model.py")
_tr  = _load("tr",  "16_tts_training.py")

PhonemeTokenizer  = _tp.PhonemeTokenizer
normalize_text    = _tp.normalize_text
PHONEMES          = _tp.PHONEMES
PHONEME2ID        = _tp.PHONEME2ID
TTSConfig         = _tts.TTSConfig
TTSModel          = _tts.TTSModel
SyntheticTTSDataset = _tr.SyntheticTTSDataset
SAMPLE_TEXTS        = _tr.SAMPLE_TEXTS
euler_solve         = _fm.euler_solve

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots", "tts")
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = plt.cm.tab10.colors
CMAP   = "magma"

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  Saved: {path}")


def synthetic_mel(n_mels=80, T=200, seed=42, speech_like=True):
    """Generate a plausible-looking mel spectrogram."""
    rng = np.random.default_rng(seed)
    mel = rng.standard_normal((n_mels, T)).astype(np.float32)
    if speech_like:
        # Low frequencies are louder
        envelope = np.exp(-np.arange(n_mels) / 20).reshape(-1, 1)
        mel = mel * 0.3 + envelope * np.sin(
            np.linspace(0, 6 * np.pi, T)[None, :] +
            np.linspace(0, np.pi, n_mels)[:, None]
        )
        # Add voiced/unvoiced structure
        voiced = (np.sin(np.linspace(0, 4 * np.pi, T)) > 0).astype(float)
        mel += 0.5 * envelope * voiced[None, :]
    return mel - mel.mean()


# ──────────────────────────────────────────────
# Figure 1 — Text Normalization & Tokenization
# ──────────────────────────────────────────────

def fig_text_processing():
    print("[Fig 1] Text normalization & tokenization")
    tok    = PhonemeTokenizer()
    texts  = [
        "Dr. Smith trained 42 models.",
        "The 1st dataset has 100 samples.",
        "Hello world!",
        "Text-to-speech is amazing.",
    ]
    fig, axes = plt.subplots(len(texts), 1, figsize=(12, 6))
    fig.suptitle("Fig 1 — Text Normalization & Phoneme Tokenization", fontsize=13, fontweight="bold")

    for ax, text in zip(axes, texts):
        norm  = normalize_text(text)
        phons = tok.text_to_phonemes(text)
        ids   = tok.encode(text)

        ax.axis("off")
        row = f"In: {text!r}\n"
        row += f"  Normalized: {norm!r}\n"
        row += f"  Phonemes ({len(phons)}): {' '.join(phons[:18])}{'…' if len(phons)>18 else ''}\n"
        row += f"  IDs ({len(ids)}): {ids[:12]}{'…' if len(ids)>12 else ''}"
        ax.text(0.01, 0.5, row, transform=ax.transAxes,
                fontsize=8.5, va="center", family="monospace",
                bbox=dict(boxstyle="round", fc="#f0f4ff", ec="#aab4cc"))

    plt.tight_layout()
    save(fig, "fig01_text_processing.png")


# ──────────────────────────────────────────────
# Figure 2 — Phoneme Vocabulary
# ──────────────────────────────────────────────

def fig_phoneme_vocab():
    print("[Fig 2] Phoneme vocabulary")
    tok    = PhonemeTokenizer()
    counts = {p: 0 for p in PHONEMES}
    for text in SAMPLE_TEXTS:
        for p in tok.text_to_phonemes(text):
            if p in counts:
                counts[p] += 1

    phonemes = list(counts.keys())
    freqs    = list(counts.values())
    colors   = [COLORS[0] if p[0] in "AEIOU" else COLORS[1] for p in phonemes]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(phonemes, freqs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("ARPAbet Phoneme")
    ax.set_ylabel("Frequency in sample texts")
    ax.set_title("Fig 2 — ARPAbet Phoneme Distribution", fontweight="bold")
    ax.tick_params(axis="x", rotation=75, labelsize=8)
    patches = [
        mpatches.Patch(color=COLORS[0], label="Vowel"),
        mpatches.Patch(color=COLORS[1], label="Consonant"),
    ]
    ax.legend(handles=patches)
    plt.tight_layout()
    save(fig, "fig02_phoneme_vocab.png")


# ──────────────────────────────────────────────
# Figure 3 — Mel Spectrogram
# ──────────────────────────────────────────────

def fig_mel_spectrogram():
    print("[Fig 3] Mel spectrogram")
    mel = synthetic_mel(n_mels=80, T=250)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle("Fig 3 — Mel Spectrogram", fontweight="bold")

    im0 = axes[0].imshow(mel, aspect="auto", origin="lower",
                          cmap=CMAP, vmin=-2, vmax=2)
    axes[0].set_ylabel("Mel bin")
    axes[0].set_title("Log-Mel Spectrogram (synthetic speech)")
    plt.colorbar(im0, ax=axes[0], label="log energy")

    # Energy over time
    energy = mel.max(0)
    axes[1].fill_between(range(len(energy)), energy, alpha=0.6, color=COLORS[0])
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Peak mel energy")
    axes[1].set_title("Energy profile (voiced / unvoiced regions)")

    plt.tight_layout()
    save(fig, "fig03_mel_spectrogram.png")


# ──────────────────────────────────────────────
# Figure 4 — RVQ Codebook Usage
# ──────────────────────────────────────────────

def fig_rvq_codebook():
    print("[Fig 4] RVQ codebook usage")
    N_Q, K, T = 4, 512, 200
    rng = np.random.default_rng(0)

    # Simulate: lower codebooks use more evenly-distributed codes
    all_codes = []
    for q in range(N_Q):
        concentration = 2.0 ** q        # higher levels → more concentrated
        alpha         = np.ones(K) / concentration
        code_probs    = rng.dirichlet(alpha)
        codes         = rng.choice(K, size=T, p=code_probs)
        all_codes.append(codes)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle("Fig 4 — RVQ Codebook Usage per Quantizer Level", fontweight="bold")

    for q, (ax, codes) in enumerate(zip(axes.flat, all_codes)):
        hist, _ = np.histogram(codes, bins=50, range=(0, K))
        ax.bar(range(len(hist)), hist, color=COLORS[q], alpha=0.8)
        entropy = -np.sum(
            (hist / hist.sum()) * np.log(hist / hist.sum() + 1e-8)
        )
        ax.set_title(f"Codebook {q+1}  (H={entropy:.2f} nats)")
        ax.set_xlabel("Code index")
        ax.set_ylabel("Count")
    plt.tight_layout()
    save(fig, "fig04_rvq_codebook.png")


# ──────────────────────────────────────────────
# Figure 5 — Codec Reconstruction
# ──────────────────────────────────────────────

def fig_codec_reconstruction():
    print("[Fig 5] Codec reconstruction (forward pass)")
    device = torch.device("cpu")
    cfg    = TTSConfig(size="small")
    model  = TTSModel(cfg).to(device)
    model.eval()

    # Synthetic waveform (2 seconds @ 8kHz equivalent)
    wav   = torch.randn(1, 1, 8000)
    with torch.no_grad():
        recon, codes, _ = model.forward_codec(wav)
        mel_orig  = model.mel_extractor(wav)
        mel_recon = model.mel_extractor(recon[..., :wav.shape[-1]])

    mel_o = mel_orig[0].numpy()
    mel_r = mel_recon[0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Fig 5 — Audio Codec: Original vs Reconstructed Mel", fontweight="bold")

    for ax, mel, title in zip(
        axes,
        [mel_o, mel_r, mel_o - mel_r],
        ["Original", "Reconstructed (codec)", "Residual (|Δ|)"],
    ):
        cmap = CMAP if "Residual" not in title else "RdBu_r"
        im   = ax.imshow(mel, aspect="auto", origin="lower",
                         cmap=cmap, vmin=-2, vmax=2)
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel bin" if "Original" in title else "")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save(fig, "fig05_codec_reconstruction.png")


# ──────────────────────────────────────────────
# Figure 6 — AR Attention Heatmap
# ──────────────────────────────────────────────

def fig_ar_attention():
    print("[Fig 6] AR model attention heatmap")
    T_text, T_audio = 12, 20

    # Simulate a reasonable attention pattern:
    # audio tokens attend back to recent audio + text positions
    rng   = np.random.default_rng(42)
    total = T_text + 1 + T_audio
    attn  = rng.random((T_audio, total))

    # Text alignment: each audio frame peaks at one text token
    for i in range(T_audio):
        text_peak = int(i * T_text / T_audio)
        attn[i, text_peak] += 3.0
    # Causal: zero out future audio positions
    for i in range(T_audio):
        attn[i, T_text + 1 + i + 1:] = 0.0

    attn = attn / (attn.sum(1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(attn, aspect="auto", cmap="Blues")
    ax.set_xlabel("Key position  [text tokens | sep | audio tokens →]")
    ax.set_ylabel("Audio query position")
    ax.set_title("Fig 6 — AR Model Attention (text-audio alignment)", fontweight="bold")
    ax.axvline(T_text - 0.5, color="red", linewidth=2, linestyle="--", label="text|audio boundary")
    ax.axvline(T_text + 0.5, color="orange", linewidth=2, linestyle="--", label="sep token")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, label="attention weight")
    plt.tight_layout()
    save(fig, "fig06_ar_attention.png")


# ──────────────────────────────────────────────
# Figure 7 — Token Probability Distribution
# ──────────────────────────────────────────────

def fig_token_probs():
    print("[Fig 7] AR sampling — token probability distributions")
    K   = 512
    rng = np.random.default_rng(7)

    temperatures = [0.5, 1.0, 1.5]
    logits_raw   = rng.standard_normal(K).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Fig 7 — AR Token Probability Distributions (top-50 shown)",
                 fontweight="bold")

    for ax, temp in zip(axes, temperatures):
        logits = torch.tensor(logits_raw) / temp
        probs  = F.softmax(logits, dim=-1).numpy()
        top_idx = np.argsort(probs)[-50:][::-1]
        ax.bar(range(50), probs[top_idx], color=COLORS[0], alpha=0.8)
        ax.set_title(f"Temperature = {temp}")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Probability" if temp == 0.5 else "")
        ax.set_ylim(0, probs[top_idx[0]] * 1.2)

    plt.tight_layout()
    save(fig, "fig07_token_probs.png")


# ──────────────────────────────────────────────
# Figure 8 — NAR Multi-codebook Prediction
# ──────────────────────────────────────────────

def fig_nar_codebooks():
    print("[Fig 8] NAR multi-codebook prediction")
    N_Q, T, K = 4, 60, 512
    rng = np.random.default_rng(8)

    codes = [rng.integers(0, K, size=T) for _ in range(N_Q)]

    fig, axes = plt.subplots(N_Q, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Fig 8 — NAR Multi-Codebook Predictions (per-frame tokens)",
                 fontweight="bold")

    for q, (ax, c) in enumerate(zip(axes, codes)):
        ax.step(range(T), c, where="mid", color=COLORS[q], linewidth=1.2)
        ax.fill_between(range(T), c, alpha=0.2, color=COLORS[q], step="mid")
        ax.set_ylabel(f"C{q+1}")
        ax.set_ylim(-10, K + 10)
        ax.set_yticks([0, K // 2, K])
        # AR prediction line
        if q == 0:
            ax.set_title("Codebook 1 (AR — autoregressive)")
        else:
            ax.set_title(f"Codebook {q+1} (NAR — parallel)")

    axes[-1].set_xlabel("Audio frame")
    plt.tight_layout()
    save(fig, "fig08_nar_codebooks.png")


# ──────────────────────────────────────────────
# Figure 9 — Flow Matching Trajectory
# ──────────────────────────────────────────────

def fig_flow_trajectory():
    print("[Fig 9] Flow matching trajectory")
    device = torch.device("cpu")
    cfg    = TTSConfig(size="small")
    model  = TTSModel(cfg).to(device)
    model.eval()

    n_mels, T = cfg.n_mels, 40
    cond  = torch.tensor(synthetic_mel(n_mels, T)).unsqueeze(0)
    x_0   = torch.randn(1, n_mels, T)
    x_1   = cond.clone()

    # Interpolate 5 snapshots along the ODE trajectory (without a trained model,
    # we use the straight-line ground-truth path for illustration)
    steps   = 6
    snapshots = []
    for i in range(steps):
        t  = i / (steps - 1)
        xt = (1 - t) * x_1 + t * x_0
        snapshots.append(xt[0].numpy())

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle("Fig 9 — Flow Matching Trajectory  (noise→mel, t=0→1)",
                 fontweight="bold")

    titles = [f"t={i/(steps-1):.2f}" for i in range(steps)]
    for ax, snap, title in zip(axes.flat, snapshots, titles):
        im = ax.imshow(snap, aspect="auto", origin="lower", cmap=CMAP,
                        vmin=-2, vmax=2)
        ax.set_title(title)
        ax.axis("off")

    plt.colorbar(im, ax=axes.flat[-1], label="amplitude")
    plt.tight_layout()
    save(fig, "fig09_flow_trajectory.png")


# ──────────────────────────────────────────────
# Figure 10 — Griffin-Lim Waveform
# ──────────────────────────────────────────────

def fig_waveform():
    print("[Fig 10] Griffin-Lim waveform output")
    device = torch.device("cpu")
    cfg    = TTSConfig(size="small")
    model  = TTSModel(cfg).to(device)
    model.eval()

    log_mel = torch.tensor(synthetic_mel(cfg.n_mels, 80)).unsqueeze(0)
    with torch.no_grad():
        gl = _fm.GriffinLim(
            n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, n_iter=10
        )
        wav = gl(log_mel)[0].numpy()

    sr = cfg.sample_rate
    t  = np.arange(len(wav)) / sr

    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    fig.suptitle("Fig 10 — Griffin-Lim Vocoder Output", fontweight="bold")

    axes[0].plot(t, wav, linewidth=0.6, color=COLORS[0], alpha=0.85)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Synthesized Waveform")
    axes[0].set_xlabel("Time (s)")

    # Spectrogram of the output
    N_FFT = 512
    win   = np.hanning(N_FFT)
    hop   = N_FFT // 4
    n_fr  = (len(wav) - N_FFT) // hop + 1
    spec  = np.zeros((N_FFT // 2 + 1, n_fr))
    for i in range(n_fr):
        frame       = wav[i * hop: i * hop + N_FFT] * win
        spec[:, i]  = np.abs(np.fft.rfft(frame))
    spec = np.log1p(spec)

    axes[1].imshow(spec, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_ylabel("Frequency bin")
    axes[1].set_xlabel("Frame")
    axes[1].set_title("Spectrogram of Griffin-Lim output")

    plt.tight_layout()
    save(fig, "fig10_waveform.png")


# ──────────────────────────────────────────────
# Figure 11 — Training Curves
# ──────────────────────────────────────────────

def fig_training_curves():
    print("[Fig 11] Training curves")
    torch.manual_seed(0)

    # Use simulated smooth curves for the figure
    def smooth_loss(start, end, n, noise=0.08):
        t    = np.linspace(0, 1, n)
        base = start + (end - start) * t
        return base + noise * (np.random.default_rng(99).random(n) - 0.5)

    epochs  = np.arange(1, 9)
    codec_l = smooth_loss(1.2, 0.35, 8)
    ar_l    = smooth_loss(6.3, 5.8, 8)
    nar_l   = smooth_loss(6.1, 5.6, 8)
    fm_l    = smooth_loss(0.9, 0.12, 8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Fig 11 — Training Loss Curves (3 phases)", fontweight="bold")

    axes[0].plot(epochs, codec_l, "o-", color=COLORS[0], label="codec")
    axes[0].set_title("Phase 1: Codec")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, ar_l,  "o-", color=COLORS[1], label="AR (cross-entropy)")
    axes[1].plot(epochs, nar_l, "s--", color=COLORS[2], label="NAR (cross-entropy)")
    axes[1].set_title("Phase 2: VALLE")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(epochs, fm_l, "o-", color=COLORS[3], label="flow matching (MSE)")
    axes[2].set_title("Phase 3: Flow Matching")
    axes[2].set_xlabel("Epoch"); axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlim(1, 8)

    plt.tight_layout()
    save(fig, "fig11_training_curves.png")


# ──────────────────────────────────────────────
# Figure 12 — Architecture Diagram
# ──────────────────────────────────────────────

def fig_architecture():
    print("[Fig 12] TTS pipeline architecture diagram")
    fig = plt.figure(figsize=(14, 7))
    ax  = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)

    def box(x, y, w, h, label, color, fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="gray", linewidth=1.5,
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", wrap=True, zorder=3)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            zorder=1
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
            ax.text(mx, my, label, ha="center", fontsize=7.5, color="#444")

    # Boxes
    box(0.3, 5.2, 2.0, 1.2, "Raw Text\n\"Hello world.\"",     "#dce9ff")
    box(0.3, 3.4, 2.0, 1.2, "Phoneme\nTokenizer\n(ARPAbet)",  "#ffecd2")
    box(0.3, 1.5, 2.0, 1.2, "Text\nEncoder\n(Transformer)",   "#d2f5e3")
    box(3.0, 3.0, 2.8, 2.2, "VALL-E\nAcoustic Model\nAR → c₁\nNAR → c₂…cₙ", "#fff0d2")
    box(6.5, 3.5, 2.2, 1.2, "Audio Codec\n(RVQ decode)",       "#f0d2ff")
    box(6.5, 1.5, 2.2, 1.2, "Flow Matching\nVocoder",          "#d2edff")
    box(9.5, 3.5, 2.2, 1.2, "Rough Mel\nSpectrogram",          "#e8e8e8")
    box(9.5, 1.5, 2.2, 1.2, "Refined Mel\nSpectrogram",        "#e8e8e8")
    box(12.2, 2.4, 1.5, 1.2, "Waveform\n🔊",                   "#d2f5e3")

    # Arrows
    arrow(1.3, 5.2, 1.3, 4.6, "")
    arrow(1.3, 3.4, 1.3, 2.7, "")
    arrow(1.3, 1.5, 3.0, 4.1, "text\nhidden")
    arrow(2.3, 3.4, 3.0, 4.0, "token IDs")
    arrow(5.8, 4.1, 6.5, 4.1, "RVQ codes\n(B,T,N_Q)")
    arrow(7.6, 3.5, 9.5, 4.1, "")
    arrow(7.6, 2.1, 9.5, 2.1, "refined mel")
    arrow(8.7, 4.1, 8.7, 2.7, "rough mel\n(cond)")
    arrow(11.7, 4.1, 12.2, 3.0, "")
    arrow(11.7, 2.1, 12.2, 2.7, "")

    # Labels
    ax.text(7, 0.7, "Griffin-Lim", ha="center", fontsize=8, style="italic", color="#666")
    ax.text(7, 6.5,
            "Qwen3-TTS Style Pipeline  "
            "(Phoneme Tokenizer → TextEncoder → VALLE AR+NAR → Codec → FlowMatching → Waveform)",
            ha="center", fontsize=10, fontweight="bold", color="#222")

    plt.tight_layout()
    save(fig, "fig12_architecture.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TTS VISUALIZATIONS — Generating 12 figures")
    print("=" * 60)

    fig_text_processing()
    fig_phoneme_vocab()
    fig_mel_spectrogram()
    fig_rvq_codebook()
    fig_codec_reconstruction()
    fig_ar_attention()
    fig_token_probs()
    fig_nar_codebooks()
    fig_flow_trajectory()
    fig_waveform()
    fig_training_curves()
    fig_architecture()

    print(f"\nAll figures saved to: {PLOTS_DIR}")
