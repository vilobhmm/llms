"""
TTS Chapter 7: Training Pipeline
==================================
Three-phase training strategy (common in real TTS systems):

  Phase 1 — Codec pretraining
  ────────────────────────────
  Train AudioCodec (Encoder + RVQ + Decoder) to reconstruct waveforms.
  Loss: spectral reconstruction + commitment

  Phase 2 — Acoustic model training
  ────────────────────────────────────
  Freeze codec.  Train VALLE (AR + NAR) to predict codec tokens from text.
  Loss_AR  = CrossEntropy(c1_pred, c1_true)
  Loss_NAR = sum_n CrossEntropy(cn_pred, cn_true)

  Phase 3 — Flow matching vocoder
  ────────────────────────────────
  Freeze codec + VALLE.  Train FlowMatchingNet to refine mel quality.
  Loss = MSE(v_θ(x_t, t, cond), u_t)   (CFM objective)

Synthetic dataset:
  Real training requires large (LibriTTS, LJSpeech, etc.) datasets.
  For this demo we synthesize data:
    • Text → phoneme token IDs (from our tokenizer)
    • "Audio" → small random tensors with realistic shapes
  This validates that all data paths, losses, and gradient flows work.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import random, math, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))

# ── Load full TTS model from chapter 6 ───────────────────────
import importlib.util as _ilu

def _load(name: str, fname: str):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = _ilu.spec_from_file_location(name, os.path.join(here, fname))
    m    = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_tp  = _load("tp",  "10_text_processing.py")
_tts = _load("tts", "15_tts_model.py")
_fm  = _load("fm",  "14_flow_matching.py")

PhonemeTokenizer  = _tp.PhonemeTokenizer
TTSModel          = _tts.TTSModel
TTSConfig         = _tts.TTSConfig
flow_matching_loss = _fm.flow_matching_loss


# ──────────────────────────────────────────────
# 1.  Synthetic Dataset
# ──────────────────────────────────────────────

SAMPLE_TEXTS = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Deep learning is transforming speech synthesis.",
    "Text to speech models convert text into audio.",
    "Neural vocoders produce high quality waveforms.",
    "The mel spectrogram captures frequency content.",
    "Attention mechanisms align text and audio.",
    "Flow matching learns to denoise mel spectrograms.",
    "Residual vector quantization encodes audio.",
    "The encoder maps text to hidden representations.",
    "Hello, how are you today?",
    "Speech synthesis has improved dramatically.",
    "Language models can now generate natural speech.",
    "The voice sounds warm and expressive.",
    "Training requires large datasets and compute.",
]


class SyntheticTTSDataset(Dataset):
    """
    Synthetic dataset that creates fake (text, audio) pairs.
    Shapes match a real dataset (LJSpeech-like) but values are random.

    Each sample:
      text_ids   : (T_text,)      phoneme token IDs
      audio_codes: (T_audio, N_Q) fake RVQ codes
      mel        : (n_mels, T_mel) fake mel spectrogram
      waveform   : (1, T_wav)     fake waveform
    """

    def __init__(
        self,
        cfg:       TTSConfig,
        n_samples: int = 128,
        seed:      int = 42,
    ):
        self.cfg    = cfg
        self.tok    = PhonemeTokenizer()
        rng         = random.Random(seed)

        self.samples: List[Dict] = []
        for i in range(n_samples):
            text = rng.choice(SAMPLE_TEXTS)
            ids  = self.tok.encode(text, add_bos=True, add_eos=True)
            # Fake "audio" — random duration between 50 and 150 codec frames
            T_audio = rng.randint(50, 150)
            T_mel   = T_audio
            T_wav   = T_audio * cfg.hop_length

            self.samples.append({
                "text":       text,
                "text_ids":   ids,
                "T_audio":    T_audio,
                "T_mel":      T_mel,
                "T_wav":      T_wav,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s   = self.samples[idx]
        cfg = self.cfg
        rng = torch.Generator()
        rng.manual_seed(idx)

        return {
            "text_ids":    torch.tensor(s["text_ids"], dtype=torch.long),
            "audio_c1":    torch.randint(0, cfg.codebook_size, (s["T_audio"],)),
            "audio_codes": torch.randint(0, cfg.codebook_size,
                                         (s["T_audio"], cfg.num_quantizers)),
            "mel":         torch.randn(cfg.n_mels, s["T_mel"]),
            "waveform":    torch.randn(1, s["T_wav"]),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences in a batch."""
    # Text
    max_text = max(x["text_ids"].shape[0] for x in batch)
    text_pad = torch.stack([
        F.pad(x["text_ids"], (0, max_text - x["text_ids"].shape[0]))
        for x in batch
    ])

    # Audio codebook-1 (AR targets)
    max_a = max(x["audio_c1"].shape[0] for x in batch)
    c1_pad = torch.stack([
        F.pad(x["audio_c1"], (0, max_a - x["audio_c1"].shape[0]))
        for x in batch
    ])

    # Audio all codes (NAR inputs)
    max_a2 = max(x["audio_codes"].shape[0] for x in batch)
    N_Q    = batch[0]["audio_codes"].shape[1]
    codes_pad = torch.stack([
        F.pad(x["audio_codes"], (0, 0, 0, max_a2 - x["audio_codes"].shape[0]))
        for x in batch
    ])

    # Mel
    max_mel = max(x["mel"].shape[-1] for x in batch)
    n_mels  = batch[0]["mel"].shape[0]
    mel_pad = torch.stack([
        F.pad(x["mel"], (0, max_mel - x["mel"].shape[-1]))
        for x in batch
    ])

    # Waveform
    max_wav = max(x["waveform"].shape[-1] for x in batch)
    wav_pad = torch.stack([
        F.pad(x["waveform"], (0, max_wav - x["waveform"].shape[-1]))
        for x in batch
    ])

    return {
        "text_ids":    text_pad,
        "audio_c1":    c1_pad,
        "audio_codes": codes_pad,
        "mel":         mel_pad,
        "waveform":    wav_pad,
    }


# ──────────────────────────────────────────────
# 2.  Training Utilities
# ──────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""
    def __init__(
        self,
        optimizer:   optim.Optimizer,
        warmup_steps: int,
        total_steps:  int,
        eta_min:      float = 1e-6,
    ):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.eta_min      = eta_min
        self._step        = 0
        self.base_lrs     = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            scale = self._step / max(1, self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.eta_min + 0.5 * (1 - self.eta_min) * (
                1 + math.cos(math.pi * progress)
            )
        for g, lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = lr * scale

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class AverageMeter:
    """Track running average of a metric."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


# ──────────────────────────────────────────────
# 3.  Phase 1: Codec Training
# ──────────────────────────────────────────────

def train_codec(
    model:      TTSModel,
    loader:     DataLoader,
    optimizer:  optim.Optimizer,
    scheduler:  WarmupCosineScheduler,
    device:     torch.device,
    n_epochs:   int = 3,
    grad_clip:  float = 1.0,
    log_every:  int = 10,
) -> List[Dict]:
    """
    Train the AudioCodec (Encoder + RVQ + Decoder).
    Uses spectral + commitment losses.
    """
    model.codec.train()
    history = []

    for epoch in range(1, n_epochs + 1):
        loss_m = AverageMeter()
        t0 = time.time()

        for step, batch in enumerate(loader, 1):
            wav = batch["waveform"].to(device)         # (B, 1, T)

            recon, codes, commit_loss = model.forward_codec(wav)

            # Match lengths
            min_len = min(wav.shape[-1], recon.shape[-1])
            wav_t   = wav[..., :min_len]
            recon_t = recon[..., :min_len]

            # L1 waveform loss
            wav_loss = F.l1_loss(recon_t, wav_t)

            # Spectral loss (MSE on magnitude spectrum)
            def _stft(x):
                return torch.stft(
                    x.squeeze(1), n_fft=512, hop_length=128,
                    win_length=512,
                    window=torch.hann_window(512, device=x.device),
                    return_complex=True
                ).abs()

            spec_loss = F.mse_loss(_stft(recon_t), _stft(wav_t))
            total_loss = wav_loss + 0.1 * spec_loss + commit_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.codec.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            loss_m.update(total_loss.item())
            if step % log_every == 0:
                print(f"  Codec | ep{epoch} step{step:4d} | "
                      f"loss={loss_m.avg:.4f} lr={scheduler.get_lr():.2e}")

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "codec_loss": loss_m.avg, "time": elapsed})
        print(f"  Codec epoch {epoch}/{n_epochs}: loss={loss_m.avg:.4f} ({elapsed:.1f}s)")

    return history


# ──────────────────────────────────────────────
# 4.  Phase 2: VALLE Training (AR + NAR)
# ──────────────────────────────────────────────

def train_valle(
    model:      TTSModel,
    loader:     DataLoader,
    optimizer:  optim.Optimizer,
    scheduler:  WarmupCosineScheduler,
    device:     torch.device,
    n_epochs:   int = 3,
    grad_clip:  float = 1.0,
    log_every:  int = 10,
) -> List[Dict]:
    """
    Train the VALLE AR + NAR models.
    Codec is frozen; only VALLE parameters updated.
    """
    model.codec.eval()
    for p in model.codec.parameters():
        p.requires_grad_(False)

    model.valle.train()
    model.text_encoder.train()
    history = []

    for epoch in range(1, n_epochs + 1):
        ar_m  = AverageMeter()
        nar_m = AverageMeter()
        t0    = time.time()

        for step, batch in enumerate(loader, 1):
            text_ids   = batch["text_ids"].to(device)    # (B, T_text)
            audio_c1   = batch["audio_c1"].to(device)    # (B, T_audio) codebook-1
            audio_codes = batch["audio_codes"].to(device) # (B, T_audio, N_Q)

            # ── AR loss ───────────────────────────────
            # Teacher-force: predict c1[t] given c1[0..t-1]
            ar_in     = audio_c1[:, :-1]                 # shift right
            ar_target = audio_c1[:, 1:]
            ar_logits = model.forward_valle_ar(text_ids, ar_in)

            # Only compute on valid (non-pad) positions
            ar_loss = F.cross_entropy(
                ar_logits.reshape(-1, model.cfg.codebook_size),
                ar_target.reshape(-1),
                ignore_index=0,
            )

            # ── NAR loss ──────────────────────────────
            # Randomly choose which codebook stage to predict
            N_Q   = audio_codes.shape[2]
            stage = random.randint(1, N_Q - 1)
            nar_logits = model.forward_valle_nar(
                text_ids, audio_codes[:, :, :stage], target_stage=stage
            )
            nar_loss = F.cross_entropy(
                nar_logits.reshape(-1, model.cfg.codebook_size),
                audio_codes[:, :, stage].reshape(-1),
                ignore_index=0,
            )

            total_loss = ar_loss + nar_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.valle.parameters()) +
                list(model.text_encoder.parameters()),
                grad_clip
            )
            optimizer.step()
            scheduler.step()

            ar_m.update(ar_loss.item())
            nar_m.update(nar_loss.item())

            if step % log_every == 0:
                print(f"  VALLE | ep{epoch} step{step:4d} | "
                      f"AR={ar_m.avg:.4f} NAR={nar_m.avg:.4f} "
                      f"lr={scheduler.get_lr():.2e}")

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "ar_loss": ar_m.avg,
            "nar_loss": nar_m.avg,
            "time": elapsed,
        })
        print(f"  VALLE epoch {epoch}/{n_epochs}: "
              f"AR={ar_m.avg:.4f} NAR={nar_m.avg:.4f} ({elapsed:.1f}s)")

    return history


# ──────────────────────────────────────────────
# 5.  Phase 3: Flow Matching Training
# ──────────────────────────────────────────────

def train_flow_matching(
    model:      TTSModel,
    loader:     DataLoader,
    optimizer:  optim.Optimizer,
    scheduler:  WarmupCosineScheduler,
    device:     torch.device,
    n_epochs:   int = 3,
    grad_clip:  float = 1.0,
    log_every:  int = 10,
) -> List[Dict]:
    """
    Train FlowMatchingNet to denoise mel spectrograms.
    Everything except flow_net is frozen.
    """
    model.codec.eval()
    model.valle.eval()
    for p in list(model.codec.parameters()) + list(model.valle.parameters()):
        p.requires_grad_(False)

    model.flow_net.train()
    history = []

    for epoch in range(1, n_epochs + 1):
        loss_m = AverageMeter()
        t0     = time.time()

        for step, batch in enumerate(loader, 1):
            mel = batch["mel"].to(device)              # (B, n_mels, T)

            # Condition = slightly noisy version (simulates codec-decoded mel)
            cond = mel + 0.1 * torch.randn_like(mel)

            fm_loss = model.forward_flow(mel, cond)

            optimizer.zero_grad()
            fm_loss.backward()
            nn.utils.clip_grad_norm_(model.flow_net.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            loss_m.update(fm_loss.item())

            if step % log_every == 0:
                print(f"  FlowMatch | ep{epoch} step{step:4d} | "
                      f"loss={loss_m.avg:.4f} lr={scheduler.get_lr():.2e}")

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "fm_loss": loss_m.avg, "time": elapsed})
        print(f"  FlowMatch epoch {epoch}/{n_epochs}: "
              f"loss={loss_m.avg:.4f} ({elapsed:.1f}s)")

    return history


# ──────────────────────────────────────────────
# 6.  Full Training Orchestrator
# ──────────────────────────────────────────────

def train_tts(
    size:        str   = "small",
    n_samples:   int   = 64,
    batch_size:  int   = 4,
    lr:          float = 3e-4,
    codec_epochs: int  = 2,
    valle_epochs: int  = 2,
    fm_epochs:    int  = 2,
    device_str:  str   = "auto",
    seed:        int   = 42,
) -> Tuple[TTSModel, Dict]:
    """
    Run all three training phases and return (model, history).
    """
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else device_str if device_str != "auto" else "cpu"
    )
    print(f"\nTraining TTS ({size}) on {device}")
    print("=" * 55)

    cfg    = TTSConfig(size=size)
    model  = TTSModel(cfg).to(device)

    dataset = SyntheticTTSDataset(cfg, n_samples=n_samples)
    loader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    history = {}

    # ── Phase 1: Codec ──────────────────────────────────────
    print("\nPhase 1: Codec pretraining")
    opt1  = optim.AdamW(model.codec.parameters(), lr=lr, weight_decay=0.01)
    sch1  = WarmupCosineScheduler(opt1,
                warmup_steps=steps_per_epoch,
                total_steps=codec_epochs * steps_per_epoch)
    history["codec"] = train_codec(
        model, loader, opt1, sch1, device, n_epochs=codec_epochs
    )

    # ── Phase 2: VALLE ──────────────────────────────────────
    print("\nPhase 2: VALLE (AR + NAR) training")
    valle_params = (
        list(model.valle.parameters()) +
        list(model.text_encoder.parameters())
    )
    opt2  = optim.AdamW(valle_params, lr=lr, weight_decay=0.01)
    sch2  = WarmupCosineScheduler(opt2,
                warmup_steps=steps_per_epoch,
                total_steps=valle_epochs * steps_per_epoch)
    history["valle"] = train_valle(
        model, loader, opt2, sch2, device, n_epochs=valle_epochs
    )

    # ── Phase 3: Flow Matching ──────────────────────────────
    print("\nPhase 3: Flow Matching vocoder")
    opt3  = optim.AdamW(model.flow_net.parameters(), lr=lr * 0.5, weight_decay=0.01)
    sch3  = WarmupCosineScheduler(opt3,
                warmup_steps=steps_per_epoch,
                total_steps=fm_epochs * steps_per_epoch)
    history["flow"] = train_flow_matching(
        model, loader, opt3, sch3, device, n_epochs=fm_epochs
    )

    print("\nTraining complete.")
    return model, history


# ──────────────────────────────────────────────
# 7.  Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TTS TRAINING MODULE — Tests")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = TTSConfig(size="small")

    # ── Dataset ──────────────────────────────────────────────
    print("\n[1] Synthetic dataset")
    ds = SyntheticTTSDataset(cfg, n_samples=16)
    print(f"  samples: {len(ds)}")
    s  = ds[0]
    for k, v in s.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v!r}")

    # ── DataLoader ────────────────────────────────────────────
    print("\n[2] DataLoader collation")
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    batch  = next(iter(loader))
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    # ── Full training run (tiny) ─────────────────────────────
    print("\n[3] Mini training run (2 samples, 1 epoch each phase)")
    model, history = train_tts(
        size="small",
        n_samples=8, batch_size=2,
        codec_epochs=1, valle_epochs=1, fm_epochs=1,
        device_str=str(device),
    )
    print("\n  Training history:")
    for phase, records in history.items():
        for rec in records:
            print(f"    {phase}: {rec}")
