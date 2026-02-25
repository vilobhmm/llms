"""
TTS Chapter 6: Full End-to-End TTS Model (Qwen3-TTS Style)
============================================================

Complete pipeline:

  Raw Text
      │
      ▼
  PhonemeTokenizer  (Chapter 1)
      │  token IDs  (B, T_text)
      ▼
  TextEncoder       (Chapter 3)
      │  hidden states  (B, T_text, D)
      ▼
  VALLEModel        (Chapter 4)
  ┌───┴──────────────────────────────────────────┐
  │  AR:  text_hidden → codec tokens (c₁)       │
  │  NAR: text_hidden + c₁ → c₂,c₃,…,cₙ        │
  └───────────────────────────────────────────────┘
      │  RVQ codes  (B, T_frames, N_quantizers)
      ▼
  AudioCodec.decode   (Chapter 2)
      │  rough waveform / decoded mel
      ▼
  FlowMatchingNet     (Chapter 5)   ← refine mel quality
      │  high-quality mel  (B, n_mels, T_frames)
      ▼
  GriffinLim / Waveform Output
      │  waveform  (B, T_samples)
      ▼
  Audio Output  🔊

Design notes:
  • TextEncoder and VALLEModel share a phoneme embedding space
  • The codec provides discrete "semantic tokens" (c₁ = coarse, c₂..cₙ = fine detail)
  • FlowMatching does not retrain the codec — it polishes the mel *after* decoding
  • All components can be trained/frozen independently (common in real systems)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

# ──────────────────────────────────────────────
# Inline constants (so this file runs standalone)
# ──────────────────────────────────────────────
_PHONEMES = [
    "AA","AE","AH","AO","AW","AX","AY","EH","ER","EY",
    "IH","IY","OW","OY","UH","UW",
    "B","CH","D","DH","F","G","HH","JH","K","L","M","N","NG","P",
    "R","S","SH","T","TH","V","W","Y","Z","ZH","SIL","SP",
]
PHONEME_VOCAB_SIZE = len(_PHONEMES) + 4

import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ──────────────────────────────────────────────
# Local imports (each module is self-contained)
# ──────────────────────────────────────────────

def _import_module(path: str):
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        os.path.basename(path).replace(".py", ""), path
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_HERE = os.path.dirname(__file__)

_tp  = _import_module(os.path.join(_HERE, "10_text_processing.py"))
_ac  = _import_module(os.path.join(_HERE, "11_audio_codec.py"))
_te  = _import_module(os.path.join(_HERE, "12_text_encoder.py"))
_am  = _import_module(os.path.join(_HERE, "13_acoustic_model.py"))
_fm  = _import_module(os.path.join(_HERE, "14_flow_matching.py"))

PhonemeTokenizer  = _tp.PhonemeTokenizer
CharTokenizer     = _tp.CharTokenizer
AudioCodec        = _ac.AudioCodec
MelSpectrogram    = _ac.MelSpectrogram
TextEncoder       = _te.TextEncoder
VALLEModel        = _am.VALLEModel
FlowMatchingNet   = _fm.FlowMatchingNet
GriffinLim        = _fm.GriffinLim
euler_solve       = _fm.euler_solve
midpoint_solve    = _fm.midpoint_solve
flow_matching_loss = _fm.flow_matching_loss


# ──────────────────────────────────────────────
# 1.  TTS Config dataclass
# ──────────────────────────────────────────────

class TTSConfig:
    """All hyperparameters in one place."""

    def __init__(self, size: str = "small"):
        # Audio
        self.sample_rate    = 24_000
        self.hop_length     = 256
        self.n_fft          = 1024
        self.n_mels         = 80

        # Codec
        self.codec_latent   = 64
        self.num_quantizers = 4
        self.codebook_size  = 512

        # Text encoder
        self.text_vocab     = PHONEME_VOCAB_SIZE
        self.text_pad_id    = 0

        if size == "small":
            self.te_d_model  = 128;  self.te_n_heads = 4; self.te_n_layers = 3; self.te_d_ff = 512
            self.ar_d_model  = 128;  self.ar_n_heads = 4; self.ar_n_layers = 3; self.ar_d_ff = 512
            self.nar_d_model = 128;  self.nar_n_heads = 4; self.nar_n_layers = 2; self.nar_d_ff = 512
            self.fm_d_model  = 64;   self.fm_n_heads = 4; self.fm_n_layers = 2; self.fm_d_ff = 256
        elif size == "medium":
            self.te_d_model  = 256;  self.te_n_heads = 4; self.te_n_layers = 4; self.te_d_ff = 1024
            self.ar_d_model  = 512;  self.ar_n_heads = 8; self.ar_n_layers = 6; self.ar_d_ff = 2048
            self.nar_d_model = 512;  self.nar_n_heads = 8; self.nar_n_layers = 4; self.nar_d_ff = 2048
            self.fm_d_model  = 256;  self.fm_n_heads = 4; self.fm_n_layers = 4; self.fm_d_ff = 1024
        else:  # large (Qwen3-TTS scale)
            self.te_d_model  = 512;  self.te_n_heads = 8; self.te_n_layers = 6; self.te_d_ff = 2048
            self.ar_d_model  = 1024; self.ar_n_heads = 16; self.ar_n_layers = 12; self.ar_d_ff = 4096
            self.nar_d_model = 1024; self.nar_n_heads = 16; self.nar_n_layers = 6; self.nar_d_ff = 4096
            self.fm_d_model  = 512;  self.fm_n_heads = 8; self.fm_n_layers = 6; self.fm_d_ff = 2048

        self.dropout    = 0.1
        self.gl_n_iter  = 30


# ──────────────────────────────────────────────
# 2.  Full TTS Model
# ──────────────────────────────────────────────

class TTSModel(nn.Module):
    """
    End-to-end Qwen3-TTS style model.

    Components:
      • tokenizer        : PhonemeTokenizer
      • text_encoder     : TextEncoder          (text → hidden states)
      • valle            : VALLEModel            (hidden → codec codes, AR+NAR)
      • codec            : AudioCodec            (codes ↔ waveform)
      • mel_extractor    : MelSpectrogram        (waveform → mel)
      • flow_net         : FlowMatchingNet       (noisy mel → clean mel)
      • griffin_lim      : GriffinLim            (mel → waveform, baseline)
    """

    def __init__(self, cfg: TTSConfig):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = PhonemeTokenizer()

        self.text_encoder = TextEncoder(
            vocab_size=cfg.text_vocab,
            d_model=cfg.te_d_model,
            n_heads=cfg.te_n_heads,
            n_layers=cfg.te_n_layers,
            d_ff=cfg.te_d_ff,
            dropout=cfg.dropout,
            pad_id=cfg.text_pad_id,
        )

        # Project text encoder output → VALLE input dim (if sizes differ)
        self.te_proj = (
            nn.Linear(cfg.te_d_model, cfg.ar_d_model, bias=False)
            if cfg.te_d_model != cfg.ar_d_model else nn.Identity()
        )

        self.valle = VALLEModel(
            text_vocab=cfg.text_vocab,
            audio_vocab=cfg.codebook_size,
            num_quantizers=cfg.num_quantizers,
            ar_d_model=cfg.ar_d_model, ar_n_heads=cfg.ar_n_heads,
            ar_n_layers=cfg.ar_n_layers, ar_d_ff=cfg.ar_d_ff,
            nar_d_model=cfg.nar_d_model, nar_n_heads=cfg.nar_n_heads,
            nar_n_layers=cfg.nar_n_layers, nar_d_ff=cfg.nar_d_ff,
            dropout=cfg.dropout,
        )

        self.codec = AudioCodec(
            latent_dim=cfg.codec_latent,
            num_quantizers=cfg.num_quantizers,
            codebook_size=cfg.codebook_size,
            hop_length=cfg.hop_length,
        )

        self.mel_extractor = MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
        )

        self.flow_net = FlowMatchingNet(
            n_mels=cfg.n_mels,
            d_model=cfg.fm_d_model,
            n_heads=cfg.fm_n_heads,
            n_layers=cfg.fm_n_layers,
            d_ff=cfg.fm_d_ff,
            dropout=cfg.dropout,
        )

        self.griffin_lim = GriffinLim(
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            sample_rate=cfg.sample_rate,
            n_iter=cfg.gl_n_iter,
        )

    # ──────────────────────────────────────────
    # Forward passes (training)
    # ──────────────────────────────────────────

    def forward_codec(
        self, waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Codec forward pass (train codec separately first).
        Returns (reconstructed, codes, codec_loss).
        """
        return self.codec(waveform)

    def forward_valle_ar(
        self,
        text_ids:  torch.Tensor,
        audio_ids: torch.Tensor,
    ) -> torch.Tensor:
        """AR stage forward → logits (B, T_audio, codebook_size)."""
        return self.valle.forward_ar(text_ids, audio_ids)

    def forward_valle_nar(
        self,
        text_ids:     torch.Tensor,
        audio_codes:  torch.Tensor,
        target_stage: int,
    ) -> torch.Tensor:
        """NAR stage forward → logits (B, T_audio, codebook_size)."""
        return self.valle.forward_nar(text_ids, audio_codes, target_stage)

    def forward_flow(
        self,
        clean_mel: torch.Tensor,
        cond_mel:  torch.Tensor,
    ) -> torch.Tensor:
        """Flow matching training loss."""
        return flow_matching_loss(self.flow_net, clean_mel, cond_mel)

    # ──────────────────────────────────────────
    # Inference pipeline
    # ──────────────────────────────────────────

    @torch.no_grad()
    def synthesize(
        self,
        text:         str,
        max_audio_len: int   = 500,
        temperature:   float = 0.8,
        top_k:         int   = 50,
        fm_steps:      int   = 20,
        use_flow:      bool  = True,
        device:        Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        text → (waveform, mel_spectrogram)

        Returns:
          waveform : (1, T_samples)
          mel      : (1, n_mels, T_frames)
        """
        if device is None:
            device = next(self.parameters()).device

        # 1. Tokenize text
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        text_ids = torch.tensor(ids, device=device).unsqueeze(0)  # (1, T_text)

        # 2. Generate codec tokens (AR + NAR)
        codes = self.valle.generate(
            text_ids,
            max_len=max_audio_len,
            temperature=temperature,
            top_k=top_k,
        )                                                # (1, T_audio, N_Q)

        # 3. Decode codec → rough waveform
        rough_wav = self.codec.decode(codes)             # (1, 1, T_samples)
        rough_mel = self.mel_extractor(rough_wav)        # (1, n_mels, T_frames)

        # 4. Refine with flow matching (optional)
        if use_flow:
            # Trim / pad to same T_frames
            T_target = rough_mel.shape[-1]
            cond = rough_mel[..., :T_target]
            refined_mel = midpoint_solve(self.flow_net, cond, n_steps=fm_steps)
        else:
            refined_mel = rough_mel

        # 5. Mel → waveform via Griffin-Lim
        waveform = self.griffin_lim(refined_mel)         # (1, T_samples)

        return waveform, refined_mel

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    def num_parameters(self, component: str = "all") -> Dict[str, int]:
        """Return parameter count per component."""
        comps = {
            "text_encoder": self.text_encoder,
            "valle_ar":     self.valle.ar_model,
            "valle_nar":    self.valle.nar_model,
            "codec":        self.codec,
            "flow_net":     self.flow_net,
        }
        if component != "all":
            return {component: sum(p.numel() for p in comps[component].parameters())}
        counts = {k: sum(p.numel() for p in v.parameters()) for k, v in comps.items()}
        counts["total"] = sum(counts.values())
        return counts

    def save(self, path: str):
        torch.save({
            "config": vars(self.cfg),
            "state":  self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TTSModel":
        ckpt = torch.load(path, map_location=device)
        cfg_dict = ckpt["config"]
        cfg = TTSConfig.__new__(TTSConfig)
        cfg.__dict__.update(cfg_dict)
        model = cls(cfg)
        model.load_state_dict(ckpt["state"])
        return model


# ──────────────────────────────────────────────
# 3.  Quick Config Summary Utility
# ──────────────────────────────────────────────

def model_summary(model: TTSModel) -> str:
    counts = model.num_parameters()
    lines  = ["=" * 55, "TTS Model Summary", "=" * 55]
    for name, n in counts.items():
        if name == "total":
            lines.append("-" * 55)
        lines.append(f"  {name:<18} {n:>12,} params")
    lines.append("=" * 55)
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 4.  Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("FULL TTS MODEL — Tests")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Instantiate ──────────────────────────────────────────
    cfg   = TTSConfig(size="small")
    model = TTSModel(cfg).to(device)
    print(model_summary(model))

    # ── Tokenizer ─────────────────────────────────────────────
    print("\n[1] Tokenizer")
    tok = model.tokenizer
    ids = tok.encode("Hello world.")
    print(f"  'Hello world.' → {len(ids)} tokens: {tok.decode(ids)}")

    # ── Codec forward ────────────────────────────────────────
    print("\n[2] Codec forward")
    wav   = torch.randn(2, 1, 8000, device=device)
    recon, codes, codec_loss = model.forward_codec(wav)
    print(f"  waveform: {wav.shape}  recon: {recon.shape}  codes: {codes.shape}")
    print(f"  codec loss: {codec_loss.item():.4f}")

    # ── VALLE AR forward ─────────────────────────────────────
    print("\n[3] VALLE AR forward (teacher-forced)")
    B, T_text, T_audio = 2, 15, 20
    text_ids  = torch.randint(4, PHONEME_VOCAB_SIZE, (B, T_text), device=device)
    audio_ids = torch.randint(0, cfg.codebook_size, (B, T_audio), device=device)
    ar_logits = model.forward_valle_ar(text_ids, audio_ids)
    ar_loss   = F.cross_entropy(
        ar_logits.reshape(-1, cfg.codebook_size),
        audio_ids.reshape(-1)
    )
    print(f"  AR logits: {ar_logits.shape}   AR loss: {ar_loss.item():.4f}")

    # ── VALLE NAR forward ────────────────────────────────────
    print("\n[4] VALLE NAR forward")
    codes_1 = torch.randint(0, cfg.codebook_size, (B, T_audio, 1), device=device)
    nar_logits = model.forward_valle_nar(text_ids, codes_1, target_stage=1)
    print(f"  NAR logits: {nar_logits.shape}")

    # ── Flow matching loss ───────────────────────────────────
    print("\n[5] Flow matching loss")
    clean_mel = torch.randn(B, cfg.n_mels, 50, device=device)
    cond_mel  = torch.randn(B, cfg.n_mels, 50, device=device)
    fm_loss   = model.forward_flow(clean_mel, cond_mel)
    print(f"  FM loss: {fm_loss.item():.4f}")

    # ── Full synthesis ────────────────────────────────────────
    print("\n[6] Full synthesis (CPU, small model)")
    model.eval()
    wav_out, mel_out = model.synthesize(
        "Hello world.",
        max_audio_len=15,
        temperature=1.0,
        fm_steps=3,
        use_flow=True,
        device=device,
    )
    print(f"  waveform: {wav_out.shape}  mel: {mel_out.shape}")
    sr = cfg.sample_rate
    dur = wav_out.shape[-1] / sr
    print(f"  Synthesized {dur:.3f}s of audio @ {sr}Hz")

    # ── Save / load ──────────────────────────────────────────
    print("\n[7] Save / Load checkpoint")
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    model.save(path)
    loaded = TTSModel.load(path, device=str(device))
    os.remove(path)
    print(f"  Saved and loaded model (params match: "
          f"{loaded.num_parameters()['total'] == model.num_parameters()['total']})")
