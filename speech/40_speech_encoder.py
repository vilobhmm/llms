"""
Speech Encoder — HuBERT-Inspired Conv1D + Transformer
======================================================
This module implements a speech feature extraction pipeline inspired by
HuBERT (Hsu et al., 2021) and wav2vec 2.0 (Baevski et al., 2020).

Design Philosophy:
------------------
Modern speech encoders operate directly on raw waveforms rather than
hand-crafted features (MFCCs). The key insight is:

  Raw waveform → CNN feature extractor → Transformer contextualizer
                                                    ↓
                              Frame-level embeddings (rich representations)

Why this architecture?
  1. CNN layers capture local acoustic patterns (phones, formants) at
     multiple time scales using strided convolutions.
  2. Transformer layers add global context — each frame attends to all
     others, capturing coarticulation, prosody, speaker identity.
  3. Unlike MFCCs, learned features adapt to the downstream task.

Components:
-----------
  WaveformFeatureExtractor : raw waveform → mel spectrogram (differentiable)
  SpeechEncoder            : Conv1D feature extractor + Transformer encoder
  SpeechProjector          : linear head to map encoder output to model dim
  VoiceActivityDetector    : per-frame voiced/silence binary classifier
  SpeechTokenizer          : VQ quantizer (k-means style) for discrete tokens

Conventions:
  - Waveforms: (B, 1, T) at 24 kHz
  - Frame rate: 75 fps (hop_length=320 samples)
  - Speech tokens: (B, T_frames, N_quantizers)

References:
  - HuBERT: https://arxiv.org/abs/2106.07447
  - wav2vec 2.0: https://arxiv.org/abs/2006.11477
  - Moshi: https://arxiv.org/abs/2410.00037
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Mel Spectrogram Extractor (inlined from audio/11_audio_codec.py concept)
# ─────────────────────────────────────────────────────────────────────────────

class MelSpectrogram(nn.Module):
    """
    Differentiable mel-spectrogram extractor using a fixed (non-learnable)
    triangular filterbank.

    The mel scale compresses frequency resolution at high frequencies (matching
    human auditory perception), making it a compact yet perceptually relevant
    representation for speech.

    Input:  (B, 1, T) or (B, T) raw waveform
    Output: (B, n_mels, T_frames) log-mel spectrogram
    """

    def __init__(
        self,
        sample_rate: int   = 24_000,
        n_fft:       int   = 1024,
        hop_length:  int   = 256,
        n_mels:      int   = 80,
        f_min:       float = 0.0,
        f_max:       float = 8_000.0,
    ):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.n_mels     = n_mels

        fb = self._mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("fb", fb)           # (n_mels, n_fft//2+1)
        self.register_buffer("window", torch.hann_window(n_fft))

    @staticmethod
    def _hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    @staticmethod
    def _mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _mel_filterbank(
        self, sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float
    ) -> torch.Tensor:
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts  = torch.tensor([self._mel_to_hz(m.item()) for m in mel_pts])
        bins    = torch.floor((n_fft + 1) * hz_pts / sr).long()

        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        for m in range(1, n_mels + 1):
            lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
            for k in range(lo, mid):
                if mid > lo:
                    fb[m - 1, k] = float(k - lo) / float(mid - lo)
            for k in range(mid, hi):
                if hi > mid:
                    fb[m - 1, k] = float(hi - k) / float(hi - mid)
        return fb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, 1, T) or (B, T)
        Returns:  (B, n_mels, T_frames)
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        stft = torch.stft(
            waveform, self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            pad_mode="reflect",
        )                                        # (B, n_fft//2+1, T_frames)
        mag = stft.abs()
        mel = torch.einsum("mf,bft->bmt", self.fb, mag)
        return torch.log(mel.clamp(min=1e-5))   # (B, n_mels, T_frames)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Causal Conv1D building block
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Causal (streaming-safe) 1-D convolution.
    Pads only the left side so that the output at position t depends only
    on inputs at positions ≤ t.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        stride:       int = 1,
        dilation:     int = 1,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Conv1D Feature Extractor (CNN backbone)
# ─────────────────────────────────────────────────────────────────────────────

class ConvFeatureExtractor(nn.Module):
    """
    CNN-based local feature extractor operating on mel spectrogram frames.

    Architecture:
      4 Conv1D layers with increasing receptive field via dilation.
      Each layer: Conv → GroupNorm → GELU

    The CNN captures:
      - Layer 1: fine-grained spectral transitions
      - Layer 2-3: phoneme-level patterns
      - Layer 4: syllable-level rhythmic features

    Input:  (B, n_mels, T_frames)  — mel spectrogram
    Output: (B, hidden_dim, T_frames)  — local feature map
    """

    def __init__(
        self,
        in_channels:  int = 80,
        hidden_dim:   int = 256,
        num_layers:   int = 4,
        kernel_size:  int = 5,
    ):
        super().__init__()
        dilations = [1, 2, 4, 8][:num_layers]
        channels  = [in_channels] + [hidden_dim] * num_layers

        layers = []
        for i, dil in enumerate(dilations):
            layers.append(nn.Sequential(
                CausalConv1d(channels[i], channels[i + 1], kernel_size, dilation=dil),
                nn.GroupNorm(max(1, channels[i + 1] // 8), channels[i + 1]),
                nn.GELU(),
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, T)
        Returns: (B, hidden_dim, T)
        """
        for layer in self.layers:
            x = layer(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Positional Encoding (sinusoidal)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).
    Added to frame embeddings before the Transformer.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Speech Encoder (CNN + Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class SpeechEncoder(nn.Module):
    """
    Full speech encoder: Mel → CNN → Transformer → frame embeddings.

    Architecture inspired by HuBERT and wav2vec 2.0:
      1. WaveformFeatureExtractor: raw waveform → mel spectrogram
      2. ConvFeatureExtractor: mel → local features (CNN, 4 layers)
      3. Linear projection: local features → transformer dim
      4. Positional encoding
      5. Transformer encoder: 4 bidirectional attention layers
         (during inference we use causal mask for streaming)

    Output shape: (B, T_frames, hidden_dim)

    The encoder is pretrained with masked prediction (HuBERT-style):
      - Mask ~30% of frames
      - Predict discrete targets from a k-means model
      - The encoder learns to predict masked frames from context
    """

    def __init__(
        self,
        n_mels:       int = 80,
        hidden_dim:   int = 256,
        n_heads:      int = 4,
        n_layers:     int = 4,
        ffn_dim:      int = 1024,
        dropout:      float = 0.1,
        hop_length:   int = 320,
        sample_rate:  int = 24_000,
        causal:       bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.causal     = causal

        # Step 1: mel spectrogram
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # Step 2: CNN feature extractor
        self.conv_extractor = ConvFeatureExtractor(
            in_channels=n_mels,
            hidden_dim=hidden_dim,
            num_layers=4,
            kernel_size=5,
        )

        # Step 3: positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, dropout=dropout)

        # Step 4: Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, T, D) layout
            norm_first=True,    # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # Layer norm after CNN
        self.pre_norm = nn.LayerNorm(hidden_dim)

    def _make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask for causal (streaming) attention."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        waveform: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        waveform: (B, 1, T_samples) at 24 kHz
        Returns:  (B, T_frames, hidden_dim)
        """
        # Mel spectrogram: (B, n_mels, T_frames)
        mel = self.mel(waveform)

        # CNN features: (B, hidden_dim, T_frames)
        features = self.conv_extractor(mel)

        # Transpose to (B, T_frames, hidden_dim)
        x = features.permute(0, 2, 1)
        x = self.pre_norm(x)

        # Add positional encoding
        x = self.pos_enc(x)

        # Causal mask for streaming
        mask = None
        if self.causal:
            mask = self._make_causal_mask(x.size(1), x.device)

        # Transformer: (B, T_frames, hidden_dim)
        x = self.transformer(x, mask=mask)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Speech Projector
# ─────────────────────────────────────────────────────────────────────────────

class SpeechProjector(nn.Module):
    """
    Projects speech encoder output to the downstream model's hidden dimension.

    Used when the speech encoder dimension differs from the dialogue LM dimension.
    Applies a 2-layer MLP with GELU activation (same as CLIP projector).

    Input:  (B, T, encoder_dim)
    Output: (B, T, model_dim)
    """

    def __init__(self, encoder_dim: int, model_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Voice Activity Detector
# ─────────────────────────────────────────────────────────────────────────────

class VoiceActivityDetector(nn.Module):
    """
    Per-frame binary classifier: voiced (1) vs. silence (0).

    Used to:
      - Determine when the user starts/stops speaking
      - Compute frame-level loss mask (ignore silence in reconstruction)
      - Drive turn-taking decisions in the dialogue LM

    Architecture: 2-layer MLP on top of speech encoder features.
    Trained with Binary Cross-Entropy loss.

    Input:  (B, T, hidden_dim)  — speech encoder output
    Output: (B, T)              — probability of voice activity per frame
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, T, hidden_dim)
        Returns:  (B, T) logits (use sigmoid for probabilities)
        """
        return self.classifier(features).squeeze(-1)

    def compute_loss(
        self,
        features:  torch.Tensor,
        vad_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        features:   (B, T, hidden_dim)
        vad_labels: (B, T) float — 1.0 for voiced, 0.0 for silence
        Returns: (loss scalar, predictions (B, T) probabilities)
        """
        logits = self.forward(features)
        loss   = self.loss_fn(logits, vad_labels)
        probs  = torch.sigmoid(logits)
        return loss, probs


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Vector Quantizer (for speech tokenization)
# ─────────────────────────────────────────────────────────────────────────────

class VectorQuantizerEMA(nn.Module):
    """
    Vector quantizer with Exponential Moving Average (EMA) codebook update.

    EMA updates are more stable than gradient-based codebook updates because:
      - No gradient needed through quantization step
      - Codebook converges without collapse (via Laplace smoothing)
      - Straight-through estimator passes gradients to encoder

    This is the VQ used in VQ-VAE-2, wav2vec 2.0, and Moshi's codec.

    Input:  (B, T, D)
    Output: quantized (B, T, D), indices (B, T), commitment_loss scalar
    """

    def __init__(
        self,
        num_embeddings: int   = 512,
        embedding_dim:  int   = 256,
        commitment_cost: float = 0.25,
        ema_decay:       float = 0.99,
    ):
        super().__init__()
        self.K    = num_embeddings
        self.D    = embedding_dim
        self.beta = commitment_cost
        self.decay = ema_decay

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed",        embed)
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embed_avg",    embed.clone())

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = z.shape
        flat = z.reshape(-1, D)

        # L2 distance to all codebook vectors
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2.0 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)           # (B*T,)
        quantized = self.embed[indices]        # (B*T, D)

        if self.training:
            one_hot = F.one_hot(indices, self.K).float()
            self.cluster_size.mul_(self.decay).add_(
                one_hot.sum(0) * (1 - self.decay)
            )
            embed_sum = one_hot.t() @ flat
            self.embed_avg.mul_(self.decay).add_(
                embed_sum * (1 - self.decay)
            )
            n = self.cluster_size.sum()
            smooth = (self.cluster_size + 1e-5) / (n + self.K * 1e-5) * n
            self.embed.data.copy_(self.embed_avg / smooth.unsqueeze(1))

        loss = self.beta * F.mse_loss(z, quantized.detach().reshape(B, T, D))
        quantized = z + (quantized.reshape(B, T, D) - z).detach()

        return quantized, indices.reshape(B, T), loss


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Residual VQ for multi-level speech tokenization
# ─────────────────────────────────────────────────────────────────────────────

class ResidualVQ(nn.Module):
    """
    Residual Vector Quantizer: stack of N VQ codebooks.

    Each codebook quantizes the residual left by the previous codebooks.
    This allows fine-grained reconstruction with a compact per-frame
    representation of N integers.

    Used in: EnCodec, SoundStream, Moshi's speech codec.

    Returns:
      quantized   : (B, T, D) — sum of all quantized embeddings
      all_indices : (B, T, N) — code index per codebook level
      total_loss  : scalar
    """

    def __init__(
        self,
        num_quantizers: int   = 4,
        num_embeddings: int   = 512,
        embedding_dim:  int   = 256,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.N = num_quantizers
        self.vqs = nn.ModuleList([
            VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual    = z
        quantized   = torch.zeros_like(z)
        all_indices: List[torch.Tensor] = []
        total_loss  = torch.tensor(0.0, device=z.device)

        for vq in self.vqs:
            q, idx, loss = vq(residual)
            residual   = residual - q.detach()
            quantized  = quantized + q
            all_indices.append(idx)
            total_loss = total_loss + loss

        return quantized, torch.stack(all_indices, dim=-1), total_loss

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """z → codes (B, T, N), no gradients."""
        residual = z
        codes: List[torch.Tensor] = []
        B, T, D = z.shape
        for vq in self.vqs:
            flat = residual.reshape(-1, D)
            dist = (
                flat.pow(2).sum(1, keepdim=True)
                - 2.0 * flat @ vq.embed.t()
                + vq.embed.pow(2).sum(1)
            )
            idx = dist.argmin(dim=1).reshape(B, T)
            q   = vq.embed[idx.reshape(-1)].reshape(B, T, D)
            residual = residual - q
            codes.append(idx)
        return torch.stack(codes, dim=-1)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (B, T, N) → quantized embedding (B, T, D)."""
        quantized = torch.zeros(
            codes.shape[0], codes.shape[1], self.vqs[0].D,
            device=codes.device,
        )
        for i, vq in enumerate(self.vqs):
            quantized = quantized + vq.embed[codes[..., i]]
        return quantized


# ─────────────────────────────────────────────────────────────────────────────
# 10. Speech Tokenizer (encoder + RVQ, end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class SpeechTokenizer(nn.Module):
    """
    End-to-end speech tokenizer:
      waveform → encoder → RVQ → discrete speech tokens

    This is equivalent to the "semantic tokenizer" in VALL-E or the
    "EnCodec encoder" in Moshi, but using the Transformer encoder above
    rather than a pure convolutional encoder.

    The discrete tokens can then be fed to a language model (DialogueLM)
    as a sequence of integers per frame.

    Usage:
      tokenizer = SpeechTokenizer(...)
      tokens = tokenizer.encode(waveform)   # (B, T_frames, N_quantizers)
      recon  = tokenizer.decode(tokens)     # (B, T_frames, hidden_dim)
    """

    def __init__(
        self,
        encoder:         SpeechEncoder,
        num_quantizers:  int = 4,
        num_embeddings:  int = 512,
    ):
        super().__init__()
        self.encoder = encoder
        self.rvq = ResidualVQ(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=encoder.hidden_dim,
        )

    def forward(
        self, waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        waveform: (B, 1, T)
        Returns:
          embeddings : (B, T_frames, hidden_dim)   — continuous
          tokens     : (B, T_frames, N_quantizers) — discrete
          vq_loss    : scalar
        """
        embeddings        = self.encoder(waveform)
        _, tokens, vq_loss = self.rvq(embeddings)
        return embeddings, tokens, vq_loss

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform → tokens (B, T_frames, N_quantizers)"""
        embeddings = self.encoder(waveform)
        return self.rvq.encode(embeddings)

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, T_frames, N_quantizers) → embeddings (B, T_frames, D)"""
        return self.rvq.decode_codes(tokens)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# __main__ smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SPEECH ENCODER MODULE — Smoke Tests")
    print("=" * 60)

    B, T = 2, 4800   # 0.2 s at 24 kHz
    wav  = torch.randn(B, 1, T)

    # ── MelSpectrogram ──────────────────────────────────────────
    print("\n[1] MelSpectrogram")
    mel_fn = MelSpectrogram(sample_rate=24000, hop_length=320, n_mels=80)
    mel    = mel_fn(wav)
    print(f"  waveform : {wav.shape}  →  mel : {mel.shape}")
    assert mel.dim() == 3, "Expected (B, n_mels, T_frames)"

    # ── ConvFeatureExtractor ─────────────────────────────────────
    print("\n[2] ConvFeatureExtractor")
    cnn = ConvFeatureExtractor(in_channels=80, hidden_dim=64, num_layers=4)
    out = cnn(mel)
    print(f"  mel      : {mel.shape}  →  features : {out.shape}")

    # ── SpeechEncoder ───────────────────────────────────────────
    print("\n[3] SpeechEncoder")
    enc = SpeechEncoder(n_mels=80, hidden_dim=64, n_heads=4, n_layers=2, ffn_dim=128)
    out = enc(wav)
    print(f"  waveform : {wav.shape}  →  frames : {out.shape}")
    assert out.shape[0] == B
    assert out.shape[2] == 64

    # ── VoiceActivityDetector ───────────────────────────────────
    print("\n[4] VoiceActivityDetector")
    vad    = VoiceActivityDetector(hidden_dim=64)
    T_f    = out.shape[1]
    labels = torch.randint(0, 2, (B, T_f)).float()
    loss, probs = vad.compute_loss(out, labels)
    print(f"  frames : {out.shape}  →  probs : {probs.shape}  loss : {loss.item():.4f}")

    # ── SpeechTokenizer ─────────────────────────────────────────
    print("\n[5] SpeechTokenizer")
    tokenizer = SpeechTokenizer(encoder=enc, num_quantizers=4, num_embeddings=128)
    embeddings, tokens, vq_loss = tokenizer(wav)
    print(f"  waveform   : {wav.shape}")
    print(f"  embeddings : {embeddings.shape}")
    print(f"  tokens     : {tokens.shape}  (B, T_frames, N_quantizers)")
    print(f"  vq_loss    : {vq_loss.item():.4f}")
    print(f"  params     : {tokenizer.num_parameters():,}")

    # ── SpeechProjector ─────────────────────────────────────────
    print("\n[6] SpeechProjector")
    proj = SpeechProjector(encoder_dim=64, model_dim=128)
    projected = proj(embeddings)
    print(f"  embeddings : {embeddings.shape}  →  projected : {projected.shape}")

    print("\nAll smoke tests passed.")
