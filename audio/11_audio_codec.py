"""
TTS Chapter 2: Audio Codec — Residual Vector Quantizer (RVQ)
=============================================================
Qwen3-TTS (and VALL-E, SoundStorm, etc.) represent audio as *discrete
token sequences* produced by a neural codec.  The codec has two parts:

  Encoder  →  continuous latent z  →  RVQ  →  discrete codes c₁,c₂,...,cₙ
  Decoder  ←  reconstructed z'    ←  sum of codebook vectors

Why RVQ?
  A single VQ codebook of size K cannot capture the full bandwidth of
  speech.  RVQ (Residual Vector Quantization) stacks N codebooks where
  each stage quantizes the *residual* of the previous stage.

  Bit-rate  ≈  N × log₂(K) bits per frame
  e.g.  8 codebooks × log₂(1024) = 80 bits / frame  @ 75 fps  ≈  6 kbps

Architecture (EnCodec-inspired):

  Waveform  →  [Conv1D encoder stack]  →  z  (B, T_frames, D)
                                              │
                         ┌────────────────────┤
                    RVQ  │  q₁ = VQ(z)        │
                         │  r₁ = z − q₁       │
                         │  q₂ = VQ(r₁)       │
                         │  r₂ = r₁ − q₂      │
                         │  …                  │
                         │  qₙ = VQ(rₙ₋₁)     │
                         └────────────────────┘
                              codes: (B, T_frames, N)
                                              │
  Waveform ← [ConvTranspose1D decoder stack] ← sum(q₁..qₙ)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# ──────────────────────────────────────────────
# 1.  Vector Quantizer (single codebook)
# ──────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """
    Straight-Through Estimator VQ.

    Forward returns:
        quantized  : tensor same shape as input   (gradient passes through)
        indices    : long tensor of codebook indices
        loss       : commitment + codebook losses
    """

    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim:  int = 128,
        commitment_cost: float = 0.25,
        ema_decay:       float = 0.99,   # use EMA updates (more stable)
    ):
        super().__init__()
        self.K   = num_embeddings
        self.D   = embedding_dim
        self.beta = commitment_cost
        self.decay = ema_decay

        # Codebook
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed",       embed)
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embed_avg",   embed.clone())

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, T, D)
        Returns: quantized (B, T, D), indices (B, T), loss scalar
        """
        B, T, D = z.shape
        flat = z.reshape(-1, D)                          # (B*T, D)

        # Distances to codebook entries  ‖z − e‖²
        dist = (
            flat.pow(2).sum(1, keepdim=True)             # (B*T, 1)
            - 2 * flat @ self.embed.t()                  # (B*T, K)
            + self.embed.pow(2).sum(1)                   # (K,)
        )
        indices = dist.argmin(dim=1)                     # (B*T,)
        quantized = self.embed[indices]                  # (B*T, D)

        # EMA codebook update (training only)
        if self.training:
            one_hot = F.one_hot(indices, self.K).float() # (B*T, K)
            self.cluster_size.mul_(self.decay).add_(
                one_hot.sum(0) * (1 - self.decay)
            )
            embed_sum = one_hot.t() @ flat               # (K, D)
            self.embed_avg.mul_(self.decay).add_(
                embed_sum * (1 - self.decay)
            )
            # Laplace smoothing for empty clusters
            n = self.cluster_size.sum()
            smooth = (self.cluster_size + 1e-5) / (n + self.K * 1e-5) * n
            self.embed.data.copy_(self.embed_avg / smooth.unsqueeze(1))

        # Commitment loss: encoder output → quantized (no codebook gradient)
        loss = self.beta * F.mse_loss(z, quantized.detach().reshape(B, T, D))

        # Straight-through: copy gradient from quantized → z
        quantized = z + (quantized.reshape(B, T, D) - z).detach()

        return quantized, indices.reshape(B, T), loss


# ──────────────────────────────────────────────
# 2.  Residual Vector Quantizer
# ──────────────────────────────────────────────

class ResidualVQ(nn.Module):
    """
    Stack of N VQ codebooks where each quantizes the residual of the last.

    Returns:
        quantized   : (B, T, D)  — sum of all quantized embeddings
        all_indices : (B, T, N)  — code index per codebook
        total_loss  : scalar
    """

    def __init__(
        self,
        num_quantizers: int = 8,
        num_embeddings: int = 1024,
        embedding_dim:  int = 128,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.N = num_quantizers
        self.vqs = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual   = z
        quantized  = torch.zeros_like(z)
        all_indices: List[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=z.device)

        for vq in self.vqs:
            q, idx, loss = vq(residual)
            residual  = residual - q.detach()
            quantized = quantized + q
            all_indices.append(idx)
            total_loss = total_loss + loss

        all_indices_t = torch.stack(all_indices, dim=-1)   # (B, T, N)
        return quantized, all_indices_t, total_loss

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Returns codes only (no gradient, no loss) — for inference."""
        residual = z
        codes: List[torch.Tensor] = []
        for vq in self.vqs:
            dist = (
                residual.reshape(-1, z.shape[-1]).pow(2).sum(1, keepdim=True)
                - 2 * residual.reshape(-1, z.shape[-1]) @ vq.embed.t()
                + vq.embed.pow(2).sum(1)
            )
            idx = dist.argmin(dim=1).reshape(z.shape[0], z.shape[1])
            q   = vq.embed[idx.reshape(-1)].reshape(*z.shape)
            residual = residual - q
            codes.append(idx)
        return torch.stack(codes, dim=-1)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (B, T, N) → quantized embedding (B, T, D)."""
        quantized = torch.zeros(
            codes.shape[0], codes.shape[1], self.vqs[0].D,
            device=codes.device
        )
        for i, vq in enumerate(self.vqs):
            quantized = quantized + vq.embed[codes[..., i]]
        return quantized


# ──────────────────────────────────────────────
# 3.  Causal Conv1D block (for the codec)
# ──────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Causal (streaming-safe) 1-D convolution."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            dilation=dilation, padding=0
        )

    def forward(self, x):                      # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block with dilated causal convolutions."""
    def __init__(self, channels: int, dilations=(1, 3, 9)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.ELU(),
                CausalConv1d(channels, channels, kernel_size=3, dilation=d),
                nn.ELU(),
                nn.Conv1d(channels, channels, kernel_size=1),
            )
            for d in dilations
        ])

    def forward(self, x):
        for c in self.convs:
            x = x + c(x)
        return x


# ──────────────────────────────────────────────
# 4.  Codec Encoder
# ──────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """
    Waveform → continuous latent (before RVQ).

    Input:  (B, 1, T_samples)   — raw audio @ 24 kHz
    Output: (B, T_frames, D)    — T_frames = T_samples / hop_length
    """

    def __init__(
        self,
        in_channels:  int = 1,
        base_channels: int = 32,
        latent_dim:   int = 128,
        hop_length:   int = 320,   # 24000/320 = 75 fps
    ):
        super().__init__()
        self.hop = hop_length

        # Strided convolutions to downsample in time
        stride_sizes = self._factorize(hop_length)  # e.g. [4, 4, 4, 5]
        channels = [in_channels] + [
            base_channels * (2 ** i) for i in range(len(stride_sizes))
        ]

        layers = []
        for i, s in enumerate(stride_sizes):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1],
                          kernel_size=2*s, stride=s, padding=s//2),
                ResBlock(channels[i + 1]),
            ]
        layers.append(nn.Conv1d(channels[-1], latent_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _factorize(n: int) -> List[int]:
        """Factorize hop_length into small strides."""
        factors = []
        for p in [5, 4, 4, 2]:
            while n % p == 0 and len(factors) < 4:
                factors.append(p)
                n //= p
        if n > 1:
            factors.append(n)
        return factors[:4] or [4, 4, 5, 4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)                        # (B, D, T_frames)
        return z.permute(0, 2, 1)              # (B, T_frames, D)


# ──────────────────────────────────────────────
# 5.  Codec Decoder
# ──────────────────────────────────────────────

class AudioDecoder(nn.Module):
    """
    Quantized latent → reconstructed waveform.

    Input:  (B, T_frames, D)
    Output: (B, 1, T_samples)
    """

    def __init__(
        self,
        latent_dim:    int = 128,
        base_channels: int = 32,
        hop_length:    int = 320,
    ):
        super().__init__()
        stride_sizes = AudioEncoder._factorize(hop_length)[::-1]
        n = len(stride_sizes)
        channels = [base_channels * (2 ** (n - 1 - i)) for i in range(n)] + [1]

        layers: List[nn.Module] = [
            nn.Conv1d(latent_dim, base_channels * (2 ** (n - 1)), kernel_size=1)
        ]
        for i, s in enumerate(stride_sizes):
            layers += [
                ResBlock(channels[i]),
                nn.ConvTranspose1d(
                    channels[i], channels[i + 1],
                    kernel_size=2*s, stride=s, padding=s//2
                ),
            ]
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.permute(0, 2, 1)                # (B, D, T_frames)
        return self.net(x)                     # (B, 1, T_samples)


# ──────────────────────────────────────────────
# 6.  Mel Spectrogram Extractor (no codec needed)
#     Used as input to acoustic models
# ──────────────────────────────────────────────

class MelSpectrogram(nn.Module):
    """
    Differentiable mel-spectrogram extractor.
    Uses a fixed (non-learnable) filterbank.
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

        # Build mel filterbank
        fb = self._mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("fb", fb)      # (n_mels, n_fft//2+1)

        # Hann window
        self.register_buffer("window", torch.hann_window(n_fft))

    @staticmethod
    def _hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    @staticmethod
    def _mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _mel_filterbank(
        self, sr, n_fft, n_mels, f_min, f_max
    ) -> torch.Tensor:
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts  = torch.tensor([self._mel_to_hz(m.item()) for m in mel_pts])
        bins    = torch.floor((n_fft + 1) * hz_pts / sr).long()

        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        for m in range(1, n_mels + 1):
            f_m_minus = bins[m - 1]
            f_m       = bins[m]
            f_m_plus  = bins[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus:
                    fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m:
                    fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return fb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, T) or (B, 1, T)
        Returns:  (B, n_mels, T_frames)
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        # STFT
        stft = torch.stft(
            waveform, self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )                                      # (B, n_fft//2+1, T_frames)
        mag = stft.abs()
        # Mel filterbank
        mel = torch.einsum("mf,bft->bmt", self.fb, mag)
        # Log mel
        mel = torch.log(mel.clamp(min=1e-5))
        return mel                             # (B, n_mels, T_frames)


# ──────────────────────────────────────────────
# 7.  Full AudioCodec
# ──────────────────────────────────────────────

class AudioCodec(nn.Module):
    """
    End-to-end codec:
      encode(waveform) → codes   (B, T_frames, N_quantizers)
      decode(codes)    → waveform (B, 1, T_samples)
    """

    def __init__(
        self,
        latent_dim:     int = 128,
        num_quantizers: int = 8,
        codebook_size:  int = 1024,
        hop_length:     int = 320,
    ):
        super().__init__()
        self.encoder = AudioEncoder(latent_dim=latent_dim, hop_length=hop_length)
        self.rvq     = ResidualVQ(num_quantizers, codebook_size, latent_dim)
        self.decoder = AudioDecoder(latent_dim=latent_dim, hop_length=hop_length)
        self.hop     = hop_length

    def forward(
        self, waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        waveform: (B, 1, T)
        Returns:  reconstructed (B, 1, T), codes (B, T_f, N), loss scalar
        """
        z         = self.encoder(waveform)
        q, codes, loss = self.rvq(z)
        recon     = self.decoder(q)
        # Match length
        min_len   = min(waveform.shape[-1], recon.shape[-1])
        recon     = recon[..., :min_len]
        return recon, codes, loss

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform → codes (B, T_frames, N_quantizers)"""
        z = self.encoder(waveform)
        return self.rvq.encode(z)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes → waveform (B, 1, T_samples)"""
        q = self.rvq.decode_codes(codes)
        return self.decoder(q)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# 8.  Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO CODEC MODULE — Tests")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── VQ ───────────────────────────────────────────────────
    print("\n[1] VectorQuantizer")
    vq = VectorQuantizer(num_embeddings=512, embedding_dim=64).to(device)
    z  = torch.randn(2, 50, 64, device=device)
    q, idx, loss = vq(z)
    print(f"  input  : {z.shape}")
    print(f"  quantized: {q.shape}  indices: {idx.shape}  loss: {loss.item():.4f}")

    # ── RVQ ──────────────────────────────────────────────────
    print("\n[2] ResidualVQ (4 codebooks)")
    rvq = ResidualVQ(num_quantizers=4, num_embeddings=512, embedding_dim=64).to(device)
    q, codes, loss = rvq(z)
    print(f"  quantized: {q.shape}  codes: {codes.shape}  loss: {loss.item():.4f}")

    # ── Codec ─────────────────────────────────────────────────
    print("\n[3] AudioCodec  (tiny, hop=256)")
    codec = AudioCodec(
        latent_dim=64, num_quantizers=4, codebook_size=512, hop_length=256
    ).to(device)
    wav   = torch.randn(2, 1, 8000, device=device)
    recon, codes, loss = codec(wav)
    print(f"  waveform : {wav.shape}")
    print(f"  recon    : {recon.shape}")
    print(f"  codes    : {codes.shape}")
    print(f"  loss     : {loss.item():.4f}")
    print(f"  params   : {codec.num_parameters():,}")

    # ── Encode / Decode roundtrip ────────────────────────────
    print("\n[4] Encode → Decode roundtrip")
    enc_codes = codec.encode(wav)
    dec_wav   = codec.decode(enc_codes)
    print(f"  codes    : {enc_codes.shape}  (B, T_frames, N_quantizers)")
    print(f"  decoded  : {dec_wav.shape}")

    # ── MelSpectrogram ───────────────────────────────────────
    print("\n[5] MelSpectrogram")
    mel_fn = MelSpectrogram(sample_rate=24000, hop_length=256, n_mels=80).to(device)
    mel    = mel_fn(wav)
    print(f"  waveform : {wav.shape}  →  mel: {mel.shape}  (B, n_mels, T_frames)")
    print(f"  mel range: [{mel.min():.2f}, {mel.max():.2f}]")
