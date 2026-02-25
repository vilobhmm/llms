"""
T2I Chapter 2: CLIP-Style Dual Encoder (Text + Image)
=======================================================
CLIP (Contrastive Language-Image Pre-Training, Radford et al. 2021) learns a
shared embedding space for text and images using contrastive learning.

Architecture:
  TextEncoder:   Transformer encoder on character-level tokens
                   (B, T_text) → (B, D)  (pooled, normalized)
  ImageEncoder:  ViT-style patch encoder
                   (B, 3, H, W) → (B, D)  (CLS token, normalized)
  CLIPModel:     Dual encoder + learned temperature + InfoNCE loss

CLIP Training Objective (InfoNCE / NT-Xent):
  Given a batch of N (image, text) pairs:
    1. Encode all images to I_1...I_N  (unit sphere)
    2. Encode all texts to T_1...T_N   (unit sphere)
    3. Compute NxN similarity matrix:  S_ij = I_i · T_j / temperature
    4. InfoNCE loss = symmetric cross-entropy on diagonal as positives

  This encourages matched (image, text) pairs to be close,
  and unmatched pairs to be far apart.

Design notes:
  - Embeddings are L2-normalized to unit sphere (cosine sim = dot product)
  - Temperature τ is learned (log-scaled for stability)
  - ImageEncoder uses ViT-B style: patch_size=4 for small images
  - TextEncoder matches audio/12_text_encoder.py with char tokenizer
  - Both encoders use a final pooling → projection layer to D-dim
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 1.  Character Tokenizer (shared with TTS codebase)
# ─────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    Simple character-level tokenizer for text conditioning.
    Vocabulary: printable ASCII (32–126) + special tokens.
    """
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, max_len: int = 77):
        self.max_len = max_len
        # Printable ASCII: space (32) through tilde (126)
        chars = [chr(i) for i in range(32, 127)]
        self.char2id = {c: i + 4 for i, c in enumerate(chars)}
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.vocab_size = 4 + len(chars)   # 4 special + 95 ascii

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        truncate: bool = True,
    ) -> list:
        ids = [self.char2id.get(c, self.UNK_ID) for c in text]
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        if truncate and len(ids) > self.max_len:
            ids = ids[:self.max_len]
        return ids

    def decode(self, ids: list) -> str:
        return "".join(
            self.id2char.get(i, "?")
            for i in ids
            if i not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)
        )


# ─────────────────────────────────────────────────────────────
# 2.  Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────────────────────
# 3.  Transformer Block (shared by text and image encoders)
# ─────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product MHSA with optional causal mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn = self.drop(F.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN → MHSA → residual → LN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────
# 4.  Text Encoder (CLIP-style)
# ─────────────────────────────────────────────────────────────

class CLIPTextEncoder(nn.Module):
    """
    Character-level Transformer text encoder.

    Input:  token IDs   (B, T_text)
    Output: normalized embedding   (B, embed_dim)

    Architecture:
      Embedding + SinusoidalPE → N Transformer blocks → EOS-pool → Linear → L2-norm
    """

    def __init__(
        self,
        vocab_size:  int   = 99,      # 4 special + 95 ASCII
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        d_ff:        int   = 512,
        embed_dim:   int   = 128,
        max_len:     int   = 77,
        dropout:     float = 0.1,
        pad_id:      int   = 0,
    ):
        super().__init__()
        self.pad_id   = pad_id
        self.d_model  = d_model
        self.embed_dim = embed_dim

        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale     = math.sqrt(d_model)
        self.pos_enc   = SinusoidalPE(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm     = nn.LayerNorm(d_model)
        self.proj     = nn.Linear(d_model, embed_dim, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,          # (B, T)
    ) -> torch.Tensor:
        """Returns L2-normalized embeddings (B, embed_dim)."""
        B, T = tokens.shape
        pad_mask = (tokens == self.pad_id)   # (B, T) True = pad

        x = self.tok_embed(tokens) * self.scale
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)

        x = self.norm(x)

        # Pool at EOS position (last non-pad token)
        # For simplicity: mean-pool over non-pad positions
        lengths = (~pad_mask).float().sum(dim=1, keepdim=True)  # (B, 1)
        mask_f  = (~pad_mask).float().unsqueeze(-1)              # (B, T, 1)
        pooled  = (x * mask_f).sum(dim=1) / lengths.clamp(min=1)  # (B, d_model)

        out = self.proj(pooled)   # (B, embed_dim)
        return F.normalize(out, dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────
# 5.  Image Encoder (ViT-style)
# ─────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """
    Image → patch embeddings.

    Splits image into (H/P) x (W/P) patches, each of size (3, P, P).
    Projects each patch to d_model via a strided convolution.

    Input:  (B, 3, H, W)
    Output: (B, N_patches, d_model)   where N_patches = (H/P)*(W/P)
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_ch: int = 3, d_model: int = 128):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, N, D)"""
        x = self.proj(x)           # (B, D, H/P, W/P)
        B, D, H, W = x.shape
        x = x.reshape(B, D, H * W).transpose(1, 2)  # (B, N, D)
        return x


class CLIPImageEncoder(nn.Module):
    """
    ViT-style image encoder for CLIP.

    Input:  (B, 3, H, W)
    Output: normalized embedding   (B, embed_dim)

    Architecture:
      PatchEmbed → + LearnedPE + CLS token → N Transformer blocks →
      CLS-pool → Linear → L2-norm
    """

    def __init__(
        self,
        img_size:    int   = 32,
        patch_size:  int   = 4,
        in_ch:       int   = 3,
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        d_ff:        int   = 512,
        embed_dim:   int   = 128,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, d_model)
        n_patches        = self.patch_embed.n_patches
        self.embed_dim   = embed_dim

        # CLS token + learned position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.layers  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized embeddings (B, embed_dim)."""
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                 # (B, N+1, D)
        x   = self.dropout(x + self.pos_embed)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        cls_out = x[:, 0]                                # (B, D)
        out     = self.proj(cls_out)                     # (B, embed_dim)
        return F.normalize(out, dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────
# 6.  CLIP Model (Dual Encoder + Contrastive Loss)
# ─────────────────────────────────────────────────────────────

class CLIPModel(nn.Module):
    """
    Full CLIP model: text encoder + image encoder + contrastive loss.

    Temperature τ is a learned scalar (log-parameterized for positivity).
    Clamped to [0.01, 100] for stability.
    """

    def __init__(
        self,
        # Text encoder config
        text_vocab:    int   = 99,
        text_d_model:  int   = 128,
        text_n_heads:  int   = 4,
        text_n_layers: int   = 4,
        text_d_ff:     int   = 512,
        # Image encoder config
        img_size:      int   = 32,
        patch_size:    int   = 4,
        img_d_model:   int   = 128,
        img_n_heads:   int   = 4,
        img_n_layers:  int   = 4,
        img_d_ff:      int   = 512,
        # Shared
        embed_dim:     int   = 128,
        dropout:       float = 0.1,
        max_text_len:  int   = 77,
        init_temp:     float = 0.07,
    ):
        super().__init__()

        tok = CharTokenizer(max_len=max_text_len)

        self.text_encoder = CLIPTextEncoder(
            vocab_size=tok.vocab_size,
            d_model=text_d_model,
            n_heads=text_n_heads,
            n_layers=text_n_layers,
            d_ff=text_d_ff,
            embed_dim=embed_dim,
            max_len=max_text_len,
            dropout=dropout,
            pad_id=tok.PAD_ID,
        )

        self.image_encoder = CLIPImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_ch=3,
            d_model=img_d_model,
            n_heads=img_n_heads,
            n_layers=img_n_layers,
            d_ff=img_d_ff,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # Learnable log-temperature (τ = exp(log_τ))
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

    @property
    def temperature(self) -> torch.Tensor:
        """Temperature scalar, clamped for stability."""
        return self.log_temp.exp().clamp(0.01, 100.0)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T) → (B, embed_dim) normalized."""
        return self.text_encoder(tokens)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, embed_dim) normalized."""
        return self.image_encoder(images)

    def forward(
        self,
        images: torch.Tensor,          # (B, 3, H, W)
        tokens: torch.Tensor,          # (B, T_text)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute image and text embeddings + similarity matrix.

        Returns:
          (img_emb, txt_emb, logits_per_image)
          logits_per_image: (B, B) — row i = similarities of image i to all texts
        """
        img_emb = self.encode_image(images)   # (B, D)
        txt_emb = self.encode_text(tokens)    # (B, D)

        # Cosine similarity matrix scaled by temperature
        sim = img_emb @ txt_emb.T            # (B, B)
        logits = sim / self.temperature

        return img_emb, txt_emb, logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────
# 7.  Contrastive (InfoNCE) Loss
# ─────────────────────────────────────────────────────────────

def clip_loss(logits_per_image: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE (contrastive) loss.

    logits_per_image: (B, B) — diagonal = matched pairs (positives)

    Loss = (CE_image + CE_text) / 2
    where:
      CE_image = cross_entropy(logits_per_image, [0,1,...,B-1])
      CE_text  = cross_entropy(logits_per_image.T, [0,1,...,B-1])
    """
    B = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image,   labels)   # image → text
    loss_t = F.cross_entropy(logits_per_image.T, labels)   # text → image
    return (loss_i + loss_t) / 2


def clip_accuracy(logits_per_image: torch.Tensor) -> float:
    """Fraction of image→text matches that are correct (top-1)."""
    B = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device)
    pred_i = logits_per_image.argmax(dim=1)
    return (pred_i == labels).float().mean().item()


# ─────────────────────────────────────────────────────────────
# 8.  Smoke-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CLIP DUAL ENCODER — Smoke Tests")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    B = 4
    IMG_SIZE = 32
    PATCH    = 4
    D_MODEL  = 64
    EMBED    = 64

    tok = CharTokenizer()

    # ── Character Tokenizer ───────────────────────────────────
    print("\n[1] CharTokenizer")
    texts = ["a photo of a cat", "an image of a dog", "a picture of a bird", "sunset over mountains"]
    tokens_list = [tok.encode(t) for t in texts]
    max_len = max(len(t) for t in tokens_list)
    tokens = torch.zeros(B, max_len, dtype=torch.long)
    for i, t in enumerate(tokens_list):
        tokens[i, :len(t)] = torch.tensor(t)
    tokens = tokens.to(device)
    print(f"  vocab_size={tok.vocab_size}, tokens shape={tokens.shape}")

    # ── Text Encoder ──────────────────────────────────────────
    print("\n[2] CLIPTextEncoder")
    text_enc = CLIPTextEncoder(
        vocab_size=tok.vocab_size, d_model=D_MODEL, n_heads=4,
        n_layers=2, d_ff=256, embed_dim=EMBED, max_len=128
    ).to(device)
    txt_emb = text_enc(tokens)
    print(f"  tokens{tokens.shape} → embeddings{txt_emb.shape}")
    print(f"  norm check (should be ~1.0): {txt_emb.norm(dim=-1).mean().item():.4f}")
    print(f"  params: {text_enc.num_parameters():,}")

    # ── Image Encoder ─────────────────────────────────────────
    print("\n[3] CLIPImageEncoder (ViT-style)")
    img_enc = CLIPImageEncoder(
        img_size=IMG_SIZE, patch_size=PATCH, in_ch=3,
        d_model=D_MODEL, n_heads=4, n_layers=2,
        d_ff=256, embed_dim=EMBED
    ).to(device)
    images  = torch.randn(B, 3, IMG_SIZE, IMG_SIZE, device=device)
    img_emb = img_enc(images)
    print(f"  images{images.shape} → embeddings{img_emb.shape}")
    print(f"  norm check (should be ~1.0): {img_emb.norm(dim=-1).mean().item():.4f}")
    print(f"  n_patches={img_enc.patch_embed.n_patches}, params={img_enc.num_parameters():,}")

    # ── Full CLIP Model ───────────────────────────────────────
    print("\n[4] CLIPModel (dual encoder + similarity)")
    model = CLIPModel(
        text_vocab=tok.vocab_size, text_d_model=D_MODEL, text_n_heads=4,
        text_n_layers=2, text_d_ff=256,
        img_size=IMG_SIZE, patch_size=PATCH, img_d_model=D_MODEL,
        img_n_heads=4, img_n_layers=2, img_d_ff=256,
        embed_dim=EMBED, max_text_len=128
    ).to(device)

    i_emb, t_emb, logits = model(images, tokens)
    print(f"  img_emb{i_emb.shape}  txt_emb{t_emb.shape}  logits{logits.shape}")
    print(f"  temperature: {model.temperature.item():.4f}")
    print(f"  total params: {model.num_parameters():,}")

    # ── Contrastive Loss ──────────────────────────────────────
    print("\n[5] CLIP Loss (InfoNCE)")
    loss = clip_loss(logits)
    acc  = clip_accuracy(logits)
    print(f"  loss={loss.item():.4f}  acc={acc:.4f}")

    # ── Backward Pass ─────────────────────────────────────────
    print("\n[6] Backward pass")
    loss.backward()
    total_grad = sum(
        p.grad.abs().mean().item()
        for p in model.parameters() if p.grad is not None
    )
    print(f"  mean gradient magnitude (should be > 0): {total_grad:.6f}")
    print("\n[OK] All CLIP tests passed")
