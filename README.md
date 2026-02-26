# 🚀 LLM-from-Scratch: Complete Multimodal AI Pipeline

A comprehensive, educational implementation of modern multimodal large language models (LLMs) built from scratch. This project implements **Text**, **Image**, **Video**, and **Audio/Speech** modalities with full training pipelines, inference loops, and visualization utilities.

Inspired by modern architectures like FLUX, Stable Diffusion 3, and Flow Matching approaches.

---

## 📋 Overview

This repository provides self-contained, modular implementations of:
- **Text-to-Text LLM**: Transformer-based language model with RLHF, DPO, and PPO training
- **Text-to-Image (T2I)**: Full latent diffusion pipeline (VAE → CLIP → DiT denoiser)
- **Text-to-Video (T2V)**: Video generation with temporal attention and 3D tokenization
- **Audio-to-Audio (A2A)**: Speech synthesis with flow matching and dialogue models

Each module is **self-contained**, includes training loops, inference code, and visualization utilities for educational purposes.

---

## 📁 Project Structure

```
llms/
├── text/                    # Text modality (LLM from scratch)
│   ├── 01_data_cleaning.py
│   ├── 02_embeddings.py
│   ├── 03_attention.py
│   ├── 04_transformer_model.py
│   ├── 05_pretraining.py
│   ├── 06_sft.py             # Supervised Fine-Tuning
│   ├── 07_reward_model.py
│   ├── 08_rlhf_ppo_dpo.py    # RLHF, PPO, DPO training
│   ├── 09_visualizations.py
│   ├── llm_modules.py        # importlib loader for digit-prefixed modules
│   └── run_llm.py            # Main entry point
│
├── image/                   # Image modality (T2I)
│   ├── 20_image_vae.py       # Convolutional VAE encoder/decoder
│   ├── 21_clip_encoder.py    # CLIP text/image encoder
│   ├── 22_dit_denoiser.py    # Diffusion Transformer denoiser
│   ├── 23_ddpm_scheduler.py  # DDPM/Flow Matching noise scheduler
│   └── 24_t2i_model.py       # Full Text-to-Image pipeline
│
├── video/                   # Video modality (T2V)
│   ├── 30_video_tokenizer.py      # 3D VAE video tokenizer
│   ├── 31_temporal_attention.py    # Temporal attention for video
│   └── 32_video_dit.py             # Video DiT denoiser
│
├── speech/                  # Speech modality (A2A)
│   ├── 40_speech_encoder.py        # Speech encoder (Mel-Spectrogram)
│   └── 41_dialogue_lm.py           # Dialogue language model
│
├── audio/                   # Audio modality (TTS)
│   ├── 10_text_processing.py       # Text tokenization & cleaning
│   ├── 11_audio_codec.py           # Audio compression (Encodec-like)
│   ├── 12_text_encoder.py          # Text-to-embedding encoder
│   ├── 13_acoustic_model.py        # Acoustic modeling
│   ├── 14_flow_matching.py         # Flow Matching training
│   ├── 15_tts_model.py             # Full TTS model
│   ├── 16_tts_training.py          # Training loop
│   ├── 17_tts_visualizations.py    # Visualizations
│   ├── run_tts.py                  # TTS entry point
│   └── (also in speech/ for dialogue)
│
├── checkpoints/             # Model checkpoints & saved weights
├── plots/                   # Generated visualizations & plots
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## 🔑 Key Modules by Modality

### 📝 **Text Modality** (`text/`)
Building a transformer-based LLM from scratch:

| File | Description |
|------|-------------|
| `01_data_cleaning.py` | Tokenization, cleaning, and dataset preparation |
| `02_embeddings.py` | Token & positional embeddings |
| `03_attention.py` | Multi-head self-attention & causal masking |
| `04_transformer_model.py` | Transformer encoder/decoder blocks |
| `05_pretraining.py` | Next-token prediction pretraining |
| `06_sft.py` | Supervised Fine-Tuning (instruction tuning) |
| `07_reward_model.py` | Reward model for RLHF |
| `08_rlhf_ppo_dpo.py` | RLHF (PPO) and DPO training methods |
| `09_visualizations.py` | Loss curves, attention heatmaps, token distributions |

**Key Pipeline:**
```
Raw Text → Tokenize → Embed → Transformer Blocks → Pretraining
                                                  ↓
                                            Fine-Tune (SFT)
                                                  ↓
                                    Train Reward Model (optional)
                                                  ↓
                                      RLHF/PPO/DPO Training
```

### 🖼️ **Image Modality** (`image/`)
Text-to-Image generation via latent diffusion:

| File | Description |
|------|-------------|
| `20_image_vae.py` | Convolutional VAE: compress images to latent space |
| `21_clip_encoder.py` | CLIP-like encoder for text → embeddings |
| `22_dit_denoiser.py` | Diffusion Transformer denoiser in latent space |
| `23_ddpm_scheduler.py` | Noise scheduler (DDPM, DDIM, or Flow Matching) |
| `24_t2i_model.py` | Full inference pipeline (text → image) |

**Key Pipeline:**
```
Text Prompt → CLIP Encoder → Text Embeddings
                                   ↓
                         DiT Denoiser (iterative)
                                   ↓
                         VAE Decoder → Image
```

### 🎬 **Video Modality** (`video/`)
Text-to-Video generation with temporal dynamics:

| File | Description |
|------|-------------|
| `30_video_tokenizer.py` | 3D VAE for video compression |
| `31_temporal_attention.py` | Temporal attention across frames |
| `32_video_dit.py` | Video DiT with spatial-temporal denoising |

**Key Pipeline:**
```
Text Prompt → Encoder → Embeddings
                          ↓
                    Video DiT (3D conv + temporal attn)
                          ↓
                    Inverse 3D-VAE → Video
```

### 🎙️ **Audio Modality** (`audio/` + `speech/`)
Text-to-Speech and dialogue models:

| File | Description |
|------|-------------|
| `10_text_processing.py` | Text tokenization (phone-level) |
| `11_audio_codec.py` | Audio compression (Encodec-style) |
| `12_text_encoder.py` | Text → latent embeddings |
| `13_acoustic_model.py` | Acoustic modeling (duration, pitch) |
| `14_flow_matching.py` | Flow Matching for continuous generation |
| `15_tts_model.py` | Full TTS inference pipeline |
| `16_tts_training.py` | Training loop with synthetic data |
| `40_speech_encoder.py` | Speech encoder (audio → embeddings) |
| `41_dialogue_lm.py` | Language model for dialogue responses |

**Key Pipeline:**
```
Text → Phone Encoder → Duration/Pitch Predictor
                          ↓
                    Flow Matching (acoustic features)
                          ↓
                    Audio Codec Decoder → Waveform
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd llms

# Install dependencies
pip install -r requirements.txt

# (Optional) Install development tools
pip install jupyter ipython
```

**Dependencies:**
```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tiktoken>=0.5.0
datasets>=2.14.0
transformers>=4.35.0
scikit-learn>=1.3.0
tqdm>=4.65.0
pandas>=2.0.0
scipy>=1.11.0
```

---

## 🚀 Quick Start

### 1️⃣ Text-to-Text LLM

```python
# Training
python text/run_llm.py --mode train --epochs 10 --batch_size 32

# Inference
python text/run_llm.py --mode inference --prompt "Once upon a time"
```

### 2️⃣ Text-to-Image (T2I)

```python
from image import image_vae, clip_encoder, dit_denoiser, ddpm_scheduler, t2i_model

# Load model (or train from scratch)
model = t2i_model.LatentDiffusionModel(config_size="small")
model.load_checkpoint()

# Generate image
image = model.synthesize(
    text="a painting of a serene lake at sunset",
    num_steps=50,
    guidance_scale=7.5
)
```

### 3️⃣ Text-to-Video (T2V)

```python
from video import video_tokenizer, temporal_attention, video_dit

# Similar workflow to T2I, but with temporal denoising
video = model.synthesize_video(
    text="a cat jumping over a fence",
    num_frames=16,
    num_steps=30
)
```

### 4️⃣ Text-to-Speech (TTS)

```bash
python audio/run_tts.py --text "Hello, world!" --output speech.wav

# Or dialogue
python speech/41_dialogue_lm.py --user_input "What is machine learning?"
```

---

## 📊 Module Architecture Overview

### Attention Mechanism
```
Query, Key, Value → Compute Attention Weights → Multiply by Values
↓
Supports multi-head, causal masking, and cross-attention
```

### Transformer Block
```
Input → Multi-Head Attention → FFN → LayerNorm → Output
```

### Diffusion Process
```
Image (x₀) → Add Noise → Noisy Latent (xₜ) → Predict Noise (εθ)
                                              → Iteratively Denoise
                                              → Clean Image
```

### Flow Matching
```
Data Distribution → Straight-Line Path → Noise Distribution
Learns ODE to follow this path (continuous alternative to discrete diffusion)
```

---

## ✨ Features

✅ **Self-Contained Modules** — Each file is independent and runnable
✅ **Educational Design** — Clear, well-commented code following ML fundamentals
✅ **Full Training Pipelines** — Pretraining, fine-tuning, RLHF, DPO, PPO
✅ **Multiple Modalities** — Text, Image, Video, and Audio in one codebase
✅ **Visualization Tools** — Attention heatmaps, loss curves, generated samples
✅ **Checkpoint Management** — Save/load trained weights
✅ **Flexible Configs** — Tiny/Small/Medium/Large model sizes
✅ **Modern Techniques** — Flow Matching, DPO, temporal attention, 3D convolutions

---

## 📈 Training & Evaluation

### Text Pretraining
- **Task:** Next-token prediction on unlabeled corpus
- **Loss:** Cross-entropy on vocabulary prediction
- **Metrics:** Perplexity, accuracy

### Image Generation
- **Task:** Denoise Gaussian noise conditioned on text
- **Loss:** MSE between predicted and actual noise
- **Metrics:** FID, CLIP score, visual quality

### Audio Synthesis
- **Task:** Generate mel-spectrogram from text
- **Loss:** Flow Matching ODE loss
- **Metrics:** MCD (Mel-Cepstral Distortion), subjective quality

---

## 🎯 Model Sizes

Each modality supports configurable sizes:

| Config | Params | Speed | Quality | Use Case |
|--------|--------|-------|---------|----------|
| **Tiny** | ~1-10M | Very Fast | Low | Testing, debugging |
| **Small** | ~50-100M | Fast | Medium | Research |
| **Medium** | ~300-500M | Moderate | High | Production |
| **Large** | ~1B+ | Slow | Very High | SOTA results |

---

## 📚 References & Inspiration

- **Raschka ML Book**: https://github.com/rasbt/machine-learning-book
- **Attention is All You Need**: Vaswani et al., 2017
- **Latent Diffusion Models**: Rombach et al., 2021
- **FLUX.1**: Open-source image model
- **Stable Diffusion 3**: Text conditioning approach
- **Flow Matching**: Albergo et al., 2023
- **DPO (Direct Preference Optimization)**: Rafailov et al., 2023
- **Qwen3-TTS**: Flow Matching for audio

---

## 🔧 Configuration

Edit model sizes and hyperparameters in each module's config sections:

```python
# Example: text/04_transformer_model.py
CONFIG = {
    "tiny": {
        "vocab_size": 50257,
        "context_length": 256,
        "embedding_dim": 64,
        "num_heads": 2,
        "num_layers": 2,
        "feedforward_dim": 256,
    },
    "small": {
        "vocab_size": 50257,
        "context_length": 512,
        "embedding_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "feedforward_dim": 1024,
    },
    # ... more configs
}
```

---

## 💾 Checkpoints & Data

- **`checkpoints/`** — Pre-trained weights (download separately if large)
- **`plots/`** — Generated visualizations and training curves

Large checkpoint files are stored separately. Training on smaller datasets (e.g., TinyStories) is recommended for quick prototyping.

---

## 🐛 Debugging & Visualization

Each module includes visualization functions:

```python
from text import visualizations

# Plot training curves
visualizations.plot_training_curves(losses, val_losses)

# Plot attention heatmap
visualizations.plot_attention_heatmap(attention_weights)

# Plot token distribution
visualizations.plot_token_distribution(logits)
```

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Open issues for bugs or questions
- Submit PRs for improvements
- Fork for your own implementations

---

## 📄 License

This project is provided for educational purposes. Refer to individual module headers for specific licensing information.

---

## 🎓 Learning Path

**Recommended order to understand the codebase:**

1. Start with **`text/03_attention.py`** — Understand self-attention
2. Read **`text/04_transformer_model.py`** — Build intuition on architecture
3. Explore **`text/05_pretraining.py`** — Learn training mechanics
4. Then dive into **`image/20_image_vae.py`** — Understand latent spaces
5. Finally **`image/24_t2i_model.py`** — See full pipeline in action

---

## 🚦 Status

✅ **Text**: Complete (pretraining, SFT, RLHF, DPO, PPO)
✅ **Image**: Complete (VAE, CLIP, DiT, DDPM/Flow Matching)
✅ **Video**: Core implementation (tokenizer, temporal attention, DiT)
✅ **Audio/Speech**: TTS pipeline with Flow Matching + dialogue model

**Next Steps**:
- Integration tests across modalities
- Multi-modal fusion (text+image → better image)
- End-to-end training recipes

---

## 📞 Support

For questions or issues:
1. Check the module docstrings
2. Review the inline comments in each file
3. Refer to the cited papers and implementations above
4. Open a GitHub issue with details

---

**Happy Learning! 🎉**

Built with ❤️ for the ML community.
