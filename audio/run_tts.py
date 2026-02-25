"""
run_tts.py — End-to-End TTS Demo (Qwen3-TTS from scratch)
============================================================
Executes every chapter in order and validates the full pipeline:

  Chapter 1  (10_text_processing.py)  — Text normalization, phoneme tokenizer
  Chapter 2  (11_audio_codec.py)      — RVQ audio codec
  Chapter 3  (12_text_encoder.py)     — Transformer text encoder
  Chapter 4  (13_acoustic_model.py)   — VALL-E AR + NAR acoustic model
  Chapter 5  (14_flow_matching.py)    — Conditional flow matching vocoder
  Chapter 6  (15_tts_model.py)        — Full end-to-end TTS model
  Chapter 7  (16_tts_training.py)     — Three-phase training
  Chapter 8  (17_tts_visualizations.py) — 12 diagnostic plots

Usage:
  cd /home/user/llms
  python audio/run_tts.py
  python audio/run_tts.py --size medium   # larger model
  python audio/run_tts.py --skip-train    # skip training (faster)
  python audio/run_tts.py --skip-viz      # skip visualizations
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


# ──────────────────────────────────────────────
# Loader helper
# ──────────────────────────────────────────────

def _load(fname: str):
    path = os.path.join(HERE, fname)
    name = fname.replace(".py", "").replace("/", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def section(title: str):
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def ok(label: str, detail: str = ""):
    tick = "[OK]"
    line = f"  {tick}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)


def fail(label: str, exc: Exception):
    print(f"  [FAIL] {label}")
    traceback.print_exc()
    raise SystemExit(1) from exc


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    t_start = time.time()

    print("\n" + "=" * 62)
    print("  Qwen3-TTS From Scratch — Full Pipeline Demo")
    print("=" * 62)
    print(f"  device : {device}")
    print(f"  size   : {args.size}")

    # ──────────────────────────────────────────
    # Chapter 1 — Text Processing
    # ──────────────────────────────────────────
    section("Chapter 1: Text Processing & Phoneme Tokenizer")
    try:
        tp   = _load("10_text_processing.py")
        tok  = tp.PhonemeTokenizer()
        norm = tp.normalize_text("Dr. Smith has 42 models.")
        ids  = tok.encode("Hello world.")
        phon = tok.decode(ids)
        ok("PhonemeTokenizer", f"'Hello world.' → {len(ids)} ids, phonemes: {phon}")
        ok("normalize_text",   repr(norm))

        ctok  = tp.CharTokenizer()
        cids  = ctok.encode("hello")
        ok("CharTokenizer",    f"vocab={ctok.vocab_size}, 'hello'→{ctok.decode(cids)!r}")
    except Exception as e:
        fail("text_processing", e)

    # ──────────────────────────────────────────
    # Chapter 2 — Audio Codec
    # ──────────────────────────────────────────
    section("Chapter 2: Audio Codec (RVQ)")
    try:
        ac    = _load("11_audio_codec.py")

        vq    = ac.VectorQuantizer(num_embeddings=256, embedding_dim=64)
        z     = torch.randn(2, 20, 64)
        q, idx, loss = vq(z)
        ok("VectorQuantizer", f"{z.shape} → q{q.shape} idx{idx.shape} loss={loss.item():.4f}")

        rvq   = ac.ResidualVQ(num_quantizers=4, num_embeddings=256, embedding_dim=64)
        q, codes, loss = rvq(z)
        ok("ResidualVQ", f"codes{codes.shape}")

        codec = ac.AudioCodec(latent_dim=64, num_quantizers=4,
                               codebook_size=256, hop_length=256).to(device)
        wav   = torch.randn(2, 1, 8000).to(device)
        recon, codes, _ = codec(wav)
        ok("AudioCodec.forward", f"recon{recon.shape}")

        enc_c = codec.encode(wav)
        dec_w = codec.decode(enc_c)
        ok("AudioCodec encode/decode roundtrip",
           f"codes{enc_c.shape} → wav{dec_w.shape}")

        mel_fn = ac.MelSpectrogram(sample_rate=24000, hop_length=256).to(device)
        mel    = mel_fn(wav)
        ok("MelSpectrogram", f"{wav.shape} → {mel.shape}")
    except Exception as e:
        fail("audio_codec", e)

    # ──────────────────────────────────────────
    # Chapter 3 — Text Encoder
    # ──────────────────────────────────────────
    section("Chapter 3: Text Encoder (Transformer)")
    try:
        te_mod = _load("12_text_encoder.py")
        te     = te_mod.TextEncoder(
            vocab_size=tp.PHONEME_VOCAB_SIZE,
            d_model=128, n_heads=4, n_layers=2, d_ff=512
        ).to(device)
        toks   = torch.randint(4, tp.PHONEME_VOCAB_SIZE, (2, 20)).to(device)
        h      = te(toks)
        ok("TextEncoder", f"tokens{toks.shape} → hidden{h.shape}  "
                          f"params={te.num_parameters():,}")

        xattn  = te_mod.CrossAttention(128, 4).to(device)
        audio  = torch.randn(2, 50, 128).to(device)
        out, w = xattn(audio, h)
        ok("CrossAttention", f"attn_out{out.shape}  weights{w.shape}")
    except Exception as e:
        fail("text_encoder", e)

    # ──────────────────────────────────────────
    # Chapter 4 — VALL-E Acoustic Model
    # ──────────────────────────────────────────
    section("Chapter 4: VALL-E AR + NAR Acoustic Model")
    try:
        am = _load("13_acoustic_model.py")

        ar = am.ARModel(
            text_vocab=tp.PHONEME_VOCAB_SIZE, audio_vocab=256,
            d_model=128, n_heads=4, n_layers=2, d_ff=512
        ).to(device)
        text_t  = torch.randint(4, tp.PHONEME_VOCAB_SIZE, (2, 15)).to(device)
        audio_t = torch.randint(0, 256, (2, 20)).to(device)
        logits  = ar(text_t, audio_t)
        ok("ARModel.forward", f"logits{logits.shape}  params={ar.num_parameters():,}")

        c1 = ar.generate(text_t, max_len=10, temperature=1.0, top_k=20)
        ok("ARModel.generate", f"c1{c1.shape}")

        nar  = am.NARModel(
            text_vocab=tp.PHONEME_VOCAB_SIZE, audio_vocab=256,
            num_quantizers=4, d_model=128, n_heads=4, n_layers=2, d_ff=512
        ).to(device)
        codes_so_far = c1.unsqueeze(-1)
        nar_logits   = nar(text_t, codes_so_far, target_stage=1)
        ok("NARModel.forward", f"nar_logits{nar_logits.shape}")

        valle = am.VALLEModel(
            text_vocab=tp.PHONEME_VOCAB_SIZE, audio_vocab=256,
            num_quantizers=4,
            ar_d_model=128, ar_n_heads=4, ar_n_layers=2, ar_d_ff=512,
            nar_d_model=128, nar_n_heads=4, nar_n_layers=2, nar_d_ff=512,
        ).to(device)
        all_codes = valle.generate(text_t, max_len=8, temperature=0.8, top_k=20)
        ok("VALLEModel.generate", f"all_codes{all_codes.shape}  "
                                   f"params={valle.num_parameters():,}")
    except Exception as e:
        fail("acoustic_model", e)

    # ──────────────────────────────────────────
    # Chapter 5 — Flow Matching Vocoder
    # ──────────────────────────────────────────
    section("Chapter 5: Flow Matching Vocoder")
    try:
        fm_mod = _load("14_flow_matching.py")

        te_e = fm_mod.TimestepEmbedding(dim=128).to(device)
        t    = torch.rand(2).to(device)
        emb  = te_e(t)
        ok("TimestepEmbedding", f"{t.shape} → {emb.shape}")

        fm_net = fm_mod.FlowMatchingNet(
            n_mels=80, d_model=64, n_heads=4, n_layers=2, d_ff=256
        ).to(device)
        x_t  = torch.randn(2, 80, 40).to(device)
        cond = torch.randn(2, 80, 40).to(device)
        v    = fm_net(x_t, t, cond)
        ok("FlowMatchingNet.forward", f"velocity{v.shape}  params={fm_net.num_parameters():,}")

        loss = fm_mod.flow_matching_loss(fm_net, x_t, cond)
        ok("flow_matching_loss", f"loss={loss.item():.4f}")

        gen  = fm_mod.euler_solve(fm_net, cond, n_steps=5)
        ok("euler_solve", f"generated{gen.shape}")

        gen2 = fm_mod.midpoint_solve(fm_net, cond, n_steps=3)
        ok("midpoint_solve", f"generated{gen2.shape}")

        gl  = fm_mod.GriffinLim(n_fft=512, hop_length=128, n_mels=80, n_iter=5)
        wav = gl(torch.randn(2, 80, 40))
        ok("GriffinLim", f"waveform{wav.shape}")
    except Exception as e:
        fail("flow_matching", e)

    # ──────────────────────────────────────────
    # Chapter 6 — Full TTS Model
    # ──────────────────────────────────────────
    section("Chapter 6: Full TTS Model")
    try:
        tts_mod = _load("15_tts_model.py")
        cfg     = tts_mod.TTSConfig(size=args.size)
        model   = tts_mod.TTSModel(cfg).to(device)

        print(tts_mod.model_summary(model))

        # Forward passes
        B, T_t, T_a = 2, 12, 15
        text_ids  = torch.randint(4, tp.PHONEME_VOCAB_SIZE, (B, T_t)).to(device)
        audio_ids = torch.randint(0, cfg.codebook_size, (B, T_a)).to(device)

        ar_logits = model.forward_valle_ar(text_ids, audio_ids)
        ok("forward_valle_ar", f"logits{ar_logits.shape}")

        codes_1 = torch.randint(0, cfg.codebook_size, (B, T_a, 1)).to(device)
        nar_lg  = model.forward_valle_nar(text_ids, codes_1, target_stage=1)
        ok("forward_valle_nar", f"logits{nar_lg.shape}")

        clean_mel = torch.randn(B, cfg.n_mels, 30).to(device)
        cond_mel  = torch.randn(B, cfg.n_mels, 30).to(device)
        fm_loss   = model.forward_flow(clean_mel, cond_mel)
        ok("forward_flow", f"loss={fm_loss.item():.4f}")

        # Full synthesis
        model.eval()
        wav_out, mel_out = model.synthesize(
            "Hello world.",
            max_audio_len=10,
            fm_steps=2,
            use_flow=True,
            device=device,
        )
        dur = wav_out.shape[-1] / cfg.sample_rate
        ok("synthesize", f"wav{wav_out.shape}  mel{mel_out.shape}  dur={dur:.3f}s")
    except Exception as e:
        fail("tts_model", e)

    # ──────────────────────────────────────────
    # Chapter 7 — Training
    # ──────────────────────────────────────────
    section("Chapter 7: Three-Phase Training")
    if args.skip_train:
        print("  [SKIP] --skip-train passed")
    else:
        try:
            tr_mod = _load("16_tts_training.py")
            t_tr   = time.time()

            trained_model, history = tr_mod.train_tts(
                size="small",
                n_samples=16, batch_size=2,
                codec_epochs=1, valle_epochs=1, fm_epochs=1,
                device_str=str(device),
            )

            for phase, records in history.items():
                for r in records:
                    key = [k for k in r if "loss" in k]
                    losses = ", ".join(f"{k}={r[k]:.4f}" for k in key)
                    ok(f"Phase {phase} ep{r['epoch']}", losses)

            ok("training complete", f"{time.time() - t_tr:.1f}s")
        except Exception as e:
            fail("tts_training", e)

    # ──────────────────────────────────────────
    # Chapter 8 — Visualizations
    # ──────────────────────────────────────────
    section("Chapter 8: Visualizations (12 figures)")
    if args.skip_viz:
        print("  [SKIP] --skip-viz passed")
    else:
        try:
            viz = _load("17_tts_visualizations.py")
            viz.fig_text_processing()
            viz.fig_phoneme_vocab()
            viz.fig_mel_spectrogram()
            viz.fig_rvq_codebook()
            viz.fig_codec_reconstruction()
            viz.fig_ar_attention()
            viz.fig_token_probs()
            viz.fig_nar_codebooks()
            viz.fig_flow_trajectory()
            viz.fig_waveform()
            viz.fig_training_curves()
            viz.fig_architecture()
            ok("All 12 figures saved", viz.PLOTS_DIR)
        except Exception as e:
            fail("tts_visualizations", e)

    # ──────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 62)
    print(f"  All chapters completed in {elapsed:.1f}s")
    print("  Repository structure:")
    print("    llms/")
    print("      text/   — LLM modules (Ch01–09)")
    print("      audio/  — TTS modules (Ch10–17, run_tts.py)")
    print("      plots/")
    print("        llm/  — LLM figures")
    print("        tts/  — TTS figures (12 plots)")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS from scratch demo")
    parser.add_argument(
        "--size", choices=["small", "medium", "large"], default="small",
        help="Model size (default: small)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip the training phase (faster)"
    )
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Skip the visualization phase"
    )
    args = parser.parse_args()
    main(args)
