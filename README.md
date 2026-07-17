# Neural Audio Codec: What Determines the Rate–Distortion Ceiling of Scalar-Quantized Speech Codecs?

A causal, streaming-capable neural audio codec built as the instrument for a controlled empirical study, not as an attempt to outperform state-of-the-art systems. The question: when a neural speech codec uses scalar quantization and a fixed, non-learned entropy coder, what sets the quality ceiling — quantizer resolution, or the information content the training objective places in the latent?

**Finding:** the ceiling is set by latent entropy — controlled by the training objective — not by quantization resolution. Adding more quantization bits past a point barely moves quality; deliberately regularizing entropy down (via a KL term) moves quality down with it; deliberately training for richer perceptual detail moves both up together, every time. This repository contains the codec, the eight-phase controlled training curriculum that produced this evidence, and every supporting experiment.

Developed at TU Ilmenau, Faculty of Electrical Engineering and Information Technology, under the supervision of Prof. Gerald Schuller. **Status:** project Exposé submitted; manuscript in preparation.

![Entropy–quality tension across the training curriculum](plots/fig_04.png)

---

## Contents

- [The finding, in three pieces of evidence](#the-finding-in-three-pieces-of-evidence)
- [Results](#results)
- [Supporting experiments](#supporting-experiments)
- [Known limitations — disclosed](#known-limitations--disclosed)
- [Future work](#future-work)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Separate encoder and decoder](#separate-encoder-and-decoder)
- [Training curriculum](#training-curriculum)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Dependencies](#dependencies)
- [References](#references)

---

## The finding, in three pieces of evidence

### 1. Entropy and quality move together — in both directions

Every phase measures two things after training: perceptual quality (PESQ-WB, STOI) and the Shannon entropy of the quantized latent, per dimension. Across five of six phases, both rise together as the training objective is made more perceptually sophisticated:

| Phase | Change | PESQ-WB | STOI | Bitrate | Mean latent entropy |
|---|---|---|---|---|---|
| C | Baseline (MSE + noise augmentation) | 1.202 | 0.733 | 5.7 kbps | 1.462 bits |
| D | Alternative quantization proxy | 1.222 | 0.731 | 5.8 kbps | 1.455 bits |
| **D-VAE** | **+ KL regularization** | **1.181** | **0.702** | **4.6 kbps** | **1.090 bits** |
| E | Log-magnitude spectral loss | 1.262 | 0.751 | 5.9 kbps | 1.533 bits |
| F | Combined triple spectral loss | 1.265 | 0.766 | 5.9 kbps | 1.521 bits |
| G | Fine-polish (best model) | 1.279 | 0.766 | 5.9 kbps | 1.520 bits |

Phase D-VAE is the deliberate exception, and it's the piece that turns this from a correlation into evidence: a KL-divergence term directly penalizes the latent's entropy, with no change to the reconstruction objective. Entropy drops sharply (1.090 vs. ~1.5 bits elsewhere) — and quality drops with it. This is the one experiment in the curriculum where entropy was pushed in the *opposite* direction from every other phase, on purpose, and quality followed it down anyway.

![Quality metrics across the full 8-phase curriculum](plots/fig_01.png)

The effect isn't concentrated in a few latent dimensions — it shows up broadly across nearly all 32:

![Per-dimension entropy across phases](plots/fig_10.png)

### 2. Adding quantization bits stops helping — the ceiling isn't resolution

Phase G's trained weights, swept from 1-bit to 6-bit quantization at inference time with no retraining:

| Bits | Levels | Theoretical kbps | Effective kbps | PESQ-WB | STOI |
|---|---|---|---|---|---|
| 1 | 2 | 3.2 | 3.55 | 1.082 | 0.456 |
| 2 | 4 | 6.4 | 3.78 | 1.100 | 0.589 |
| **3 (trained)** | 8 | 9.6 | 5.87 | 1.279 | 0.766 |
| 4 | 16 | 12.8 | 8.79 | 1.366 | 0.793 |
| 5 | 32 | 16.0 | 12.17 | 1.397 | 0.804 |
| 6 | 64 | 19.2 | 15.18 | 1.405 | 0.806 |

Going from 1-bit to 3-bit produces real gains. Past 3-bit, bitrate *triples* (5.87 → 15.18 kbps) while STOI moves only 0.793 → 0.806. If the ceiling were a resolution problem, more bits would keep helping. It doesn't — it plateaus hard, meaning the latent had already run out of exploitable information well before the quantizer ran out of levels.

![Rate-distortion sweep: PESQ-WB and STOI vs bitrate, 1-bit through 6-bit, EnCodec shown for reference](plots/fig_02.png)

Reproduce with `python scripts/13_rd_sweep.py`.

### 3. Causality isn't the reason for the remaining gap to EnCodec

A non-causal ablation (bidirectional attention, fine-tuned from Phase G) shows removing the real-time constraint entirely changes almost nothing:

| Model | Bitrate | PESQ-WB | STOI |
|---|---|---|---|
| Phase G (causal) | 5.87 kbps | 1.279 | 0.766 |
| Phase NC (bidirectional) | 5.78 kbps | 1.269 | 0.758 |

The direction of the (tiny) delta flips per speaker — the signature of measurement noise, not a real effect. This rules out "we're causal and EnCodec effectively isn't" as the explanation for the quality gap below; the paper's argument is that the gap comes from EnCodec's adversarial training producing a fundamentally different latent-shaping signal than any reconstruction-based loss used here can supply.

---

## Results

Evaluated on LibriSpeech `test-clean` (5 speakers, 5-second clips, 16 kHz mono).

| Codec | Bitrate | PESQ-WB | STOI |
|---|---|---|---|
| AAC | ~16 kbps | 1.641 | 0.855 |
| EnCodec 1.5 kbps (Meta) | 1.5 kbps | 1.611 | 0.829 |
| EnCodec 3.0 kbps (Meta) | 3.0 kbps | 2.148 | 0.880 |
| EnCodec 6.0 kbps (Meta) | 6.0 kbps | 2.842 | 0.922 |
| Ours — Phase C | 5.7 kbps | 1.202 | 0.733 |
| **Ours — Phase G (default)** | **5.9 kbps** | **1.279** | **0.766** |

**On the gap to EnCodec:** EnCodec uses residual vector quantization and adversarial training — neither replicated here. Section 3 rules out causality as the explanation for the gap. The contribution is the controlled evidence for *why* scalar-quantization codecs hit their quality ceiling, not closing the gap to systems with structurally different architectures.

To reproduce this table: `python scripts/08b_phaseG_eval.py`

> **Note on PESQ:** the `pesq` package compiles a C extension at install time. See [Dependencies](#dependencies) for platform-specific build tool requirements. Without it, PESQ shows `n/a` and STOI is reported instead.

---

## Supporting experiments

Three additional experiments characterize the latent and rule out alternative explanations.

**Speaker identity is not disentangled from content.** A linear probe on the frozen, mean-pooled latent recovers speaker identity at 35.8% accuracy against a 3.1% chance baseline (32 speakers) — expected, since reconstruction-only training has no mechanism to separate "what is said" from "who said it."

![Speaker identity linear probe, per-speaker recall](plots/fig_08.png)

**The bitstream fails completely, not gracefully, under corruption.** zlib's CRC-32 checksum means a single flipped bit causes total decode failure rather than degraded audio — 0% decode success at bit error rate ≥ 0.1%. A real deployment needs a channel-coding layer (e.g. Reed–Solomon) underneath this codec; this repository does not include one.

![Bitstream corruption robustness](plots/fig_09.png)

**Effective bitrate tracks signal complexity automatically, even out-of-distribution**, despite training exclusively on clean speech:

![Bitrate and intelligibility across signal types](plots/fig_05.png)

A pure tone compresses to 0.34 kbps; white/pink noise approaches the 9.6 kbps theoretical cap — bitrate is a direct, mechanical readout of latent entropy (Section 1), and that holds for signals the model never saw in training.

---

## Known limitations — disclosed

- **The intended 200-frame (100 ms) sliding attention window does not function.** `torch.triu` where `torch.tril` was needed makes the window mask a no-op — the model trained on full unbounded causal attention across the entire ~1,995-frame chunk in every phase. Reported metrics reflect this actual behavior; the architecture description below has been corrected. Detail in [12_attention_statistics.md](docs/report_results/12_attention_statistics.md).
- **No positional encoding.** Temporal order comes from causal convolutions and the causal attention mask only.
- **Dropout was never active.** All training scripts passed `dropout=0.0`. Regularization came from noise augmentation and Phase D-VAE's KL term only.
- **Latent width (`bottleneck_dim=32`) — quality ordering confirmed at 16 and 64 dims; controlled D-VAE replication at those widths is pending.** Full A→G curricula at 16-dim and 64-dim confirm monotonic quality scaling (G-16: PESQ 1.135 < G-32: 1.256 < G-64: 1.272), but the D-VAE ablation (β·KL) has only been run at 32 dims. Whether the entropy-quality coupling holds at other widths is scoped to the MS thesis extension.

---

## Future work

The 20 CP research project is complete. The MS thesis (30 CP) extends it by testing whether the entropy-quality coupling generalises across additional axes.

### MS Thesis extensions

| # | Experiment | What it tests | Status |
|---|---|---|---|
| 1 | Soft entropy penalty training (D-Entropy) | Coupling holds under a second independent mechanism — not VAE-specific | **Closed.** D-Entropy vs D: ΔPESQ=−0.070, ΔSTOI=−0.057, p<0.0001*** (n=40). Larger effect than D-VAE. |
| 2 | Music evaluation — MUSDB18-HQ, SI-SDR | Modality independence — coupling holds beyond speech | **Closed.** D-VAE = highest compression (1.440×) + lowest SI-SDR (−7.35 dB) on 40 tracks. D-VAE vs D p<0.0001*** on both metrics. |
| 3 | Bottleneck width ablation (16 / 64 dims) | Coupling holds regardless of latent width | **Partially closed.** Full A→G curricula at 16-dim and 64-dim confirm quality ordering. D-VAE ablation at those widths pending. |
| 4 | VQ comparison (replace SQ with RVQ, same encoder) | Coupling holds regardless of quantizer class — not SQ-specific | **Not started.** Requires a full RVQ curriculum from scratch. This is the strongest remaining open objection to the generality of the coupling claim. |

### Open housekeeping (resolve before numbers go in the paper)

| Item | What is needed |
|---|---|
| Phase A/B PESQ | Run `scripts/eval_phaseAB.py` on Windows (~30 min); PESQ wheel cannot build on this Linux machine. |
| Phase G canonical entropy | Pick one value: 1.520 bits (5-speaker canonical set, 2026-06-30) vs 1.5944 bits (4-speaker recompute, 2026-07-07). Recommend 5-speaker set for consistency with all other headline numbers. |
| Bitrate standardisation | 5.87 kbps (2026-07-01 eval, speaker 121 present) vs 5.97 kbps (OOD eval, speaker 1320 substituted). Use a consistent speaker set throughout. |

---

## Architecture

[![Neural Audio Codec — Architecture](architecture.png)](architecture.png)

```
Waveform (16 kHz)
  → CausalConv encoder     [4 layers, k=7,7,7,3, s=2,2,2,1 → 2000 Hz latent rate]
  → Transformer            [6 layers, d=384, 8 heads — causal; intended 200-frame
                             window is non-functional, see Known Limitations]
  → Linear(384 → 32)       [spatial bottleneck: 12× dimension reduction]
  → Conv1d stride=20       [temporal bottleneck: 2000 Hz → 100 Hz]
  ─── 3-bit quantise + zlib ───   (theoretical cap: 32×3×100 = 9.6 kbps; ~5.9 kbps effective)
  → ConvTranspose1d ×20
  → Linear(32 → 384)
  → Transformer decoder    [identical configuration to encoder]
  → CausalConv decoder
  → Waveform (16 kHz)
```

zlib is used deliberately for its lack of learned adaptivity: because it exploits only generic statistical redundancy, its achieved compression ratio functions as an unbiased probe of the latent's own Shannon entropy, and as a conservative lower bound relative to what a learned entropy model could achieve on the same representation.

**Key properties**

| Property | Value |
|---|---|
| Quantization | 3-bit uniform (8 levels) + zlib entropy coding |
| Sample rate | 16 kHz mono |
| Typical bitrate (Phase G) | ~5.9 kbps |
| Total parameters | ~38M |
| Streaming chunking | Set by inference chunk size (`encode.py` default: 1s), independent of the attention mechanism — see Known Limitations |

---

## Quick start

```
git clone https://gitlab.tu-ilmenau.de/muaw1874/audio_cod.git
cd audio_cod

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Downloads checkpoints (~480 MB) and LibriSpeech test-clean (~346 MB) if not present,
# then verifies the full setup end-to-end
python bootstrap.py
```

Once bootstrap completes with no failures, run inference:

```
# Default: uses first file from LibriSpeech test-clean
python scripts/infer_offline.py

# Or point to any audio file
python scripts/infer_offline.py --input /path/to/speech.wav
```

Each run writes to `inference_runs/<timestamp>/`:

```
source.wav           resampled input fed to the encoder
compressed.nacodec   compressed bitstream (decodable with scripts/decode.py)
reconstructed.wav    decoded output
metrics.json         bitrate, PESQ-WB, STOI, SNR, run metadata
```

Pass `--save 0` to discard `compressed.nacodec` after the run.

---

## Separate encoder and decoder

`infer_offline.py` saves the compressed bitstream as `compressed.nacodec` alongside the other outputs by default (pass `--save 0` to discard it). To encode and decode as two fully separate steps — simulating a transmit/receive pipeline — use the standalone scripts:

```
# Step 1 — encode: audio file → compressed binary
python scripts/encode.py input.wav compressed.nacodec

# Step 2 — decode: compressed binary → reconstructed audio (no original needed)
python scripts/decode.py compressed.nacodec reconstructed.wav
```

The `.nacodec` file is the actual compressed bitstream: a 28-byte header (magic bytes, sample rate, chunk dimensions) followed by the 3-bit scalar-quantised, zlib-compressed latent frames. Everything needed to reconstruct the audio is contained in this file; the original `input.wav` is not required at decode time.

Example output from the encoder:

```
checkpoint:  temporal_phaseG/best.pt  (phase=G, d_model=384, bottleneck=32)
input:       input.wav  (5.00s @ 16000 Hz, 80000 samples)
chunk   1/5  latent=(32, 100)  compressed=1181B
chunk   2/5  latent=(32, 100)  compressed=1173B
...
bitrate:     5.87 kbps
file_size:   3.7 KB  (157 KB uncompressed PCM, 42× reduction)
output:      compressed.nacodec
```

Example output from the decoder:

```
checkpoint:  temporal_phaseG/best.pt  (phase=G, d_model=384, bottleneck=32)
input:       compressed.nacodec  (5.00s @ 16000Hz, 5 chunks)
chunk   1/5  latent=(32, 100)  1181B  → 16000 samples
...
bitrate:     5.87 kbps
output:      reconstructed.wav
```

Both scripts accept `--checkpoint path/to/best.pt` to select a specific phase and `--device cpu` to run without a GPU.

---

## Training curriculum

This is the controlled experiment, not just a list of scripts: each phase loads the previous phase's `best.pt` and changes exactly one aspect of the training objective, holding architecture and data fixed. This single-variable-at-a-time design is what makes the entropy-quality evidence in Section 1 attributable to a specific cause rather than an aggregate correlation. Requires LibriSpeech `train-clean-100` (~6 GB) under `../datasets/LibriSpeech/train-clean-100/`.

```
python scripts/01_phaseA_train.py     # Phase A     — float32 baseline, no quantization
python scripts/02_phaseB_train.py     # Phase B     — 3-bit STE quantisation-aware training
python scripts/03a_phaseC_train.py    # Phase C     — noise augmentation (white/pink/babble)
python scripts/04a_phaseD_train.py    # Phase D     — uniform noise proxy (differentiable QAT)
python scripts/05a_phaseDvae_train.py # Phase D-VAE — variational bottleneck (KL-regularized)
python scripts/06a_phaseE_train.py    # Phase E     — log-magnitude STFT loss
python scripts/07a_phaseF_train.py    # Phase F     — triple combined spectral loss (40 epochs)
python scripts/08a_phaseG_train.py    # Phase G     — fine-polish pass (LR=2e-7, 20 epochs)
```

To evaluate a phase against its baseline:

```
python scripts/03b_phaseC_eval.py     # Phase C vs AAC vs EnCodec
python scripts/04b_phaseD_eval.py     # Phase D vs Phase C
python scripts/05b_phaseDvae_eval.py  # Phase D-VAE vs Phase C
python scripts/06b_phaseE_eval.py     # Phase E vs Phase C
python scripts/07b_phaseF_eval.py     # Phase F vs Phase C
python scripts/08b_phaseG_eval.py     # Phase G vs Phase F vs Phase C
python scripts/13_rd_sweep.py         # Rate-distortion sweep, 1-bit through 6-bit
```

Each eval script writes audio samples, a metrics CSV, and a summary report to `comparisons/`.

---

## Dataset

Inference uses LibriSpeech `test-clean` by default. `bootstrap.py` downloads and extracts it automatically (~346 MB) if not already present, placed at:

```
../datasets/LibriSpeech/test-clean/
```

i.e. one directory above the project root, as a sibling of `audio_cod/`.

If you prefer to download it manually (or if the automatic download fails):

```
mkdir -p ../datasets/LibriSpeech
cd ../datasets/LibriSpeech
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
```

To run inference on your own audio instead:

```
python scripts/infer_offline.py --input /path/to/speech.wav
```

---

## Project structure

```
audio_cod/
├── bootstrap.py                    Entry point — run once after cloning
├── requirements.txt
├── architecture.png
├── plots/                          Figures referenced throughout this README
├── config/
│   ├── paths.yaml                  Dataset and checkpoint path overrides
│   └── training.yaml
├── src/
│   ├── model.py                    Core architecture (encoder, bottleneck, decoder)
│   ├── model_noncausal.py          Bidirectional variant used for the NC ablation
│   ├── losses.py                   Shared loss functions (STFT, spectral, noise utils)
│   ├── codec_utils.py              Shared inference utilities (load_model, encode_decode)
│   ├── train.py                    Base training loop and dataset classes
│   └── paths.py                    Dataset/checkpoint path resolution
├── scripts/
│   ├── infer_offline.py            Offline inference: encode → decode → metrics
│   ├── encode.py                   Standalone encoder  (audio → .nacodec)
│   ├── decode.py                   Standalone decoder  (.nacodec → audio)
│   ├── inspect_nacodec.py          Inspect a .nacodec file (header, per-chunk stats, bitrate)
│   ├── download_checkpoints.py     Download pre-trained weights from Google Drive
│   ├── 01_phaseA_train.py  …  08a_phaseG_train.py / 08b_phaseG_eval.py
│   └── 13_rd_sweep.py              Rate-distortion sweep across quantization bit-depths
├── checkpoints_active/             Downloaded by bootstrap.py — not tracked in git
│   ├── temporal_phaseC/best.pt
│   ├── temporal_phaseD/best.pt
│   ├── temporal_phaseD_vae/best.pt
│   ├── temporal_phaseE/best.pt
│   ├── temporal_phaseF/best.pt
│   └── temporal_phaseG/best.pt
└── inference_runs/                 Per-run artifacts from infer_offline.py — not tracked
```

---

## Dependencies

Core requirements are installed automatically by `bootstrap.py`. Key packages:

| Package | Purpose |
|---|---|
| `torch`, `torchaudio` | Model training and inference |
| `soundfile` | Audio I/O |
| `numpy`, `scipy` | Numerical computing |
| `av` | AAC encode/decode (Phase C evaluation) |
| `pyyaml` | Configuration |
| `tqdm` | Progress bars |
| `gdown` | Checkpoint download from Google Drive |
| `pystoi` | STOI metric |
| `pesq` | PESQ metric (requires build tools — see below) |

> **Note on PESQ:** the `pesq` package compiles a C extension at install time and requires platform build tools:
> - **Linux:** `sudo apt install python3-dev` (Debian/Ubuntu) or `sudo dnf install python3-devel` (RHEL/CentOS/Fedora), then `pip install pesq`
> - **macOS:** install Xcode Command Line Tools (`xcode-select --install`), then `pip install pesq`
> - **Windows:** install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select the "Desktop development with C++" workload), then `pip install pesq`
>
> Without it, PESQ shows `n/a` and STOI is reported instead. `bootstrap.py` reports which metrics are available on your machine.

---

## References

- A. Brendel, N. Pia, K. Gupta, L. Behringer, G. Fuchs, and M. Multrus, "Neural Speech Coding for Real-Time Communications Using Constant Bitrate Scalar Quantization," *IEEE Journal of Selected Topics in Signal Processing*, 2024.
- A. Défossez, J. Copet, G. Synnaeve, and Y. Adi, "High Fidelity Neural Audio Compression" (EnCodec), *Transactions on Machine Learning Research*, 2023.
- V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books," *ICASSP*, 2015.
- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural Discrete Representation Learning" (VQ-VAE), *NeurIPS*, 2017.
- J. Xu, Z. Cheng, F. Zhang, Y. Liu, L. Song, and W. Zhang, "Benchmarking Neural Speech Compression from a Rate-Distortion Perspective," arXiv:2606.11631, 2026.
- N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, "SoundStream: An End-to-End Neural Audio Codec," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 30, pp. 495–507, 2021.
