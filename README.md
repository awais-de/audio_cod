# Neural Audio Codec — Low-Bitrate Speech Compression

A transformer-based neural audio codec targeting real-time speech communication at sub-10 kbps, streaming in causal, fixed-size chunks (1 second by default) for real-time use. See [Known Issues](#known-issues) for a note on the transformer's attention window and what actually governs latency.

Developed at TU Ilmenau under the supervision of Prof. Gerald Schuller.

---

## Replication — Quick Start

```bash
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

```bash
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

## Separate Encoder and Decoder

`infer_offline.py` saves the compressed bitstream as `compressed.nacodec` alongside the other outputs by default (pass `--save 0` to discard it). To encode and decode as two fully separate steps — simulating a transmit/receive pipeline — use the standalone scripts:

```bash
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

## Results

Evaluated on LibriSpeech test-clean (5 speakers, 5-second clips, 16 kHz mono).

| Codec | Bitrate | PESQ-WB | STOI |
|---|---|---|---|
| AAC | ~16 kbps | 1.641 | 0.855 |
| EnCodec 1.5 kbps (Meta) | 1.5 kbps | 1.611 | 0.829 |
| EnCodec 3.0 kbps (Meta) | 3.0 kbps | 2.148 | 0.880 |
| EnCodec 6.0 kbps (Meta) | 6.0 kbps | 2.842 | 0.922 |
| Ours — Phase C | 5.7 kbps | 1.202 | 0.733 |
| Ours — Phase G (default) | 5.9 kbps | 1.279 | 0.766 |

Phase G is the best-performing checkpoint across all phases. The training progression from Phase C to Phase G yields a consistent +0.077 PESQ and +0.033 STOI improvement through combined spectral losses and fine-polish training. The gap versus EnCodec at matched bitrate is explained by latent entropy structure — a rate-distortion sweep (1-bit through 6-bit) confirms quality plateaus from 4-bit onward regardless of added bits, with the ceiling set by the training objective rather than quantization resolution.

To reproduce this table:

```bash
python scripts/08b_phaseG_eval.py
```

> **Note on PESQ:** the `pesq` package compiles a C extension at install time and requires platform build tools:
> - **Linux:** `sudo apt install python3-dev` (Debian/Ubuntu) or `sudo dnf install python3-devel` (RHEL/CentOS/Fedora), then `pip install pesq`
> - **macOS:** install Xcode Command Line Tools (`xcode-select --install`), then `pip install pesq`
> - **Windows:** install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select the "Desktop development with C++" workload), then `pip install pesq`
>
> Without it, PESQ shows `n/a` and STOI is reported instead. `bootstrap.py` reports which metrics are available on your machine.

---

## Architecture

![Neural Audio Codec — Architecture](architecture.png)

```
Waveform (16 kHz)
  → CausalConv encoder   [3× stride-2  →  2000 Hz latent rate]
  → Transformer          [6 layers, d=384, causal self-attention over the full chunk (~1,995 frames)]
  → Linear(384 → 32)    [spatial bottleneck: 12× dimension reduction]
  → Conv1d stride=20    [temporal bottleneck: 2000 Hz → 100 Hz]
  ─── 3-bit quantise + zlib ───   (theoretical cap: 32×3×100 = 9.6 kbps)
  → ConvTranspose1d ×20
  → Linear(32 → 384)
  → Transformer decoder
  → CausalConv decoder
  → Waveform (16 kHz)
```

**Key properties**

| Property | Value |
|---|---|
| Streaming latency | Set by inference chunk size — 1 s default in `encode.py`, configurable |
| Transformer attention | Causal, unrestricted over the full input chunk (see [Known Issues](#known-issues)) |
| Quantization | 3-bit uniform (8 levels) + zlib |
| Sample rate | 16 kHz mono |
| Typical bitrate (Phase G) | ~5.9 kbps |

---

## Known Issues

**Attention window — confirmed non-functional.** The transformer was designed with a 200-frame (100 ms) sliding attention window for low latency, as stated in the code's own docstring. Direct inspection of the trained model and source confirmed this was never enforced: the windowing mask was implemented as `torch.triu(diagonal=window_size+1)` (masking keys far *ahead* of the query — already excluded by the causal mask) instead of a `torch.tril` masking keys far *behind* the query, which is what limiting past lookback actually requires. Since a standard 1-second training chunk produces `seq_len ≈ 1,995` — roughly 10× the intended window — this bug was active for every training step, in every phase, from the start.

The model was trained on, and relies on, **unrestricted causal attention across the full chunk**, not a 200-frame window. Per-layer attention measured at the real operating length (seq_len≈1,995) is close to uniform across each layer's visible history (avg. distance 441–511 frames, close to the theoretical uniform baseline T/4≈499) — not concentrated on recent frames.

This does **not** invalidate any PESQ/STOI/entropy/bitrate result reported below — every metric measures the model's actual trained behavior either way, bug included. It does mean the "100 ms window" describes an architectural *intention*, not the trained model's real internal behavior. Deployable streaming latency is unaffected by this bug: it's governed by inference **chunk size**, set independently in `encode.py` (1-second default, configurable), not by the attention window internals.

---

## Training Curriculum

Each phase loads from the previous phase's `best.pt`. Requires LibriSpeech `train-clean-100` (~6 GB) under `../datasets/LibriSpeech/train-clean-100/`.

```bash
python scripts/01_phaseA_train.py    # Phase A  — float32 temporal compression baseline
python scripts/02_phaseB_train.py    # Phase B  — 3-bit STE quantisation-aware training
python scripts/03a_phaseC_train.py   # Phase C  — noise augmentation (white/pink/babble)
python scripts/04a_phaseD_train.py   # Phase D  — uniform noise proxy (differentiable QAT)
python scripts/05a_phaseDvae_train.py # Phase D-VAE — variational bottleneck
python scripts/06a_phaseE_train.py   # Phase E  — log-magnitude STFT loss
python scripts/07a_phaseF_train.py   # Phase F  — triple combined spectral loss (40 epochs)
python scripts/08a_phaseG_train.py   # Phase G  — fine-polish pass (LR=2e-7, 20 epochs)
```

To evaluate a phase against the previous baseline:

```bash
python scripts/03b_phaseC_eval.py    # Phase C vs AAC vs EnCodec
python scripts/04b_phaseD_eval.py    # Phase D vs Phase C
python scripts/05b_phaseDvae_eval.py # Phase D-VAE vs Phase C
python scripts/06b_phaseE_eval.py    # Phase E vs Phase C
python scripts/07b_phaseF_eval.py    # Phase F vs Phase C
python scripts/08b_phaseG_eval.py    # Phase G vs Phase F vs Phase C
```

Each eval script writes audio samples, a metrics CSV, and a summary report to `comparisons/`.

---

## Dataset

Inference uses LibriSpeech `test-clean` by default. `bootstrap.py` downloads and extracts it automatically (~346 MB) if not already present. It is placed at:

```
../datasets/LibriSpeech/test-clean/
```

i.e. one directory above the project root, as a sibling of `audio_cod/`.

If you prefer to download it manually (or if the automatic download fails):

```bash
mkdir -p ../datasets/LibriSpeech
cd ../datasets/LibriSpeech
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
```

To run inference on your own audio instead:

```bash
python scripts/infer_offline.py --input /path/to/speech.wav
```

---

## Project Structure

```
audio_cod/
├── bootstrap.py                    Entry point — run once after cloning
├── requirements.txt
├── config/
│   ├── paths.yaml                  Dataset and checkpoint path overrides
│   └── training.yaml
├── src/
│   ├── model.py                    Core architecture (encoder, bottleneck, decoder)
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
│   ├── 01_phaseA_train.py
│   ├── 02_phaseB_train.py
│   ├── 03a_phaseC_train.py  /  03b_phaseC_eval.py
│   ├── 04a_phaseD_train.py  /  04b_phaseD_eval.py
│   ├── 05a_phaseDvae_train.py  /  05b_phaseDvae_eval.py
│   ├── 06a_phaseE_train.py  /  06b_phaseE_eval.py
│   ├── 07a_phaseF_train.py  /  07b_phaseF_eval.py
│   └── 08a_phaseG_train.py  /  08b_phaseG_eval.py
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
| `pesq` | PESQ metric (requires `python3-dev` on Linux) |

---

## References

- Défossez et al., *High Fidelity Neural Audio Compression* (EnCodec, Meta 2022)
- Zeghidour et al., *SoundStream: An End-to-End Neural Audio Codec* (Google 2021)
- Panayotov et al., *LibriSpeech: An ASR Corpus Based on Public Domain Audio Books* (ICASSP 2015)
