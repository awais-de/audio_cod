# Neural Audio Codec — Low-Bitrate Speech Compression

A transformer-based neural audio codec targeting real-time speech communication at sub-10 kbps with under 100ms latency.

Developed at TU Ilmenau under the supervision of Prof. Gerald Schuller.

---

## Results

Evaluated on LibriSpeech test-clean (5 speakers, 5-second clips, 16 kHz mono):

| Codec | Bitrate | PESQ | STOI | Latency |
|---|---|---|---|---|
| AAC (actual floor at 16 kHz mono) | 15.6 kbps | 2.763 | 0.879 | N/A |
| EnCodec 1.5 kbps (Meta) | 1.5 kbps | 2.539 | 0.812 | N/A |
| EnCodec 3.0 kbps (Meta) | 3.0 kbps | 2.681 | 0.854 | N/A |
| EnCodec 6.0 kbps (Meta) | 6.0 kbps | 2.785 | 0.885 | N/A |
| **Ours (3-bit QAT, Phase C)** | **5.8 kbps** | **2.645** | **0.843** | **100ms** |

Our codec at 5.8 kbps is competitive with EnCodec at 3.0 kbps, and within 0.14 PESQ of EnCodec at the same bitrate — using a significantly simpler quantization scheme (3-bit uniform + zlib vs. residual vector quantization + GAN training).

---

## Architecture

```
Waveform (16 kHz)
  → CausalConv encoder  [3× stride-2 → 2000 Hz latent rate]
  → Transformer         [6 layers, d=384, causal window=200 frames = 100ms]
  → Linear(384 → 32)   [spatial bottleneck: 12× dimension reduction]
  → Conv1d stride=20   [temporal bottleneck: 2000 Hz → 100 Hz]
  ─── 3-bit quantize + zlib ───   (bitrate cap: 32×3×100 = 9.6 kbps)
  → ConvTranspose1d ×20
  → Linear(32 → 384)
  → Transformer decoder
  → CausalConv decoder
  → Waveform (16 kHz)
```

**Key properties:**
- Algorithmic delay: **100ms** (causal window size)
- Minimum streaming block: **100ms**
- Quantization: 3-bit uniform (8 levels) + zlib entropy coding
- Training: Two-phase curriculum — Phase A (float32 compression) → Phase B/C (3-bit QAT + noise augmentation)

---

## Installation

```bash
git clone https://github.com/awais-de/audio_cod.git
cd audio_cod
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start — Inference

**Encode + decode in one step:**
```bash
python scripts/infer_file.py speech.wav
# output: speech_reconstructed.wav
```

**Or as separate steps:**
```bash
python scripts/encode.py speech.wav speech.nacodec
python scripts/decode.py speech.nacodec reconstructed.wav
```

**Real-time microphone demo:**
```bash
python scripts/infer_stream.py
```

All scripts read model architecture from the checkpoint automatically — no config file needed.

> **Note:** Pre-trained checkpoints (~84 MB each) are not stored in this repository.  
> Download from: [TU Ilmenau SharePoint](https://tuilmenau365-my.sharepoint.com/:f:/g/personal/m_awais_tu-ilmenau_de/IgDKZN_RnOjaTrwfdK77ocxuAZxSG3XjEyz0cB3_VOwCYZs?e=HodLGX)  
> Place under `checkpoints_active/temporal_phaseC/best.pt`

---

## Reproduce the Comparison

```bash
python scripts/03b_phaseC_eval.py
```

Runs AAC vs EnCodec vs Ours on 5 LibriSpeech test-clean speakers at multiple bitrates.

---

## Training

To reproduce the full training curriculum from scratch:

```bash
# Phase A: Learn temporal compression (float32, no quantization)
python scripts/01_phaseA_train.py

# Phase B: Fine-tune with 3-bit QAT
python scripts/02_phaseB_train.py

# Phase C: Continue with noise augmentation
python scripts/03a_phaseC_train.py

# Phase D: Uniform noise proxy (VAE-style QAT)
python scripts/04a_phaseD_train.py

# Phase E: Log-magnitude STFT loss
python scripts/06a_phaseE_train.py

# Phase F: Triple combined spectral loss
python scripts/07a_phaseF_train.py

# Phase G: Fine-polish pass (lower LR)
python scripts/08a_phaseG_train.py
```

Each phase loads from the previous phase's `best.pt`. Requires LibriSpeech `train-clean-100` (~6 GB). Update paths in `config/paths.yaml`.

---

## Project Structure

```
audio_cod/
├── src/
│   ├── model.py               Core architecture (encoder, bottleneck, decoder)
│   ├── train.py               Base training loop and dataset utilities
│   ├── quantization.py        Uniform quantizer and entropy coder
│   ├── rate_distortion_loss.py  R-D loss (distortion + rate penalty)
│   ├── entropy_model.py       Learned GMM entropy model
│   └── paths.py               Dataset/checkpoint path resolution
├── scripts/
│   ├── encode.py              Compress audio to .nacodec binary
│   ├── decode.py              Decompress .nacodec to audio
│   ├── infer_file.py          Encode + decode in one command
│   ├── infer_stream.py        Real-time microphone demo
│   ├── 01_phaseA_train.py     Phase A: float32 temporal compression
│   ├── 02_phaseB_train.py     Phase B: 3-bit QAT
│   ├── 03a_phaseC_train.py    Phase C: noise augmentation
│   ├── 03b_phaseC_eval.py     Phase C: evaluation vs AAC / EnCodec
│   ├── 04a_phaseD_train.py    Phase D: uniform noise proxy
│   ├── 04b_phaseD_eval.py
│   ├── 05a_phaseDvae_train.py Phase D-VAE: variational bottleneck
│   ├── 05b_phaseDvae_eval.py
│   ├── 06a_phaseE_train.py    Phase E: log-magnitude STFT loss
│   ├── 06b_phaseE_eval.py
│   ├── 07a_phaseF_train.py    Phase F: triple combined spectral loss
│   ├── 07b_phaseF_eval.py
│   ├── 08a_phaseG_train.py    Phase G: fine-polish pass
│   └── 08b_phaseG_eval.py
├── checkpoints_active/        Trained weights (not tracked — see SharePoint link above)
├── config/
│   ├── paths.yaml             Dataset paths
│   ├── training.yaml          Base training config
│   └── training_improved.yaml
├── requirements.txt
└── setup.py
```

---

## Dependencies

```
torch>=2.0
torchaudio>=2.0
soundfile
numpy
pyyaml
tqdm
pesq
pystoi
encodec          # for comparison only
av               # for AAC comparison only (pip install av)
sounddevice      # for real-time streaming demo only
```

---

## References

- Défossez et al., *High Fidelity Neural Audio Compression* (EnCodec, Meta 2022)
- Zeghidour et al., *SoundStream: An End-to-End Neural Audio Codec* (Google 2021)
- LibriSpeech corpus: Panayotov et al., ICASSP 2015
