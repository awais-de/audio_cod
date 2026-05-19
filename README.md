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
git clone https://github.com/talhar007/audio_cod.git
cd audio_cod
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start — Inference

```bash
python scripts/inference.py --input speech.wav --output reconstructed.wav
```

The script reads all architecture parameters from the checkpoint automatically. No config file needed.

**Example output:**
```
Bitrate   : 5.5 kbps  (cap: 9.6 kbps)
Latency   : 100 ms algo delay
PESQ (WB) : 2.740
STOI      : 0.872
```

**With a specific checkpoint:**
```bash
python scripts/inference.py \
    --input speech.wav \
    --output reconstructed.wav \
    --checkpoint checkpoints_active/temporal_phaseC/best.pt
```

> **Note:** Pre-trained checkpoints (~84 MB each) are not stored in this repository.  
> Download from: [TU Ilmenau SharePoint](https://tuilmenau365-my.sharepoint.com/:f:/g/personal/m_awais_tu-ilmenau_de/IgDKZN_RnOjaTrwfdK77ocxuAZxSG3XjEyz0cB3_VOwCYZs?e=HodLGX)  
> Place under `checkpoints_active/temporal_phaseC/best.pt`

---

## Reproduce the Comparison

```bash
# AAC vs EnCodec vs Ours (5 speakers, 5 bitrates)
python scripts/compare_encodec.py

# AAC vs Ours only
python scripts/compare_phaseB.py
```

Audio output saved to `comparisons/encodec_comparison/` — each speaker folder contains:
`source.wav`, `aac_Xkbps.wav`, `encodec_1.5kbps.wav`, `encodec_3.0kbps.wav`, `encodec_6.0kbps.wav`, `ours_3bit_Xkbps.wav`

---

## Training

To reproduce training from scratch:

```bash
# Phase A: Learn temporal compression (float32, no quantization)
python scripts/finetune_temporal_phaseA.py

# Phase B: Fine-tune with 3-bit QAT
python scripts/finetune_temporal_phaseB.py

# Phase C: Continue with noise augmentation
python scripts/finetune_temporal_phaseC.py
```

Requires LibriSpeech `train-clean-100` (~6 GB). Update paths in `config/paths.yaml`.

---

## Project Structure

```
audio_cod/
├── src/
│   ├── model.py              Core architecture (encoder, bottleneck, decoder)
│   └── paths.py              Dataset/checkpoint path resolution
├── scripts/
│   ├── inference.py          ← Run this for compression/reconstruction
│   ├── compare_encodec.py    3-way comparison: AAC vs EnCodec vs Ours
│   ├── finetune_temporal_phaseA.py   Phase A training
│   ├── finetune_temporal_phaseB.py   Phase B QAT training
│   ├── finetune_temporal_phaseC.py   Phase C noise augmentation
│   └── archive/              Old experimental scripts
├── checkpoints_active/
│   ├── phase1_base/          Base model (PESQ=4.27, no compression)
│   ├── temporal_phaseA/      Phase A best checkpoint
│   ├── temporal_phaseB/      Phase B best checkpoint
│   ├── temporal_phaseC/      Phase C best checkpoint ← current best
│   └── archive/              All previous runs preserved
├── comparisons/
│   └── encodec_comparison/   Audio examples + metrics CSV + report
├── runs/
│   ├── RUNS.md               Master log of all training runs
│   └── temporal_phaseC/      Phase C training history + log
├── docs/
│   └── progress_report.md    Report submitted to supervisor
├── config/paths.yaml         Dataset paths
└── requirements.txt
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
```

---

## References

- Défossez et al., *High Fidelity Neural Audio Compression* (EnCodec, Meta 2022)
- Zeghidour et al., *SoundStream: An End-to-End Neural Audio Codec* (Google 2021)
- LibriSpeech corpus: Panayotov et al., ICASSP 2015
