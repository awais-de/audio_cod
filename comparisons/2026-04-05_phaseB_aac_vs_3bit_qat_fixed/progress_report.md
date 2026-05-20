# Neural Audio Codec — Progress Report
**Date:** April 5, 2026  
**Author:** Muhammad Awais  
**Supervisor:** Prof. Gerald Schuller

---

## 1. Objectives

The professor's requirements for this iteration were:

- Reduce bitrate further toward **10 kbps**
- Reduce latency to **below 100ms** (for real-time communication)
- Keep **STOI high**
- Provide **audio examples** from both AAC and the neural model

All four objectives have been addressed.

---

## 2. Architecture

The model extends a pre-trained transformer-based neural audio codec (Phase 1 baseline, PESQ=4.27) with two compression stages stacked after the encoder:

```
Waveform (16 kHz)
    → AudioEncoder         [6-layer causal transformer, d=384, window=100ms]
    → Linear(384 → 32)     [spatial bottleneck: 12× dimension reduction]
    → Conv1d stride=20     [temporal bottleneck: 2000 Hz → 100 Hz frame rate]
    ─── QUANTIZE HERE ───  [3-bit uniform, 8 levels per dimension per frame]
    → ConvTranspose1d ×20  [temporal upsample back to 2000 Hz]
    → Linear(32 → 384)     [spatial expansion]
    → AudioDecoder         [6-layer causal transformer]
    → Waveform (16 kHz)
```

**Hard bitrate cap:**  
32 dimensions × 3 bits × 100 Hz = **9,600 bps = 9.6 kbps**

After entropy coding (zlib, level 9), the real measured bitrate is **~6.3 kbps**.

**Latency:**  
The attention window is fixed at 200 frames × 0.5ms = **100ms** of context.  
Measured encode+decode latency: **58ms** per 1-second chunk.

---

## 3. Training Strategy: Two-Phase Approach

A single-phase cold-start training (randomly initialised bottleneck layers + 3-bit QAT simultaneously) caused the loss to stall at 0.308 — the new layers could not learn meaningful temporal compression and survive quantization at the same time.

The fix was a **two-phase curriculum**:

### Phase A — Float32 Compression (No Quantization)
- Loaded Phase 1 checkpoint; new bottleneck layers randomly initialised
- Forward pass uses **float32 latents** — no quantization constraint
- Goal: teach the model to compress speech temporally before constraining it
- Optimiser: Adam, LR=3×10⁻⁵, **cosine annealing** to 1×10⁻⁷ over 30 epochs
- Best loss reached: **0.192** (epoch 24)

### Phase B — 3-bit QAT Fine-tuning
- Loaded Phase A best checkpoint (all layers already trained)
- Forward pass: 3-bit Straight-Through Estimator (STE) quantization applied to latents
  - Forward: round to 8 uniform levels between z_min and z_max
  - Backward: identity (gradients flow through as if no quantization)
- Optimiser: Adam, LR=1×10⁻⁵, cosine annealing over 20 epochs
- Best loss reached: **0.251** (epoch 19)

---

## 4. Quantization & Bitrate Measurement

**3-bit STE quantization (training):**
```
z_min, z_max = z.min(), z.max()
scale = (z_max - z_min) / 7          # 8 levels → 7 intervals
q = round((z - z_min) / scale)       # integer in {0,...,7}
z_quant = q * scale + z_min          # dequantized
z_ste = z + (z_quant - z).detach()  # STE: real forward, identity backward
```

**Real bitrate measurement (evaluation):**
1. Encode audio in 1-second chunks with `model.encode()`
2. Apply 3-bit uniform quantization → uint8 array
3. Compress with `zlib.compress(level=9)` as entropy coder
4. Bitrate = compressed bytes × 8 / audio duration

This is the actual deployable pipeline — no theoretical assumptions.

---

## 5. Results

Evaluated on **LibriSpeech test-clean**, 5 speakers, 5-second clips. Neither the test speakers nor their audio were seen during training.

| Codec | Bitrate | PESQ (WB) | STOI | Latency |
|---|---|---|---|---|
| AAC @ 10 kbps target (actual floor: ~15–16 kbps) | 15.6 kbps | 1.490 | 0.497 | N/A |
| **Neural 3-bit QAT (ours)** | **6.3 kbps** | **1.794** | **0.588** | **58ms** |

**Per-speaker breakdown:**

| Speaker | AAC kbps | AAC PESQ | AAC STOI | Neural kbps | Neural PESQ | Neural STOI |
|---|---|---|---|---|---|---|
| 1089 | 16.7 | 1.490 | 0.497 | 5.9 | 1.854 | 0.606 |
| 1188 | 15.4 | 1.501 | 0.500 | 6.1 | 1.747 | 0.574 |
| 1221 | 15.2 | 1.497 | 0.499 | 6.5 | 1.762 | 0.579 |
| 1284 | 15.6 | 1.485 | 0.496 | 6.1 | 1.936 | 0.631 |
| 1320 | 15.4 | 1.477 | 0.493 | 7.0 | 1.669 | 0.551 |

**Key observations:**
- Neural codec operates at **6.3 kbps** — well below the 10 kbps target, and 2.5× lower than AAC's minimum achievable bitrate at 16 kHz mono (~15–16 kbps).
- Neural codec **outperforms AAC on both PESQ and STOI** despite the much lower bitrate.
- Latency of **58ms** satisfies the <100ms real-time communication requirement.

---

## 6. Bitrate Reduction Journey

| Stage | Real Bitrate | PESQ | Notes |
|---|---|---|---|
| Phase 1 baseline | 88.5 kbps | 4.27 | No compression; full float32 latents |
| Bottleneck v1 (32-dim) | ~10.6 kbps | 1.865 | Spatial bottleneck only |
| 1-bit QAT attempt | 37 kbps | — | Abandoned; latents spread apart |
| Temporal stride v1 (cold-start QAT) | 5.6 kbps | 1.937 | Loss stalled at 0.308 |
| **Two-phase (Phase A + B)** | **6.3 kbps** | **1.794** | Current best deployable model |

Total bitrate reduction: **88.5 kbps → 6.3 kbps (14× reduction)**

---

## 7. Audio Examples

Attached audio files compare the two codecs on 5 speakers from LibriSpeech test-clean.  
Each speaker folder contains three files:

| File | Description |
|---|---|
| `source.wav` | Original uncompressed audio (reference) |
| `aac_XYkbps.wav` | AAC encoded + decoded at 10 kbps target (actual ~15–16 kbps) |
| `neural_3bit_Xkbps.wav` | Neural codec: 3-bit quantized + zlib compressed + reconstructed |

Speakers: 1089, 1188, 1221, 1284, 1320 (5-second clips each).

---

## 8. Possible Next Steps

- **Entropy model**: replace zlib with a learned arithmetic coder trained on the quantized latent distribution — potentially reducing bitrate to 3–4 kbps.
- **Perceptual loss**: add a discriminator (GAN-style) or mel-spectrogram loss weighted toward speech frequencies to improve perceived quality.
- **More quantization levels**: experiment with 4-bit (16 levels) at the cost of slightly higher bitrate (~12.8 kbps cap) for improved quality.
- **Streaming evaluation**: benchmark real-time streaming latency on embedded/CPU hardware.
