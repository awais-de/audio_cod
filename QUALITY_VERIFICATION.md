# Audio Quality Verification Report

**Date:** January 27, 2026  
**Model:** Neural Audio Codec (best_model.pt)  
**Test Set:** LibriSpeech train-clean-100 (6 samples evaluated)  
**Targets:** PESQ ≥ 3.5, STOI ≥ 0.9

---

## Executive Summary

❌ **RESULT: FAILED** - The Neural Audio Codec does NOT meet the audio quality requirements.

### Key Findings

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **PESQ** | ≥ 3.5 | **1.04-1.10** | ❌ **67% below target** |
| **STOI** | ≥ 0.9 | **0.37-0.41** | ❌ **58% below target** |
| **SNR** | > 20 dB (typical) | **-26.8 to -27.4 dB** | ❌ Poor |

---

## Detailed Results (Sample of 6 Files)

| File | Duration | SNR (dB) | PESQ | STOI | MSE | LSD |
|------|----------|----------|------|------|-----|-----|
| 2007-149877-0049 | 9.99s | -27.37 | 1.044 | 0.379 | 0.9988 | 3.135 |
| 2007-149877-0033 | 9.99s | -27.11 | 1.097 | 0.413 | 0.9985 | 2.431 |
| 2007-149877-0043 | 9.99s | -26.82 | 1.068 | 0.365 | 0.9990 | 3.065 |
| (3 more files) | ~10s each | ~-27 | ~1.07 | ~0.38 | ~0.999 | ~2.9 |

**Averages (from 6 samples):**
- PESQ: ~1.07 (Target: ≥3.5) ❌
- STOI: ~0.39 (Target: ≥0.9) ❌  
- SNR: ~-27 dB (Target: >20 dB) ❌

---

## Analysis

### What Do These Results Mean?

#### 1. **PESQ (Perceptual Evaluation of Speech Quality)**
- **Scale:** 1.0 (Bad) to 4.5 (Excellent)
- **Target:** ≥ 3.5 (Good quality)
- **Achieved:** ~1.07 (Very poor quality)
- **Interpretation:** The reconstructed audio has severe perceptual degradation. Users would rate the quality as "bad" or "unacceptable" for teleconferencing.

#### 2. **STOI (Short-Time Objective Intelligibility)**
- **Scale:** 0.0 (Unintelligible) to 1.0 (Fully intelligible)
- **Target:** ≥ 0.9 (90% intelligibility)
- **Achieved:** ~0.39 (39% intelligibility)
- **Interpretation:** Only about 39% of speech is intelligible. This is completely unacceptable for communication applications.

#### 3. **SNR (Signal-to-Noise Ratio)**
- **Expected for good codec:** 20-30 dB
- **Achieved:** -27 dB (NEGATIVE!)
- **Interpretation:** The "noise" (reconstruction error) is actually LOUDER than the original signal. This indicates the model is producing mostly noise/distortion rather than reconstructing the audio.

#### 4. **MSE (Mean Squared Error)**
- **Achieved:** ~0.999
- **Interpretation:** Nearly 1.0 means the reconstructed signal is almost completely different from the original. This confirms the model is not learning proper reconstruction.

---

## Root Cause Analysis

### Why Is Quality So Poor?

Based on the metrics, several issues are likely:

1. **Insufficient Training**
   - Model was trained for 100 epochs (22.11 hours)
   - May need significantly more training time
   - Training loss converged, but to a poor local minimum

2. **Model Capacity Issues**
   - Model has 6.7M parameters (optimized from 51.8M)
   - May have been over-optimized for speed at the expense of quality
   - Smaller models (d_model=256, n_layers=4) may lack capacity for high-quality reconstruction

3. **Loss Function**
   - Multi-scale spectral loss + L1 loss may not be sufficient
   - May need adversarial training (GAN) or perceptual losses
   - Current loss may not correlate well with perceptual quality

4. **Data Issues**
   - Training on LibriSpeech (clean speech only)
   - Model may be underfitting
   - Need to verify training data quality and diversity

5. **Architecture Limitations**
   - Sliding window attention (256 frames) may limit context
   - Causal design may prevent full context modeling
   - Compression ratio (32x) may be too aggressive

---

## Comparison with Requirements

| Requirement | Target | Current Status | Gap |
|-------------|--------|----------------|-----|
| **Latency** | < 20ms | ✅ ~10ms | **MET** (+100%) |
| **PESQ** | ≥ 3.5 | ❌ ~1.07 | **NOT MET** (-69%) |
| **STOI** | ≥ 0.9 | ❌ ~0.39 | **NOT MET** (-57%) |
| **Bitrate** | 8-16 kbps | ⏳ Not tested | Pending quantization |
| **Real-time** | RTF < 1.0 | ✅ 0.07-0.70x | **MET** |

**Overall:** 2 of 5 criteria met (Latency, Real-time capability)

---

## Recommendations

### Immediate Actions (Critical Path)

#### Option A: Improve Current Model (2-3 weeks)

1. **Extended Training** (3-5 days)
   - Train for 500-1000 epochs instead of 100
   - Monitor PESQ/STOI on validation set
   - Early stopping based on quality metrics, not just loss

2. **Increase Model Capacity** (1-2 days)
   - Restore to original size: d_model=512, n_layers=8
   - Trade latency for quality (may still meet <20ms target)
   - Re-run latency benchmarks after

3. **Improve Loss Function** (2-3 days)
   - Add perceptual loss (VGGish features)
   - Add adversarial loss (discriminator)
   - Weight losses to optimize PESQ/STOI directly

4. **Data Augmentation** (1-2 days)
   - Add noise robustness training
   - Use multiple datasets (LibriSpeech + VCTK + others)
   - Increase training data volume

#### Option B: Use Baseline Codec for Demo (Faster, 2-3 days)

1. **Demonstrate with Opus/AAC**
   - Use industry-standard codec that meets requirements
   - Show neural codec as "research prototype"
   - Focus on latency/infrastructure demonstration

2. **Neural Codec as Supplementary**
   - Show the neural architecture works (low latency)
   - Acknowledge quality needs improvement
   - Discuss future research directions

---

## Next Steps (Recommended Order)

### Priority 1: Diagnose Training Issue (1 day)

1. **Check Training Curves**
   - Review training/validation loss over 100 epochs
   - Verify loss actually decreased
   - Check for overfitting or underfitting

2. **Inspect Model Outputs**
   - Listen to reconstructed audio samples
   - Visualize spectrograms (original vs reconstructed)
   - Identify artifacts (noise, distortion, missing frequencies)

3. **Sanity Check**
   - Test model on training set samples
   - If training set also has poor quality, model didn't learn
   - If training set is good but validation bad, overfitting issue

### Priority 2: Quick Quality Improvement (3-5 days)

1. **Increase model size** to d_model=512, n_layers=6
2. **Train for 300 epochs** (~3-4 days with optimization)
3. **Add perceptual losses** (STFT magnitude, phase)
4. **Test on validation set** every 50 epochs

### Priority 3: Parallel Development (Ongoing)

1. **Build real-time demo infrastructure** (can use baseline codec initially)
2. **Implement bitrate quantization** (INT8, vector quantization)
3. **Prepare baseline comparisons** (Opus at 16kbps)

---

## Technical Details

### Test Configuration
- **Model:** 6.7M parameters (d_model=256, n_layers=4, n_heads=8)
- **Sample Rate:** 16 kHz mono
- **Processing:** 2-second chunks with GPU
- **Test Duration:** 10 seconds per file (to avoid OOM)
- **Metrics:**
  - PESQ: Wide-band mode (8-16 kHz)
  - STOI: Standard mode (not extended)
  - SNR: Power-based calculation

### Measurement Methodology
- Processed 6 audio files from LibriSpeech
- Each file ~10 seconds duration
- Chunked processing (2s chunks) to avoid GPU OOM
- Metrics calculated on full reconstructed sequence

---

## Context: What Is Acceptable Quality?

### PESQ Scale Reference
- **4.5:** Excellent (transparent, indistinguishable)
- **4.0:** Good (minor impairments)
- **3.5:** Fair (noticeable but acceptable) ← **TARGET**
- **3.0:** Poor (annoying)
- **2.5:** Bad (very annoying)
- **1.0:** Unacceptable ← **CURRENT: ~1.07**

### STOI Scale Reference
- **1.0 (100%):** Perfect intelligibility
- **0.9 (90%):** Good intelligibility ← **TARGET**
- **0.7 (70%):** Moderate intelligibility
- **0.5 (50%):** Poor intelligibility
- **0.39 (39%):** Very poor ← **CURRENT**

### SNR Reference
- **30+ dB:** Excellent quality
- **20-30 dB:** Good quality (typical for codecs)
- **10-20 dB:** Acceptable quality
- **0-10 dB:** Poor quality
- **Negative:** Noise louder than signal ← **CURRENT: -27 dB**

---

## Conclusion

While the Neural Audio Codec successfully meets the **latency requirement** (<20ms), it **fails to meet audio quality requirements** by a significant margin.

**Critical Issues:**
- ❌ PESQ: 1.07 vs 3.5 target (69% below)
- ❌ STOI: 0.39 vs 0.9 target (57% below)  
- ❌ SNR: -27 dB vs 20+ dB expected

**The model produces audio that is:**
- Barely intelligible (39% STOI)
- Very poor perceptual quality (1.07 PESQ)
- More noise than signal (-27 dB SNR)

**Recommended Path Forward:**
1. **Immediate:** Diagnose training issue (1 day)
2. **Short-term:** Retrain with larger model + more epochs (3-5 days)
3. **Alternative:** Use baseline codec for demo, show neural codec as prototype

**Time Estimate to Meet Quality Requirements:** 1-3 weeks of focused development and training.

---

**Report Generated:** January 27, 2026  
**Test Configuration:** best_model.pt (6.7M params, 100 epochs)  
**Recommendation:** DO NOT PROCEED to demo with current model quality
