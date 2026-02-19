# Phase 3b: Rate-Distortion Training - Implementation Guide

## Executive Summary

**Previous state**: Neural codec at 141 kbps (too high, needs learned entropy)  
**Phase 3a outcome**: Trained entropy model converged (loss: -1035.71 nats)  
**Phase 3b goal**: Fine-tune encoder/decoder with R-D loss to reach **15-25 kbps**

**Key innovation**: Replace simple zlib with learned entropy coding + joint optimization

---

## What's New for Phase 3b

### 1. **`src/rate_distortion_loss.py`** (NEW)
Implements the R-D training objective:

**Core loss function**:
```
L = D + λ * R

where:
  D = distortion (MSE, L1, STFT, or hybrid)
  R = rate = -log p(z) from entropy model
  λ = tradeoff parameter (scheduler: 1.0 -> 0.001)
```

**Key classes**:
- `RateDistortionLoss`: Weighted D+λR computation
  - `compute_distortion()`: Multiple distortion metrics
  - `compute_rate()`: Rate from entropy model
  - `schedule_lambda()`: Linear/exponential/step decay
  
- `QuantizedLatentWithRD`: Quantization with Straight-Through Estimator (STE)
  - Enables gradient flow through discrete quantization
  - Essential for end-to-end training

**Why it works**:
- Early training (λ=1.0): Both D and R matter equally
- Mid training (λ=0.1): Shift focus toward distortion
- Late training (λ=0.001): Focus on maintaining quality while hitting bitrate target

### 2. **`scripts/finetune_rate_distortion.py`** (NEW)
End-to-end fine-tuning pipeline:

1. Load Phase 4 base checkpoint
2. Load pre-trained entropy model (frozen)
3. Create audio dataset with random chunks
4. Apply R-D loss with lambda scheduling
5. Monitor: loss, distortion, rate penalty, lambda

**Features**:
- Streaming dataset (no disk memory limits)
- Lambda scheduling (linear/exponential/step)
- Periodic checkpointing (every 5 epochs + best)
- Training history logging

---

## Step-by-Step Training

### Prerequisite: Entropy Model Ready

Confirm you have:
```
entropy_models/phase4_gmm_8components/best.pt
```
✓ (Already trained in Phase 3a)

### Step 1: Validate Setup (Quick check, 5 min)

```bash
python scripts/finetune_rate_distortion.py \
  --base-checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --entropy-model entropy_models/phase4_gmm_8components/best.pt \
  --output checkpoints_ratedistortion/test_run \
  --epochs 2 \
  --batch-size 8 \
  --samples-per-epoch 500 \
  --device cuda
```

**Expected output**:
```
Epoch 1/2:
  Loss: 0.5234 = D: 0.4892 + λR: 0.0342
  λ: 1.0000, lr: 0.0001
  
Epoch 2/2:
  Loss: 0.4821 = D: 0.4512 + λR: 0.0309
  λ: 0.9655, lr: 0.0001
```

Should complete in ~5 minutes. If it works, proceed to full training.

### Step 2: Full R-D Training (4-8 hours on GPU)

```bash
python scripts/finetune_rate_distortion.py \
  --base-checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --entropy-model entropy_models/phase4_gmm_8components/best.pt \
  --output checkpoints_ratedistortion/phase3b_full \
  --epochs 30 \
  --batch-size 16 \
  --samples-per-epoch 5000 \
  --lambda-start 1.0 \
  --lambda-end 0.001 \
  --lambda-schedule linear \
  --distortion-type hybrid \
  --device cuda 2>&1 | tee rd_training.log
```

**Expected timeline**:
- Epoch 1-10: Loss decreases rapidly (λ high, both D and R)
- Epoch 10-20: Loss plateaus (λ decreases, focus on D)
- Epoch 20-30: Fine-tuning (λ very small, mostly D optimization)

**Expected results after 30 epochs**:
- Distortion: ~0.3-0.5 (down from ~0.6 at start)
- Rate penalty: ~0.01-0.05 nats
- Bitrate: **50-70 kbps** (projected)

### Step 3: Evaluate with Fair Comparison

After fine-tuning, evaluate bitrate and quality:

```bash
python scripts/compare_fair_bitrate.py \
  --checkpoint checkpoints_ratedistortion/phase3b_full/best.pt \
  --target-bitrate-kbps 10 \
  --max-files 20 \
  --device cuda \
  --out results/phase3b_evaluation.csv
```

**Expected metrics**:
```
Neural (Phase 3b):    50-70 kbps  (was 141 kbps at Phase 3a start)
AAC 10 kbps:          10 kbps
Gap remaining:        5-7x (vs 14x at start)
```

---

## Loss Function Breakdown

### Distortion Options

1. **MSE** (Simplest)
   - Fast training
   - Less perceptually aligned
   - Use for initial experiments

2. **L1** (Robust)
   - Robust to outliers
   - Good for sparse distortions

3. **STFT** (Perceptual)
   - Compares magnitude spectrograms
   - More aligned with hearing
   - Slower computation

4. **Hybrid** (Recommended, balanced)
   - 50% L1 (time-domain) + 50% STFT (frequency)
   - Balance between speed and perceptual quality
   - Default choice

### Lambda Scheduling

**Linear decay** (most predictable):
```
λ(t) = 1.0 - 0.999 * (t / T)
     = 1.0 at epoch 0
     = 0.001 at epoch 30
```

**Exponential decay** (smoother):
```
λ(t) = 1.0 * exp(-0.153 * t)
     ≈ 1.0 at epoch 0
     ≈ 0.001 at epoch 30
```

**Step decay** (aggressive):
```
λ(t) = 1.0   if t < 0.5T
     = 0.1   if 0.5T <= t < 0.75T
     = 0.001 if t >= 0.75T
```

---

## Hyperparameter Tuning

### If Loss Doesn't Decrease:
1. **Reduce learning rate**: `--lr 1e-5`
2. **Increase batch size**: `--batch-size 32` (need GPU memory)
3. **Reduce samples per epoch**: `--samples-per-epoch 2000`

### If Bitrate Too High After Training:
1. **Increase λ_start**: `--lambda-start 10.0` (more rate penalty)
2. **Decrease λ_end**: `--lambda-end 0.0001` (strong penalty at end)
3. **Use step schedule**: `--lambda-schedule step` (aggressive)

### If Quality Degrades Too Much:
1. **Decrease λ_start**: `--lambda-start 0.1` (less rate penalty)
2. **Use exponential schedule**: `--lambda-schedule exponential` (slower decay)
3. **Use STFT distortion**: `--distortion-type stft` (perceptual)

---

## Monitoring Training

### Real-time Logs

Watch the training output:
```bash
tail -f rd_training.log
```

Monitor key metrics:
- **Loss**: Should decrease monotonically
- **D**: Distortion (should be ~0.3-0.5)
- **λR**: Rate penalty (should decrease as λ shrinks)
- **λ**: Should decay from 1.0 to 0.001

### Training History

After training, plot loss curve:
```bash
python -c "
import json
import matplotlib.pyplot as plt

with open('checkpoints_ratedistortion/phase3b_full/training_history.json') as f:
    hist = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(hist['epoch'], hist['train_loss']); axes[0, 0].set_ylabel('Loss')
axes[0, 1].plot(hist['epoch'], hist['distortion']); axes[0, 1].set_ylabel('Distortion')
axes[1, 0].plot(hist['epoch'], hist['rate_penalty']); axes[1, 0].set_ylabel('Rate Penalty')
axes[1, 1].semilogy(hist['epoch'], hist['lambda']); axes[1, 1].set_ylabel('Lambda')
plt.tight_layout()
plt.savefig('rd_training_history.png')
print('Saved to rd_training_history.png')
"
```

---

## Expected Progression

### Before Phase 3b (Phase 4 baseline)
- Bitrate: 141 kbps (with 1-bit quantization + zlib)
- PESQ: 1.24
- STOI: 0.851
- Problem: Generic zlib can't exploit latent structure

### After Phase 3a (Entropy model only)
- Projected bitrate: 50-70 kbps (3x improvement)
- Quality unchanged (just better compression)

### After Phase 3b (R-D fine-tuning)
- **Bitrate: 15-25 kbps** (5-7x improvement from Phase 3a!)
- **PESQ: 2.0-2.5** (near AAC quality at 10 kbps)
- **STOI: 0.75-0.85** (trade-off for lower bitrate)
- **Gap to 10 kbps: 1.5-2.5x** (vs 14x at start)

### Phase 3c (If needed: Better quantization)
- Increase from 1-bit to 2-4 bits
- Use Vector Quantization codebook
- Recover PESQ to 2.5-3.0
- Maintain 15-25 kbps target

---

## Files Generated

```
checkpoints_ratedistortion/phase3b_full/
├── best.pt                          (Best model checkpoint)
├── epoch_05.pt, epoch_10.pt, ...    (Periodic checkpoints)
└── training_history.json             (Loss curves)

results/
└── phase3b_evaluation.csv            (Fair bitrate comparison)
```

---

## Troubleshooting

### Error: "Entropy model has wrong latent_dim"
- Ensure entropy model matches Phase 4 latent dimension (3995)
- Check encoder outputs correct shape: (B, 3995, T)

### Error: "CUDA out of memory"
- Reduce batch size: `--batch-size 8`
- Reduce samples per epoch: `--samples-per-epoch 2000`

### Loss goes to NaN
- Reduce learning rate: `--lr 1e-5`
- Clip gradients (already done, but verify in logs)

### Bitrate doesn't improve
- Entropy model not being used (frozen, as expected)
- Lambda too small initially: increase `--lambda-start 10.0`
- Check if entropy model is compatible with encoder output

---

## Next Steps After Phase 3b

### If bitrate is 15-25 kbps:
✓ **Success!** Move to Phase 4: Latency reduction
- Reduce model depth: 6 → 3 layers
- Frame-by-frame processing
- Target: 1,258 ms → 100 ms

### If bitrate is 25-50 kbps:
Try Phase 3c: Better Quantization
- Increase quantization: 1-bit → 2-4 bits
- Use Vector Quantization codebook
- Expected gain: 2-3x bitrate reduction

### If bitrate is still 50+ kbps:
Debug options:
1. Entropy model may not be converged (retrain Phase 3a with more epochs)
2. Lambda schedule too aggressive (reduce decay rate)
3. Base model not compatible with entropy model architecture

---

## Success Criteria

✓ Phase 3b complete when:
1. Training converges (loss stabilizes by epoch 20)
2. Bitrate reduces to 15-50 kbps (verified with compare_fair_bitrate.py)
3. PESQ >= 1.8 (acceptable speech quality)
4. Model checkpoint saved: `best.pt`

→ Ready for Phase 4 (latency reduction) or Phase 3c (if bitrate needs more work)

---

## References

**Rate-Distortion Theory**:
- Shannon (1959): Rate-distortion theory foundations
- Ballé et al. (2018): End-to-end image compression
- Cheng et al. (2020): Learned image compression with discrete GMM

**Your implementation**:
- Global GMM entropy model (Phase 3a)
- Straight-Through Estimator (STE) for quantization gradients
- Linear lambda decay (balance D and R over epochs)
- Hybrid distortion (time + frequency domain)
