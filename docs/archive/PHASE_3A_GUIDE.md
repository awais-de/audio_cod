# Phase 3a: Learned Entropy Model - Implementation Guide

## Executive Summary

**Problem**: Neural codec achieves **141 kbps** at 1-bit quantization (target: 10 kbps).  
**Root cause**: zlib compression doesn't understand the structure of neural latent distributions.  
**Solution**: Train a learned entropy model (Gaussian Mixture Model) to replace zlib.  
**Expected gain**: 2-3x bitrate reduction → **50-70 kbps** (3-5x closer to target).

---

## What's Been Created

### 1. **`src/entropy_model.py`** (NEW)
Learnable entropy models for latent distribution:
- **`GaussianMixtureModel`**: Global mixture of Gaussians (simplest, proven)
- **`ConditionalGMM`**: Factorized per-dimension mixture (more expressive)
- **`EntropyBottleneck`**: Wrapper for rate-distortion training

**Key methods**:
- `log_prob(z)`: Compute log probability of latent vector
- `entropy_loss(z, beta)`: Training loss for learning distribution
- `compute_rate_in_bits(z)`: Calculate compression rate

**Why**: Learned model can exploit latent structure that generic zlib misses.

### 2. **`scripts/train_entropy_model.py`** (NEW)
Self-contained training pipeline:
1. Load trained neural codec checkpoint
2. Extract latents from dataset (frozen encoder)
3. Train entropy model to fit latent distribution
4. Save best checkpoint for Phase 3b integration

**Usage**:
```bash
python scripts/train_entropy_model.py \
  --checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --output entropy_models/phase4_gmm_8components \
  --epochs 20 \
  --batch-size 128 \
  --model-type global_gmm \
  --num-components 8
```

### 3. **`scripts/diagnose_latent_distribution.py`** (NEW)
Diagnostic tool to:
- Extract latents from codec
- Analyze their statistics (mean, std, entropy)
- Estimate compression gains from learned entropy
- Compare: zlib vs Gaussian vs GMM

**Usage**:
```bash
python scripts/diagnose_latent_distribution.py \
  --checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --max-files 20
```

---

## Step-by-Step Implementation Plan

### Phase 3a.1: Validate Current Baseline (1 hour)

**Goal**: Establish reference metrics before training anything.

```bash
# 1. Diagnose latent distribution
python scripts/diagnose_latent_distribution.py \
  --checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --max-files 20

# Expected output:
# - Current bitrate: ~141 kbps
# - Estimated Gaussian entropy: ~80 kbps
# - Estimated GMM entropy: ~50-60 kbps
# - Gap to 10 kbps: 5-14x
```

This tells you:
- ✓ Latent distribution structure
- ✓ Realistic gain from entropy modeling
- ✓ Whether GMM will bring you to Phase 3b threshold

### Phase 3a.2: Train Entropy Model (2-4 hours on GPU)

**Goal**: Learn p(z) from latent distribution.

```bash
# 2. Train entropy model on latents
python scripts/train_entropy_model.py \
  --checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --output entropy_models/phase4_gmm_8components \
  --max-files 100 \
  --chunk-sec 2.0 \
  --epochs 20 \
  --batch-size 128 \
  --model-type global_gmm \
  --num-components 8 \
  --device cuda
```

**Monitoring**:
- Loss should decrease over epochs (see `entropy_models/phase4_gmm_8components/training_history.json`)
- Lower loss = better latent distribution fit
- Check mean loss: ~0.1-0.5 nats is typical for well-fitting GMM

**Output**:
- `entropy_models/phase4_gmm_8components/best.pt`: Trained entropy model checkpoint
  - Contains: `entropy_model_state_dict`, `latent_dim`, `num_components`
  - Ready for Phase 3b integration

### Phase 3a.3: Integration Preview (Optional)

Once entropy model is trained, you can preview bitrate reduction:

```python
# Load entropy model
entropy_ckpt = torch.load('entropy_models/phase4_gmm_8components/best.pt')
entropy_model = EntropyBottleneck(...)
entropy_model.load_state_dict(entropy_ckpt['entropy_model_state_dict'])

# On latent z:
rate_bits = entropy_model.compute_rate_in_bits(z)
estimated_bitrate = rate_bits.mean().item() * frame_rate
print(f"Estimated bitrate with learned entropy: {estimated_bitrate:.1f} kbps")
```

---

## Why This Works: The Math

### Current Method (zlib + 1-bit quantization)
```
Latent z (float32) 
  → Scalar quantize to 1-bit
  → zlib compress (generic entropy coding)
  → Loss: ~0.5-1.0 bits per latent value (zlib doesn't know latent distribution)
  
Actual bitrate = num_latents * 0.5 / duration_seconds ≈ 141 kbps
```

### New Method (learned entropy model)
```
Latent z (float32)
  → Scalar quantize to 1-bit
  → Learn p(z) from data using GMM
  → Arithmetic coding with learned model
  → Loss: ~0.1-0.3 bits per latent value (optimal for that distribution)
  
Estimated bitrate = num_latents * 0.15 / duration_seconds ≈ 50 kbps
```

**Key insight**: Entropy coding rate = -log₂(p(z)). If GMM fits latents well, probabilities are higher, entropy is lower.

---

## Expected Results & Timeline

### During Training
- Initial loss: ~1.0-1.5 nats
- Final loss: ~0.2-0.5 nats (good fit)
- Training should take: 2-4 hours on GPU (20 epochs)

### After Integration (Phase 3b)
Expected metrics with learned entropy model:

| Metric | Current (zlib) | After GMM | Target |
|--------|---|---|---|
| Bitrate (1-bit quant) | 141 kbps | 50-70 kbps | 10 kbps |
| PESQ | 1.24 | ~1.5-1.8 | 2.0 |
| STOI | 0.851 | ~0.85 | 0.75 |

The entropy model alone **won't** reach 10 kbps (still need rate-distortion training), but it:
- ✓ Proves learned entropy works
- ✓ Reduces bitrate 2-3x (biggest quick win)
- ✓ Enables Phase 3b to target bitrate effectively

---

## Troubleshooting

### Issue: "No latents extracted"
- **Cause**: Dataset path issue
- **Fix**: Check dataset exists, use `--max-files 1` with known file path

### Issue: Loss doesn't decrease
- **Cause**: Learning rate too high or model capacity too low
- **Fix**: Reduce `--lr 1e-4`, increase `--num-components 16`

### Issue: Model trains but bitrate unchanged
- **Cause**: Not integrated into compression pipeline yet
- **Fix**: Phase 3a only trains the model. Phase 3b (finetune_rate_distortion.py) uses it.

---

## Next Steps After Phase 3a

Once entropy model is trained and you confirm bitrate reduction:

1. **Phase 3b: Rate-Distortion Training**
   - Use entropy model in loss function
   - Add quantization layer with gradient flow
   - Fine-tune encoder/decoder jointly
   - Target: 50 kbps → 15-25 kbps

2. **Phase 3c: Increase Quantization** (if needed)
   - 1-bit → 2-4 bit quantization
   - Recover PESQ quality from 1.24 → 2.5+

3. **Phase 4: Latency Reduction**
   - Parallel to above (independent)
   - Reduce model depth, use causal attention
   - Target: 1,258 ms → ~100 ms

---

## References

**Key papers on learned entropy coding**:
- Ballé et al. (2018): "End-to-end optimized image compression"
  - Introduced variational entropy bottleneck
  - Used entropy models to replace naive compression

- Cheng et al. (2020): "Learned image compression with discretized Gaussian mixture"
  - Practical GMM-based entropy models
  - Achieved near-optimal rate-distortion

**Your implementation**:
- Global GMM: Simple, effective, proven
- Factorized GMM: More expressive if needed later
- Both compatible with arithmetic coding libraries (e.g., `torchac`)

---

## Files Ready to Use

```
src/
├── entropy_model.py          ← Entropy models (NEW)
├── quantization.py           ← Existing quantization
└── model.py                  ← Existing neural codec

scripts/
├── train_entropy_model.py           ← Training script (NEW)
├── diagnose_latent_distribution.py  ← Diagnostic tool (NEW)
├── compare_fair_bitrate.py          ← Evaluation (existing)
└── finetune_rate_distortion.py      ← Phase 3b (not yet created)

entropy_models/
└── (will be created after training)
```

---

## Estimated GPU Time

On V100 GPU (16GB):
- Latent extraction: 30-60 min (50-100 files)
- Entropy model training: 2-4 hours (20 epochs, 128 batch)
- **Total Phase 3a: 3-5 hours**

On CPU (not recommended):
- Training will be 10-20x slower
- Consider only if testing locally

---

## Success Criteria

✓ Phase 3a complete when:
1. Entropy model converges (loss < 0.5 nats)
2. Bitrate reduces to 50-70 kbps (verified with diagnostic)
3. Model checkpoint saved and ready for Phase 3b

→ Move to Phase 3b to reach 10-25 kbps target.
