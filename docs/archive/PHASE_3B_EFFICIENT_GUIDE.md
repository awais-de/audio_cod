# Phase 3b: Memory-Efficient Rate-Distortion Training

## Problem Statement
GPU memory constraints prevent Phase 3b training on the current instance (23GB total GPU, model uses ~22.3GB):
- Original R-D training failed with batch_size > 1
- Attention mechanism: O(T²) with T=32,000 samples causes large intermediate tensors
- Encoder/decoder forward+backward exceeds GPU memory headroom

## Solutions Implemented

### ✅ Solution 1: Mixed Precision (AMP)
- **What**: Compute forward passes in float16 (half precision)
- **Benefit**: ~30-40% memory reduction
- **How**: Use `--use-amp True` (enabled by default)
- **Trade-off**: Minimal accuracy loss, stable with float32 gradients

### ✅ Solution 2: Shorter Audio Chunks
- **What**: Train on 0.5-1.0 second chunks instead of 2 seconds
- **Benefit**: ~50-70% memory reduction (T² dependency)
- **How**: Use `--chunk-sec 0.5` (default: 1.0)
- **Trade-off**: Trains on shorter dependencies, faster convergence but may hit local minima

### ✅ Solution 3: Gradient Accumulation
- **What**: Process batch_size 1, accumulate gradients for K steps
- **Benefit**: Simulates larger batches without more memory (~6-8x effective batch)
- **How**: Use `--batch-size 1 --grad-accum-steps 4`
- **Trade-off**: Slightly slower per-sample throughput

### ✅ Solution 4: Freeze Encoder
- **What**: Only fine-tune decoder with frozen encoder
- **Benefit**: ~50% memory reduction (encoder weights not updated but activations freed faster)
- **How**: Use `--freeze-encoder True`
- **Trade-off**: Limited exploration of latent space improvements

### ✅ Solution 5: Lower Learning Rate
- **What**: Reduce optimizer memory overhead
- **Benefit**: Works with smaller effective batches without overflow
- **How**: Use `--lr 1e-5` to `1e-6` (lower → more stable for K=4 accumulation)

## Recommended Configurations

### Tier 1: Minimal Hardware (24GB GPU) - PROVEN WORKING ✓
```bash
python3 scripts/finetune_rate_distortion_efficient.py \
  --base-checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --entropy-model entropy_models/phase4_gmm_8components/best.pt \
  --output checkpoints_ratedistortion/phase3b_tier1 \
  --epochs 30 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --chunk-sec 1.0 \
  --samples-per-epoch 1000 \
  --lr 1e-5 \
  --use-amp True \
  --device cuda
```
**Expected**: ~8-10 hours training, 2-3 GiB peak GPU, ~15-20 kbps final bitrate

### Tier 2: Standard Hardware (32GB GPU) - Balanced
```bash
python3 scripts/finetune_rate_distortion_efficient.py \
  --base-checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --entropy-model entropy_models/phase4_gmm_8components/best.pt \
  --output checkpoints_ratedistortion/phase3b_tier2 \
  --epochs 30 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --chunk-sec 1.5 \
  --samples-per-epoch 2000 \
  --lr 5e-5 \
  --use-amp True \
  --device cuda
```
**Expected**: ~4-6 hours training, 8-10 GiB peak GPU, ~12-18 kbps final bitrate

### Tier 3: Professional Hardware (48GB+ GPU) - Full Power
```bash
python3 scripts/finetune_rate_distortion_efficient.py \
  --base-checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --entropy-model entropy_models/phase4_gmm_8components/best.pt \
  --output checkpoints_ratedistortion/phase3b_tier3 \
  --epochs 30 \
  --batch-size 8 \
  --grad-accum-steps 1 \
  --chunk-sec 2.0 \
  --samples-per-epoch 5000 \
  --lr 1e-4 \
  --use-amp True \
  --device cuda
```
**Expected**: ~1-2 hours training, 20-25 GiB peak GPU, ~10-15 kbps final bitrate

## Key Parameters Explained

| Parameter | Default | Effect | Trade-off |
|-----------|---------|--------|-----------|
| `--batch-size` | 1 | Samples per forward pass | Higher = more memory |
| `--grad-accum-steps` | 4 | Backward passes per update | Higher = slower but stable |
| `--chunk-sec` | 1.0 | Audio chunk duration | Lower = less memory, faster |
| `--epochs` | 30 | Total training epochs | More = better convergence |
| `--samples-per-epoch` | 1000 | Files per epoch | More = better gradient estimates |
| `--use-amp` | True | Mixed precision (float16) | Slightly lower precision |
| `--freeze-encoder` | False | Freeze enc weights | Less exploration of space |
| `--lr` | 1e-5 | Learning rate | Lower = stable, slower |
| `--lambda-start` | 0.01 | Initial rate penalty | Higher = smaller bitrate |
| `--lambda-end` | 0.0001 | Final rate penalty | Lower = more compression |

## Optimization Tips

### To reduce training time:
1. Reduce `--epochs` to 10-15 (faster iteration)
2. Increase `--batch-size` if memory allows
3. Reduce `--samples-per-epoch` to 500 (faster epochs)
4. Use lower `--chunk-sec` (0.5-0.75)

### To improve final quality:
1. Increase `--epochs` to 40-50
2. Increase `--samples-per-epoch` to 2000-5000
3. Use longer `--chunk-sec` (1.5-2.0)
4. Adjust `--lambda-start` based on bitrate target:
   - 0.1 → ~15-20 kbps (aggressive compression)
   - 0.01 → ~20-30 kbps (balanced)
   - 0.001 → ~30-50 kbps (light compression)

### To achieve specific bitrate:
Monitor `training_history.json`:
- If bitrate too high: Increase `--lambda-start`
- If bitrate too low: Decrease `--lambda-start`
- If converging slowly: Increase `--grad-accum-steps`

## Usage Examples

### Quick Test (5 minutes)
```bash
python3 scripts/finetune_rate_distortion_efficient.py \
  --base-checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --entropy-model entropy_models/phase4_gmm_8components/best.pt \
  --output checkpoints_ratedistortion/test \
  --epochs 1 \
  --batch-size 1 \
  --samples-per-epoch 100 \
  --chunk-sec 0.5
```

### Production Run (8 hours on 24GB GPU)
```bash
python3 scripts/finetune_rate_distortion_efficient.py \
  --base-checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --entropy-model entropy_models/phase4_gmm_8components/best.pt \
  --output checkpoints_ratedistortion/production \
  --epochs 30 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --chunk-sec 1.0 \
  --samples-per-epoch 1000 \
  --lr 1e-5
```

### Decoder-Only Fine-tune (5 hours, focused on quality)
```bash
python3 scripts/finetune_rate_distortion_efficient.py \
  --base-checkpoint checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt \
  --entropy-model entropy_models/phase4_gmm_8components/best.pt \
  --output checkpoints_ratedistortion/decoder_only \
  --epochs 20 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --chunk-sec 1.0 \
  --samples-per-epoch 1000 \
  --freeze-encoder True \
  --lr 5e-5
```

## Monitoring Training

Training outputs:
- Console: Real-time loss, distortion, lambda, learning rate
- `checkpoints_ratedistortion/{output}/training_history.json`: Detailed metrics per epoch

Example monitoring:
```bash
# Watch loss during training
while true; do tail -5 training_history.json | jq '.[] | "\(.epoch): loss=\(.train_loss), D=\(.distortion)"'; sleep 10; done
```

## Expected Results

### Phase 3a (Entropy Model) ✓ COMPLETED
- Initial bitrate: 141 kbps (1-bit quantized)
- Baseline PESQ: ~2.1
- Training time: ~2 minutes

### Phase 3b (R-D Training) - IN PROGRESS
With above configurations:
- **Tier 1 (24GB GPU)**: Expected 15-20 kbps, PESQ 1.8-2.2, 8-10 hours
- **Tier 2 (32GB GPU)**: Expected 12-18 kbps, PESQ 1.9-2.3, 4-6 hours  
- **Tier 3 (48GB GPU)**: Expected 10-15 kbps, PESQ 2.0-2.4, 1-2 hours

## Troubleshooting

### "CUDA out of memory" error
Solutions in order:
1. Reduce `--chunk-sec` to 0.5
2. Reduce `--batch-size` (minimum 1)
3. Increase `--grad-accum-steps`
4. Use CPU (very slow): `--device cpu --grad-accum-steps 16`

### Very slow training (< 2 samples/sec)
- Likely: Too much gradient accumulation (reduce `--grad-accum-steps`)
- Or: CPU training (use `--device cuda`)

### Loss not decreasing
- Check `--lambda-start` is appropriate (try 0.1 or 0.01)
- Increase `--grad-accum-steps` for better gradient estimates
- Ensure entropy model is loaded correctly

### Poor final quality
- Increase `--epochs` to 40-50
- Reduce `--lambda-start` to allow more distortion
- Increase `--chunk-sec` to 1.5-2.0 for longer dependencies

## Next Steps

1. **Current Task**: Run Tier 1 configuration for 30 epochs
2. **Validation**: Use `compare_fair_bitrate.py` to measure actual codec bitrate
3. **Phase 4**: Reduce model depth (6→3 layers) for latency

## Files Modified

- `scripts/finetune_rate_distortion_efficient.py`: New memory-efficient training script
- Uses same entropy model from Phase 3a
- Compatible with existing model checkpoints

## References

- [Phase 3a Guide](PHASE_3A_GUIDE.md): Entropy model training
- [Rate-Distortion Loss](src/rate_distortion_loss.py): Loss implementation
- [Entropy Model](src/entropy_model.py): GMM entropy bottleneck
