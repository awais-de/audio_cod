# QUICK START GUIDE

## One-Command Setup & Training

```bash
# Make script executable
chmod +x train.sh

# Run everything (setup + sanity check + training)
./train.sh
```

This single command will:
1. ✓ Create virtual environment
2. ✓ Install all dependencies
3. ✓ Run sanity checks
4. ✓ Download dataset if needed
5. ✓ Start training

---

## Manual Setup (if not using train.sh)

### 1. Create & Activate Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR on Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Setup
```bash
python scripts/sanity_check.py
```

### 4. Train Model
```bash
python src/train.py
```

---

## Key Commands

| Command | Purpose |
|---------|---------|
| `python scripts/sanity_check.py` | Verify setup (GPU, packages, config, dataset) |
| `python src/train.py` | Start training |
| `python scripts/inference.py --audio input.wav --output output.wav` | Test model on audio |
| `nvidia-smi` | Check GPU usage during training |
| `tail -f training.log` | Monitor training logs |

---

## Configuration

All hyperparameters are in `config/training.yaml`:

```yaml
batch_size: 32           # Increase to 64 for faster training on large GPUs
learning_rate: 0.0001    # Lower for more stable training
segment_length: 6000     # Increase to 8000 for better quality
num_workers: 4           # Decrease to 0 if out of memory
```

### Common Adjustments

**If Out of Memory:**
```yaml
batch_size: 16
segment_length: 4000
num_workers: 2
```

**For Faster Training (with 50GB+ VRAM):**
```yaml
batch_size: 64
segment_length: 8000
num_workers: 8
```

---

## Expected Performance

| Metric | Value |
|--------|-------|
| GPU Memory | ~8-12GB (for batch_size=32) |
| Time per Epoch | 45 min - 1.5 hours |
| Total Training (100 epochs) | 1.5 - 3 hours |
| Model Size | 12M parameters |

---

## Troubleshooting

### Dataset Not Found
```bash
python scripts/sanity_check.py
# Follow instructions to download
```

### Out of Memory Error
1. Reduce `batch_size` in `config/training.yaml`
2. Reduce `segment_length`
3. Set `num_workers: 0`
4. Restart training

### Slow Training
- Increase `batch_size` if GPU utilization < 70%
- Check `nvidia-smi` for GPU memory usage
- Increase `num_workers` for faster data loading

### CUDA Not Found
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Project Structure

```
audio_cod/
├── src/
│   ├── model.py          ← Model architecture
│   └── train.py          ← Training script
├── config/
│   └── training.yaml     ← Configuration (EDIT THIS)
├── scripts/
│   ├── sanity_check.py   ← Setup verification
│   └── inference.py      ← Run inference
├── checkpoints/          ← Model checkpoints (auto-created)
├── requirements.txt      ← Dependencies
├── train.sh              ← One-command training
└── README.md             ← Full documentation
```

---

## Inference

After training, test the model on audio:

```bash
python scripts/inference.py \
  --audio /path/to/audio.wav \
  --output /path/to/output.wav \
  --checkpoint checkpoints/best_model.pt
```

---

## Tips for Best Results

1. **GPU Optimization**: The code automatically enables:
   - CUDNN benchmark mode
   - TF32 for faster computation
   - Tensor cores (if available)

2. **Data Loading**: Optimized with:
   - Parallel workers (num_workers=4)
   - Pinned memory (pin_memory=True)
   - Prefetching (prefetch_factor=2)

3. **Training**: Features:
   - Gradient clipping (max_norm=1.0)
   - Cosine annealing scheduler
   - NaN/Inf detection and handling
   - Periodic checkpoints every 10 epochs

---

## Next Steps

1. ✓ Run `./train.sh` to start training
2. ✓ Monitor loss curves (should decrease smoothly)
3. ✓ After training completes, run inference to test
4. ✓ Adjust config and retrain if needed

For more details, see [README.md](README.md) and [docs/OPTIMIZATION_GUIDE.md](docs/OPTIMIZATION_GUIDE.md)
