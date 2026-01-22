# GPU Setup Guide for Neural Audio Codec

## Quick Check: Is GPU Working?

Run this command to check if PyTorch detects your GPU:

```bash
python check_gpu.py
```

This will show:
- ✅ If GPU is detected and working
- GPU name, memory, and specifications
- Performance comparison (GPU vs CPU)
- Recommended training settings for your GPU

## If GPU is NOT Detected

### Windows Users

1. **Check if you have an NVIDIA GPU:**
   - Open Task Manager (Ctrl+Shift+Esc)
   - Go to "Performance" tab
   - Look for "GPU 0" or "GPU 1" - should say NVIDIA

2. **Install NVIDIA Drivers** (if not already installed):
   - Visit: https://www.nvidia.com/download/index.aspx
   - Select your GPU model
   - Download and install latest drivers

3. **Reinstall PyTorch with CUDA support:**

```bash
# Uninstall CPU-only version
pip uninstall torch torchaudio

# Install CUDA version (for CUDA 12.1)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. **Verify installation:**

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Should output: `CUDA Available: True`

### Linux Users

1. **Install NVIDIA drivers:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535

# Check driver installation
nvidia-smi
```

2. **Install PyTorch with CUDA:**
```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### macOS Users

**Note:** macOS doesn't support NVIDIA GPUs. For Apple Silicon (M1/M2/M3):

```bash
# Install PyTorch with MPS support
pip install torch torchaudio

# The code will automatically use MPS backend if available
```

## GPU Memory Requirements

| GPU Memory | Recommended Settings | Notes |
|------------|---------------------|-------|
| 24+ GB | batch_size=32, full model | Best for large datasets |
| 16 GB | batch_size=16, full model | Good for most use cases |
| 12 GB | batch_size=8, full model | Standard training |
| 8 GB | batch_size=4-8 | May need to reduce segment_length |
| 6 GB | batch_size=2-4, smaller model | Consider d_model=256 |
| 4 GB | batch_size=1-2, small model | Limited, use d_model=256, n_layers=4 |

## Optimizing for Your GPU

Edit `config.yaml` based on your GPU:

```yaml
# For 24GB+ GPU
training:
  batch_size: 32
  
model:
  d_model: 512
  n_layers: 8

# For 8GB GPU
training:
  batch_size: 4
  segment_length: 8000  # Reduce from 16000
  
model:
  d_model: 512
  n_layers: 8

# For 6GB GPU
training:
  batch_size: 2
  segment_length: 8000
  
model:
  d_model: 256  # Smaller model
  n_layers: 6
```

## GPU Acceleration Tips

The training scripts automatically enable:
- ✅ cuDNN benchmark mode (`torch.backends.cudnn.benchmark = True`)
- ✅ TF32 mode for faster computation
- ✅ Automatic mixed precision (future enhancement)

## Monitoring GPU During Training

### Windows

Use Task Manager:
1. Open Task Manager (Ctrl+Shift+Esc)
2. Go to "Performance" tab
3. Select your GPU
4. Watch GPU utilization, memory usage

Or use:
```bash
nvidia-smi -l 1
```

### Linux

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

## Common GPU Issues

### Issue 1: "CUDA out of memory"

**Solution:**
- Reduce `batch_size` in config.yaml
- Reduce `segment_length` 
- Use smaller model (d_model=256, n_layers=6)

### Issue 2: GPU not being used (CPU training instead)

**Solution:**
1. Run `python check_gpu.py`
2. Reinstall PyTorch with CUDA support
3. Check that PyTorch CUDA version matches your drivers

### Issue 3: Slow GPU training

**Possible causes:**
- Batch size too small (increase if you have memory)
- Data loading bottleneck (increase num_workers)
- Model too small to benefit from GPU

### Issue 4: "CUDA driver version insufficient"

**Solution:**
- Update NVIDIA drivers to latest version
- Or install older PyTorch version matching your drivers

## Performance Expectations

With GPU acceleration:

| GPU | Training Speed (per epoch) | Total Time (100 epochs) |
|-----|---------------------------|-------------------------|
| RTX 4090 | 2-3 min | 3-5 hours |
| RTX 3090 | 3-5 min | 5-8 hours |
| RTX 3080 | 4-6 min | 7-10 hours |
| RTX 3060 | 6-10 min | 10-17 hours |
| GTX 1080 Ti | 8-12 min | 13-20 hours |

Without GPU (CPU only): **1-2 weeks** for 100 epochs ⚠️

## Multi-GPU Training (Future)

For multiple GPUs, you can use DataParallel:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

This is not yet implemented but can be added to `train.py`.

## Troubleshooting Commands

```bash
# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check if GPU is visible
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check GPU memory
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1e9)"

# Run full diagnostic
python check_gpu.py
```

## Next Steps

1. ✅ Run `python check_gpu.py` to verify GPU is working
2. ✅ Adjust batch_size in `config.yaml` based on GPU memory
3. ✅ Run `python train_quick.py` to test GPU training (5-10 min)
4. ✅ Monitor GPU usage during training
5. ✅ Start full training with `python train.py`

With GPU acceleration, training will be **50-100x faster** than CPU! 🚀
