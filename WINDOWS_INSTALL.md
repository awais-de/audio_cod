# Windows Installation Guide

## Quick Start for Windows Users

### Option 1: Install with soundfile (Recommended)

```bash
pip install torch torchaudio soundfile scipy numpy
```

Then run:
```bash
python inference.py
```

### Option 2: Minimal Installation (scipy fallback)

If you encounter issues with `soundfile`, the code will automatically fall back to `scipy` for saving WAV files:

```bash
pip install torch torchaudio scipy numpy
```

## Common Issues on Windows

### Issue 1: torchcodec/FFmpeg Error

**Error message:**
```
RuntimeError: Could not load libtorchcodec. Likely causes:
1. FFmpeg is not properly installed...
```

**Solution:** 
This is now fixed! The updated `inference.py` uses alternative backends that don't require FFmpeg. Just make sure you have one of these installed:
- `soundfile` (recommended): `pip install soundfile`
- `scipy` (fallback): Already included in requirements

### Issue 2: CUDA/GPU Issues

If you get CUDA errors and want to run on CPU only:

```bash
# Install CPU-only PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Or set device to CPU in the code or config:
```python
device = "cpu"
```

### Issue 3: Module Not Found Errors

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

If that fails, install them one by one:

```bash
pip install torch
pip install torchaudio
pip install numpy
pip install scipy
pip install pyyaml
pip install soundfile
```

## Testing the Installation

After installation, test with:

```bash
python inference.py
```

You should see output like:
```
Model created with 20,XXX,XXX parameters
Estimated latency: XX.XX ms
Testing with synthetic audio (1 second)...
Input shape: torch.Size([1, 1, 16000])
Latent shape: torch.Size([1, 2000, 512])
...
Saved: test_original.wav, test_reconstructed.wav
```

## Training on Windows

For training, you'll also need:

```bash
pip install matplotlib tensorboard
```

Update `config.yaml` with your audio dataset paths (use Windows paths):

```yaml
data:
  train_dir: "C:/Users/YourName/audio_data/train"
  val_dir: "C:/Users/YourName/audio_data/val"
```

Then run:
```bash
python train.py
```

## Performance Tips for Windows

1. **Use CUDA if available:**
   - Check: `python -c "import torch; print(torch.cuda.is_available())"`
   - Install CUDA version: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`

2. **Reduce memory usage if needed:**
   - Edit `config.yaml` and reduce `batch_size` to 4 or 2
   - Reduce `segment_length` to 8000 (0.5 seconds)

3. **Speed up training:**
   - Use `num_workers: 0` in config if you get DataLoader errors on Windows
   - Enable GPU if available

## Getting Audio Datasets

Popular speech datasets for training:

1. **LibriSpeech** (English, free):
   - Download: https://www.openslr.org/12
   - Extract to `data/train/` and `data/val/`

2. **VCTK** (Multi-speaker English):
   - Download: https://datashare.ed.ac.uk/handle/10283/3443

3. **Your own recordings:**
   - Record with Windows Voice Recorder
   - Convert to WAV format
   - Place in `data/train/` folder

## Troubleshooting Checklist

- [ ] Python 3.8+ installed
- [ ] PyTorch installed (`import torch` works)
- [ ] At least one audio backend installed (soundfile or scipy)
- [ ] All files from the project downloaded
- [ ] Run `python inference.py` to test

## Need Help?

If you still have issues:

1. Check your Python version: `python --version` (should be 3.8+)
2. Check PyTorch: `python -c "import torch; print(torch.__version__)"`
3. Check audio backend: `python -c "import soundfile; print('soundfile OK')"`
4. Try the scipy fallback: `python -c "import scipy; print('scipy OK')"`

The code now includes automatic fallbacks, so it should work with minimal dependencies!
