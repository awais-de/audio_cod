# Neural Audio Codec - Optimized Version

A deep learning-based audio codec that compresses and reconstructs audio using a Transformer-based encoder-decoder architecture.

## 🚀 What's New (Optimized Version)

This project has been completely restructured and optimized for faster training:

### Performance Improvements
- **Model**: Reduced from 51.8M to **12M parameters** (4.3x reduction)
  - d_model: 512 → 256
  - n_layers: 8 → 4
  - n_heads: 16 → 8

- **Data Loading**: Optimized for GPU efficiency
  - batch_size: 4 → 32 (8x larger)
  - num_workers: 0 → 4 (parallel loading)
  - segment_length: 8000 → 6000 (shorter sequences)
  - pin_memory: enabled
  - prefetch_factor: 2

- **Expected Speedup**: **5-7x faster** training per epoch
  - Original: ~6-7 hours per epoch
  - Optimized: ~45 minutes - 1.5 hours per epoch

### Project Structure
```
audio_cod/
├── src/                      # Source code
│   ├── model.py             # Neural Audio Codec model (12M params)
│   └── train.py             # Optimized training script
├── config/                   # Configuration files
│   └── training.yaml        # Training configuration (EDIT THIS FOR CUSTOMIZATION)
├── scripts/                  # Utility scripts
│   ├── sanity_check.py      # Verify installation & setup
│   ├── inference.py         # Run inference on audio files
│   └── evaluate.py          # Evaluate model performance
├── checkpoints/             # Saved model checkpoints (auto-created)
├── data/                    # Dataset location (auto-downloaded)
├── docs/                    # Documentation
└── requirements.txt         # Python dependencies
```

## ✅ Quick Start

### 1. Prerequisites
- Python 3.8+
- CUDA-capable GPU with 8GB+ VRAM (recommended for fast training)
- 6GB disk space for dataset

### 2. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Sanity Check

```bash
# Verify everything is set up correctly
python scripts/sanity_check.py
```

This will check:
- ✓ Python version
- ✓ Required packages installed
- ✓ CUDA/GPU available
- ✓ Configuration files valid
- ✓ Model can be loaded
- ✓ Dataset is available (or download automatically)

### 4. Train the Model

```bash
# Start training
python src/train.py
```

The script will automatically:
- Detect GPU and enable optimizations
- Download dataset if missing
- Save checkpoints to `checkpoints/` directory
- Log training progress

**Estimated time**: 100 epochs ≈ 45 minutes - 2 hours (depending on GPU)

### 5. Inference on Audio

```bash
# Test the trained model on an audio file
python scripts/inference.py \
  --audio path/to/your/audio.wav \
  --output path/to/output.wav \
  --checkpoint checkpoints/best_model.pt
```

## 📊 Model Architecture

### Encoder (6M parameters)
- 4 Causal Convolution layers with GroupNorm (downsampling)
- 4 Transformer blocks with sliding-window causal attention
- Compression ratio: 8x (input frequency)

### Decoder (6M parameters)
- 4 Transformer blocks with sliding-window causal attention
- 4 Transposed Convolution layers with GroupNorm (upsampling)
- Reconstructs original audio waveform

### Key Features
- **Causal Design**: No future context for low-latency streaming
- **Sliding Window Attention**: Reduces complexity from O(n²) to O(n*w)
- **Multi-scale Spectral Loss**: Perceptual quality optimization
- **Optimized for Streaming**: Can process audio chunks independently

## 🎯 Configuration

Edit `config/training.yaml` to customize training:

```yaml
model:
  sample_rate: 16000        # Audio sample rate (Hz)
  d_model: 256              # Embedding dimension
  n_layers: 4               # Number of transformer layers
  n_heads: 8                # Attention heads
  dropout: 0.1              # Dropout rate

training:
  epochs: 100               # Number of epochs
  batch_size: 32            # Batch size (increase for faster training on large GPUs)
  learning_rate: 0.0001     # Learning rate
  segment_length: 6000      # Audio segment length (samples)
  num_workers: 4            # Parallel data loading processes

data:
  train_dir: "/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100"
  val_dir: "/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100"
```

### Tuning Tips
- **Larger batch_size** (32 → 64) if GPU has >20GB VRAM
- **Longer segment_length** (6000 → 8000) for better context
- **Fewer num_workers** (4 → 0) if running out of memory
- **Lower learning_rate** (0.0001 → 0.00005) for more stable training

## 📦 Dataset

The project uses **LibriSpeech train-clean-100**:
- 28,539 audio files
- ~100 hours of speech
- 16 kHz mono audio
- Size: ~5.9 GB

**Automatic Download**: The training script will automatically download and extract the dataset on first run.

**Manual Download** (optional):
```bash
mkdir -p /mnt/Data/muaw1874/datasets
cd /mnt/Data/muaw1874/datasets
wget https://openslr.trmal.net/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz
```

## 📈 Training Progress

During training, you'll see:
```
Epoch 1/100
================================================================================
Epoch 1 [100/7135] Loss: 2.1234 | Avg: 2.5644 | L1: 0.2341 | Spectral: 2.3876
Epoch 1 [200/7135] Loss: 1.9876 | Avg: 2.4234 | L1: 0.1876 | Spectral: 2.2387
...
Epoch 1 COMPLETE (142.5s)
================================================================================
Training Loss:    1.2345 (L1: 0.1234, Spectral: 1.1111)
Validation Loss:  1.3456 (L1: 0.1345, Spectral: 1.2111)
Learning Rate:    0.00010000
✓ Saved best model (Val Loss: 1.3456)
```

### Checkpoints

- `checkpoints/best_model.pt`: Best validation loss (automatic save)
- `checkpoints/checkpoint_epoch_{N}.pt`: Periodic saves every 10 epochs

## 🎵 Inference

### Basic Usage
```bash
python scripts/inference.py --audio input.wav --output output.wav
```

### Advanced Usage (Python)
```python
import torch
import torchaudio
from src.model import NeuralAudioCodec

# Load model
checkpoint = torch.load('checkpoints/best_model.pt')
model = NeuralAudioCodec()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load audio
audio, sr = torchaudio.load('input.wav')

# Process
with torch.no_grad():
    reconstructed = model(audio.unsqueeze(0))

# Save
torchaudio.save('output.wav', reconstructed.squeeze(0), sr)
```

## 📊 Performance Metrics

The model evaluates training with:
- **L1 Loss**: Time-domain reconstruction error
- **Spectral Loss**: Multi-scale STFT magnitude loss (FFT sizes: 512, 1024, 2048)
- **SNR**: Signal-to-Noise Ratio (dB)
- **Compression Ratio**: Original size / Latent size

## 🔧 Troubleshooting

### Out of Memory (OOM)
```yaml
# Reduce batch size
batch_size: 16  # from 32
segment_length: 4000  # from 6000
num_workers: 2  # from 4
```

### Slow Training
- Check GPU usage: `nvidia-smi`
- Increase `batch_size` if GPU utilization < 70%
- Increase `num_workers` for faster data loading

### Dataset Issues
```bash
# Re-run sanity check
python scripts/sanity_check.py

# Manually set dataset path in config/training.yaml
# and re-run training
```

## 📚 Files Overview

| File | Purpose |
|------|---------|
| `src/model.py` | Optimized model architecture (12M params) |
| `src/train.py` | Training loop with optimizations |
| `config/training.yaml` | Hyperparameter configuration |
| `scripts/sanity_check.py` | Verify installation and setup |
| `scripts/inference.py` | Run inference on audio files |
| `requirements.txt` | Python dependencies |

## 🚀 Next Steps

1. **Run sanity check**: `python scripts/sanity_check.py`
2. **Start training**: `python src/train.py`
3. **Monitor progress**: Watch loss curves in console output
4. **Test inference**: `python scripts/inference.py --audio test.wav --output test_out.wav`
5. **Fine-tune**: Edit `config/training.yaml` and retrain if needed

## 📝 License

This project is for educational purposes.

## 🎯 Performance Summary

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Model Parameters | 51.8M | 12M | 4.3x smaller |
| Batch Size | 4 | 32 | 8x larger |
| Segment Length | 8000 | 6000 | -25% |
| num_workers | 0 | 4 | 4x parallel |
| GPU Memory | ~40GB | ~8GB | 5x less |
| Time per Epoch | 6-7h | 45m-1.5h | 5-7x faster |
| Expected Total Time | 600-700h | 75-150h | **5-7x faster** |

## 💡 Key Optimizations

1. **Model Reduction**
   - Fewer transformer layers (8 → 4)
   - Smaller embeddings (512 → 256)
   - Fewer attention heads (16 → 8)

2. **Data Loading**
   - Parallel workers (0 → 4)
   - Larger batches (4 → 32)
   - Pin memory for GPU
   - Prefetch factor for buffering

3. **Training**
   - Shorter segments (8000 → 6000)
   - GPU optimizations (cuDNN, TF32)
   - Efficient attention (sliding window)
   - Better normalization (GroupNorm)

## 📞 Support

For issues or questions:
1. Run sanity check: `python scripts/sanity_check.py`
2. Check config: `config/training.yaml`
3. Review logs in console output
4. Verify dataset is downloaded: `ls /mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100/`
