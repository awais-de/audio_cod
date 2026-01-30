# Neural Audio Codec - Setup Guide

Comprehensive setup script that handles all dependencies, FFmpeg support, dataset verification, model checkpoint validation, and sanity checks.

## Overview

The setup script (`setup.py`) or wrapper (`setup.sh`) performs 7 automatic checks:

1. **Virtual Environment Creation** - Creates isolated Python environment
2. **Dependency Installation** - Installs all packages from `requirements.txt`
3. **FFmpeg Support** - Detects and configures audio codec libraries
4. **Dataset Verification** - Checks for LibriSpeech train-clean-100 and test-clean
5. **Model Checkpoints** - Validates all 4 trained model checkpoints
6. **Sanity Checks** - Verifies PyTorch, model architecture, GPU/device, and critical imports
7. **Demo Scripts** - Confirms evaluation and demo scripts are available

## Quick Start

### Linux / macOS

```bash
cd /path/to/audio_cod
python3 setup.py
```

Or using the bash wrapper:

```bash
bash setup.sh
```

### Windows

```cmd
cd C:\path\to\audio_cod
python setup.py
```

## Detailed Usage

### What the Script Does

#### Step 1: Virtual Environment
- Creates `venv/` directory if it doesn't exist
- Isolates project dependencies from system Python
- Provides activation instructions for your OS

#### Step 2: Dependencies
- Reads `requirements.txt`
- Installs all Python packages with version constraints
- Includes: PyTorch, Torchaudio, NumPy, SciPy, PESQ, STOI, scikit-learn, etc.
- Automatically upgrades pip first

#### Step 3: FFmpeg Support
- Checks if FFmpeg is installed system-wide
- Provides OS-specific installation instructions if missing:
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt-get install ffmpeg libavformat-dev libavcodec-dev`
  - **Windows**: `choco install ffmpeg`
- Attempts to install Python FFmpeg packages (`torchcodec`, `audioread`)

#### Step 4: Dataset Verification
- Checks for `datasets/LibriSpeech/train-clean-100/`
- Checks for `datasets/LibriSpeech/test-clean/`
- Counts FLAC/WAV files in each
- Provides download commands if datasets are missing

#### Step 5: Model Checkpoints
- Validates all 4 phase checkpoints:
  - Phase 1: Multi-scale Spectral Loss
  - Phase 2: Perceptual Fine-tuning
  - Phase 3: Extended Data + Augmentation
  - Phase 4: Adversarial GAN
- Checks file existence and reports sizes
- Verifies best.pt files in checkpoint directories

#### Step 6: Sanity Checks
- **Imports**: Verifies PyTorch, Torchaudio, NumPy, SciPy, PESQ, STOI, tqdm
- **Model Architecture**: Tests loading NeuralAudioCodec with correct dimensions
- **Compute Device**: Detects available GPU (CUDA/MPS) or CPU fallback

#### Step 7: Demo Scripts
- Confirms presence of key evaluation scripts:
  - `scripts/eval_testclean.py` - Real audio evaluation
  - `scripts/eval_synthetic.py` - Synthetic audio evaluation
  - `scripts/ams_codec.py` - AMS codec implementation
  - `scripts/demo_server.py` - Real-time demo server
  - `scripts/demo_client.py` - Interactive demo client

## Output Format

The script provides colored output for quick scanning:

```
✅ Success      - Feature/check passed
❌ Error       - Critical check failed
⚠️  Warning    - Optional feature unavailable
ℹ️  Info       - Informational message
```

## Final Report

After completing all 7 steps, the script generates:

1. **Summary Table**: Counts of passed/failed checks
2. **Details**: Lists all specific results
3. **Recommendations**: Next steps for incomplete items
4. **Status**: Overall setup completion level

Example:
```
✅ Checks Passed: 17
❌ Checks Failed: 1
⚠️  Warnings: 7

✅ SETUP COMPLETE - Ready for inference and evaluation!
(or)
⚠️  SETUP PARTIAL - Some optional features may be unavailable
```

## Next Steps After Setup

### 1. Activate Virtual Environment

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

### 2. Run Evaluation

**With real test-clean dataset:**
```bash
python scripts/eval_testclean.py
```

**With synthetic audio (no dataset needed):**
```bash
python scripts/eval_synthetic.py
```

### 3. Start Demo Server

**Terminal 1 - Start server:**
```bash
python scripts/demo_server.py
```

**Terminal 2 - Start client:**
```bash
python scripts/demo_client.py
```

### 4. View Documentation

- `FINAL_EVALUATION_REPORT.md` - Comprehensive Phase analysis
- `LOCAL_EVALUATION_GUIDE.md` - Local setup for datasets
- `FFMPEG_INSTALL.md` - System FFmpeg installation
- `README.md` - Project overview

## Troubleshooting

### "Failed to install dependencies"
- Check internet connection
- Ensure Python 3.7+ is installed
- Try manual installation: `pip install -r requirements.txt`

### "FFmpeg not installed"
- Install system FFmpeg (see Step 3 output)
- Then reinstall codec packages: `pip install torchcodec audioread`

### "Models not found"
- Phase 1-4 checkpoints should be in `checkpoints_emergency/`
- Check directory structure with: `find checkpoints_emergency -name "best.pt"`

### "Datasets not found"
- Download from OpenSLR (see Step 4 output)
- Expected location: `datasets/LibriSpeech/{train-clean-100,test-clean}/`

### "PyTorch import failed"
- Ensure venv is activated
- Try: `pip install --upgrade torch torchaudio`

### "GPU not detected"
- Check NVIDIA drivers: `nvidia-smi` (Linux/macOS) or GPU manager (Windows)
- For Apple Silicon: Verify PyTorch-Metal support installed
- Script will fall back to CPU automatically

## File Locations

```
audio_cod/
├── setup.py                    # Main setup script
├── setup.sh                    # Bash wrapper
├── requirements.txt            # Python dependencies
├── venv/                       # Virtual environment (created)
├── src/
│   └── model.py               # NeuralAudioCodec architecture
├── scripts/
│   ├── eval_testclean.py      # Real audio evaluation
│   ├── eval_synthetic.py      # Synthetic audio evaluation
│   ├── demo_server.py         # Demo server
│   └── demo_client.py         # Demo client
├── checkpoints_emergency/      # Model checkpoints
│   ├── phase1_multiscale_*/
│   ├── phase2_perceptual_*/
│   ├── phase3_extended_data_*/
│   └── phase4_adversarial_*/
└── datasets/
    └── LibriSpeech/
        ├── train-clean-100/   # ~28K files
        └── test-clean/        # ~2K files
```

## Command Reference

| Task | Command |
|------|---------|
| Full setup | `python3 setup.py` |
| Activate venv | `source venv/bin/activate` |
| Evaluate on test-clean | `python scripts/eval_testclean.py` |
| Evaluate synthetically | `python scripts/eval_synthetic.py` |
| Start demo server | `python scripts/demo_server.py` |
| Start demo client | `python scripts/demo_client.py` |
| Install FFmpeg (macOS) | `brew install ffmpeg` |
| Install FFmpeg (Linux) | `sudo apt-get install ffmpeg` |

## Python Version Support

Tested on:
- ✅ Python 3.7+
- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11

## OS Support

- ✅ Linux (Ubuntu, Debian, etc.)
- ✅ macOS (Intel and Apple Silicon)
- ✅ Windows 10/11

## Advanced Options

### Skip venv (use system Python)
Edit `setup.py` and comment out `setup_venv()` call

### Custom requirements file
```bash
pip install -r custom_requirements.txt
```

### Manual dependency installation
```bash
pip install torch==2.10.0 torchaudio==2.10.0
pip install -r requirements.txt
```

### Check without installation
```bash
python3 setup.py
```
(The script will report what's missing without installing)

## Support

For issues or questions:
1. Check `setup.py` output for specific failures
2. Review relevant section above
3. Consult `FINAL_EVALUATION_REPORT.md` for model details
4. See `LOCAL_EVALUATION_GUIDE.md` for dataset setup
