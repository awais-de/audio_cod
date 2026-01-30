# Local Evaluation Setup Guide

## What You Need to Do

### 1. Download Datasets on Your Local PC

```bash
# Create a datasets directory
mkdir -p ~/audio_codec_data/datasets
cd ~/audio_codec_data/datasets

# Download LibriSpeech train-clean-100
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# Download LibriSpeech test-clean
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz

# Final structure:
# ~/audio_codec_data/datasets/LibriSpeech/
#   ├── train-clean-100/
#   └── test-clean/
```

### 2. Copy Model Checkpoints from Server

Copy these files from the server to your local machine:

```
From server:
/home/muaw1874/Desktop/ac_proj/audio_cod/

Copy to your machine:
~/audio_codec_data/

Directory structure needed:
~/audio_codec_data/
├── src/
│   ├── __init__.py
│   └── model.py
├── scripts/
│   ├── eval_testclean.py
│   └── eval_synthetic.py
├── checkpoints_emergency/
│   ├── phase1_multiscale_20260129_124452/
│   │   └── best.pt
│   ├── phase2_perceptual_20260129_210723/
│   │   └── best.pt
│   ├── phase3_extended_data_20260129_213522/
│   │   └── best.pt
│   └── phase4_adversarial_20260130_063348/
│       └── best.pt
├── datasets/
│   └── LibriSpeech/
│       ├── train-clean-100/
│       └── test-clean/
```

### 3. Install Required Packages on Your Local Machine

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchaudio torchvision
pip install pesq pystoi numpy tqdm

# Optional but recommended for better audio support
pip install librosa soundfile
```

### 4. Run Evaluation Scripts

**For Test-Clean (Real Audio) - BEST OPTION:**

```bash
# Navigate to your local directory
cd ~/audio_codec_data

# Edit the eval_testclean.py script to update paths:
# Change:
#   test_path = Path('/home/muaw1874/Desktop/ac_proj/datasets/LibriSpeech/test-clean')
# To:
#   test_path = Path('./datasets/LibriSpeech/test-clean')

# Run evaluation
python scripts/eval_testclean.py
```

**For Synthetic Audio (No Audio Files Needed):**

```bash
cd ~/audio_codec_data

# Edit eval_synthetic.py to update paths (same as above)

# Run evaluation
python scripts/eval_synthetic.py
```

### 5. What You'll Get

- ✅ Real PESQ scores on test-clean audio (1,970 files)
- ✅ Real STOI scores  
- ✅ Comparison of all 4 phases on actual speech data
- ✅ Proper evaluation without synthetic data

---

## Files to Copy from Server

**Essential:**
1. `src/model.py` - Model architecture
2. `src/__init__.py` - Empty init file
3. `scripts/eval_testclean.py` - Test-clean evaluation script
4. `scripts/eval_synthetic.py` - Synthetic evaluation script
5. All checkpoint `.pt` files from `checkpoints_emergency/`

**Optional but helpful:**
6. `FINAL_EVALUATION_REPORT.md` - Summary of findings
7. `eval_synthetic_results.txt` - Our synthetic evaluation results

---

## Expected Output

When you run locally, you should see:

```
================================================================================
EVALUATION ON LIBRISPEECH TEST-CLEAN
================================================================================
Device: cuda  (or cpu if no GPU)
Sample length: 16000 samples (1.0s at 16000Hz)
Max samples: 50

Found 1970 test-clean audio files

Evaluating Phase 1...
Evaluating: 100%|████████| 50/50
  ✅ PESQ: X.XXXX ± 0.XXXX (50 samples)
  ✅ STOI: X.XXXX ± 0.XXXX (50 samples)

... (same for Phase 2-4)

================================================================================
COMPARISON TABLE
================================================================================
Model                     PESQ                 STOI                
--------
Phase 1                   X.XXXX±X.XXXX        X.XXXX±X.XXXX       
Phase 2                   X.XXXX±X.XXXX        X.XXXX±X.XXXX       
Phase 3                   X.XXXX±X.XXXX        X.XXXX±X.XXXX       
Phase 4                   X.XXXX±X.XXXX        X.XXXX±X.XXXX       
================================================================================
```

---

## Troubleshooting

**If you get "PESQ not installed":**
```bash
pip install pesq --upgrade
```

**If you get "cannot load FLAC file":**
- On Linux: `sudo apt-get install ffmpeg libavformat-dev`
- On macOS: `brew install ffmpeg`
- On Windows: Download from https://ffmpeg.org/download.html

**If you get "CUDA out of memory":**
- Edit scripts to reduce `NUM_SAMPLES` from 50 to 10
- Or use CPU: change `DEVICE = 'cpu'` in scripts

---

## Quick Copy Commands

**From your server terminal (on local machine with SSH access):**

```bash
# Copy the entire project
scp -r muaw1874@<server-ip>:/home/muaw1874/Desktop/ac_proj/audio_cod ~/audio_codec_data

# Or copy specific directories
scp -r muaw1874@<server-ip>:/home/muaw1874/Desktop/ac_proj/audio_cod/src ~/audio_codec_data/
scp -r muaw1874@<server-ip>:/home/muaw1874/Desktop/ac_proj/audio_cod/scripts ~/audio_codec_data/
scp -r muaw1874@<server-ip>:/home/muaw1874/Desktop/ac_proj/audio_cod/checkpoints_emergency ~/audio_codec_data/
```

---

This approach will give you:
✅ **Real test-clean evaluation** with 1,970 files  
✅ **Proper FFmpeg support** on your local machine  
✅ **Complete metrics** (PESQ/STOI) without workarounds  
✅ **Best possible environment** for model testing  

Let me know when you've done this and I can help you verify the results!
