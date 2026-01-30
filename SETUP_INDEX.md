# Setup System - Complete Index

## 📦 What Was Created

### Core Setup Files

| File | Size | Type | Purpose |
|------|------|------|---------|
| [setup.py](setup.py) | 18 KB | Python | **Main setup script** - Run this for complete setup |
| [setup.sh](setup.sh) | 858 B | Bash | Wrapper for setup.py (optional) |

### Documentation Files

| File | Size | Type | Purpose |
|------|------|------|---------|
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | 7.7 KB | Markdown | Detailed guide with all explanations |
| [SETUP_QUICK_REF.md](SETUP_QUICK_REF.md) | 4.1 KB | Markdown | Quick reference card for common tasks |
| [SETUP_SUMMARY.txt](SETUP_SUMMARY.txt) | 7.0 KB | Text | Overview of what the setup does |
| [SETUP_INDEX.md](SETUP_INDEX.md) | - | Markdown | This file - Navigation guide |

---

## 🚀 Quick Start (Choose One)

### Option 1: Python Script (Recommended)
```bash
python3 setup.py
```

### Option 2: Bash Wrapper
```bash
bash setup.sh
```

### Option 3: Windows
```cmd
python setup.py
```

---

## 📋 The 7 Setup Steps

When you run `setup.py`, it performs these steps automatically:

1. **Virtual Environment** → Creates isolated Python environment
2. **Dependencies** → Installs all packages from requirements.txt
3. **FFmpeg Support** → Checks/installs audio codec libraries
4. **Dataset Check** → Verifies LibriSpeech datasets
5. **Model Validation** → Confirms all 4 phase checkpoints exist
6. **Sanity Tests** → Verifies PyTorch, model loading, GPU detection
7. **Scripts Check** → Confirms demo/eval scripts are available

---

## 📖 Which Document to Read?

| Need | Read This |
|------|-----------|
| **Just run it!** | [SETUP_QUICK_REF.md](SETUP_QUICK_REF.md) (2 min read) |
| **Complete details** | [SETUP_GUIDE.md](SETUP_GUIDE.md) (10 min read) |
| **Quick overview** | [SETUP_SUMMARY.txt](SETUP_SUMMARY.txt) (5 min read) |
| **Troubleshooting** | [SETUP_GUIDE.md](SETUP_GUIDE.md#troubleshooting) section |
| **Command reference** | [SETUP_QUICK_REF.md](SETUP_QUICK_REF.md#-quick-commands) |
| **FFmpeg help** | [FFMPEG_INSTALL.md](../FFMPEG_INSTALL.md) |
| **Dataset setup** | [LOCAL_EVALUATION_GUIDE.md](../LOCAL_EVALUATION_GUIDE.md) |

---

## ✅ What Gets Checked

### Environment
- [ ] Python version (3.7+)
- [ ] Virtual environment (created automatically)
- [ ] Operating system detection

### Dependencies
- [ ] PyTorch & Torchaudio
- [ ] NumPy, SciPy
- [ ] Audio metrics (PESQ, STOI)
- [ ] Utility libraries (tqdm, scikit-learn, etc.)

### System Components
- [ ] FFmpeg system library
- [ ] Compute device (GPU/CPU/MPS)
- [ ] Python codec packages (torchcodec, audioread)

### Project Assets
- [ ] Phase 1 checkpoint (Multi-scale Spectral)
- [ ] Phase 2 checkpoint (Perceptual)
- [ ] Phase 3 checkpoint (Extended Data)
- [ ] Phase 4 checkpoint (Adversarial)
- [ ] LibriSpeech train-clean-100 dataset
- [ ] LibriSpeech test-clean dataset

### Code & Scripts
- [ ] Model architecture (src/model.py)
- [ ] Evaluation scripts (eval_*.py)
- [ ] Demo server/client (demo_*.py)
- [ ] AMS codec (ams_codec.py)

---

## 🎯 After Setup Completes

### If Status is ✅ SETUP COMPLETE
You're ready to go! Next steps:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run evaluation (choose one)
python scripts/eval_synthetic.py      # No dataset needed
python scripts/eval_testclean.py      # Needs test-clean

# 3. Start demo (in 2 terminals)
python scripts/demo_server.py         # Terminal 1
python scripts/demo_client.py         # Terminal 2
```

### If Status is ⚠️ SETUP PARTIAL
Some optional features unavailable, but core features work:

1. Check the warnings section of setup output
2. Most common issue: FFmpeg not installed
3. Fix: Follow FFmpeg installation commands in output
4. You can still run `eval_synthetic.py` without FFmpeg

---

## 🔧 Custom Setup Options

### Skip Virtual Environment
```bash
# Edit setup.py and comment out setup_venv() call
# Then reinstall packages: pip install -r requirements.txt
```

### Install Additional Packages
```bash
source venv/bin/activate
pip install <package-name>
```

### Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 setup.py  # Just runs checks
```

### Update Requirements
```bash
pip install --upgrade -r requirements.txt
```

---

## 📊 Output Legend

The setup script uses these symbols:

| Symbol | Meaning |
|--------|---------|
| ✅ | Check passed / Success |
| ❌ | Check failed / Error (may need fix) |
| ⚠️  | Warning / Optional feature missing |
| ℹ️  | Information / Status message |

---

## 🆘 Troubleshooting Quick Links

| Problem | Solution |
|---------|----------|
| venv creation failed | Check Python installation: `python3 --version` |
| Package installation fails | Check internet, try: `pip install --upgrade pip` |
| FFmpeg not found | See [FFMPEG_INSTALL.md](../FFMPEG_INSTALL.md) |
| Models not found | Run: `find checkpoints_emergency -name "best.pt"` |
| Datasets not found | Download from [LOCAL_EVALUATION_GUIDE.md](../LOCAL_EVALUATION_GUIDE.md) |
| GPU not detected | Check NVIDIA drivers: `nvidia-smi` |
| Import errors | Activate venv: `source venv/bin/activate` |

---

## 📁 Project Structure After Setup

```
audio_cod/
├── setup.py                       ← Run this
├── setup.sh                       ← Or this
├── SETUP_GUIDE.md                 ← Read for details
├── SETUP_QUICK_REF.md             ← Quick commands
├── SETUP_SUMMARY.txt              ← Overview
├── SETUP_INDEX.md                 ← This file
│
├── venv/                          ← Created by setup.py
│   ├── bin/python                 ← Python executable
│   ├── lib/python3.x/site-packages/  ← Installed packages
│
├── src/
│   └── model.py                   ← Model architecture
│
├── scripts/
│   ├── eval_synthetic.py          ← Evaluation (no dataset)
│   ├── eval_testclean.py          ← Evaluation (real audio)
│   ├── demo_server.py             ← Demo server
│   └── demo_client.py             ← Demo client
│
├── checkpoints_emergency/
│   ├── phase1_multiscale_*/
│   ├── phase2_perceptual_*/
│   ├── phase3_extended_data_*/
│   └── phase4_adversarial_*/
│
└── datasets/                      ← Download here if needed
    └── LibriSpeech/
        ├── train-clean-100/
        └── test-clean/
```

---

## 💡 Pro Tips

1. **First time?** Read [SETUP_QUICK_REF.md](SETUP_QUICK_REF.md)
2. **Need details?** Read [SETUP_GUIDE.md](SETUP_GUIDE.md)
3. **Run it multiple times** - It's safe and idempotent
4. **Check GPU before starting** - See CUDA version in setup output
5. **Use eval_synthetic.py first** - No dataset download needed
6. **Activate venv before each session** - Required for all Python commands

---

## 🔗 Related Documentation

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Comprehensive setup guide
- [SETUP_QUICK_REF.md](SETUP_QUICK_REF.md) - Command reference card
- [SETUP_SUMMARY.txt](SETUP_SUMMARY.txt) - Feature overview
- [FFMPEG_INSTALL.md](../FFMPEG_INSTALL.md) - Audio codec setup
- [LOCAL_EVALUATION_GUIDE.md](../LOCAL_EVALUATION_GUIDE.md) - Dataset/model setup
- [FINAL_EVALUATION_REPORT.md](../FINAL_EVALUATION_REPORT.md) - Model analysis
- [README.md](../README.md) - Project overview

---

## ✨ Features

✅ **One Command** - Single setup command does everything  
✅ **Cross-Platform** - Works on Linux, macOS, Windows  
✅ **Automatic Detection** - Finds OS and requirements  
✅ **Comprehensive Checks** - 7-step verification  
✅ **Smart Errors** - Clear feedback on what failed  
✅ **Zero Config** - No manual configuration needed  
✅ **Idempotent** - Safe to run multiple times  
✅ **Educational** - Explains what each step does  

---

## 📞 Support

1. **Check setup.py output** - Usually has the answer
2. **Read relevant section** of [SETUP_GUIDE.md](SETUP_GUIDE.md)
3. **Search quick ref** in [SETUP_QUICK_REF.md](SETUP_QUICK_REF.md)
4. **Check troubleshooting** above
5. **Review project docs** linked above

---

**Status:** ✅ Ready to Use  
**Created:** January 30, 2026  
**Python Version:** 3.7+ required  
**OS Support:** Linux, macOS, Windows
