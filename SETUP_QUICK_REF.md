# Quick Setup Reference Card

## 🚀 One-Liner Setup

```bash
python3 setup.py
```

That's it! The script handles everything.

---

## 📋 What Gets Checked

| Check | Details | Status |
|-------|---------|--------|
| **venv** | Virtual environment creation | Auto-created |
| **Dependencies** | PyTorch, audio libraries, metrics | Auto-installed |
| **FFmpeg** | System audio codec library | ⚠️ May need manual install |
| **Models** | All 4 phase checkpoints | ✅ Should exist |
| **Datasets** | LibriSpeech train-clean-100 & test-clean | ⚠️ Download if needed |
| **PyTorch** | Core framework import | ✅ Auto-tested |
| **PESQ/STOI** | Audio quality metrics | ✅ Auto-tested |
| **GPU/Device** | CUDA/MPS/CPU detection | ✅ Auto-detected |

---

## ⚡ Quick Commands

### Setup & Activation
```bash
python3 setup.py              # Run full setup
source venv/bin/activate      # Activate (Linux/macOS)
venv\Scripts\activate         # Activate (Windows)
```

### Run Evaluation
```bash
python scripts/eval_synthetic.py     # No dataset needed ✅
python scripts/eval_testclean.py     # Needs test-clean ⚠️
```

### Start Demo
```bash
python scripts/demo_server.py         # Terminal 1
python scripts/demo_client.py         # Terminal 2
```

---

## 🔧 Install FFmpeg (If Needed)

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libavformat-dev libavcodec-dev
```

**Windows:**
```cmd
choco install ffmpeg -y
```

---

## 📥 Download Datasets (If Needed)

```bash
cd datasets
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf train-clean-100.tar.gz
tar -xzf test-clean.tar.gz
```

---

## ✅ Success Indicators

✅ **Setup Complete when you see:**
- "✅ PyTorch imported successfully"
- "✅ Model architecture loads correctly"  
- "Device: CUDA: NVIDIA RTX A5000" (or your GPU)
- "✅ All 4 demo scripts found"

⚠️ **Warnings are OK:**
- "FFmpeg not installed" → Just run demo without real audio
- "test-clean: Not found" → Use eval_synthetic.py instead
- "torchvision installation failed" → Not needed for audio codec

---

## 🎯 Typical Workflow

```
1. python3 setup.py
   ↓
2. source venv/bin/activate
   ↓
3. python scripts/eval_synthetic.py
   ↓
4. python scripts/demo_server.py & python scripts/demo_client.py
```

---

## 📊 Output Legend

```
✅ = Check passed / Installation successful
❌ = Check failed / Critical error (fix needed)
⚠️  = Warning / Optional feature unavailable
ℹ️  = Information / Status message
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "venv not found" | Run: `python3 setup.py` again |
| "pip: command not found" | Activate venv first: `source venv/bin/activate` |
| "ModuleNotFoundError: torch" | Reinstall: `pip install torch torchaudio` |
| "No module named 'model'" | Check: `cd audio_cod` and try again |
| "FFmpeg not found" | Install from commands above |
| "Dataset not found" | Download using commands above |

---

## 📁 Expected Directory Structure

```
audio_cod/
├── setup.py
├── venv/                    # Created by setup.py
├── src/model.py
├── scripts/eval_*.py
├── checkpoints_emergency/   # Phase 1-4 models
└── datasets/LibriSpeech/    # Download if needed
    ├── train-clean-100/
    └── test-clean/
```

---

## 💡 Pro Tips

- Keep setup.py output for reference (takes 3-5 minutes)
- Activate venv before running any scripts
- Use `eval_synthetic.py` first (no dependencies)
- Check GPU with: `nvidia-smi` (Linux) or Activity Monitor (macOS)
- For Windows, use PowerShell or Git Bash (not cmd.exe)

---

## 📖 Full Documentation

- `SETUP_GUIDE.md` - Detailed setup guide (this file)
- `FINAL_EVALUATION_REPORT.md` - Model performance analysis
- `LOCAL_EVALUATION_GUIDE.md` - Local evaluation setup
- `FFMPEG_INSTALL.md` - FFmpeg installation details

---

**Last Updated:** Jan 30, 2026  
**Status:** ✅ Ready for Production
