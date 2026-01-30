# FFmpeg Support - Compact Installation Commands

## Linux (Ubuntu/Debian)
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswresample-dev && pip install torchcodec audioread soundfile librosa
```

## macOS
```bash
brew install ffmpeg && pip install torchcodec audioread soundfile librosa
```

## Windows (PowerShell)
```powershell
# First download FFmpeg from https://ffmpeg.org/download.html or use chocolatey:
choco install ffmpeg -y
# Then:
pip install torchcodec audioread soundfile librosa
```

## All Packages from requirements.txt
```bash
pip install -r requirements.txt torchcodec audioread
```

## Minimal FFmpeg Support (without torchcodec)
```bash
pip install soundfile librosa audioread
```

## Complete Setup (System + Python)
### Ubuntu/Debian:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswresample-dev && pip install -r requirements.txt torchcodec audioread
```

### macOS:
```bash
brew install ffmpeg && pip install -r requirements.txt torchcodec audioread
```

## Verify Installation
```bash
python -c "import torchaudio; import soundfile; import librosa; print('✅ All audio libraries OK')"
```

## Check FFmpeg
```bash
ffmpeg -version
```
