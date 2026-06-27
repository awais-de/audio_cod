#!/usr/bin/env python3
"""Quick bitrate check on the current best checkpoint."""
import torch, sys, zlib
import numpy as np
import soundfile as sf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths

ckpt_path = PROJECT_ROOT / 'checkpoints_ratedistortion/bottleneck_v1/best.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
model = NeuralAudioCodec(
    d_model=ckpt['d_model'], n_layers=ckpt['n_layers'], n_heads=ckpt['n_heads'],
    window_size=ckpt['window_size'], bottleneck_dim=ckpt['bottleneck_dim'], dropout=0.0
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Epoch {ckpt['epoch']}, loss={ckpt['train_loss']:.5f}, "
      f"bottleneck={ckpt['bottleneck_dim']}, window={ckpt['window_size']}")

files = sorted(get_dataset_paths()['train_clean_100'].rglob('*.flac'))[:8]
total_bits, total_dur = 0, 0.0
for f in files:
    audio, sr = sf.read(f)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    chunks = []
    for start in range(0, len(audio), 16000):
        chunk = audio[start:start+16000]
        if len(chunk) < 160: continue
        x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            z = model.encode(x).squeeze(0).cpu().numpy()
        chunks.append(z)
    z_np = np.concatenate(chunks, axis=0)
    threshold = (z_np.min() + z_np.max()) / 2
    z_bin = (z_np > threshold).astype(np.uint8)
    compressed = zlib.compress(z_bin.tobytes(), level=9)
    kbps = len(compressed) * 8 / (len(audio)/sr) / 1000
    print(f"  {f.name}: {kbps:.1f} kbps")
    total_bits += len(compressed) * 8
    total_dur += len(audio) / sr

print(f"Mean: {total_bits/total_dur/1000:.1f} kbps  (raw cap: {ckpt['bottleneck_dim']*2:.0f} kbps)")
