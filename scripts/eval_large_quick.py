#!/usr/bin/env python3
"""Quick eval for large model checkpoint"""
import sys
import os
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
import random
from pesq import pesq
from pystoi import stoi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import NeuralAudioCodec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = Path('checkpoints_large/best_large_model.pt')

# Create large model
model = NeuralAudioCodec(d_model=512, n_layers=8, n_heads=8, window_size=512).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load test files
data_root = Path('/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100')
files = list(data_root.rglob('*.flac'))
random.shuffle(files)
files = files[:20]

pesq_scores = []
stoi_scores = []

with torch.no_grad():
    for i, f in enumerate(files, 1):
        audio, sr = sf.read(str(f))
        if audio.ndim > 1:
            audio = audio[:, 0]
        
        # Use 4s segment
        seg_len = 64000
        if len(audio) < seg_len:
            audio = np.pad(audio, (0, seg_len - len(audio)))
        start = random.randint(0, max(1, len(audio) - seg_len))
        segment = audio[start:start + seg_len]
        
        x = torch.from_numpy(segment).float().unsqueeze(0).unsqueeze(0).to(device)
        recon = model(x)
        
        min_len = min(x.shape[-1], recon.shape[-1])
        orig = x[..., :min_len].squeeze().cpu().numpy()
        rec = recon[..., :min_len].squeeze().cpu().numpy()
        
        try:
            p = pesq(16000, orig, rec, 'wb')
            pesq_scores.append(p)
        except:
            pass
        try:
            s = stoi(orig, rec, 16000, extended=False)
            stoi_scores.append(s)
        except:
            pass
        
        if i % 5 == 0:
            print(f"[{i}/20] PESQ={p:.3f} STOI={s:.3f}")

mean_pesq = float(np.mean(pesq_scores)) if pesq_scores else 0
mean_stoi = float(np.mean(stoi_scores)) if stoi_scores else 0

print(f"\nFinal: PESQ={mean_pesq:.3f}, STOI={mean_stoi:.3f}")

Path('checkpoints_large/eval_large_best.txt').write_text(
    f"checkpoint={ckpt_path}\nfiles=20\nPESQ={mean_pesq:.3f}\nSTOI={mean_stoi:.3f}\n"
)
