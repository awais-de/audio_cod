#!/usr/bin/env python3
"""Simple evaluation on test-clean using existing eval script approach"""

import torch
import torchaudio
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / '..' / 'src'))
from model import NeuralAudioCodec

try:
    from pesq import pesq
    HAS_PESQ = True
except:
    HAS_PESQ = False

# Get first 10 test files
test_path = Path('/home/muaw1874/Desktop/ac_proj/datasets/LibriSpeech/test-clean')
test_files = sorted(list(test_path.glob('**/*.flac')))[:10]
print(f"Testing with {len(test_files)} files\n")

# Load model
print("Loading Phase 1 model...")
model = NeuralAudioCodec(d_model=384, n_layers=6)
state_dict = torch.load(
    '/home/muaw1874/Desktop/ac_proj/audio_cod/checkpoints_emergency/phase1_multiscale_20260129_124452/best.pt',
    map_location='cpu', weights_only=False
)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded\n")

pesq_scores = []
for i, audio_file in enumerate(test_files):
    try:
        # Load audio
        waveform, sr = torchaudio.load(str(audio_file))
        
        # Resample
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # First 1 second
        waveform = waveform[:, :16000]
        
        # Normalize
        waveform_max = torch.abs(waveform).max()
        if waveform_max > 0:
            waveform = waveform / waveform_max
        
        # Reconstruct
        with torch.no_grad():
            recon = model(waveform.unsqueeze(0))
        
        recon_max = torch.abs(recon).max()
        if recon_max > 0:
            recon = recon / recon_max
        
        orig_np = waveform.squeeze().numpy()
        recon_np = recon.squeeze().detach().cpu().numpy()
        
        print(f"File {i+1}: {audio_file.name}")
        print(f"  Original: shape={orig_np.shape}, min={orig_np.min():.4f}, max={orig_np.max():.4f}")
        print(f"  Recon:    shape={recon_np.shape}, min={recon_np.min():.4f}, max={recon_np.max():.4f}")
        
        # PESQ
        if HAS_PESQ:
            try:
                p = pesq(16000, orig_np, recon_np, 'nb')
                pesq_scores.append(p)
                print(f"  PESQ: {p:.4f}")
            except Exception as e:
                print(f"  PESQ error: {type(e).__name__}: {e}")
        else:
            print("  PESQ not available")
        print()
        
    except Exception as e:
        print(f"File {i+1} error: {e}\n")
        continue

if pesq_scores:
    print(f"\nPESQ scores: {[f'{p:.4f}' for p in pesq_scores]}")
    print(f"Mean PESQ: {np.mean(pesq_scores):.4f} ± {np.std(pesq_scores):.4f}")
else:
    print("No PESQ scores computed")
