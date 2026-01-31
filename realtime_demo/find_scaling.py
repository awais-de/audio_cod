#!/usr/bin/env python3
"""Find the correct output amplitude scaling."""

import torch
import numpy as np
from model import NeuralAudioCodec
import os

checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt')
model = NeuralAudioCodec(
    sample_rate=16000, hop_length=160, d_model=384, n_layers=6, n_heads=8,
    window_size=256, dropout=0.1
)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# Test with different input amplitudes
for input_amp in [0.1, 0.3, 0.5, 0.7, 0.9]:
    t = np.linspace(0, 0.02, 320)
    x = torch.from_numpy(np.sin(2*np.pi*440*t)[np.newaxis, np.newaxis, :] * input_amp).float()
    
    with torch.no_grad():
        latent = model.encoder(x)
        output = model.decoder(latent)
    
    out_amp = output.abs().max().item()
    scale_factor = input_amp / out_amp if out_amp > 0 else 1.0
    
    print(f"Input amplitude: {input_amp:.1f} → Output amplitude: {out_amp:.4f} → Scale factor: {scale_factor:.2f}x")
