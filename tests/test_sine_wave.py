#!/usr/bin/env python3
"""Test with real audio signal (sine wave) instead of random noise."""

import torch
import numpy as np
from model import NeuralAudioCodec
import os

# Load model
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

print("="*60)
print("TESTING WITH SINE WAVE (REAL AUDIO)")
print("="*60)

# Test 1: 6000 samples of 440Hz sine wave
print("\n[TEST 1] 6000 SAMPLES OF 440Hz SINE WAVE:")
t_6k = np.linspace(0, 0.375, 6000)  # 0.375 seconds
x_6k = torch.from_numpy(np.sin(2*np.pi*440*t_6k)[np.newaxis, np.newaxis, :] * 0.3).float()

with torch.no_grad():
    latent_6k = model.encoder(x_6k)
    output_6k = model.decoder(latent_6k)

output_6k_np = output_6k.cpu().numpy().flatten()[:5960]
input_6k_np = x_6k.numpy().flatten()[:5960]

print(f"  Input:  range [{input_6k_np.min():.4f}, {input_6k_np.max():.4f}]")
print(f"  Output: range [{output_6k_np.min():.4f}, {output_6k_np.max():.4f}]")
corr_6k = np.corrcoef(input_6k_np, output_6k_np)[0, 1]
print(f"  Correlation: {corr_6k:.4f}")

# Test 2: 320 samples of same sine
print("\n[TEST 2] 320 SAMPLES OF 440Hz SINE WAVE:")
t_320 = np.linspace(0, 0.02, 320)
x_320 = torch.from_numpy(np.sin(2*np.pi*440*t_320)[np.newaxis, np.newaxis, :] * 0.3).float()

with torch.no_grad():
    latent_320 = model.encoder(x_320)
    output_320 = model.decoder(latent_320)

output_320_np = output_320.cpu().numpy().flatten()[:280]
input_320_np = x_320.numpy().flatten()[:280]

print(f"  Input:  range [{input_320_np.min():.4f}, {input_320_np.max():.4f}]")
print(f"  Output: range [{output_320_np.min():.4f}, {output_320_np.max():.4f}]")
corr_320 = np.corrcoef(input_320_np, output_320_np)[0, 1]
print(f"  Correlation: {corr_320:.4f}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
if corr_6k > 0.7 and corr_320 > 0.7:
    print("✓ Model works well on both sizes with real audio")
elif corr_6k > 0.5 and corr_320 < 0.3:
    print("❌ Model FAILS on 320-sample frames!")
    print("   Solution: Buffer to 6000 samples before encoding")
else:
    print(f"? Model quality is poor: 6k={corr_6k:.2f}, 320={corr_320:.2f}")
    print("  This suggests model training issues, not frame size")
