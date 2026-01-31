#!/usr/bin/env python3
"""Verify that the model fails on 320-sample frames but works on 6000-sample segments."""

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
print("TESTING MODEL ON DIFFERENT SEGMENT SIZES")
print("="*60)

# Test 1: 6000 samples (what it was trained on)
print("\n[TEST 1] 6000 SAMPLES (TRAINING SIZE):")
x_6k = torch.randn(1, 1, 6000) * 0.3
with torch.no_grad():
    latent_6k = model.encoder(x_6k)
    output_6k = model.decoder(latent_6k)

print(f"  Input:  shape {x_6k.shape}, range [{x_6k.min():.4f}, {x_6k.max():.4f}]")
print(f"  Latent: shape {latent_6k.shape}, range [{latent_6k.min():.4f}, {latent_6k.max():.4f}]")
print(f"  Output: shape {output_6k.shape}, range [{output_6k.min():.4f}, {output_6k.max():.4f}]")

# Output is 5960, so truncate input to match
x_6k_trunc = x_6k[:, :, :5960]
corr_6k = np.corrcoef(x_6k_trunc.numpy().flatten(), output_6k.cpu().numpy().flatten())[0, 1]
mae_6k = np.mean(np.abs(x_6k_trunc.numpy() - output_6k.cpu().numpy()))
print(f"  Correlation: {corr_6k:.4f}")
print(f"  Mean absolute error: {mae_6k:.6f}")

# Test 2: 320 samples (what we're using now)
print("\n[TEST 2] 320 SAMPLES (REALTIME FRAME SIZE):")
x_320 = torch.randn(1, 1, 320) * 0.3  # Same amplitude
with torch.no_grad():
    latent_320 = model.encoder(x_320)
    output_320 = model.decoder(latent_320)

print(f"  Input:  shape {x_320.shape}, range [{x_320.min():.4f}, {x_320.max():.4f}]")
print(f"  Latent: shape {latent_320.shape}, range [{latent_320.min():.4f}, {latent_320.max():.4f}]")
print(f"  Output: shape {output_320.shape}, range [{output_320.min():.4f}, {output_320.max():.4f}]")

# Output is 280, so truncate input to match
x_320_trunc = x_320[:, :, :280]
corr_320 = np.corrcoef(x_320_trunc.numpy().flatten(), output_320.cpu().numpy().flatten())[0, 1]
mae_320 = np.mean(np.abs(x_320_trunc.numpy() - output_320.cpu().numpy()))
print(f"  Correlation: {corr_320:.4f}")
print(f"  Mean absolute error: {mae_320:.6f}")

# Analysis
print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)
print(f"Output amplitude ratio: {output_6k.abs().max() / output_320.abs().max():.1f}x")
print(f"Correlation difference: {corr_6k - corr_320:.4f}")

if abs(corr_320) < 0.5 and abs(corr_6k) > 0.5:
    print("\n❌ CONFIRMED: Model FAILS on 320-sample frames!")
    print("   Model was trained on 6000-sample segments.")
    print("   Short frames produce near-zero outputs with negative correlation.")
    print("\n   SOLUTION: Buffer frames to 6000 samples before encoding.")
else:
    print("\n✓ Model works on both sizes.")
