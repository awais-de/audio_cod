#!/usr/bin/env python3
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

# Create test signal
t = np.linspace(0, 0.02, 320)
audio = (np.sin(2 * np.pi * 440 * t) * 32000).astype(np.int16)

# Encode
x = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
x_tensor = torch.from_numpy(x[np.newaxis, np.newaxis, :]).float()
print(f"Input tensor shape: {x_tensor.shape}, range [{x_tensor.min():.4f}, {x_tensor.max():.4f}]")

with torch.no_grad():
    latent = model.encoder(x_tensor)
    print(f"Latent shape: {latent.shape}, range [{latent.min():.4f}, {latent.max():.4f}]")
    
    # Decode
    audio_float = model.decoder(latent)
    print(f"Decoder output shape: {audio_float.shape}, range [{audio_float.min():.4f}, {audio_float.max():.4f}]")
    
    # Squeeze and denormalize
    audio_np = audio_float.squeeze().cpu().numpy()
    print(f"After squeeze: shape {audio_np.shape}, range [{audio_np.min():.4f}, {audio_np.max():.4f}]")
    
    # Interpolate 280 -> 320
    if len(audio_np) < 320:
        x_old = np.linspace(0, 1, len(audio_np))
        x_new = np.linspace(0, 1, 320)
        audio_np = np.interp(x_new, x_old, audio_np)
    
    print(f"After interp: shape {audio_np.shape}, range [{audio_np.min():.4f}, {audio_np.max():.4f}]")
    
    # Denormalize
    audio_int16 = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
    print(f"After denorm: range [{audio_int16.min()}, {audio_int16.max()}]")
    
    # Check correlation
    corr = np.corrcoef(audio, audio_int16)[0,1]
    print(f"\nCorrelation with input: {corr:.4f}")
    print(f"Mean absolute error: {np.mean(np.abs(audio.astype(float) - audio_int16.astype(float))):.2f}")
