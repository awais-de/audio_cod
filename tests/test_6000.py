#!/usr/bin/env python3
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

# Create 6000 samples of 440Hz sine
t = np.linspace(0, 0.375, 6000)
audio_np = (np.sin(2*np.pi*440*t) * 20000).astype(np.int16)

# Encode
x = audio_np.astype(np.float32) / 32768.0
x_tensor = torch.from_numpy(x[np.newaxis, np.newaxis, :]).float()

with torch.no_grad():
    latent = model.encoder(x_tensor)
    output = model.decoder(latent)

# Denormalize with 3.5x scaling
audio_out = output.squeeze().cpu().numpy() * 3.5
audio_out = np.clip(audio_out * 32767.0, -32768, 32767).astype(np.int16)

# Pad to 6000
if len(audio_out) < 6000:
    audio_out = np.pad(audio_out, (0, 6000 - len(audio_out)))

corr = np.corrcoef(audio_np[:len(audio_out)], audio_out)[0,1]
print(f"6000-sample test:")
print(f"  Input:  range [{audio_np.min()}, {audio_np.max()}]")
print(f"  Output: range [{audio_out.min()}, {audio_out.max()}]")
print(f"  Correlation: {corr:.4f}")

if corr > 0.8:
    print("✅ WITH 6000 SAMPLES: Model works great!")
else:
    print("❌ Still broken even with 6000 samples")
