import numpy as np
import torch
from model import NeuralAudioCodec

model = NeuralAudioCodec(sample_rate=16000, hop_length=160, d_model=384, n_layers=6, n_heads=8, window_size=256, dropout=0.1)
checkpoint = torch.load('best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict'])
model.eval()

# Test sine wave
t = np.linspace(0, 0.02, 320)
audio_int16 = (np.sin(2 * np.pi * 440 * t) * 32000).astype(np.int16)

# OPTION 1: Normalized (current implementation)
audio_normalized = audio_int16.astype(np.float32) / 32768.0
x1 = torch.from_numpy(audio_normalized[np.newaxis, np.newaxis, :])
with torch.no_grad():
    out1 = model(x1)
out1_np = out1.squeeze().numpy()
corr1 = np.corrcoef(audio_normalized[:len(out1_np)], out1_np)[0,1]

print(f"WITH normalization (/32768):")
print(f"  Input range: [{audio_normalized.min():.3f}, {audio_normalized.max():.3f}]")
print(f"  Output range: [{out1.min():.3f}, {out1.max():.3f}]")
print(f"  Correlation: {corr1:.4f}")

# OPTION 2: Direct conversion (like evaluation script - soundfile returns float already)
audio_direct = audio_int16.astype(np.float32)
x2 = torch.from_numpy(audio_direct[np.newaxis, np.newaxis, :])
with torch.no_grad():
    out2 = model(x2)
out2_np = out2.squeeze().numpy()
corr2 = np.corrcoef(audio_direct[:len(out2_np)], out2_np)[0,1]

print(f"\nWITHOUT normalization (direct int16->float):")
print(f"  Input range: [{audio_direct.min():.1f}, {audio_direct.max():.1f}]")
print(f"  Output range: [{out2.min():.1f}, {out2.max():.1f}]")
print(f"  Correlation: {corr2:.4f}")

print(f"\n{'='*60}")
if corr2 > corr1:
    print(f"✅ FOUND IT! Model was trained WITHOUT normalization!")
    print(f"   Better correlation: {corr2:.4f} vs {corr1:.4f}")
else:
    print(f"Model expects normalized input")
