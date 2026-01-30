#!/usr/bin/env python3
import torch
from src.model import NeuralAudioCodec
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Load Phase 1
print("📦 Loading Phase 1: phase1_multiscale_20260129_124452/best.pt")
model_p1 = NeuralAudioCodec(d_model=384, n_layers=6, n_heads=8).to(device)
ckpt_p1 = torch.load("checkpoints_emergency/phase1_multiscale_20260129_124452/best.pt", map_location=device)
model_p1.load_state_dict(ckpt_p1)
model_p1.eval()
print("✅ Loaded\n")

# Load V3
print("📦 Loading V3: pesq_balanced_v3_20260129_094112/best.pt")
model_v3 = NeuralAudioCodec(d_model=384, n_layers=6, n_heads=8).to(device)
ckpt_v3 = torch.load("checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt", map_location=device)
if 'model_state_dict' in ckpt_v3:
    model_v3.load_state_dict(ckpt_v3['model_state_dict'])
else:
    model_v3.load_state_dict(ckpt_v3)
model_v3.eval()
print("✅ Loaded\n")

# Test with random waveforms
print("📊 Testing with random waveforms (batch_size=1, seq_len=6000)...\n")

snr_p1_list = []
snr_v3_list = []

for i in range(10):
    # Random input (1 batch, 6000 samples)
    x = torch.randn(1, 1, 6000).to(device)
    
    with torch.no_grad():
        y_p1 = model_p1(x)
        y_v3 = model_v3(x)
    
    # Compute SNR (reconstruction error)
    snr_p1 = 10 * torch.log10(torch.mean(x ** 2) / (torch.mean((x - y_p1) ** 2) + 1e-10)).item()
    snr_v3 = 10 * torch.log10(torch.mean(x ** 2) / (torch.mean((x - y_v3) ** 2) + 1e-10)).item()
    
    snr_p1_list.append(snr_p1)
    snr_v3_list.append(snr_v3)
    
    print(f"[{i+1}/10] P1 SNR: {snr_p1:7.2f} dB  |  V3 SNR: {snr_v3:7.2f} dB  |  Δ: {snr_p1-snr_v3:+6.2f} dB")

print(f"\n{'='*70}")
print(f"EVALUATION SUMMARY")
print(f"{'='*70}\n")

avg_snr_p1 = np.mean(snr_p1_list)
avg_snr_v3 = np.mean(snr_v3_list)
delta = avg_snr_p1 - avg_snr_v3
delta_pct = (delta / avg_snr_v3 * 100) if avg_snr_v3 != 0 else 0

print(f"Phase 1 Average SNR: {avg_snr_p1:.3f} dB")
print(f"V3 Average SNR:      {avg_snr_v3:.3f} dB")
print(f"Δ SNR:               {delta:+.3f} dB ({delta_pct:+.1f}%)")

if delta > 0:
    print(f"\n✅ Phase 1 is BETTER (higher SNR = better reconstruction)")
else:
    print(f"\n⚠️ V3 is better (Phase 1 SNR is lower)")

print(f"\nNote: SNR measured on random test signals")
print(f"Higher SNR = better reconstruction fidelity")
