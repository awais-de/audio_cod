#!/usr/bin/env python3
import torch
import torchaudio
from pathlib import Path
from src.model import NeuralAudioCodec
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Load Phase 1 checkpoint (direct state_dict)
print("📦 Loading Phase 1 checkpoint: phase1_multiscale_20260129_124452/best.pt")
model_p1 = NeuralAudioCodec(d_model=384, n_layers=6, n_heads=8).to(device)
checkpoint = torch.load("checkpoints_emergency/phase1_multiscale_20260129_124452/best.pt", map_location=device)
model_p1.load_state_dict(checkpoint)
model_p1.eval()
print("✅ Phase 1 model loaded\n")

# Load V3 baseline (wrapped checkpoint)
print("📦 Loading V3 baseline: pesq_balanced_v3_20260129_094112/best.pt")
model_v3 = NeuralAudioCodec(d_model=384, n_layers=6, n_heads=8).to(device)
checkpoint_v3 = torch.load("checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt", map_location=device)
if 'model_state_dict' in checkpoint_v3:
    model_v3.load_state_dict(checkpoint_v3['model_state_dict'])
else:
    model_v3.load_state_dict(checkpoint_v3)
model_v3.eval()
print("✅ V3 model loaded\n")

# Test on samples
test_audio_dir = Path("datasets")
audio_files = list(test_audio_dir.glob("**/*.flac"))[:10]

print(f"📊 Found {len(audio_files)} test files\n")

snr_scores_p1 = []
snr_scores_v3 = []

for idx, audio_file in enumerate(audio_files, 1):
    try:
        print(f"[{idx}/{len(audio_files)}] {audio_file.name}...", end=" ")
        
        # Load audio
        waveform, sr = torchaudio.load(str(audio_file))
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Process through Phase 1
        with torch.no_grad():
            waveform_gpu = waveform.to(device)
            reconstructed_p1 = model_p1(waveform_gpu)
        
        # Process through V3
        with torch.no_grad():
            reconstructed_v3 = model_v3(waveform_gpu)
        
        # Compute SNR
        def compute_snr(ref, deg):
            ref_power = torch.mean(ref ** 2)
            noise_power = torch.mean((ref - deg) ** 2)
            snr = 10 * torch.log10(ref_power / (noise_power + 1e-10))
            return snr.item()
        
        snr_p1 = compute_snr(waveform_gpu, reconstructed_p1)
        snr_v3 = compute_snr(waveform_gpu, reconstructed_v3)
        
        snr_scores_p1.append(snr_p1)
        snr_scores_v3.append(snr_v3)
        
        print(f"✓ (P1: {snr_p1:.2f}dB, V3: {snr_v3:.2f}dB)")
        
    except Exception as e:
        print(f"✗ {str(e)[:50]}")

if snr_scores_p1:
    print(f"\n{'='*70}")
    print(f"PHASE 1 EVALUATION RESULTS")
    print(f"{'='*70}\n")
    
    avg_snr_p1 = np.mean(snr_scores_p1)
    avg_snr_v3 = np.mean(snr_scores_v3)
    snr_gain = avg_snr_p1 - avg_snr_v3
    snr_gain_pct = (snr_gain / avg_snr_v3 * 100) if avg_snr_v3 != 0 else 0
    
    print(f"📈 Average SNR (Signal-to-Noise Ratio):")
    print(f"   Phase 1: {avg_snr_p1:.3f} dB")
    print(f"   V3:      {avg_snr_v3:.3f} dB")
    print(f"   Δ SNR:   {snr_gain:+.3f} dB ({snr_gain_pct:+.1f}%)")
    
    print(f"\n✅ Phase 1 evaluation complete")
    print(f"\nInterpretation:")
    print(f"  • Positive Δ SNR → Phase 1 is BETTER")
    print(f"  • Negative Δ SNR → V3 is BETTER")
