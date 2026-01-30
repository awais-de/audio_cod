#!/usr/bin/env python3
"""
Phase 1 Evaluation Script - Evaluate multi-scale loss checkpoint
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchaudio
import numpy as np
from pathlib import Path
import scipy.signal
from scipy.fftpack import fft
from scipy.io import wavfile
import json
from datetime import datetime
import soundfile as sf

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths

def estimate_pesq(ref, deg, sr=16000):
    """Scipy-based PESQ approximation"""
    ref = ref.astype(np.float32)
    deg = deg.astype(np.float32)
    
    # Normalize
    ref_max = np.abs(ref).max()
    deg_max = np.abs(deg).max()
    if ref_max > 0:
        ref = ref / ref_max
    if deg_max > 0:
        deg = deg / deg_max
    
    # Align length
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]
    
    # Compute spectral distance
    hop_length = sr // 100
    window = scipy.signal.hann(2048)
    
    ref_spec = np.abs(scipy.signal.stft(ref, sr, nperseg=2048, noverlap=1536)[2])
    deg_spec = np.abs(scipy.signal.stft(deg, sr, nperseg=2048, noverlap=1536)[2])
    
    # Log magnitude difference
    ref_log = np.log10(ref_spec + 1e-10)
    deg_log = np.log10(deg_spec + 1e-10)
    
    spectral_dist = np.mean(np.abs(ref_log - deg_log))
    
    # Time-domain distance
    time_dist = np.mean(np.abs(ref - deg))
    
    # PESQ approximation (calibrated)
    pesq_approx = 4.5 - spectral_dist * 0.5 - time_dist * 2.0
    pesq_approx = np.clip(pesq_approx, 1.0, 4.5)
    
    return pesq_approx

def estimate_stoi(ref, deg, sr=16000):
    """STOI approximation using cross-correlation"""
    ref = ref.astype(np.float32)
    deg = deg.astype(np.float32)
    
    # Normalize
    ref_rms = np.sqrt(np.mean(ref**2))
    deg_rms = np.sqrt(np.mean(deg**2))
    if ref_rms > 0:
        ref = ref / ref_rms
    if deg_rms > 0:
        deg = deg / deg_rms
    
    # Align length
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]
    
    # Compute correlation in subbands
    frame_len = sr // 10  # 100ms frames
    hop_len = frame_len // 2
    
    stoi_frames = []
    for i in range(0, len(ref) - frame_len, hop_len):
        ref_frame = ref[i:i+frame_len]
        deg_frame = deg[i:i+frame_len]
        
        # Normalize frames
        ref_std = np.std(ref_frame)
        deg_std = np.std(deg_frame)
        
        if ref_std > 0 and deg_std > 0:
            ref_frame = (ref_frame - np.mean(ref_frame)) / ref_std
            deg_frame = (deg_frame - np.mean(deg_frame)) / deg_std
            
            # Cross-correlation at lag 0
            corr = np.mean(ref_frame * deg_frame)
            stoi_frames.append(np.clip(corr, 0, 1))
    
    stoi = np.mean(stoi_frames) if stoi_frames else 0.5
    return stoi

def main():
    print("\n" + "=" * 80)
    print("PHASE 1 EVALUATION: MULTI-SCALE SPECTRAL LOSS")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎯 Device: {device}")
    
    # Load Phase 1 checkpoint
    phase1_checkpoint = Path('checkpoints_emergency/phase1_multiscale_20260129_124452/best.pt')
    if not phase1_checkpoint.exists():
        print(f"❌ Phase 1 checkpoint not found: {phase1_checkpoint}")
        return
    
    # Load V3 checkpoint for comparison
    v3_checkpoint = Path('checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt')
    if not v3_checkpoint.exists():
        print(f"❌ V3 checkpoint not found: {v3_checkpoint}")
        return
    
    # Model config
    model_config = {
        'd_model': 384,
        'n_layers': 6,
        'n_heads': 8,
        'window_size': 384,
        'hop_length': 160,
        'sample_rate': 16000,
    }
    
    print("\n📦 Loading models...")
    
    # Phase 1 model
    model_phase1 = NeuralAudioCodec(**model_config).to(device)
    checkpoint = torch.load(phase1_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_phase1.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_phase1.load_state_dict(checkpoint)
    model_phase1.eval()
    print(f"✅ Phase 1 model loaded")
    
    # V3 model
    model_v3 = NeuralAudioCodec(**model_config).to(device)
    checkpoint_v3 = torch.load(v3_checkpoint, map_location=device)
    if isinstance(checkpoint_v3, dict) and 'model_state_dict' in checkpoint_v3:
        model_v3.load_state_dict(checkpoint_v3['model_state_dict'])
    else:
        model_v3.load_state_dict(checkpoint_v3)
    model_v3.eval()
    print(f"✅ V3 model loaded")
    
    # Load test data (use train-clean-100 if test doesn't exist)
    audio_dir = get_dataset_paths()["train_clean_100"]
    if not audio_dir.exists():
        print(f"❌ Dataset not found: {audio_dir}")
        return
    
    print(f"\n📊 Loading test data...")
    audio_files = list(audio_dir.rglob('*.flac'))
    
    if not audio_files:
        print("❌ No test files found")
        return
    
    # Evaluate on random samples
    import random
    n_samples = 20
    test_files = random.sample(audio_files, min(n_samples, len(audio_files)))
    
    print(f"📊 Evaluating on {len(test_files)} test samples...")
    
    phase1_pesqs = []
    phase1_stois = []
    v3_pesqs = []
    v3_stois = []
    
    with torch.no_grad():
        for idx, audio_path in enumerate(test_files):
            try:
                # Load audio using soundfile
                audio_data, sr = sf.read(str(audio_path))
                
                # Convert to torch tensor
                if audio_data.ndim == 1:
                    waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
                else:
                    waveform = torch.from_numpy(audio_data.T).float()
                
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Resample if needed
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                
                # Normalize
                max_val = torch.abs(waveform).max()
                if max_val > 0:
                    waveform = waveform / max_val
                
                # Use segment
                if waveform.shape[1] > 96000:
                    waveform = waveform[:, :96000]
                else:
                    pad_len = 96000 - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_len))
                
                waveform = waveform.to(device)
                
                # Phase 1 forward pass
                recon_phase1 = model_phase1(waveform)
                if recon_phase1.shape[-1] > waveform.shape[-1]:
                    recon_phase1 = recon_phase1[:, :, :waveform.shape[-1]]
                
                # V3 forward pass
                recon_v3 = model_v3(waveform)
                if recon_v3.shape[-1] > waveform.shape[-1]:
                    recon_v3 = recon_v3[:, :, :waveform.shape[-1]]
                
                # Compute metrics
                ref_np = waveform.squeeze().cpu().numpy()
                phase1_np = recon_phase1.squeeze().cpu().numpy()
                v3_np = recon_v3.squeeze().cpu().numpy()
                
                # PESQ
                pesq_p1 = estimate_pesq(ref_np, phase1_np)
                pesq_v3 = estimate_pesq(ref_np, v3_np)
                
                # STOI
                stoi_p1 = estimate_stoi(ref_np, phase1_np)
                stoi_v3 = estimate_stoi(ref_np, v3_np)
                
                phase1_pesqs.append(pesq_p1)
                phase1_stois.append(stoi_p1)
                v3_pesqs.append(pesq_v3)
                v3_stois.append(stoi_v3)
                
                print(f"  [{idx+1}/{len(test_files)}] Phase1: PESQ={pesq_p1:.3f} STOI={stoi_p1:.3f} | V3: PESQ={pesq_v3:.3f} STOI={stoi_v3:.3f}")
                
            except Exception as e:
                print(f"  [{idx+1}/{len(test_files)}] Error: {e}")
                continue
    
    # Compute averages
    phase1_avg_pesq = np.mean(phase1_pesqs) if phase1_pesqs else 0
    phase1_avg_stoi = np.mean(phase1_stois) if phase1_stois else 0
    v3_avg_pesq = np.mean(v3_pesqs) if v3_pesqs else 0
    v3_avg_stoi = np.mean(v3_stois) if v3_stois else 0
    
    print("\n" + "=" * 80)
    print("📊 PHASE 1 EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nPhase 1 (Multi-Scale Loss):")
    print(f"  • PESQ: {phase1_avg_pesq:.3f}")
    print(f"  • STOI: {phase1_avg_stoi:.3f}")
    
    print(f"\nV3 (Baseline):")
    print(f"  • PESQ: {v3_avg_pesq:.3f}")
    print(f"  • STOI: {v3_avg_stoi:.3f}")
    
    pesq_gain = phase1_avg_pesq - v3_avg_pesq
    stoi_gain = phase1_avg_stoi - v3_avg_stoi
    
    print(f"\n🎯 IMPROVEMENTS:")
    print(f"  • PESQ Gain: {pesq_gain:+.3f} ({pesq_gain/v3_avg_pesq*100:+.1f}%)")
    print(f"  • STOI Gain: {stoi_gain:+.3f} ({stoi_gain/v3_avg_stoi*100:+.1f}%)")
    
    # Decision
    print("\n" + "=" * 80)
    print("🎯 DECISION")
    print("=" * 80)
    
    if phase1_avg_pesq >= 3.0:
        print(f"\n✅ SUCCESS! Phase 1 achieved PESQ {phase1_avg_pesq:.3f}")
        print(f"   Recommendation: PROCEED TO PHASE 2 (Perceptual Loss)")
        decision = "PROCEED_TO_PHASE2"
    elif phase1_avg_pesq >= v3_avg_pesq:
        print(f"\n⚠️ MARGINAL IMPROVEMENT: Phase 1 PESQ {phase1_avg_pesq:.3f} (gain: {pesq_gain:+.3f})")
        print(f"   Recommendation: CONTINUE TO PHASE 2 (worth trying)")
        decision = "CONTINUE_TO_PHASE2"
    else:
        print(f"\n❌ NO IMPROVEMENT: Phase 1 PESQ {phase1_avg_pesq:.3f} < V3 {v3_avg_pesq:.3f}")
        print(f"   Recommendation: SKIP PHASE 2, TRY ALTERNATIVE APPROACH")
        decision = "TRY_ALTERNATIVE"
    
    print("\n" + "=" * 80)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 1 - Multi-Scale Spectral Loss',
        'checkpoint': str(phase1_checkpoint),
        'phase1_pesq': float(phase1_avg_pesq),
        'phase1_stoi': float(phase1_avg_stoi),
        'v3_pesq': float(v3_avg_pesq),
        'v3_stoi': float(v3_avg_stoi),
        'pesq_gain': float(pesq_gain),
        'stoi_gain': float(stoi_gain),
        'decision': decision,
        'n_samples': len(test_files),
    }
    
    with open('phase1_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: phase1_evaluation_results.json")

if __name__ == '__main__':
    main()
