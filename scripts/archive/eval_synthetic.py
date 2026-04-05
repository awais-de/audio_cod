#!/usr/bin/env python3
"""
Evaluate models on synthetic test data (no FLAC dependency).
Generates random audio segments and evaluates PESQ/STOI.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from model import NeuralAudioCodec
from paths import get_checkpoint_paths

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("Warning: pesq not installed")

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    print("Warning: pystoi not installed")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SR = 16000
SEG_LEN = 16000  # 1 second
NUM_SAMPLES = 50

_ckpt_paths = get_checkpoint_paths()
CHECKPOINTS = {
    'Phase 1': str(_ckpt_paths['phase1']),
    'Phase 2': str(_ckpt_paths['phase2']),
    'Phase 3': str(_ckpt_paths['phase3']),
    'Phase 4': str(_ckpt_paths['phase4']),
}

def load_model(checkpoint_path):
    """Load model from checkpoint."""
    model = NeuralAudioCodec(d_model=384, n_layers=6)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None
    
    model = model.to(DEVICE)
    model.eval()
    return model

def generate_synthetic_audio(num_samples=10):
    """Generate synthetic audio resembling speech."""
    audio_list = []
    
    for _ in range(num_samples):
        # Generate audio with multiple frequency components (speech-like)
        t = np.linspace(0, 1, SEG_LEN)
        
        # Fundamental frequency (pitch) + harmonics
        f0 = np.random.uniform(80, 300)  # Speech pitch range
        audio = np.zeros(SEG_LEN)
        
        for harmonic in range(1, 6):
            audio += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
        
        # Add some modulation
        audio *= (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))  # Amplitude envelope
        
        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-7)
        audio = audio * 0.8  # Leave some headroom
        
        audio_list.append(audio.astype(np.float32))
    
    return audio_list

@torch.no_grad()
def evaluate_model(model, synthetic_audios):
    """Evaluate model on synthetic audio."""
    pesq_scores = []
    stoi_scores = []
    
    for audio in tqdm(synthetic_audios, desc='Evaluating', unit='sample'):
        try:
            # Convert to tensor
            waveform = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Encode and decode
            reconstructed = model(waveform)
            reconstructed = reconstructed.squeeze(0)
            
            # Convert to numpy
            orig_np = audio
            recon_np = reconstructed.squeeze(0).detach().cpu().numpy()
            
            # Clip to valid range
            orig_np = np.clip(orig_np, -1, 1)
            recon_np = np.clip(recon_np, -1, 1)
            
            # Compute PESQ
            if HAS_PESQ:
                try:
                    pesq_score = pesq(SR, orig_np, recon_np, 'nb')
                    if not np.isnan(pesq_score) and not np.isinf(pesq_score):
                        pesq_scores.append(pesq_score)
                except Exception:
                    pass
            
            # Compute STOI
            if HAS_STOI:
                try:
                    stoi_score = stoi(orig_np, recon_np, SR)
                    if not np.isnan(stoi_score) and not np.isinf(stoi_score):
                        stoi_scores.append(stoi_score)
                except Exception:
                    pass
        
        except Exception:
            continue
    
    return {
        'pesq': pesq_scores,
        'stoi': stoi_scores,
        'num_samples': len(pesq_scores),
    }

def main():
    print("=" * 80)
    print("EVALUATION ON SYNTHETIC AUDIO (No FFmpeg Dependency)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Sample length: {SEG_LEN} samples ({SEG_LEN/SR:.1f}s at {SR}Hz)")
    print(f"Num samples: {NUM_SAMPLES}")
    print()
    
    # Generate synthetic audio
    print("Generating synthetic audio...")
    synthetic_audios = generate_synthetic_audio(NUM_SAMPLES)
    print(f"Generated {len(synthetic_audios)} synthetic audio samples")
    print()
    
    # Evaluate each model
    results = {}
    for model_name, checkpoint_path in CHECKPOINTS.items():
        print(f"Evaluating {model_name}...")
        print(f"Checkpoint: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            print(f"  ❌ Checkpoint not found")
            continue
        
        model = load_model(checkpoint_path)
        if model is None:
            print(f"  ❌ Failed to load model")
            continue
        
        eval_result = evaluate_model(model, synthetic_audios)
        results[model_name] = eval_result
        
        if eval_result['pesq']:
            pesq_mean = np.mean(eval_result['pesq'])
            pesq_std = np.std(eval_result['pesq'])
            print(f"  ✅ PESQ: {pesq_mean:.4f} ± {pesq_std:.4f} ({eval_result['num_samples']} samples)")
        else:
            print(f"  ⚠️  No PESQ scores computed")
        
        if eval_result['stoi']:
            stoi_mean = np.mean(eval_result['stoi'])
            stoi_std = np.std(eval_result['stoi'])
            print(f"  ✅ STOI: {stoi_mean:.4f} ± {stoi_std:.4f} ({eval_result['num_samples']} samples)")
        else:
            print(f"  ⚠️  No STOI scores computed")
        print()
    
    # Print comparison table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<25} {'PESQ':<20} {'STOI':<20}")
    print("-" * 80)
    
    for model_name in CHECKPOINTS.keys():
        if model_name in results:
            result = results[model_name]
            
            if result['pesq']:
                pesq_mean = np.mean(result['pesq'])
                pesq_std = np.std(result['pesq'])
                pesq_str = f"{pesq_mean:.4f}±{pesq_std:.4f}"
            else:
                pesq_str = "N/A"
            
            if result['stoi']:
                stoi_mean = np.mean(result['stoi'])
                stoi_std = np.std(result['stoi'])
                stoi_str = f"{stoi_mean:.4f}±{stoi_std:.4f}"
            else:
                stoi_str = "N/A"
            
            print(f"{model_name:<25} {pesq_str:<20} {stoi_str:<20}")
    
    print("=" * 80)
    print("\n✅ EVALUATION COMPLETE")
    print("\nNote: Evaluated on synthetic audio (speech-like signals).")
    print("Real test-clean evaluation requires FFmpeg system libraries (root access).")

if __name__ == '__main__':
    main()
