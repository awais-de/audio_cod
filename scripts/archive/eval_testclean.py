#!/usr/bin/env python3
"""
Comprehensive evaluation on LibriSpeech test-clean dataset.
Compares V3 baseline with Phase 1-4 checkpoints.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from model import NeuralAudioCodec
from paths import get_dataset_paths, get_checkpoint_paths

# PESQ and STOI imports
try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    print("Warning: pesq not installed, skipping PESQ metric")
    HAS_PESQ = False

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    print("Warning: pystoi not installed, skipping STOI metric")
    HAS_STOI = False

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SR = 16000
SEG_LEN = 16000  # 1 second at 16kHz
NUM_SAMPLES = 50  # Evaluate on 50 test samples

# Model checkpoints
_ckpt_paths = get_checkpoint_paths()
CHECKPOINTS = {
    'V3 Baseline': str(_ckpt_paths['v3_baseline']),
    'Phase 1': str(_ckpt_paths['phase1']),
    'Phase 2': str(_ckpt_paths['phase2']),
    'Phase 3': str(_ckpt_paths['phase3']),
    'Phase 4': str(_ckpt_paths['phase4']),
}

def load_model(checkpoint_path):
    """Load model from checkpoint."""
    # Use correct model dimensions (d_model=384, n_layers=6) for Phase 1-4 checkpoints
    model = NeuralAudioCodec(d_model=384, n_layers=6)
    
    try:
        # Try loading with weights_only=False for older checkpoints
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        # Handle checkpoints saved with metadata wrapper
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None
    
    model = model.to(DEVICE)
    model.eval()
    return model

def get_test_files():
    """Get all test-clean audio files."""
    test_path = get_dataset_paths()["test_clean"]
    print(f"Looking for test files in: {test_path}")
    print(f"Path exists: {test_path.exists()}")
    if not test_path.exists():
        print(f"Error: test-clean directory not found at {test_path}")
        return []
    
    audio_files = sorted(list(test_path.glob('**/*.flac')))
    print(f"Found {len(audio_files)} test-clean audio files")
    if audio_files:
        # Check first file dimensions
        wf, sr = torchaudio.load(str(audio_files[0]))
        print(f"First file: {wf.shape[1]} samples ({wf.shape[1]/sr:.2f}s) @ {sr}Hz")
    return audio_files

@torch.no_grad()
def evaluate_model(model, audio_files, max_samples=None):
    """Evaluate model on test files."""
    if max_samples is None:
        max_samples = len(audio_files)
    
    pesq_scores = []
    stoi_scores = []
    error_count = 0
    skip_count = 0
    
    for audio_file in tqdm(audio_files[:max_samples], desc='Evaluating', unit='file'):
        try:
            # Load audio
            waveform, sr = torchaudio.load(str(audio_file))
            
            # Resample if needed
            if sr != SR:
                resampler = T.Resample(sr, SR)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Take first 1 second
            if waveform.shape[1] < SEG_LEN:
                skip_count += 1
                continue  # Skip short files
            
            waveform = waveform[:, :SEG_LEN]
            
            # Normalize to [-1, 1]
            waveform_max = torch.abs(waveform).max()
            if waveform_max > 0:
                waveform = waveform / waveform_max
            
            waveform = waveform.to(DEVICE)
            
            # Encode and decode
            reconstructed = model(waveform.unsqueeze(0))
            reconstructed = reconstructed.squeeze(0)
            
            # Pad reconstructed to match original length if needed
            if reconstructed.shape[1] < waveform.shape[1]:
                pad_amount = waveform.shape[1] - reconstructed.shape[1]
                reconstructed = torch.nn.functional.pad(reconstructed, (0, pad_amount))
            elif reconstructed.shape[1] > waveform.shape[1]:
                reconstructed = reconstructed[:, :waveform.shape[1]]
            
            # Normalize reconstruction to [-1, 1]
            recon_max = torch.abs(reconstructed).max()
            if recon_max > 0:
                reconstructed = reconstructed / recon_max
            
            # Convert to numpy for metrics
            orig_np = waveform.squeeze(0).detach().cpu().numpy()
            recon_np = reconstructed.squeeze(0).detach().cpu().numpy()
            
            # Ensure values are in valid range for PESQ
            orig_np = np.clip(orig_np, -1, 1)
            recon_np = np.clip(recon_np, -1, 1)
            
            # Compute PESQ (narrowband)
            if HAS_PESQ:
                try:
                    pesq_score = pesq(SR, orig_np, recon_np, 'nb')
                    if not np.isnan(pesq_score) and not np.isinf(pesq_score):
                        pesq_scores.append(pesq_score)
                except Exception as e:
                    error_count += 1
                    # print(f"PESQ error: {e}")
            
            # Compute STOI
            if HAS_STOI:
                try:
                    stoi_score = stoi(orig_np, recon_np, SR)
                    if not np.isnan(stoi_score) and not np.isinf(stoi_score):
                        stoi_scores.append(stoi_score)
                except Exception as e:
                    error_count += 1
        
        except Exception as e:
            skip_count += 1
            continue
    
    return {
        'pesq': pesq_scores,
        'stoi': stoi_scores,
        'num_samples': len(pesq_scores),
        'skip_count': skip_count,
        'error_count': error_count,
    }

def main():
    print("=" * 80)
    print("EVALUATION ON LIBRISPEECH TEST-CLEAN")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Sample length: {SEG_LEN} samples ({SEG_LEN/SR:.1f}s at {SR}Hz)")
    print(f"Max samples: {NUM_SAMPLES}")
    print()
    
    # Get test files
    test_files = get_test_files()
    if not test_files:
        print("Error: No test files found")
        return
    
    print(f"Total test-clean files: {len(test_files)}")
    print()
    
    # Evaluate each model
    results = {}
    for model_name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nEvaluating {model_name}...")
        print(f"Checkpoint: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            print(f"  ❌ Checkpoint not found")
            continue
        
        model = load_model(checkpoint_path)
        if model is None:
            print(f"  ❌ Failed to load model")
            continue
        
        eval_result = evaluate_model(model, test_files, max_samples=NUM_SAMPLES)
        results[model_name] = eval_result
        
        print(f"  Samples evaluated: {eval_result['num_samples']}, Skipped: {eval_result.get('skip_count', 0)}, Errors: {eval_result.get('error_count', 0)}")
        
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
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<25} {'PESQ':<20} {'STOI':<20}")
    print("-" * 80)
    
    v3_pesq = None
    for model_name in CHECKPOINTS.keys():
        if model_name in results:
            result = results[model_name]
            
            if result['pesq']:
                pesq_mean = np.mean(result['pesq'])
                pesq_std = np.std(result['pesq'])
                pesq_str = f"{pesq_mean:.4f}±{pesq_std:.4f}"
                
                if model_name == 'V3 Baseline':
                    v3_pesq = pesq_mean
                elif v3_pesq is not None:
                    improvement = ((pesq_mean - v3_pesq) / v3_pesq) * 100
                    pesq_str += f" ({improvement:+.1f}%)"
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
    print("\nEVALUATION COMPLETE")
    if v3_pesq is not None:
        print(f"V3 baseline PESQ: {v3_pesq:.4f}")
        print(f"Target: 3.5 PESQ")
        if v3_pesq >= 3.5:
            print("✅ TARGET ACHIEVED")
        else:
            gap = 3.5 - v3_pesq
            print(f"❌ Gap to target: {gap:.4f} PESQ")

if __name__ == '__main__':
    main()
