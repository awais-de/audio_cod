#!/usr/bin/env python3
"""
Comprehensive Model Evaluation
Compare V3, Phase 1, Phase 2, Phase 3, Phase 4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from pesq import pesq
from pystoi import stoi
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.model import NeuralAudioCodec

# Config
SR = 16000
SEG_LEN = 16000
N_FILES = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

V3_PESQ = 2.953
V3_STOI = 0.960


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    model = NeuralAudioCodec(
        d_model=384, n_layers=6, n_heads=8,
        window_size=384, hop_length=160, sample_rate=16000
    )
    model = model.to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    return model


def evaluate_model(model, audio_files, device, name):
    """Evaluate model on audio files"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating: {name}")
    logger.info(f"{'='*80}")
    
    pesq_scores = []
    stoi_scores = []
    
    with torch.no_grad():
        for i, audio_path in enumerate(audio_files):
            try:
                # Load and prepare audio
                audio, sr = sf.read(str(audio_path))
                if len(audio.shape) > 1:
                    audio = audio[:, 0]
                
                # Resample if needed
                if sr != SR:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
                
                # Normalize
                audio_norm = audio / (np.abs(audio).max() + 1e-8)
                
                # Use first 1 second
                segment = audio_norm[:SEG_LEN]
                if len(segment) < SEG_LEN:
                    segment = np.pad(segment, (0, SEG_LEN - len(segment)))
                
                # Forward pass
                x = torch.from_numpy(segment.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                y = model(x)
                
                # Get output
                recon = y.squeeze().cpu().numpy()
                
                # Match lengths
                if len(recon) > len(segment):
                    recon = recon[:len(segment)]
                elif len(recon) < len(segment):
                    recon = np.pad(recon, (0, len(segment) - len(recon)))
                
                # Clamp to valid range
                recon = np.clip(recon, -1.0, 1.0)
                segment = np.clip(segment, -1.0, 1.0)
                
                # Compute metrics
                pesq_val = pesq(SR, segment, recon, 'wb')
                stoi_val = stoi(segment, recon, SR)
                
                pesq_scores.append(pesq_val)
                stoi_scores.append(stoi_val)
                
                logger.info(f"  [{i+1:2d}/{len(audio_files)}] PESQ: {pesq_val:.4f} | STOI: {stoi_val:.4f} | {audio_path.parent.name}")
                
            except Exception as e:
                logger.warning(f"  [{i+1:2d}/{len(audio_files)}] Error: {str(e)[:60]}")
                continue
    
    if not pesq_scores:
        return None, None
    
    pesq_mean = np.mean(pesq_scores)
    pesq_std = np.std(pesq_scores)
    stoi_mean = np.mean(stoi_scores)
    stoi_std = np.std(stoi_scores)
    
    return (pesq_mean, pesq_std, pesq_scores), (stoi_mean, stoi_std, stoi_scores)


def main():
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL EVALUATION")
    logger.info("="*80)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Samples per model: {N_FILES}")
    logger.info(f"Segment length: {SEG_LEN} samples ({SEG_LEN/SR:.1f}s)")
    
    # Load test files
    test_clean = Path('/home/muaw1874/Desktop/ac_proj/datasets/LibriSpeech/test-clean')
    if not test_clean.exists():
        logger.error("Test-clean not found, using train-clean-100")
        test_clean = Path('/home/muaw1874/Desktop/ac_proj/datasets/LibriSpeech/train-clean-100')
    
    audio_files = sorted(test_clean.rglob('*.flac'))[:N_FILES]
    logger.info(f"\nLoaded {len(audio_files)} test files")
    
    # Checkpoints to evaluate
    checkpoints = {
        'V3 Baseline': Path('checkpoints_emergency/best_emergency.pt'),
        'Phase 1 (Multi-scale)': Path('checkpoints_emergency/phase1_multiscale_20260129_124452/best.pt'),
        'Phase 2 (Perceptual)': Path('checkpoints_emergency/phase2_perceptual_20260129_210723/best.pt'),
        'Phase 3 (Extended Data)': Path('checkpoints_emergency/phase3_extended_data_20260129_213522/best.pt'),
        'Phase 4 (Adversarial)': Path('checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt'),
    }
    
    # Evaluate all models
    results = {}
    for name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue
        
        try:
            model = load_model(ckpt_path, DEVICE)
            pesq_res, stoi_res = evaluate_model(model, audio_files, DEVICE, name)
            
            if pesq_res and stoi_res:
                results[name] = {
                    'pesq_mean': pesq_res[0],
                    'pesq_std': pesq_res[1],
                    'pesq_scores': pesq_res[2],
                    'stoi_mean': stoi_res[0],
                    'stoi_std': stoi_res[1],
                    'stoi_scores': stoi_res[2],
                }
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS COMPARISON")
    logger.info("="*80)
    
    logger.info("\n📊 PESQ Scores:")
    logger.info(f"{'Model':<25} {'PESQ':<12} {'Std':<10} {'vs V3':<12} {'Target':<10}")
    logger.info("-" * 70)
    
    for name, res in results.items():
        pesq_mean = res['pesq_mean']
        pesq_std = res['pesq_std']
        improvement = pesq_mean - V3_PESQ
        status = "✅ TARGET" if pesq_mean >= 3.5 else "⚠️  TARGET" if pesq_mean >= 3.0 else "❌ BELOW"
        
        logger.info(f"{name:<25} {pesq_mean:>6.4f}±{pesq_std:<5.4f} {improvement:>+6.4f} ({improvement/V3_PESQ*100:>+5.1f}%) {status:<10}")
    
    logger.info("\n📊 STOI Scores:")
    logger.info(f"{'Model':<25} {'STOI':<12} {'Std':<10} {'vs V3':<12}")
    logger.info("-" * 60)
    
    for name, res in results.items():
        stoi_mean = res['stoi_mean']
        stoi_std = res['stoi_std']
        improvement = stoi_mean - V3_STOI
        
        logger.info(f"{name:<25} {stoi_mean:>6.4f}±{stoi_std:<5.4f} {improvement:>+6.4f} ({improvement/V3_STOI*100:>+5.1f}%)")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    if 'Phase 4 (Adversarial)' in results:
        p4 = results['Phase 4 (Adversarial)']
        if p4['pesq_mean'] >= 3.5:
            logger.info("✅ TARGET ACHIEVED: PESQ >= 3.5 with Phase 4")
        elif p4['pesq_mean'] >= 3.0:
            logger.info(f"✅ SIGNIFICANT IMPROVEMENT: Phase 4 PESQ = {p4['pesq_mean']:.4f}")
        else:
            logger.info(f"⚠️  Phase 4 PESQ = {p4['pesq_mean']:.4f}, below target")
    
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()
