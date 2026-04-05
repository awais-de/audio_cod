#!/usr/bin/env python3
"""
Phase 4 Final Evaluation
Evaluate the Phase 4 adversarial fine-tuned model
Compare PESQ, STOI against V3 baseline
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
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths

# Config
SR = 16000
SEG_LEN = 16000  # 1 second at 16kHz
N_SAMPLES = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Baseline V3 metrics
V3_PESQ = 2.953
V3_STOI = 0.960


def pick_segment(audio: np.ndarray, seg_len: int) -> np.ndarray:
    """Pick a non-silent segment from audio"""
    if len(audio) <= seg_len:
        audio = np.pad(audio, (0, max(0, seg_len - len(audio))))
        return audio[:seg_len]
    
    # Try to find non-silent segment
    for _ in range(10):
        start = random.randint(0, len(audio) - seg_len)
        segment = audio[start:start + seg_len]
        if np.sqrt(np.mean(segment**2)) > 0.01:
            return segment
    
    # Fallback: middle segment
    mid = len(audio) // 2
    start = max(0, mid - seg_len // 2)
    return audio[start:start + seg_len]


def load_random_files(limit=N_SAMPLES):
    """Load random audio files"""
    paths = get_dataset_paths()
    data_dir = paths["test_clean"]
    if not data_dir.exists():
        data_dir = paths["train_clean_100"]
    
    files = list(data_dir.rglob('*.flac'))
    random.shuffle(files)
    return files[:limit]


def evaluate_model(model, files, device):
    """Evaluate model on files"""
    pesq_scores = []
    stoi_scores = []
    
    model.eval()
    with torch.no_grad():
        for i, audio_path in enumerate(files):
            try:
                # Load audio
                audio, sr = sf.read(str(audio_path))
                if sr != SR:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
                
                # Normalize
                if audio.max() > 0:
                    audio = audio / (np.abs(audio).max() + 1e-8)
                
                # Get segment
                segment = pick_segment(audio, SEG_LEN)
                
                # Reconstruct
                input_tensor = torch.from_numpy(segment).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                
                reconstructed = output.squeeze().cpu().numpy()
                
                # Handle size mismatch
                if len(reconstructed) > len(segment):
                    reconstructed = reconstructed[:len(segment)]
                elif len(reconstructed) < len(segment):
                    reconstructed = np.pad(reconstructed, (0, len(segment) - len(reconstructed)))
                
                # Clamp
                reconstructed = np.clip(reconstructed, -1.0, 1.0)
                segment = np.clip(segment, -1.0, 1.0)
                
                # Compute metrics
                try:
                    pesq_score = pesq(SR, segment, reconstructed, 'wb')
                    pesq_scores.append(pesq_score)
                except Exception as e:
                    logger.debug(f"PESQ error: {e}")
                
                try:
                    stoi_score = stoi(segment, reconstructed, SR)
                    stoi_scores.append(stoi_score)
                except Exception as e:
                    logger.debug(f"STOI error: {e}")
                
                logger.info(f"  [{i+1}/{len(files)}] PESQ: {pesq_scores[-1]:.4f} | STOI: {stoi_scores[-1]:.4f}")
            
            except Exception as e:
                logger.warning(f"Error processing {audio_path}: {e}")
                continue
    
    return pesq_scores, stoi_scores


def main():
    logger.info("=" * 80)
    logger.info("PHASE 4 EVALUATION: ADVERSARIAL FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Samples: {N_SAMPLES}")
    logger.info(f"Segment length: {SEG_LEN} samples ({SEG_LEN/SR:.1f}s)")
    logger.info("")
    
    # Load model
    logger.info("📦 Loading model...")
    model = NeuralAudioCodec(
        d_model=384,
        n_layers=6,
        n_heads=8,
        window_size=384,
        hop_length=160,
        sample_rate=16000,
    )
    model = model.to(DEVICE)
    
    # Load Phase 4 checkpoint
    phase4_dirs = sorted(Path('checkpoints_emergency').glob('phase4_adversarial_*'))
    if not phase4_dirs:
        logger.error("❌ No Phase 4 checkpoint found")
        return
    
    phase4_ckpt = phase4_dirs[-1] / 'best.pt'
    logger.info(f"📂 Loading checkpoint: {phase4_ckpt}")
    try:
        state_dict = torch.load(phase4_ckpt, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logger.info("✅ Checkpoint loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        return
    
    # Load test files
    logger.info(f"\n📊 Loading {N_SAMPLES} test samples...")
    files = load_random_files(N_SAMPLES)
    logger.info(f"✓ Loaded {len(files)} files")
    
    # Evaluate
    logger.info("\n🔄 Evaluating...")
    pesq_scores, stoi_scores = evaluate_model(model, files, DEVICE)
    
    # Results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    if pesq_scores:
        pesq_mean = np.mean(pesq_scores)
        pesq_std = np.std(pesq_scores)
        pesq_improvement = pesq_mean - V3_PESQ
        pesq_pct = (pesq_improvement / V3_PESQ) * 100
        
        logger.info(f"\n🎵 PESQ Score:")
        logger.info(f"  V3 Baseline:    {V3_PESQ:.4f}")
        logger.info(f"  Phase 4:        {pesq_mean:.4f} ± {pesq_std:.4f}")
        logger.info(f"  Improvement:    {pesq_improvement:+.4f} ({pesq_pct:+.2f}%)")
        
        if pesq_mean >= 3.5:
            logger.info(f"  ✅ TARGET ACHIEVED! (≥ 3.5)")
        elif pesq_mean >= 3.0:
            logger.info(f"  ✅ Significant improvement")
        else:
            logger.info(f"  ⚠️  Below target")
    
    if stoi_scores:
        stoi_mean = np.mean(stoi_scores)
        stoi_std = np.std(stoi_scores)
        stoi_improvement = stoi_mean - V3_STOI
        stoi_pct = (stoi_improvement / V3_STOI) * 100
        
        logger.info(f"\n🔊 STOI Score:")
        logger.info(f"  V3 Baseline:    {V3_STOI:.4f}")
        logger.info(f"  Phase 4:        {stoi_mean:.4f} ± {stoi_std:.4f}")
        logger.info(f"  Improvement:    {stoi_improvement:+.4f} ({stoi_pct:+.2f}%)")
    
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
