#!/usr/bin/env python3
"""
Quick eval: Phase C vs D-VAE vs D-Entropy
==========================================
Compares the three phases on n speakers using STOI + Shannon H + zlib ratio.
No PESQ required — runs on Linux.

Decision aid for whether D-Entropy training needs a higher entropy-max.

Usage:
  python scripts/quick_entropy_eval.py            # 15 speakers (default)
  python scripts/quick_entropy_eval.py --n 10     # faster
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.codec_utils import load_model, encode_decode, compute_compression_stats, compute_metrics
from src.paths import get_dataset_paths

PHASES = [
    ('C',        PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt'),
    ('D-VAE',    PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae/best.pt'),
    ('D-Entropy',PROJECT_ROOT / 'checkpoints_active/temporal_phaseEntropy/best.pt'),
]


def collect_speakers(test_clean: Path, n: int, clip_sec: float = 5.0, sr: int = 16000):
    speakers = {}
    for flac in sorted(test_clean.rglob('*.flac')):
        spk = flac.parts[-3]
        if spk not in speakers:
            audio, file_sr = sf.read(flac, dtype='float32')
            if audio.ndim > 1:
                audio = audio[:, 0]
            if file_sr != sr:
                continue
            n_samples = int(clip_sec * sr)
            mid = len(audio) // 2
            start = max(0, mid - n_samples // 2)
            speakers[spk] = audio[start:start + n_samples]
        if len(speakers) >= n:
            break
    return list(speakers.values())


def eval_phase(name, ckpt_path, audios, device, sr=16000):
    print(f"  Loading {name}...", flush=True)
    model, _ = load_model(ckpt_path, device)

    stois, ratios, entropies = [], [], []
    for i, audio in enumerate(audios):
        recon, _ = encode_decode(model, audio, sr, device)
        _, stoi  = compute_metrics(audio, recon, sr)
        stats    = compute_compression_stats(model, audio, sr, device)

        stois.append(stoi)
        ratios.append(stats['compression_ratio'])
        entropies.append(float(stats['dim_entropy'].mean()))
        print(f"    [{i+1:2d}/{len(audios)}] STOI={stoi:.3f}  H={entropies[-1]:.4f}bits  zlib={ratios[-1]:.3f}x",
              flush=True)

    return {
        'stoi':  np.mean(stois),
        'H':     np.mean(entropies),
        'zlib':  np.mean(ratios),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',      type=int, default=15, help='Number of speakers')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    paths = get_dataset_paths()
    test_clean = Path(paths['test_clean'])
    print(f"Dataset: {test_clean}")
    print(f"Collecting {args.n} speakers...", flush=True)
    audios = collect_speakers(test_clean, args.n)
    print(f"  Got {len(audios)} speakers.\n")

    results = {}
    for name, ckpt in PHASES:
        print(f"\n{'─'*50}")
        print(f"Phase {name}")
        results[name] = eval_phase(name, ckpt, audios, device)
        r = results[name]
        print(f"  → mean STOI={r['stoi']:.4f}  H={r['H']:.4f}bits  zlib={r['zlib']:.4f}x")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Phase':<14} {'STOI':>8} {'H (bits)':>10} {'zlib ratio':>12}")
    print(f"{'─'*60}")
    for name, r in results.items():
        print(f"{name:<14} {r['stoi']:>8.4f} {r['H']:>10.4f} {r['zlib']:>12.4f}")

    c   = results['C']
    dv  = results['D-VAE']
    de  = results['D-Entropy']

    print(f"\n{'─'*60}")
    print("Deltas vs Phase C:")
    print(f"  D-VAE:     ΔSTOI={dv['stoi']-c['stoi']:+.4f}  ΔH={dv['H']-c['H']:+.4f}bits  Δzlib={dv['zlib']-c['zlib']:+.4f}x")
    print(f"  D-Entropy: ΔSTOI={de['stoi']-c['stoi']:+.4f}  ΔH={de['H']-c['H']:+.4f}bits  Δzlib={de['zlib']-c['zlib']:+.4f}x")

    print(f"\n{'─'*60}")
    print("Decision:")
    h_drop = c['H'] - de['H']
    stoi_drop = c['stoi'] - de['stoi']
    if h_drop > 0.15 and stoi_drop > 0.02:
        print(f"  ✓ Mechanism confirmed: H dropped {h_drop:.4f} bits, STOI dropped {stoi_drop:.4f}")
        print(f"    Keep β=0.01. Proceed to full eval on Windows.")
    elif h_drop > 0.05:
        print(f"  ~ Weak signal: H dropped {h_drop:.4f} bits, STOI dropped {stoi_drop:.4f}")
        print(f"    Consider rerun with --entropy-max 0.05 for a stronger effect.")
    else:
        print(f"  ✗ No meaningful effect: H dropped only {h_drop:.4f} bits")
        print(f"    Rerun with --entropy-max 0.05 or 0.1")


if __name__ == '__main__':
    main()
