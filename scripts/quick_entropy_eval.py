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
    ('G-16dim',  PROJECT_ROOT / 'checkpoints_active/temporal_phaseG_16/best.pt'),
    ('G-32dim',  PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt'),
    ('G-64dim',  PROJECT_ROOT / 'checkpoints_active/temporal_phaseG_64/best.pt'),
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

    ref = results['G-32dim']

    print(f"\n{'─'*60}")
    print("Deltas vs 32-dim (existing baseline):")
    for name, r in results.items():
        if name == 'G-32dim':
            continue
        print(f"  {name}: ΔSTOI={r['stoi']-ref['stoi']:+.4f}  "
              f"ΔH={r['H']-ref['H']:+.4f}bits  Δzlib={r['zlib']-ref['zlib']:+.4f}x")

    print(f"\n{'─'*60}")
    print("Coupling check (does H track quality across widths?):")
    vals = list(results.values())
    h_order    = sorted(results.keys(), key=lambda k: results[k]['H'])
    stoi_order = sorted(results.keys(), key=lambda k: results[k]['stoi'])
    if h_order == stoi_order:
        print(f"  ✓ H and STOI rank identically across widths — coupling holds.")
    else:
        print(f"  H rank:    {' < '.join(h_order)}")
        print(f"  STOI rank: {' < '.join(stoi_order)}")
        print(f"  ~ Ranks differ — inspect manually.")


if __name__ == '__main__':
    main()
