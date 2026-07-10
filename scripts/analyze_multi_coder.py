#!/usr/bin/env python3
"""
Multi-coder entropy confirmation.

Runs the same encode→quantize pipeline as analyze_compression.py but measures
compression ratios under three non-learned entropy coders: zlib, LZMA, and bz2.

If the phase ordering (D-VAE most compressed, E/F/G clustered) holds under all
three coders, it demonstrates that the latent entropy floor is a property of the
representation, not of the specific compressor used to probe it.

Same speakers, same clip length, same quantisation as the 2026-06-30 baseline.

Output: comparisons/YYYY-MM-DD_multi_coder/
  report.txt    — per-phase ratios for all three coders + ordering check
  metrics.csv   — per-speaker, per-phase, per-coder raw numbers
"""

import bz2
import csv
import lzma
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.codec_utils import load_model
from src.paths import get_dataset_paths

PHASES = {
    'C':     PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt',
    'D':     PROJECT_ROOT / 'checkpoints_active/temporal_phaseD/best.pt',
    'D-VAE': PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae/best.pt',
    'E':     PROJECT_ROOT / 'checkpoints_active/temporal_phaseE/best.pt',
    'F':     PROJECT_ROOT / 'checkpoints_active/temporal_phaseF/best.pt',
    'G':     PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt',
}

NUM_LEVELS = 8
CLIP_SEC   = 5
SR         = 16000
CODERS = {
    'zlib': lambda b: zlib.compress(b, level=9),
    'lzma': lambda b: lzma.compress(b, preset=9),
    'bz2':  lambda b: bz2.compress(b, compresslevel=9),
}


def quantize(z_np: np.ndarray):
    z_min  = float(z_np.min())
    z_max  = float(z_np.max())
    scale  = (z_max - z_min) / (NUM_LEVELS - 1) + 1e-8
    q = np.clip(np.round((z_np - z_min) / scale), 0, NUM_LEVELS - 1).astype(np.uint8)
    return q


def shannon_entropy_per_dim(q: np.ndarray) -> np.ndarray:
    """q shape: (frames, dims). Returns (dims,) entropy in bits."""
    n_dims = q.shape[1]
    H = np.zeros(n_dims)
    for d in range(n_dims):
        counts = np.bincount(q[:, d], minlength=NUM_LEVELS).astype(float)
        probs  = counts / counts.sum()
        nonzero = probs[probs > 0]
        H[d] = -np.sum(nonzero * np.log2(nonzero))
    return H


def run_phase(model, audio_clips: list, device) -> dict:
    """
    Returns per-coder total compressed sizes, raw bits, per-dim entropy,
    and per-speaker per-coder ratios.
    Speaker ordering matches audio_clips.
    """
    raw_bits_total = 0
    coder_bits = {c: 0 for c in CODERS}
    all_q = []
    per_spk = []  # one dict per speaker

    chunk_size = CLIP_SEC * SR

    with torch.no_grad():
        for audio in audio_clips:
            spk_raw = 0
            spk_coder = {c: 0 for c in CODERS}
            spk_q = []

            for start in range(0, len(audio), chunk_size):
                chunk = audio[start:start + chunk_size]
                if len(chunk) < 160:
                    continue
                x  = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
                z  = model.encode(x)
                q  = quantize(z.squeeze(0).cpu().numpy())

                raw = q.size * 3
                spk_raw += raw
                raw_bits_total += raw

                qb = q.tobytes()
                for name, compress_fn in CODERS.items():
                    cb = len(compress_fn(qb)) * 8
                    spk_coder[name] += cb
                    coder_bits[name] += cb

                spk_q.append(q)

            all_q.append(spk_q)
            duration = len(audio) / SR
            spk_ratios = {c: spk_raw / spk_coder[c] for c in CODERS}
            spk_kbps   = {c: spk_coder[c] / duration / 1000 for c in CODERS}
            per_spk.append({'raw': spk_raw, 'ratios': spk_ratios, 'kbps': spk_kbps})

    # Aggregate per-dim entropy (average across speakers and chunks)
    # Flatten all chunks across all speakers
    flat_q = np.concatenate([c for spk in all_q for c in spk], axis=0)
    mean_H = shannon_entropy_per_dim(flat_q)

    total_duration = sum(len(a) / SR for a in audio_clips)
    global_ratios = {c: raw_bits_total / coder_bits[c] for c in CODERS}
    global_kbps   = {c: coder_bits[c] / total_duration / 1000 for c in CODERS}

    return {
        'ratios':   global_ratios,
        'kbps':     global_kbps,
        'mean_H':   float(mean_H.mean()),
        'min_H':    float(mean_H.min()),
        'max_H':    float(mean_H.max()),
        'dim_H':    mean_H,
        'per_spk':  per_spk,
    }


def main():
    import soundfile as sf

    device = torch.device('cpu')   # explicit CPU — avoids OOM from resident models

    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir   = PROJECT_ROOT / 'comparisons' / f'{timestamp}_multi_coder'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Same 5 speakers as the 2026-06-30 baseline ──────────────────────────
    paths    = get_dataset_paths()
    speakers = {}
    for f in sorted(paths['test_clean'].rglob('*.flac')):
        spk = f.parts[-3]
        if spk not in speakers:
            speakers[spk] = f
        if len(speakers) == 5:
            break
    test_files = list(speakers.values())
    spk_ids    = [f.parts[-3] for f in test_files]

    print(f"\n{'='*72}")
    print("MULTI-CODER ENTROPY CONFIRMATION")
    print(f"{'='*72}")
    print(f"Coders: {', '.join(CODERS)}  |  device: {device}")
    print(f"Clips:  5 speakers × {CLIP_SEC}s  |  16 kHz mono")
    print(f"Output: {out_dir}\n")

    audio_clips = []
    for f in test_files:
        audio, file_sr = sf.read(f)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            n     = int(len(audio) * SR / file_sr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio_clips.append(np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32))

    # ── Load and run each phase ───────────────────────────────────────────────
    results = {}
    for phase, ckpt_path in PHASES.items():
        if not ckpt_path.exists():
            print(f"  skip  {phase}  (checkpoint not found)")
            continue
        print(f"  loading phase {phase} ...", end=' ', flush=True)
        model, _ = load_model(ckpt_path, device)
        print("ok — running inference ...", end=' ', flush=True)
        results[phase] = run_phase(model, audio_clips, device)
        r = results[phase]
        print(
            f"zlib={r['ratios']['zlib']:.3f}×  "
            f"lzma={r['ratios']['lzma']:.3f}×  "
            f"bz2={r['ratios']['bz2']:.3f}×  "
            f"H̄={r['mean_H']:.3f} bits"
        )
        del model

    # ── Report ───────────────────────────────────────────────────────────────
    SEP = '=' * 72
    sep = '-' * 72

    lines = [
        '',
        SEP,
        'MULTI-CODER ENTROPY CONFIRMATION',
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Clips: 5 speakers × {CLIP_SEC}s  |  16 kHz mono  |  device: {device}",
        "Coders: zlib (Deflate, level 9)  |  LZMA (preset 9)  |  bz2 (level 9)",
        SEP,
        '',
        'COMPRESSION RATIOS — all coders',
        '(raw 3-bit size / compressed size  —  higher = more latent structure)',
        '',
        f"{'Phase':<10} {'zlib':>9} {'lzma':>9} {'bz2':>9} {'Mean H(d)':>11}",
        sep,
    ]
    for phase, r in results.items():
        lines.append(
            f"  {phase:<8} "
            f"{r['ratios']['zlib']:>8.3f}× "
            f"{r['ratios']['lzma']:>8.3f}× "
            f"{r['ratios']['bz2']:>8.3f}×"
            f" {r['mean_H']:>10.3f}"
        )

    # Check phase ordering consistency
    lines += [sep, '', 'ORDERING CHECK (highest ratio = most structure)']
    for coder in CODERS:
        ranked = sorted(results.items(), key=lambda x: x[1]['ratios'][coder], reverse=True)
        order  = ' > '.join(p for p, _ in ranked)
        lines.append(f"  {coder:<6}: {order}")

    # D-VAE gap
    lines += ['', 'D-VAE GAP vs next-most-compressed phase']
    if 'D-VAE' in results:
        for coder in CODERS:
            dvae_r = results['D-VAE']['ratios'][coder]
            others = {p: r['ratios'][coder] for p, r in results.items() if p != 'D-VAE'}
            second = max(others.values())
            second_phase = max(others, key=others.get)
            lines.append(
                f"  {coder:<6}: D-VAE={dvae_r:.3f}×  "
                f"next={second_phase} {second:.3f}×  "
                f"gap=+{dvae_r - second:.3f}×"
            )

    lines += ['', sep]

    # Effective kbps table
    lines += [
        '',
        'EFFECTIVE BITRATE (kbps) — all coders',
        f"{'Phase':<10} {'zlib':>9} {'lzma':>9} {'bz2':>9}",
        sep,
    ]
    for phase, r in results.items():
        lines.append(
            f"  {phase:<8} "
            f"{r['kbps']['zlib']:>8.2f}k "
            f"{r['kbps']['lzma']:>8.2f}k "
            f"{r['kbps']['bz2']:>8.2f}k"
        )
    lines.append(sep)

    report = '\n'.join(lines)
    print('\n' + report)

    with open(out_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # ── CSV ──────────────────────────────────────────────────────────────────
    with open(out_dir / 'metrics.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['speaker', 'phase', 'coder', 'compression_ratio', 'effective_kbps'])
        for phase, r in results.items():
            for i, spk in enumerate(spk_ids):
                spk_data = r['per_spk'][i]
                for coder in CODERS:
                    w.writerow([
                        spk, phase, coder,
                        f"{spk_data['ratios'][coder]:.4f}",
                        f"{spk_data['kbps'][coder]:.4f}",
                    ])

    print(f"\nreport: {out_dir}/report.txt")
    print(f"csv:    {out_dir}/metrics.csv")


if __name__ == '__main__':
    main()
