#!/usr/bin/env python3
"""
Evaluate Phase A and Phase B with real PESQ/STOI.
Same 5-speaker, 5-second test set as all 2026-07-01 comparisons.
"""

import sys
import zlib
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path('/home/muaw1874/Desktop/ac_proj/audio_cod')
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import get_dataset_paths
from src.codec_utils import load_model, encode_decode, compute_metrics


def run_phase(phase_name, ckpt_path, test_files, device, sr=16000, clip_sec=5):
    print(f"\n{'='*60}")
    print(f"  Evaluating {phase_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    model, meta = load_model(ckpt_path, device)
    print(f"  Loaded. bottleneck_dim={meta.get('bottleneck_dim')} "
          f"temporal_stride={meta.get('temporal_stride')}")

    results = []
    for audio_path in test_files:
        spk = audio_path.parts[-3]
        audio, file_sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            n = int(len(audio) * sr / file_sr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio = np.clip(audio[:clip_sec * sr], -1.0, 1.0).astype(np.float32)

        decoded, kbps = encode_decode(model, audio, sr, device, chunk_sec=1.0)
        n_common = min(len(audio), len(decoded))
        pesq_wb, stoi = compute_metrics(audio[:n_common], decoded[:n_common], sr)

        results.append({'speaker': spk, 'kbps': kbps, 'pesq_wb': pesq_wb, 'stoi': stoi})
        pesq_str = f"{pesq_wb:.3f}" if pesq_wb is not None else "n/a"
        print(f"  spk {spk:>6}  kbps={kbps:.2f}  PESQ-WB={pesq_str}  STOI={stoi:.3f}")

    mean_kbps  = float(np.mean([r['kbps']    for r in results]))
    pesq_vals  = [r['pesq_wb'] for r in results if r['pesq_wb'] is not None]
    mean_pesq  = float(np.mean(pesq_vals)) if pesq_vals else None
    mean_stoi  = float(np.mean([r['stoi']    for r in results if r['stoi']    is not None]))
    pesq_str   = f"{mean_pesq:.3f}" if mean_pesq is not None else "n/a (PESQ not available on this machine)"
    print(f"\n  MEAN  kbps={mean_kbps:.2f}  PESQ-WB={pesq_str}  STOI={mean_stoi:.3f}")
    return results, mean_kbps, mean_pesq, mean_stoi


def main():
    device = 'cpu'   # avoid CUDA OOM when other models are resident
    SR = 16000
    CLIP_SEC = 5
    TARGET_SPEAKERS = {'1089', '1188', '1221', '1284', '1320'}

    paths = get_dataset_paths()
    speakers = {}
    for f in sorted(paths['test_clean'].rglob('*.flac')):
        spk = f.parts[-3]
        if spk in TARGET_SPEAKERS and spk not in speakers:
            speakers[spk] = f
        if len(speakers) == len(TARGET_SPEAKERS):
            break
    test_files = [speakers[s] for s in sorted(speakers)]
    print(f"Test files ({len(test_files)} speakers): {[f.parts[-3] for f in test_files]}")

    phases = [
        ('Phase A', PROJECT_ROOT / 'checkpoints_active/temporal_phaseA/best.pt'),
        ('Phase B', PROJECT_ROOT / 'checkpoints_active/temporal_phaseB/best.pt'),
    ]

    summary = []
    per_speaker_all = {}

    for phase_name, ckpt_path in phases:
        results, mk, mp, ms = run_phase(phase_name, ckpt_path, test_files, device, SR, CLIP_SEC)
        summary.append({'phase': phase_name, 'kbps': mk, 'pesq_wb': mp, 'stoi': ms})
        per_speaker_all[phase_name] = results

    # Write report
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    out_dir = PROJECT_ROOT / 'comparisons' / '2026-07-10_phaseAB_eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    SEP = '=' * 68
    sep = '-' * 68
    lines = [
        '',
        '====================================================================',
        'EVAL: Phase A and Phase B — with real PESQ/STOI',
        f'Generated: {timestamp}',
        'Clips: 5 speakers x 5s  |  16 kHz mono',
        '====================================================================',
        '',
        f"{'Model':<28} {'kbps':>6}  {'PESQ-WB':>8}  {'STOI':>6}",
        sep,
    ]
    for s in summary:
        pstr = f"{s['pesq_wb']:>8.3f}" if s['pesq_wb'] is not None else "     n/a"
        lines.append(f"  {s['phase']:<26} {s['kbps']:>5.2f}k  {pstr}  {s['stoi']:>6.3f}")
    lines.append(sep)
    lines.append('')
    lines.append('NOTE: PESQ-WB not available on this machine (pesq wheel fails to build).')
    lines.append('      Run eval_phaseAB.py on the Windows machine to fill in PESQ column.')
    lines.append('')
    lines.append('PER-SPEAKER')
    lines.append(sep)
    for phase_name, results in per_speaker_all.items():
        lines.append(f'\n  {phase_name}')
        lines.append(f"  {'Speaker':>8}  {'kbps':>6}  {'PESQ-WB':>8}  {'STOI':>6}")
        lines.append(f"  {'-'*40}")
        for r in results:
            pstr = f"{r['pesq_wb']:>8.3f}" if r['pesq_wb'] is not None else "     n/a"
            lines.append(f"  {r['speaker']:>8}  {r['kbps']:>6.2f}  {pstr}  {r['stoi']:>6.3f}")
    lines.append(SEP)

    report = '\n'.join(lines)
    print('\n' + report)

    (out_dir / 'report.txt').write_text(report)
    print(f"\nSaved: {out_dir}/report.txt")

    # CSV
    with open(out_dir / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['phase', 'speaker', 'kbps', 'pesq_wb', 'stoi'])
        w.writeheader()
        for phase_name, results in per_speaker_all.items():
            for r in results:
                w.writerow({'phase': phase_name, **r})
    print(f"Saved: {out_dir}/metrics.csv")


if __name__ == '__main__':
    main()
