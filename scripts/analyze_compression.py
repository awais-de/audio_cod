#!/usr/bin/env python3
"""
Compression statistics across all trained phases.

Measures, for each checkpoint, how efficiently zlib compresses the quantized
latent representation — revealing whether training progressively organises the
latent into a more structured (more compressible) form.

Output: comparisons/YYYY-MM-DD_compression_analysis/
  report.txt   — human-readable summary
  metrics.csv  — per-speaker, per-phase raw numbers
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import get_dataset_paths
from src.codec_utils import load_model, compute_compression_stats, avg

import torch

PHASES = {
    'phaseC':    PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt',
    'phaseD':    PROJECT_ROOT / 'checkpoints_active/temporal_phaseD/best.pt',
    'phaseDvae': PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae/best.pt',
    'phaseE':    PROJECT_ROOT / 'checkpoints_active/temporal_phaseE/best.pt',
    'phaseF':    PROJECT_ROOT / 'checkpoints_active/temporal_phaseF/best.pt',
    'phaseG':    PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt',
    # MS Thesis — D-VAE entropy-quality coupling at other widths (#41)
    'phaseD16':    PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_16/best.pt',
    'phaseDvae16': PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae_16/best.pt',
    'phaseD64':    PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_64/best.pt',
    'phaseDvae64': PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae_64/best.pt',
}

CLIP_SEC = 5
SR       = 16000


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir   = PROJECT_ROOT / 'comparisons' / f'{timestamp}_compression_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────

    paths = get_dataset_paths()
    speakers = {}
    for f in sorted(paths['test_clean'].rglob('*.flac')):
        spk = f.parts[-3]
        if spk not in speakers:
            speakers[spk] = f
        if len(speakers) == 5:
            break
    test_files = list(speakers.values())

    import soundfile as sf

    print(f"\n{'='*72}")
    print("COMPRESSION ANALYSIS — all phases")
    print(f"{'='*72}")
    print(f"device: {device}  |  clips: 5 speakers × {CLIP_SEC}s  |  16kHz mono")
    print(f"output: {out_dir}\n")

    # ── Load models ───────────────────────────────────────────────────────────

    models = {}
    for name, path in PHASES.items():
        if not path.exists():
            print(f"  skip    {name}  (checkpoint not found)")
            continue
        print(f"  loading {name} ...", end=' ', flush=True)
        m, _ = load_model(path, device)
        models[name] = m
        print("ok")
    print()

    # ── Run analysis ──────────────────────────────────────────────────────────

    all_results = []

    for idx, audio_path in enumerate(test_files, 1):
        spk_id = audio_path.parts[-3]

        audio, file_sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            n     = int(len(audio) * SR / file_sr)
            audio = np.interp(np.linspace(0, len(audio), n),
                              np.arange(len(audio)), audio)
        audio = np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32)

        print(f"[{idx}/5] speaker {spk_id}")

        row = {'speaker': spk_id, 'file': audio_path.name}

        for name, model in models.items():
            stats = compute_compression_stats(model, audio, SR, device)
            row[f'{name}_ratio']      = stats['compression_ratio']
            row[f'{name}_eff_kbps']   = stats['effective_kbps']
            row[f'{name}_theo_kbps']  = stats['theoretical_kbps']
            row[f'{name}_mean_entr']  = float(stats['dim_entropy'].mean())
            row[f'{name}_min_entr']   = float(stats['dim_entropy'].min())
            row[f'{name}_max_entr']   = float(stats['dim_entropy'].max())
            row[f'{name}_mean_util']  = float(stats['dim_utilisation'].mean())
            row[f'{name}_dim_entropy'] = stats['dim_entropy']     # (32,) array

            print(f"  {name:<12}  ratio={stats['compression_ratio']:.3f}x"
                  f"  eff={stats['effective_kbps']:.2f} kbps"
                  f"  theo={stats['theoretical_kbps']:.2f} kbps"
                  f"  mean_entropy={stats['dim_entropy'].mean():.3f} bits")

        all_results.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────

    SEP = '=' * 72
    sep = '-' * 72

    lines = [
        '',
        SEP,
        'COMPRESSION ANALYSIS — all phases',
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Clips: 5 speakers × {CLIP_SEC}s  |  16 kHz mono",
        SEP,
        '',
        'COMPRESSION RATIO  (raw 3-bit size / zlib compressed size)',
        'Higher = more structure in the latent = zlib removes more redundancy',
        '',
        f"{'Phase':<14} {'Ratio':>8} {'Eff kbps':>10} {'Theo kbps':>11}"
        f" {'Mean H(d)':>10} {'Min H(d)':>10} {'Max H(d)':>10}",
        sep,
    ]

    for name in models:
        r     = avg(all_results, f'{name}_ratio')
        eff   = avg(all_results, f'{name}_eff_kbps')
        theo  = avg(all_results, f'{name}_theo_kbps')
        mh    = avg(all_results, f'{name}_mean_entr')
        minh  = avg(all_results, f'{name}_min_entr')
        maxh  = avg(all_results, f'{name}_max_entr')
        lines.append(
            f"  {name:<12} {r:>8.3f}x {eff:>9.2f}k {theo:>10.2f}k"
            f" {mh:>10.3f} {minh:>10.3f} {maxh:>10.3f}"
        )

    lines += [
        sep,
        '',
        'NOTE: H(d) = Shannon entropy of quantized values for dimension d.',
        '      Max possible = 3.000 bits (uniform use of all 8 levels).',
        '      Low entropy = dimension uses few distinct levels = zlib compresses heavily.',
        '',
    ]

    # Per-dimension entropy table (averaged across speakers)
    lines += [
        'PER-DIMENSION MEAN ENTROPY  (averaged across 5 speakers)',
        f"{'Dim':<6}" + ''.join(f"{n:>12}" for n in models),
        sep,
    ]
    n_dims = 32
    dim_entropy_per_phase = {}
    for name in models:
        # average dim_entropy across speakers
        arrays = [r[f'{name}_dim_entropy'] for r in all_results
                  if f'{name}_dim_entropy' in r]
        dim_entropy_per_phase[name] = np.mean(arrays, axis=0) if arrays else np.zeros(n_dims)

    for d in range(n_dims):
        row_str = f"  {d:<4}"
        for name in models:
            row_str += f"{dim_entropy_per_phase[name][d]:>12.3f}"
        lines.append(row_str)

    lines += [sep, '']

    # Per-speaker detail
    lines += ['PER-SPEAKER', sep]
    for r in all_results:
        lines.append(f"\n  speaker {r['speaker']}  ({r['file']})")
        lines.append(f"  {'Phase':<14} {'Ratio':>8} {'Eff kbps':>10} {'Mean H(d)':>10}")
        lines.append(f"  {'-'*44}")
        for name in models:
            if f'{name}_ratio' in r:
                lines.append(
                    f"  {name:<14} {r[f'{name}_ratio']:>8.3f}x"
                    f" {r[f'{name}_eff_kbps']:>9.2f}k"
                    f" {r[f'{name}_mean_entr']:>10.3f}"
                )

    lines.append(SEP)
    report = '\n'.join(lines)

    print(report)

    with open(out_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # CSV — one row per speaker per phase
    with open(out_dir / 'metrics.csv', 'w', encoding='utf-8') as f:
        f.write('speaker,file,phase,compression_ratio,effective_kbps,'
                'theoretical_kbps,mean_entropy,min_entropy,max_entropy,mean_utilisation\n')
        for r in all_results:
            for name in models:
                if f'{name}_ratio' not in r:
                    continue
                f.write(
                    f"{r['speaker']},{r['file']},{name},"
                    f"{r[f'{name}_ratio']:.4f},"
                    f"{r[f'{name}_eff_kbps']:.4f},"
                    f"{r[f'{name}_theo_kbps']:.4f},"
                    f"{r[f'{name}_mean_entr']:.4f},"
                    f"{r[f'{name}_min_entr']:.4f},"
                    f"{r[f'{name}_max_entr']:.4f},"
                    f"{r[f'{name}_mean_util']:.4f}\n"
                )

    # Per-dimension entropy CSV
    with open(out_dir / 'dim_entropy.csv', 'w', encoding='utf-8') as f:
        f.write('dim,' + ','.join(models.keys()) + '\n')
        for d in range(n_dims):
            row_vals = ','.join(f"{dim_entropy_per_phase[n][d]:.4f}" for n in models)
            f.write(f"{d},{row_vals}\n")

    print(f"\nreport:      {out_dir}/report.txt")
    print(f"metrics csv: {out_dir}/metrics.csv")
    print(f"entropy csv: {out_dir}/dim_entropy.csv")


if __name__ == '__main__':
    main()
