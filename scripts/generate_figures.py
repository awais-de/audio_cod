#!/usr/bin/env python3
"""Regenerate all thesis figures from canonical comparison report files.

Usage:
  python scripts/generate_figures.py              # all figures
  python scripts/generate_figures.py --fig fig_01 fig_18   # specific figures
  python scripts/generate_figures.py --list       # list available figures
  python scripts/generate_figures.py --dpi 150   # lower DPI for quick preview

Output: plots/<fig_name>.png  (300 DPI by default, IEEE two-column sizing)

Figures that require checkpoint-level inference (attention heatmaps, quantisation
gap) are skipped automatically and print instructions for generating the data first.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from figures.data_loader import load_all
from figures import plots
from figures.plots import DataNotAvailable

# ---------------------------------------------------------------------------
# Figure registry — name → (function, description)
# ---------------------------------------------------------------------------

FIGURES: dict[str, tuple] = {
    'fig_01':  (plots.fig_01_phase_progression,  'Phase PESQ+STOI bar chart C→G  (n=40, 95% CI)'),
    'fig_02':  (plots.fig_02_rd_curve,           'R-D curve: PESQ-WB + STOI vs bitrate, EntroCodec vs EnCodec'),
    'fig_03':  (plots.fig_03_dim_entropy,        'Per-dimension entropy bar chart, Phase G'),
    'fig_04':  (plots.fig_04_entropy_quality,    'Latent entropy vs PESQ/STOI scatter (cross-phase)'),
    'fig_05':  (plots.fig_05_ood,                'OOD evaluation: bitrate + STOI by signal type'),
    'fig_06':  (plots.fig_06_compression,        'zlib compression ratio + effective kbps per phase'),
    'fig_07':  (plots.fig_07_causality,          'Causal (G) vs non-causal (NC) per speaker'),
    'fig_08':  (plots.fig_08_speaker_probe,      'Speaker identity probe recall (sorted)'),
    'fig_09':  (plots.fig_09_corruption,         'Bitstream corruption robustness'),
    'fig_10':  (plots.fig_10_dim_heatmap,        '32×6 entropy heatmap: H(d) across phases'),
    'fig_11':  (plots.fig_11_attention,          '[SKIP] Attention stats — requires checkpoint inference'),
    'fig_12':  (plots.fig_12_quant_gap,          '[SKIP] Quantisation gap — requires float vs 3-bit inference'),
    'fig_14':  (plots.fig_14_attn_heatmaps,      '[SKIP] Attention heatmaps — requires saved attention weights'),
    'fig_15':  (plots.fig_15_dual_entropy,       'Dual entropy confirmation: H̄(d) vs zlib ratio  (r=−0.9965)'),
    'fig_16':  (plots.fig_16_multi_coder,        'Multi-coder comparison: zlib + lzma + bz2'),
    # MS Thesis additions
    'fig_17':  (plots.fig_17_entropy_ablation,   '[NEW] Entropy penalty ablation: D vs D-VAE vs D-Entropy'),
    'fig_18':  (plots.fig_18_channel_ablation,   '[NEW] Channel width ablation: 16 / 32 / 64 dims'),
    'fig_19':  (plots.fig_19_music_eval,         '[NEW] Music evaluation SI-SDR  (requires music eval run)'),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fig', nargs='+', metavar='NAME',
                        help='Figures to generate (default: all). E.g. --fig fig_01 fig_18')
    parser.add_argument('--out', type=Path,
                        default=PROJECT_ROOT / 'plots',
                        help='Output directory (default: plots/)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300)')
    parser.add_argument('--list', action='store_true',
                        help='List available figures and exit')
    args = parser.parse_args()

    if args.list:
        print('\nAvailable figures:\n')
        for name, (_, desc) in FIGURES.items():
            print(f'  {name:<10} {desc}')
        print()
        return

    args.out.mkdir(parents=True, exist_ok=True)

    print(f'\nLoading comparison data from {PROJECT_ROOT / "comparisons"}/ ...')
    data = load_all(PROJECT_ROOT)
    loaded = [k for k in ('metrics', 'ci', 'compression', 'rd', 'multi_coder',
                           'ood', 'speaker_probe', 'corruption', 'music')
              if k in data]
    print(f'  Loaded: {", ".join(loaded)}\n')

    targets = args.fig if args.fig else list(FIGURES)
    unknown = [t for t in targets if t not in FIGURES]
    if unknown:
        print(f'ERROR: unknown figure(s): {unknown}')
        print(f'Valid names: {list(FIGURES)}')
        sys.exit(1)

    generated, skipped = 0, 0
    for name in targets:
        fn, desc = FIGURES[name]
        print(f'  {name} — {desc}')
        try:
            fig = fn(data)
            out_path = args.out / f'{name}.png'
            fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f'           → {out_path.relative_to(PROJECT_ROOT)}')
            generated += 1
        except DataNotAvailable as e:
            print(f'           SKIP: {e}')
            skipped += 1
        except KeyError as e:
            print(f'           SKIP: missing data key {e} — source report may not exist')
            skipped += 1

    print(f'\nDone. {generated} generated, {skipped} skipped → {args.out.relative_to(PROJECT_ROOT)}/')


if __name__ == '__main__':
    main()
