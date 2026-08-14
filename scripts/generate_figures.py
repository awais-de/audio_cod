#!/usr/bin/env python3
"""Regenerate all thesis figures from canonical comparison report files.

Usage:
  python scripts/generate_figures.py                                  # all figures
  python scripts/generate_figures.py --fig fig_02_phase_progression   # specific figures
  python scripts/generate_figures.py --list                           # list available figures
  python scripts/generate_figures.py --dpi 150                        # lower DPI for quick preview

Output: plots/<fig_name>.png  (300 DPI by default, IEEE two-column sizing)

Figure numbering follows the paper's narrative order: the codec itself (1),
headline curriculum result (2), comparison vs EnCodec — quality/complexity/
delay (3-5), the entropy-mechanism cluster (6-12), OOD/generalization and
robustness (13-17), speaker privacy (18), music (19), then figures that need
checkpoint-level inference not yet run (20-22, auto-skipped).
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
#
# Keys double as the output filename stem (plots/<key>.png) and the --fig
# CLI argument, so the number and the descriptive slug both live in one
# place. Renumbering the paper's figures means editing this dict only —
# the underlying fig_XX_* function names in figures/plots.py are internal
# and intentionally not tied to paper numbering.
# ---------------------------------------------------------------------------

FIGURES: dict[str, tuple] = {
    'fig_01_architecture':            (plots.fig_20_architecture,       'Codec architecture / streaming pipeline schematic'),
    'fig_02_phase_progression':       (plots.fig_01_phase_progression,  'Phase PESQ+STOI bar chart C→G  (n=40, 95% CI)'),
    'fig_03_rd_curve':                (plots.fig_02_rd_curve,           'R-D curve: PESQ-WB + STOI vs bitrate, EntroCodec vs EnCodec'),
    'fig_04_complexity':              (plots.fig_21_complexity,         'Model complexity: params + MACs, EntroCodec vs EnCodec'),
    'fig_05_latency':                 (plots.fig_22_latency,            'End-to-end delay: algorithmic vs measured CPU latency'),
    'fig_06_dim_entropy':             (plots.fig_03_dim_entropy,        'Per-dimension entropy bar chart, Phase G'),
    'fig_07_entropy_heatmap':         (plots.fig_10_dim_heatmap,        '32×6 entropy heatmap: H(d) across phases'),
    'fig_08_entropy_quality_scatter': (plots.fig_04_entropy_quality,    'Latent entropy vs PESQ/STOI scatter (cross-phase)'),
    'fig_09_dual_entropy':            (plots.fig_15_dual_entropy,       'Dual entropy confirmation: H̄(d) vs zlib ratio  (r=−0.9965)'),
    'fig_10_multi_coder':             (plots.fig_16_multi_coder,        'Multi-coder comparison: zlib + lzma + bz2'),
    'fig_11_entropy_ablation':        (plots.fig_17_entropy_ablation,   'Entropy penalty ablation: D vs D-VAE vs D-Entropy'),
    'fig_12_channel_ablation':        (plots.fig_18_channel_ablation,   'Channel width ablation: 16 / 32 / 64 dims'),
    'fig_13_compression':             (plots.fig_06_compression,        'zlib compression ratio + effective kbps per phase'),
    'fig_14_ood_signals':             (plots.fig_05_ood,                'OOD evaluation: bitrate + STOI by signal type'),
    'fig_15_vctk_generalization':     (plots.fig_23_vctk_generalization,'VCTK cross-corpus generalization (OOD validation)'),
    'fig_16_causality':               (plots.fig_07_causality,          'Causal (G-fixed) vs non-causal, full curriculum (G-nc) per speaker'),
    'fig_17_corruption':              (plots.fig_09_corruption,         'Bitstream corruption robustness'),
    'fig_18_speaker_probe':           (plots.fig_08_speaker_probe,      'Speaker identity probe recall (sorted)'),
    'fig_19_music_eval':              (plots.fig_19_music_eval,         'Music evaluation SI-SDR  (requires music eval run)'),
    'fig_20_attention_stats':         (plots.fig_11_attention,          '[SKIP] Attention stats — requires checkpoint inference'),
    'fig_21_quant_gap':               (plots.fig_12_quant_gap,          '[SKIP] Quantisation gap — requires float vs 3-bit inference'),
    'fig_22_attn_heatmaps':           (plots.fig_14_attn_heatmaps,      '[SKIP] Attention heatmaps — requires saved attention weights'),
    'fig_23_rd_sweep_width':          (plots.fig_24_rd_sweep_width,     'R-D sweep by latent width: PESQ-WB + STOI vs bitrate, 16/32/64-dim'),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fig', nargs='+', metavar='NAME',
                        help='Figures to generate (default: all). '
                             'E.g. --fig fig_02_phase_progression fig_12_channel_ablation')
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
        width = max(len(n) for n in FIGURES) + 2
        for name, (_, desc) in FIGURES.items():
            print(f'  {name:<{width}} {desc}')
        print()
        return

    args.out.mkdir(parents=True, exist_ok=True)

    print(f'\nLoading comparison data from {PROJECT_ROOT / "comparisons"}/ ...')
    data = load_all(PROJECT_ROOT)
    loaded = [k for k in ('metrics', 'ci', 'compression', 'rd', 'multi_coder',
                           'ood', 'speaker_probe', 'corruption', 'complexity', 'vctk', 'music')
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
