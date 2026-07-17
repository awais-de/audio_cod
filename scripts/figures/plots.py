"""One function per figure. Each accepts the full data dict and returns a Figure.

Figures that require checkpoint-level inference (attention, quantisation gap)
raise DataNotAvailable with instructions for how to generate the data first.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from . import style

style.apply()


class DataNotAvailable(RuntimeError):
    """Raised when a figure needs data that can't be derived from comparison reports."""


# ---------------------------------------------------------------------------
# fig_01 — Phase progression PESQ + STOI (core 32-dim curriculum)
# ---------------------------------------------------------------------------

def fig_01_phase_progression(data: dict) -> plt.Figure:
    ci = data['ci']
    phases = [p for p in ['C', 'D', 'D-VAE', 'E', 'F', 'G'] if p in ci]

    pesq  = [ci[p]['pesq']    for p in phases]
    p_lo  = [ci[p]['pesq'] - ci[p]['pesq_lo'] for p in phases]
    p_hi  = [ci[p]['pesq_hi'] - ci[p]['pesq'] for p in phases]
    stoi  = [ci[p]['stoi']    for p in phases]
    s_lo  = [ci[p]['stoi'] - ci[p]['stoi_lo'] for p in phases]
    s_hi  = [ci[p]['stoi_hi'] - ci[p]['stoi'] for p in phases]
    colors = [style.PHASE_COLORS[p] for p in phases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(style.COL2_W, style.ROW_H))
    x = np.arange(len(phases))
    w = 0.55

    for ax, vals, errs_lo, errs_hi, ylabel, ylim in [
        (ax1, pesq, p_lo, p_hi, 'PESQ-WB', (1.10, 1.32)),
        (ax2, stoi, s_lo, s_hi, 'STOI',    (0.68, 0.83)),
    ]:
        bars = ax.bar(x, vals, w, color=colors, edgecolor='white', linewidth=0.5)
        ax.errorbar(x, vals, yerr=[errs_lo, errs_hi], fmt='none',
                    color='#333', linewidth=0.8, capsize=2)
        for bar, v, err_hi in zip(bars, vals, errs_hi):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err_hi,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=6.5, color='#333')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, axis='y')
        # Annotate D-VAE drop
        dvae_idx = phases.index('D-VAE') if 'D-VAE' in phases else None
        if dvae_idx is not None:
            ax.annotate('KL↓', xy=(dvae_idx, vals[dvae_idx]),
                        xytext=(dvae_idx, ylim[0] + 0.02 * (ylim[1] - ylim[0])),
                        ha='center', fontsize=6, color=style.PHASE_COLORS['D-VAE'],
                        arrowprops=dict(arrowstyle='->', color=style.PHASE_COLORS['D-VAE'], lw=0.8))

    fig.suptitle('Quality across the training curriculum  (n=40, 95% CI)',
                 fontsize=9, fontweight='bold')
    return fig


# ---------------------------------------------------------------------------
# fig_01b — STOI all phases A→G (A/B hatched: PESQ unavailable)
# ---------------------------------------------------------------------------

def fig_01b_stoi_all_phases(data: dict) -> plt.Figure:
    ci = data.get('ci', {})
    ab = data.get('phase_ab', {})

    phase_order = ['A', 'B', 'C', 'D', 'D-VAE', 'E', 'F', 'G']
    stoi_vals, hatches = [], []
    for p in phase_order:
        if p in ab:
            stoi_vals.append(ab[p]['stoi'])
            hatches.append('//')
        elif p in ci:
            stoi_vals.append(ci[p]['stoi'])
            hatches.append('')
        else:
            stoi_vals.append(None)
            hatches.append('')

    fig, ax = plt.subplots(figsize=(style.COL2_W, style.ROW_H))
    x = np.arange(len(phase_order))
    w = 0.6
    for i, (p, v, h) in enumerate(zip(phase_order, stoi_vals, hatches)):
        if v is None:
            continue
        color = style.PHASE_COLORS.get(p, '#888888')
        ax.bar(i, v, w, color=color, hatch=h, edgecolor='white', linewidth=0.5)
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=6.5)

    ab_patch = mpatches.Patch(facecolor='#999', hatch='//', label='A/B: STOI only (PESQ unavailable)')
    style.legend(fig, handles=[ab_patch], labels=[ab_patch.get_label()])
    ax.set_xticks(x)
    ax.set_xticklabels(phase_order)
    ax.set_ylabel('STOI')
    ax.set_ylim(0.48, 0.86)
    ax.grid(True, axis='y')
    ax.set_title('STOI across all phases  (A/B: 5-speaker; C–G: n=40)', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# fig_02 — R-D curve PESQ-WB
# ---------------------------------------------------------------------------

def fig_02_rd_curve_pesq(data: dict) -> plt.Figure:
    return _rd_curve(data, metric='pesq')


# ---------------------------------------------------------------------------
# fig_03 — Per-dimension entropy bar chart, Phase G
# ---------------------------------------------------------------------------

def fig_03_dim_entropy(data: dict) -> plt.Figure:
    per_dim = data['per_dim_h']
    vals = per_dim.get('G', [])
    if not vals:
        raise DataNotAvailable('per_dim_h["G"] is empty — run compression analysis first')

    fig, ax = plt.subplots(figsize=(style.COL2_W, style.ROW_H))
    x = np.arange(len(vals))
    colors = [style.PHASE_COLORS['D-VAE'] if v == min(vals) else style.PHASE_COLORS['G'] for v in vals]
    ax.bar(x, vals, 0.8, color=colors, edgecolor='white', linewidth=0.3)
    ax.axhline(np.mean(vals), color='#444', linewidth=0.8, linestyle='--',
               label=f'Mean {np.mean(vals):.3f} bits')
    ax.axhline(3.0, color='#aaa', linewidth=0.6, linestyle=':',
               label='Max (3.0 bits = uniform)')
    min_idx = int(np.argmin(vals))
    # Placed above, in the clear headroom over the bars, rather than off to the
    # side where it used to overlap neighbouring bars; darker red for contrast
    # against white (the bar's own salmon fill reads too faint at this size).
    ax.annotate(f'dim {min_idx}\n{min(vals):.3f} bits',
                xy=(min_idx, vals[min_idx]),
                xytext=(min_idx, 1.98),
                fontsize=6.5, ha='center', va='bottom', color='#a83232',
                arrowprops=dict(arrowstyle='->', color='#a83232', lw=0.8))
    ax.set_xlabel('Latent dimension index')
    ax.set_ylabel('Shannon entropy (bits)')
    ax.set_title('Per-dimension entropy — Phase G', fontsize=9)
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_ylim(0.8, 2.2)
    style.legend(fig, ax=ax)
    ax.grid(True, axis='y')
    return fig


# ---------------------------------------------------------------------------
# fig_04 — Entropy vs PESQ/STOI scatter (cross-phase)
# ---------------------------------------------------------------------------

def fig_04_entropy_quality(data: dict) -> plt.Figure:
    ci = data['ci']
    comp = data.get('compression', {})

    # Use phases that appear in both sources
    phases = [p for p in ['C', 'D', 'D-VAE', 'E', 'F', 'G'] if p in ci and p in comp]
    if not phases:
        raise DataNotAvailable('Need both CI and compression data')

    entropy = [comp[p]['mean_h'] for p in phases]
    pesq    = [ci[p]['pesq']    for p in phases]
    stoi    = [ci[p]['stoi']    for p in phases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(style.COL2_W, style.ROW_H))
    for ax, vals, ylabel in [(ax1, pesq, 'PESQ-WB'), (ax2, stoi, 'STOI')]:
        non_vae = [(h, v) for p, h, v in zip(phases, entropy, vals) if p != 'D-VAE']
        if len(non_vae) > 1:
            xs, ys = zip(*non_vae)
            z = np.polyfit(xs, ys, 1)
            xr = np.linspace(min(xs), max(xs), 50)
            ax.plot(xr, np.polyval(z, xr), '--', color='#aaa', linewidth=0.8)
        for p, h, v in zip(phases, entropy, vals):
            c = style.PHASE_COLORS[p]
            ax.scatter(h, v, color=c, s=60, zorder=5, edgecolors='white', linewidths=0.5)
            ax.annotate(p, (h, v), textcoords='offset points', xytext=(4, 2), fontsize=6)
        ax.set_xlabel('Mean H(d) per dimension (bits)')
        ax.set_ylabel(ylabel)
        ax.grid(True)
    fig.suptitle('Latent entropy vs reconstruction quality (cross-phase)', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# fig_05 — OOD evaluation
# ---------------------------------------------------------------------------

def fig_05_ood(data: dict) -> plt.Figure:
    rows = data['ood']
    labels = [r['label'] for r in rows]
    kbps   = [r['kbps']  for r in rows]
    stois  = [r['stoi']  for r in rows]
    colors = [style.PHASE_COLORS['G'] if r['is_speech'] else style.PHASE_COLORS['D-Entropy']
              for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(style.COL2_W, style.ROW_H + 0.5))
    cap_line = None
    for ax, vals, xlabel, xlim, lbl in [
        (ax1, kbps,  'Effective bitrate (kbps)', (0, 10.5), True),
        (ax2, stois, 'STOI',                     (-0.1, 1.0), False),
    ]:
        bars = ax.barh(labels, vals, color=colors, edgecolor='white', linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(max(bar.get_width(), 0) + 0.05 * xlim[1],
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:.2f}', va='center', ha='left', fontsize=6.5)
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.grid(True, axis='x')
        if lbl:
            cap_line = ax.axvline(9.6, color='#aaa', linestyle=':', linewidth=0.8,
                                   label='9.6 kbps cap')

    # ax2 repeats ax1's category labels by default — drop them so they don't
    # bleed into the middle of the figure, on top of ax1's bars.
    ax2.set_yticklabels([])

    legend_els = [
        mpatches.Patch(color=style.PHASE_COLORS['G'],         label='Speech signal'),
        mpatches.Patch(color=style.PHASE_COLORS['D-Entropy'], label='Synthetic signal'),
        cap_line,
    ]
    style.legend(fig, handles=legend_els, labels=[h.get_label() for h in legend_els], ncol=3)
    fig.suptitle('OOD evaluation — Phase G', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# fig_06 — Compression ratio + effective kbps per phase
# ---------------------------------------------------------------------------

def fig_06_compression(data: dict) -> plt.Figure:
    comp = data['compression']
    ci   = data.get('ci', {})
    phases = [p for p in ['C', 'D', 'D-VAE', 'E', 'F', 'G'] if p in comp]
    ratios  = [comp[p]['ratio']    for p in phases]
    kbps    = [comp[p]['eff_kbps'] for p in phases]
    colors  = [style.PHASE_COLORS[p] for p in phases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(style.COL2_W, style.ROW_H))
    x = np.arange(len(phases))
    w = 0.55
    for ax, vals, ylabel in [(ax1, ratios, 'zlib compression ratio'), (ax2, kbps, 'Effective kbps')]:
        bars = ax.bar(x, vals, w, color=colors, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(vals),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y')
    fig.suptitle('Bitstream compression per phase', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# fig_07 — Causality ablation: G vs NC per-speaker
# ---------------------------------------------------------------------------

def fig_07_causality(data: dict) -> plt.Figure:
    metrics = data.get('metrics')
    if metrics is None:
        raise DataNotAvailable('metrics DataFrame not loaded')
    g  = metrics[metrics['phase'] == 'G'].set_index('speaker')
    nc = metrics[metrics['phase'] == 'NC'].set_index('speaker')
    common = g.index.intersection(nc.index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(style.COL2_W, style.ROW_H))
    for ax, col, label in [(ax1, 'pesq_wb', 'PESQ-WB'), (ax2, 'stoi', 'STOI')]:
        g_vals  = g.loc[common, col].values
        nc_vals = nc.loc[common, col].values
        ax.scatter(nc_vals, g_vals, s=20, color=style.PHASE_COLORS['G'],
                   edgecolors='white', linewidths=0.4, zorder=5)
        lo = min(g_vals.min(), nc_vals.min()) - 0.02
        hi = max(g_vals.max(), nc_vals.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], '--', color='#aaa', linewidth=0.7)
        ax.set_xlabel(f'Phase NC  {label}')
        ax.set_ylabel(f'Phase G  {label}')
        ax.grid(True)
    fig.suptitle('Causal (G) vs non-causal (NC) — per speaker  (n=40, ns)', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# fig_08 — Speaker probe recall (sorted)
# ---------------------------------------------------------------------------

def fig_08_speaker_probe(data: dict) -> plt.Figure:
    probe = data['speaker_probe']
    rows  = sorted(probe['per_speaker'], key=lambda r: r['recall'])
    spks  = [r['speaker'] for r in rows]
    recs  = [r['recall']  for r in rows]
    colors = ['#d65f5f' if v == 0 else ('#27ae60' if v == 100 else style.PHASE_COLORS['G'])
              for v in recs]

    fig, ax = plt.subplots(figsize=(style.COL1_W, max(2.5, len(spks) * 0.22)))
    ax.barh(spks, recs, 0.7, color=colors, edgecolor='white', linewidth=0.4)
    ax.axvline(3.125, color='#aaa', linestyle='--', linewidth=0.8, label='Chance (3.1%)')
    ax.axvline(probe['accuracy'], color=style.PHASE_COLORS['D-Entropy'], linestyle='-',
               linewidth=0.9, label=f'Mean ({probe["accuracy"]:.1f}%)')
    ax.set_xlabel('Recall (%)')
    ax.set_xlim(0, 115)
    style.legend(fig, ax=ax)
    ax.grid(True, axis='x')
    ax.set_title('Speaker identity probe — Phase G', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# fig_09 — Corruption robustness
# ---------------------------------------------------------------------------

def fig_09_corruption(data: dict) -> plt.Figure:
    rows = data['corruption']
    rates   = [r['rate']    for r in rows]
    success = [r['success'] for r in rows]

    fig, ax = plt.subplots(figsize=(style.COL1_W, style.ROW_H))
    ax.step(rates, success, where='post', color=style.PHASE_COLORS['G'], linewidth=1.5)
    ax.fill_between(rates, success, step='post', color=style.PHASE_COLORS['G'], alpha=0.15)
    nonzero = [(r, s) for r, s in zip(rates, success) if r > 0]
    if nonzero:
        ax.set_xscale('log')
    ax.set_xlabel('Byte corruption rate (log scale)')
    ax.set_ylabel('Decompression success (%)')
    ax.set_ylim(-5, 115)
    ax.grid(True, which='both')
    ax.set_title('Bitstream corruption robustness  (zlib CRC)', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# fig_10 — Per-dim entropy heatmap 32 dims × 6 phases
# ---------------------------------------------------------------------------

def fig_10_dim_heatmap(data: dict) -> plt.Figure:
    per_dim = data['per_dim_h']
    phases  = [p for p in ['C', 'D', 'D-VAE', 'E', 'F', 'G'] if p in per_dim]
    mat = np.array([per_dim[p] for p in phases])

    fig, ax = plt.subplots(figsize=(style.COL2_W, style.ROW_H + 0.3))
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0.6, vmax=1.9)
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(phases, fontsize=7)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([str(i) for i in range(mat.shape[1])], fontsize=5)
    ax.set_xlabel('Latent dimension index')
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label='H(d) bits')
    ax.set_title('Per-dimension entropy across phases  (max = 3 bits)', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# fig_11 — Attention statistics (requires checkpoint inference)
# ---------------------------------------------------------------------------

def fig_11_attention(_data: dict) -> plt.Figure:
    raise DataNotAvailable(
        'fig_11 requires running scripts/12_attention_stats.py against the Phase G checkpoint. '
        'Run: python scripts/12_attention_stats.py --checkpoint checkpoints_active/temporal_phaseG/best.pt '
        '--output comparisons/attention_stats/report.txt'
    )


# ---------------------------------------------------------------------------
# fig_12 — Quantisation gap (requires float vs 3-bit inference)
# ---------------------------------------------------------------------------

def fig_12_quant_gap(_data: dict) -> plt.Figure:
    raise DataNotAvailable(
        'fig_12 requires float and 3-bit inference against Phase G. '
        'Run scripts/quantisation_gap.py first to produce comparisons/quant_gap/report.txt'
    )


# ---------------------------------------------------------------------------
# fig_13 — R-D curve STOI
# ---------------------------------------------------------------------------

def fig_13_rd_curve_stoi(data: dict) -> plt.Figure:
    return _rd_curve(data, metric='stoi')


# ---------------------------------------------------------------------------
# fig_14a/b/c — Attention heatmaps (require per-utterance attention weights)
# ---------------------------------------------------------------------------

def fig_14_attn_heatmaps(_data: dict) -> plt.Figure:
    raise DataNotAvailable(
        'fig_14 requires per-utterance attention weights. '
        'Run scripts/12_attention_stats.py with --save-heatmaps flag.'
    )


# ---------------------------------------------------------------------------
# fig_15 — Dual entropy confirmation: H̄(d) vs zlib ratio
# ---------------------------------------------------------------------------

def fig_15_dual_entropy(data: dict) -> plt.Figure:
    comp   = data['compression']
    phases = [p for p in ['C', 'D', 'D-VAE', 'E', 'F', 'G'] if p in comp]
    entropy = [comp[p]['mean_h'] for p in phases]
    ratio   = [comp[p]['ratio']  for p in phases]

    z = np.polyfit(entropy, ratio, 1)
    xr = np.linspace(min(entropy), max(entropy), 50)

    fig, ax = plt.subplots(figsize=(style.COL1_W, style.ROW_H))
    ax.plot(xr, np.polyval(z, xr), '--', color='#aaa', linewidth=0.8)
    for p, h, r in zip(phases, entropy, ratio):
        ax.scatter(h, r, color=style.PHASE_COLORS[p], s=60, zorder=5,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(p, (h, r), textcoords='offset points', xytext=(4, 2), fontsize=6)
    ax.set_xlabel('Mean H(d) per dimension (bits)')
    ax.set_ylabel('zlib compression ratio')
    ax.grid(True)
    # Compute and display Pearson r
    r_val = np.corrcoef(entropy, ratio)[0, 1]
    ax.text(0.97, 0.97, f'r = {r_val:.4f}', transform=ax.transAxes,
            fontsize=7, ha='right', va='top')
    ax.set_title('Dual entropy confirmation: H̄(d) vs zlib ratio', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# fig_16 — Multi-coder comparison
# ---------------------------------------------------------------------------

def fig_16_multi_coder(data: dict) -> plt.Figure:
    rows   = data['multi_coder']
    phases = [r['phase'] for r in rows]
    zlib   = [r['zlib']  for r in rows]
    lzma   = [r['lzma']  for r in rows]
    bz2    = [r['bz2']   for r in rows]

    x  = np.arange(len(phases))
    w  = 0.25
    c1 = style.PHASE_COLORS['G']
    c2 = style.PHASE_COLORS['E']
    c3 = style.PHASE_COLORS['D']

    fig, ax = plt.subplots(figsize=(style.COL2_W, style.ROW_H))
    ax.bar(x - w,   zlib, w, color=c1, label='zlib',  edgecolor='white', linewidth=0.4)
    ax.bar(x,       lzma, w, color=c2, label='lzma',  edgecolor='white', linewidth=0.4)
    ax.bar(x + w,   bz2,  w, color=c3, label='bz2',   edgecolor='white', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel('Compression ratio')
    style.legend(fig, ax=ax)
    ax.grid(True, axis='y')
    ax.set_title('Compression ratio across coders — D-VAE ranks highest in all', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# fig_17 — Entropy penalty ablation: D vs D-VAE vs D-Entropy
# ---------------------------------------------------------------------------

def fig_17_entropy_ablation(data: dict) -> plt.Figure:
    ci = data['ci']
    phases = [p for p in ['D', 'D-VAE', 'D-Entropy'] if p in ci]

    # ylim tuned so bar differences are clearly visible
    ylims = {
        'pesq': (1.08, 1.28),
        'stoi': (0.66, 0.79),
        'kbps': (0.0, 6.5),
    }

    fig, axes = plt.subplots(1, 3, figsize=(style.COL2_W, style.ROW_H),
                              gridspec_kw={'wspace': 0.45})
    for ax, metric, label in [
        (axes[0], 'pesq', 'PESQ-WB'),
        (axes[1], 'stoi', 'STOI'),
        (axes[2], 'kbps', 'Eff. kbps'),
    ]:
        vals   = [ci[p][metric]  for p in phases]
        colors = [style.PHASE_COLORS[p] for p in phases]
        errs_lo = [ci[p][metric] - ci[p].get(f'{metric}_lo', ci[p][metric]) for p in phases]
        errs_hi = [ci[p].get(f'{metric}_hi', ci[p][metric]) - ci[p][metric] for p in phases]
        lo, hi  = ylims[metric]
        bars = ax.bar(range(len(phases)), vals, 0.55, color=colors,
                      edgecolor='white', linewidth=0.5, bottom=0)
        if metric != 'kbps':
            ax.errorbar(range(len(phases)), vals, yerr=[errs_lo, errs_hi],
                        fmt='none', color='#333', linewidth=0.8, capsize=2)
        span = hi - lo
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01 * span,
                    f'{v:.3f}' if metric != 'kbps' else f'{v:.2f}',
                    ha='center', va='bottom', fontsize=6.5)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=7)
        ax.set_ylabel(label)
        ax.set_ylim(lo, hi)
        ax.grid(True, axis='y')

    fig.suptitle('Entropy penalty ablation: KL (D-VAE) vs soft entropy (D-Entropy) vs base (D)\n'
                 '(n=40, 95% CI; both vs D: p<0.0001***)', fontsize=8.5)
    return fig


# ---------------------------------------------------------------------------
# fig_18 — Channel width ablation: 16 / 32 / 64 dims
# ---------------------------------------------------------------------------

def fig_18_channel_ablation(data: dict) -> plt.Figure:
    ci = data['ci']

    # Full progression for each width
    widths = {
        '16-dim': (['C-16', 'D-16', 'E-16', 'F-16', 'G-16'], '#4472c8'),
        '32-dim': (['C',    'D',    'E',    'F',    'G'   ], style.PHASE_COLORS['G']),
        '64-dim': (['C-64', 'D-64', 'E-64', 'F-64', 'G-64'], '#003c5a'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(style.COL2_W, style.ROW_H))
    stage_labels = ['C', 'D', 'E', 'F', 'G']
    x = np.arange(len(stage_labels))
    w = 0.25
    offsets = {'16-dim': -w, '32-dim': 0, '64-dim': +w}

    for ax, metric, ylabel in [
        (axes[0], 'pesq', 'PESQ-WB'),
        (axes[1], 'stoi', 'STOI'),
    ]:
        for width_label, (phases, color) in widths.items():
            vals = [ci[p][metric] if p in ci else float('nan') for p in phases]
            offset = offsets[width_label]
            ax.bar(x + offset, vals, w, color=color, edgecolor='white',
                   linewidth=0.4, label=width_label, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.set_xlabel('Curriculum phase')
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y')

    style.legend(fig, ax=axes[0], ncol=3)

    fig.suptitle('Channel width ablation: 16 / 32 / 64 bottleneck dimensions\n'
                 '(G-16 vs G: p<0.0001***;  G-64 vs G: p=0.006**;  n=40)',
                 fontsize=8.5)
    return fig


# ---------------------------------------------------------------------------
# fig_19 — Music evaluation (SI-SDR)
# ---------------------------------------------------------------------------

def fig_19_music_eval(data: dict) -> plt.Figure:
    music = data.get('music')
    if music is None:
        raise DataNotAvailable(
            'Music eval results not found. '
            'Run: python scripts/eval_music.py --dataset <path/to/MUSDB18-HQ>'
        )
    phase_order = [p for p in ['C', 'D', 'D-VAE', 'G'] if p in music['phase'].values]
    grp    = music.groupby('phase')
    si_sdr = grp['si_sdr'].mean().reindex(phase_order)
    ratio  = grp['zlib_ratio'].mean().reindex(phase_order)
    colors = [style.PHASE_COLORS.get(p, '#888888') for p in phase_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(style.COL2_W, style.ROW_H),
                                   constrained_layout=True)
    x = np.arange(len(phase_order))
    w = 0.55

    # Left: SI-SDR mean per phase (variance is track-difficulty-dominated, not shown)
    bars1 = ax1.bar(x, si_sdr.values, w, color=colors, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars1, si_sdr.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, v - 0.12,
                 f'{v:.2f}', ha='center', va='top', fontsize=6.5, color='white')
    ax1.set_xticks(x); ax1.set_xticklabels(phase_order)
    ax1.set_ylabel('SI-SDR (dB)')
    ax1.set_ylim(-9.5, 0.5)
    ax1.grid(True, axis='y')
    ax1.text(0.97, 0.97, 'higher = better', transform=ax1.transAxes,
             fontsize=6, color='#555', va='top', ha='right')

    # Right: zlib compression ratio
    bars2 = ax2.bar(x, ratio.values, w, color=colors, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars2, ratio.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{v:.3f}×', ha='center', va='bottom', fontsize=6.5)
    ax2.set_xticks(x); ax2.set_xticklabels(phase_order)
    ax2.set_ylabel('zlib compression ratio')
    ax2.set_ylim(1.0, 1.6)
    ax2.grid(True, axis='y')
    ax2.text(0.97, 0.04, 'higher = more compressed', transform=ax2.transAxes,
             fontsize=6, color='#555', va='bottom', ha='right')

    fig.suptitle('Music eval — MUSDB18-HQ test set, n=40 tracks\n'
                 'D-VAE: highest compression + lowest quality  (p<0.0001***)',
                 fontsize=8.5)
    return fig


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _rd_curve(data: dict, metric: str) -> plt.Figure:
    rd = data['rd']
    ours    = rd['ours']
    encodec = rd['encodec']

    our_kbps  = [r['eff_kbps'] for r in ours]
    our_vals  = [r[metric]     for r in ours]
    enc_kbps  = [r['kbps']     for r in encodec]
    enc_vals  = [r[metric]     for r in encodec]
    trained_idx = next(i for i, r in enumerate(ours) if r['bits'] == 3)

    ylabel = 'PESQ-WB' if metric == 'pesq' else 'STOI'
    fig, ax = plt.subplots(figsize=(style.COL1_W, style.ROW_H))
    ax.plot(our_kbps, our_vals, 'o-', color=style.PHASE_COLORS['G'], linewidth=1.2,
            markersize=4, label='Ours (Phase G, SQ)')
    # 1-bit and 2-bit points sit only ~0.2 kbps apart — label them directly so
    # the pair doesn't read as a single marker.
    ax.annotate('1-bit', (our_kbps[0], our_vals[0]), textcoords='offset points',
                xytext=(-8, 7), fontsize=6, ha='right', va='bottom',
                color=style.PHASE_COLORS['G'])
    ax.annotate('2-bit', (our_kbps[1], our_vals[1]), textcoords='offset points',
                xytext=(8, 7), fontsize=6, ha='left', va='bottom',
                color=style.PHASE_COLORS['G'])
    ax.scatter([our_kbps[trained_idx]], [our_vals[trained_idx]],
               color=style.PHASE_COLORS['G'], s=60, zorder=6, marker='*',
               label=f'Trained (3-bit, {our_kbps[trained_idx]:.1f} kbps)')
    ax.plot(enc_kbps, enc_vals, 's--', color=style.PHASE_COLORS['D-VAE'], linewidth=1.0,
            markersize=4, label='EnCodec (RVQ)')
    ax.set_xlabel('Effective bitrate (kbps)')
    ax.set_ylabel(ylabel)
    style.legend(fig, ax=ax)
    ax.grid(True)
    ax.set_title(f'Rate-distortion: {ylabel}', fontsize=9)
    return fig
