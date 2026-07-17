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
# fig_02 — R-D curve: PESQ-WB and STOI vs bitrate, ours vs EnCodec
# ---------------------------------------------------------------------------

def fig_02_rd_curve(data: dict) -> plt.Figure:
    rd = data['rd']
    ours    = rd['ours']
    encodec = rd['encodec']

    our_kbps    = [r['eff_kbps'] for r in ours]
    enc_kbps    = [r['kbps']     for r in encodec]
    trained_idx = next(i for i, r in enumerate(ours) if r['bits'] == 3)

    fig, axes = plt.subplots(1, 2, figsize=(style.COL2_W, style.ROW_H))
    for ax, metric, ylabel in [(axes[0], 'pesq', 'PESQ-WB'), (axes[1], 'stoi', 'STOI')]:
        our_vals = [r[metric] for r in ours]
        enc_vals = [r[metric] for r in encodec]

        ax.plot(our_kbps, our_vals, 'o-', color=style.PHASE_COLORS['G'], linewidth=1.2,
                markersize=4, label='Ours (Phase G, SQ)')
        # 1-bit and 2-bit points sit only ~0.2 kbps apart — label them
        # directly so the pair doesn't read as a single marker.
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
        ax.set_title(ylabel, fontsize=9)
        ax.grid(True)

    style.legend(fig, ax=axes[0], ncol=3)
    fig.suptitle('Rate-distortion: ours vs EnCodec', fontsize=9)
    return fig


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
    # Scaled to include the 3.0-bit (uniform) reference line rather than
    # cropping it out — the whole point of that line is to show how far
    # these dims sit from the theoretical max.
    ax.set_ylim(0.8, 3.15)
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
        ax.set_xlabel('Mean H(d) per dimension (bits)')
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # Phase labels used to sit next to each point — with several phases
    # clustered close together in H(d), the text overlapped. A shared legend
    # below avoids that regardless of how tight the cluster is.
    legend_handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=6,
                              markerfacecolor=style.PHASE_COLORS[p], markeredgecolor='white',
                              label=p) for p in phases]
    style.legend(fig, handles=legend_handles, labels=phases, ncol=len(phases))
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
    for ax, vals, xlabel, xlim, lbl in [
        (ax1, kbps,  'Effective bitrate (kbps)', (0, 12.0), True),
        (ax2, stois, 'STOI',                     (-0.1, 1.0), False),
    ]:
        bars = ax.barh(labels, vals, color=colors, edgecolor='white', linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(max(bar.get_width(), 0) + 0.02 * xlim[1],
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:.2f}', va='center', ha='left', fontsize=6.5)
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.grid(True, axis='x')
        if lbl:
            # Annotated directly rather than via the legend — it's only
            # relevant to this panel, not to the STOI panel alongside it.
            ax.axvline(9.6, color='#aaa', linestyle=':', linewidth=0.8)
            # Pushed just clear of the widest bar-end value label ("8.30").
            ax.text(9.85, 0.97, '9.6 kbps cap', transform=ax.get_xaxis_transform(),
                    rotation=90, va='top', ha='left', fontsize=6, color='#888')

    # ax2 repeats ax1's category labels by default — drop them so they don't
    # bleed into the middle of the figure, on top of ax1's bars.
    ax2.set_yticklabels([])

    legend_els = [
        mpatches.Patch(color=style.PHASE_COLORS['G'],         label='Speech signal'),
        mpatches.Patch(color=style.PHASE_COLORS['D-Entropy'], label='Synthetic signal'),
    ]
    style.legend(fig, handles=legend_els, labels=[h.get_label() for h in legend_els], ncol=2)
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
        # Same range on both axes (not just auto-scaled independently) so the
        # dashed y=x reference line is an actual diagonal, not a distorted one.
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Non-causal  (Phase NC)')
        ax.set_ylabel('Causal  (Phase G)')
        ax.grid(True)
    fig.suptitle('Causal vs non-causal — per speaker  (n=40, ns)', fontsize=9)
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
    ax.axvline(3.125, color='#aaa', linestyle='--', linewidth=0.8, label='Chance')
    ax.axvline(probe['accuracy'], color=style.PHASE_COLORS['D-Entropy'], linestyle='-',
               linewidth=0.9, label='Mean')
    # Values shown on the chart itself, next to their line, in the empty
    # bottom corner (lowest-recall speakers have no/short bars there) —
    # the legend just needs to say which line is which, not repeat the number.
    ax.annotate('3.1%', xy=(3.125, 0.015), xycoords=ax.get_xaxis_transform(),
                xytext=(4, 0), textcoords='offset points',
                ha='left', va='bottom', fontsize=6.5, color='#888')
    ax.annotate(f'{probe["accuracy"]:.1f}%', xy=(probe['accuracy'], 0.015),
                xycoords=ax.get_xaxis_transform(), xytext=(4, 0), textcoords='offset points',
                ha='left', va='bottom', fontsize=6.5, color=style.PHASE_COLORS['D-Entropy'])
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
    ax.set_xlabel('Mean H(d) per dimension (bits)')
    ax.set_ylabel('zlib compression ratio')
    ax.grid(True)
    # Compute and display Pearson r
    r_val = np.corrcoef(entropy, ratio)[0, 1]
    ax.text(0.97, 0.97, f'r = {r_val:.4f}', transform=ax.transAxes,
            fontsize=7, ha='right', va='top')
    ax.set_title('Dual entropy confirmation: H̄(d) vs zlib ratio', fontsize=9)

    # Phase labels used to sit next to each point — C/D and E/F/G cluster
    # tightly together in H(d), so the text overlapped. A shared legend
    # below avoids that regardless of how tight the cluster is.
    legend_handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=6,
                              markerfacecolor=style.PHASE_COLORS[p], markeredgecolor='white',
                              label=p) for p in phases]
    style.legend(fig, handles=legend_handles, labels=phases, ncol=len(phases))
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
    ax.bar(x - w,   zlib, w, color=c1, label='zlib (baseline)', edgecolor='#c0392b', linewidth=1.4)
    ax.bar(x,       lzma, w, color=c2, label='lzma',  edgecolor='white', linewidth=0.4)
    ax.bar(x + w,   bz2,  w, color=c3, label='bz2',   edgecolor='white', linewidth=0.4)
    # zlib is the baseline used everywhere else in this analysis (fig_06,
    # fig_15) — outlined in red, with a flat bracket to bz2 (straight, not a
    # diagonal connecting two different bar heights) marking the gain the
    # strongest coder offers over that baseline, for every phase.
    for xi, (z, l, b) in enumerate(zip(zlib, lzma, bz2)):
        y_bar = max(z, l, b) + 0.06
        ax.plot([xi - w, xi - w, xi + w, xi + w], [z, y_bar, y_bar, b],
                color='#c0392b', linewidth=0.9, zorder=6)
        ax.text(xi, y_bar + 0.015, f'Δ {b - z:.3f}', ha='center', va='bottom',
                fontsize=6.5, color='#c0392b')
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
                              gridspec_kw={'wspace': 0.32})
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
        # "D-VAE" and "D-Entropy" are wide enough at this bar spacing to run
        # into each other unrotated ("D-VAED-Entropy") — angle them instead.
        ax.set_xticklabels(phases, fontsize=7, rotation=20, ha='right')
        ax.set_ylabel(label)
        ax.set_ylim(lo, hi)
        ax.grid(True, axis='y')

    fig.suptitle('Penalizing latent entropy trades quality for bitrate\n'
                 '(D: no penalty · D-VAE: KL penalty · D-Entropy: soft-entropy penalty; '
                 'both vs D: p<0.0001***, n=40)', fontsize=8.5)
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
        vals = {}
        for width_label, (phases, color) in widths.items():
            v = [ci[p][metric] if p in ci else float('nan') for p in phases]
            vals[width_label] = v
            offset = offsets[width_label]
            baseline = width_label == '32-dim'
            ax.bar(x + offset, v, w, color=color, alpha=0.85, label=width_label,
                   edgecolor='#c0392b' if baseline else 'white',
                   linewidth=1.4 if baseline else 0.4)

        # 32-dim is the baseline used throughout the rest of the thesis —
        # outlined in red, with brackets to 16-dim and 64-dim (the two
        # widths the experiment is actually about) marking the local
        # gain/loss against it, per phase. Labels sit over their own outer
        # bar (16-dim / 64-dim) rather than at the bracket's midpoint next
        # to the shared baseline bar — that's twice the horizontal room
        # (0.5 units vs 0.25) and is what stopped the two deltas colliding.
        # Colored by direction relative to baseline: red = 16-dim loses
        # quality, green = 64-dim gains it.
        loss_color, gain_color = '#c0392b', '#27ae60'
        v16, v32, v64 = vals['16-dim'], vals['32-dim'], vals['64-dim']
        span = max(v16 + v32 + v64) - min(v16 + v32 + v64)
        lift, pad = 0.1 * span, 0.06 * span
        for xi, (a, m, b) in enumerate(zip(v16, v32, v64)):
            y_bar = max(a, m, b) + lift
            ax.plot([xi - w, xi - w, xi, xi], [a, y_bar, y_bar, m],
                    color=loss_color, linewidth=0.9, zorder=6)
            # No "Δ" prefix — color already tells the reader which
            # comparison this is, and the shorter string leaves more gap
            # before the neighbouring phase group's own labels. Anchored at
            # 1.0*w (not further out) — pushing labels out past their own
            # bar shrinks the gap to the *next* group's facing label just as
            # much as it grows the gap within this group, so it's a wash;
            # only a shorter string (or smaller font) actually helps.
            ax.text(xi - w, y_bar + pad, f'{a - m:+.2f}', ha='center',
                    va='bottom', fontsize=5, color=loss_color)
            ax.plot([xi, xi, xi + w, xi + w], [m, y_bar, y_bar, b],
                    color=gain_color, linewidth=0.9, zorder=6)
            ax.text(xi + w, y_bar + pad, f'{b - m:+.2f}', ha='center',
                    va='bottom', fontsize=5, color=gain_color)

        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.set_xlabel('Curriculum phase')
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y')

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_labels = [l if l != '32-dim' else '32-dim (baseline)' for l in legend_labels]
    style.legend(fig, handles=legend_handles, labels=legend_labels, ncol=3)

    fig.suptitle('Bottleneck width vs quality, relative to the 32-dim baseline\n'
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
        # Just past the bar tip, dark text — matches how the ratio panel
        # labels its bars, instead of pale text sitting inside the bar.
        ax1.text(bar.get_x() + bar.get_width() / 2, v - 0.15,
                 f'{v:.2f}', ha='center', va='top', fontsize=6.5, color='#333')
    ax1.set_xticks(x); ax1.set_xticklabels(phase_order)
    ax1.set_ylabel('SI-SDR (dB)')
    ax1.set_ylim(-9.5, 0.5)
    ax1.grid(True, axis='y')
    # Bottom-left: C's bar is the shortest, leaving that corner clear.
    ax1.text(0.03, 0.03, 'higher = better', transform=ax1.transAxes,
             fontsize=6, color='#555', va='bottom', ha='left')

    # Right: zlib compression ratio
    bars2 = ax2.bar(x, ratio.values, w, color=colors, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars2, ratio.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{v:.3f}×', ha='center', va='bottom', fontsize=6.5)
    ax2.set_xticks(x); ax2.set_xticklabels(phase_order)
    ax2.set_ylabel('zlib compression ratio')
    ax2.set_ylim(1.0, 1.6)
    ax2.grid(True, axis='y')
    # Top-right: G's bar is the shortest here, leaving that corner clear
    # (bottom-right used to sit on every bar's base, since they all start at 1.0).
    ax2.text(0.97, 0.97, 'higher = more compressed', transform=ax2.transAxes,
             fontsize=6, color='#555', va='top', ha='right')

    fig.suptitle('Music eval — MUSDB18-HQ test set, n=40 tracks\n'
                 'D-VAE: highest compression + lowest quality  (p<0.0001***)',
                 fontsize=8.5)
    return fig
