"""Shared matplotlib style, sizing, and phase palette for thesis figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# IEEE two-column sizing (inches)
COL1_W = 3.5
COL2_W = 7.16
ROW_H  = 2.5

# Fixed phase colour order — never cycled, never reused for different meaning.
# 32-dim curriculum uses saturated hues; 16-dim lighter; 64-dim darker.
PHASE_COLORS: dict[str, str] = {
    'A':         '#bbbbbb',
    'B':         '#999999',
    'C':         '#4878d0',
    'D':         '#6acc65',
    'D-VAE':     '#d65f5f',
    'D-Entropy': '#ee854a',
    'E':         '#b47cc7',
    'F':         '#c4ad66',
    'G':         '#77bedb',
    'NC':        '#aaaaaa',
    # 16-dim ablation: lighter tints of the 32-dim hues
    'C-16':      '#a0bef0',
    'D-16':      '#a8e0a0',
    'E-16':      '#d4b8e8',
    'F-16':      '#ddd0a0',
    'G-16':      '#4472c8',
    # 64-dim ablation: darker shades
    'C-64':      '#2050a8',
    'D-64':      '#2e8a30',
    'E-64':      '#7030a0',
    'F-64':      '#7a6000',
    'G-64':      '#003c5a',
}

# EntroCodec vs EnCodec comparisons (fig_03, fig_04, fig_05) used to borrow
# PHASE_COLORS['G']/['D-VAE'] (blue/red) — but those two colors now carry a
# separate meaning elsewhere (blue = normal curriculum phase, red = the
# D-VAE anomaly), so reusing them here reads as a phase callout by mistake.
# Dedicated, unrelated hues instead.
CODEC_COLORS: dict[str, str] = {
    'EntroCodec': '#2a9d8f',  # teal
    'EnCodec':    '#d9822b',  # orange
}

# Curriculum phases in canonical display order
CORE_PHASES    = ['C', 'D', 'D-VAE', 'E', 'F', 'G']
THESIS_PHASES  = ['C', 'D', 'D-VAE', 'D-Entropy', 'E', 'F', 'G']
ALL_PHASES     = ['A', 'B', 'C', 'D', 'D-VAE', 'D-Entropy', 'E', 'F', 'G', 'NC']
DIM16_PHASES   = ['C-16', 'D-16', 'E-16', 'F-16', 'G-16']
DIM64_PHASES   = ['C-64', 'D-64', 'E-64', 'F-64', 'G-64']


def apply() -> None:
    mpl.rcParams.update({
        'font.family':        'serif',
        'font.size':          8,
        'axes.titlesize':     9,
        'axes.labelsize':     8,
        'xtick.labelsize':    7,
        'ytick.labelsize':    7,
        'legend.fontsize':    7,
        'axes.linewidth':     0.6,
        'xtick.major.width':  0.6,
        'ytick.major.width':  0.6,
        'lines.linewidth':    1.2,
        'grid.linewidth':     0.4,
        'grid.alpha':         0.35,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'figure.constrained_layout.use': True,
    })


def new_fig(w: float = COL2_W, h: float = ROW_H) -> plt.Figure:
    apply()
    return plt.figure(figsize=(w, h))


def legend(fig: plt.Figure, ax=None, handles=None, labels=None, ncol: int | None = None) -> None:
    """Place a figure's legend in one fixed slot — outside the axes, centered
    below the plot — so it never overlaps plotted data and looks the same
    across every figure. Every figure that needs a legend should call this
    instead of ax.legend()/fig.legend() directly.

    handles/labels default to whatever the given (or first) axes has labelled
    via label= kwargs on its artists.
    """
    if handles is None:
        src = ax if ax is not None else fig.axes[0]
        handles, labels = src.get_legend_handles_labels()
    if ncol is None:
        ncol = len(handles)
    # 'outside' placement only reserves space correctly under constrained
    # layout — figure functions in this module don't otherwise enable it.
    fig.set_layout_engine('constrained')
    fig.legend(handles, labels, loc='outside lower center', ncol=ncol, fontsize=7)
