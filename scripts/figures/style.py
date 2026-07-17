"""Shared matplotlib style, sizing, and phase palette for thesis figures."""
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
