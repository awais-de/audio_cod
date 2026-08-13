#!/usr/bin/env python3
"""
Confidence interval evaluation — Tier 2 statistical strengthening.

Evaluates phases C, D, D-VAE, E, F, G, NC on N LibriSpeech test-clean
speakers.  Outputs bootstrap 95% CIs and paired Wilcoxon tests for the two
key contrasts:
  - G vs NC  (causality ablation)
  - D-VAE vs D  (entropy-suppression quality cost)

Usage (paths.yaml configured, or test-clean next to project root):
  python scripts/eval_confidence_intervals.py

With explicit path (e.g. on Windows):
  python scripts/eval_confidence_intervals.py --librispeech "C:\\data\\LibriSpeech\\test-clean"

All output is written to a single file:
  comparisons/<date>_confidence_intervals/report.txt

Copy that one file back from the evaluation machine.

Requirements:
  pip install pesq pystoi scipy soundfile torch numpy
  (pesq is optional — STOI CIs still run without it)
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.codec_utils import load_model, encode_decode, compute_metrics
from src.paths import get_dataset_paths

try:
    from scipy.stats import wilcoxon as _wilcoxon
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# All phases evaluated — checkpoints must exist under checkpoints_active/
PHASES = [
    # Original curriculum — 32-dim (20 CP project)
    ('C',         PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt'),
    ('D',         PROJECT_ROOT / 'checkpoints_active/temporal_phaseD/best.pt'),
    ('D-VAE',     PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae/best.pt'),
    ('E',         PROJECT_ROOT / 'checkpoints_active/temporal_phaseE/best.pt'),
    ('F',         PROJECT_ROOT / 'checkpoints_active/temporal_phaseF/best.pt'),
    ('G',         PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt'),
    ('NC',        PROJECT_ROOT / 'checkpoints_active/temporal_phaseNC/best.pt'),
    # MS Thesis — second causal mechanism (32-dim)
    ('D-Entropy', PROJECT_ROOT / 'checkpoints_active/temporal_phaseEntropy/best.pt'),
    # MS Thesis — channel width ablation: 16-dim full progression
    ('C-16',      PROJECT_ROOT / 'checkpoints_active/temporal_phaseC_16/best.pt'),
    ('D-16',      PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_16/best.pt'),
    ('E-16',      PROJECT_ROOT / 'checkpoints_active/temporal_phaseE_16/best.pt'),
    ('F-16',      PROJECT_ROOT / 'checkpoints_active/temporal_phaseF_16/best.pt'),
    ('G-16',      PROJECT_ROOT / 'checkpoints_active/temporal_phaseG_16/best.pt'),
    # MS Thesis — channel width ablation: 64-dim full progression
    ('C-64',      PROJECT_ROOT / 'checkpoints_active/temporal_phaseC_64/best.pt'),
    ('D-64',      PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_64/best.pt'),
    ('E-64',      PROJECT_ROOT / 'checkpoints_active/temporal_phaseE_64/best.pt'),
    ('F-64',      PROJECT_ROOT / 'checkpoints_active/temporal_phaseF_64/best.pt'),
    ('G-64',      PROJECT_ROOT / 'checkpoints_active/temporal_phaseG_64/best.pt'),
    # MS Thesis — D-VAE entropy-quality coupling at other widths (#41)
    ('D-VAE-16',  PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae_16/best.pt'),
    ('D-VAE-64',  PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae_64/best.pt'),
    # Attention window fix (#27/#28/#29) — corrected 200-frame causal window,
    # 32-dim, retrained end to end. Originals above are untouched for comparison.
    ('C-fixed',         PROJECT_ROOT / 'checkpoints_active/temporal_phaseC_fixed/best.pt'),
    ('D-fixed',         PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_fixed/best.pt'),
    ('D-VAE-fixed',     PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae_fixed/best.pt'),
    ('D-Entropy-fixed', PROJECT_ROOT / 'checkpoints_active/temporal_phaseEntropy_fixed/best.pt'),
    ('E-fixed',         PROJECT_ROOT / 'checkpoints_active/temporal_phaseE_fixed/best.pt'),
    ('F-fixed',         PROJECT_ROOT / 'checkpoints_active/temporal_phaseF_fixed/best.pt'),
    ('G-fixed',         PROJECT_ROOT / 'checkpoints_active/temporal_phaseG_fixed/best.pt'),
    ('NC-fixed',        PROJECT_ROOT / 'checkpoints_active/temporal_phaseNC_fixed/best.pt'),
    # Full non-causal curriculum (#27/#45) — trained A through G from scratch,
    # not a 30-epoch fine-tune, for a fair depth-matched comparison to G-fixed.
    ('G-nc',            PROJECT_ROOT / 'checkpoints_active/temporal_phaseG_nc/best.pt'),
]

# Contrasts for Wilcoxon signed-rank tests.
# Each entry: (label, phase_A, phase_B, interpretation)
CONTRASTS = [
    # 20 CP project
    ('G vs NC',        'G',         'NC',      'causal vs non-causal (smaller G = causality costs quality)'),
    ('D-VAE vs D',     'D-VAE',     'D',       'KL penalty suppresses entropy → quality drops'),
    # MS Thesis: second causal mechanism
    ('D-Entr vs D',    'D-Entropy', 'D',       'soft entropy penalty vs base (independent mechanism)'),
    # MS Thesis: channel width — endpoint comparison
    ('G-16 vs G',      'G-16',      'G',       'narrower bottleneck vs 32-dim at best checkpoint'),
    ('G-64 vs G',      'G-64',      'G',       'wider bottleneck vs 32-dim at best checkpoint'),
    # MS Thesis: channel width — within-width progression (base vs best)
    ('G-16 vs C-16',   'G-16',      'C-16',    '16-dim: full curriculum gain (same pattern as 32-dim?)'),
    ('G-64 vs C-64',   'G-64',      'C-64',    '64-dim: full curriculum gain (same pattern as 32-dim?)'),
    # MS Thesis: does the D-VAE entropy-quality coupling hold at other widths? (#41)
    ('D-VAE-16 vs D-16', 'D-VAE-16', 'D-16',   'KL penalty at 16-dim: does entropy drop come with quality drop?'),
    ('D-VAE-64 vs D-64', 'D-VAE-64', 'D-64',   'KL penalty at 64-dim: does entropy drop come with quality drop?'),
    # Attention window fix (#27/#28/#29): how much do corrected numbers actually
    # move, and does the core coupling claim survive under the real fix?
    ('G-fixed vs G',            'G-fixed',     'G',       'corrected 200-frame window vs the original triu/tril bug'),
    ('NC-fixed vs G-fixed',     'NC-fixed',    'G-fixed', 'causality ablation redone under the corrected window'),
    ('D-VAE-fixed vs D-fixed',  'D-VAE-fixed', 'D-fixed', 'KL penalty coupling under the corrected window -- still holds?'),
    ('D-Entr-fixed vs D-fixed', 'D-Entropy-fixed', 'D-fixed', 'soft entropy penalty coupling under the corrected window'),
    # Full non-causal retrain (#45): fair depth-matched causality comparison --
    # G-nc trained the full A-G curriculum, not a fine-tune from G-fixed.
    ('G-fixed vs G-nc', 'G-fixed', 'G-nc', 'causality ablation, full non-causal curriculum (fair depth match)'),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=10_000, ci=95, rng=None):
    """Bootstrap CI on the mean. Returns (mean, lo, hi). Skips None entries."""
    if rng is None:
        rng = np.random.default_rng(42)
    vals = np.array([v for v in values if v is not None], dtype=float)
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boots, [alpha, 100.0 - alpha])
    return float(vals.mean()), float(lo), float(hi)


def wilcoxon_test(a_vals, b_vals):
    """
    Paired Wilcoxon signed-rank test on matched lists. Returns (stat, p).
    Skips pairs where either value is None.
    """
    if not SCIPY_OK:
        return None, None
    pairs = [(a, b) for a, b in zip(a_vals, b_vals)
             if a is not None and b is not None]
    if len(pairs) < 5:
        return None, None
    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    if np.all(a_arr == b_arr):
        return 0.0, 1.0
    stat, p = _wilcoxon(a_arr, b_arr, alternative='two-sided')
    return float(stat), float(p)


def sig_stars(p):
    if p is None:
        return 'n/a'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def collect_speakers(test_clean_path: Path, n: int, clip_sec: int, sr: int):
    """
    Return list of (speaker_id, audio_array) for the first `n` speakers
    found under test_clean_path, sorted numerically by speaker ID.
    One utterance per speaker (first .flac file found).
    """
    by_speaker = {}
    for f in sorted(test_clean_path.rglob('*.flac')):
        spk = f.parts[-3]
        if spk not in by_speaker:
            by_speaker[spk] = f
    # Sort numerically where possible, else lexicographically
    def spk_key(s):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)
    sorted_ids = sorted(by_speaker, key=spk_key)[:n]
    speakers = []
    for spk in sorted_ids:
        audio, file_sr = sf.read(by_speaker[spk])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            n_samples = int(len(audio) * sr / file_sr)
            audio = np.interp(
                np.linspace(0, len(audio), n_samples),
                np.arange(len(audio)), audio
            )
        audio = np.clip(audio[:clip_sec * sr], -1.0, 1.0).astype(np.float32)
        speakers.append((spk, audio))
    return speakers


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--librispeech', type=Path, default=None,
                   help='Path to LibriSpeech test-clean directory. '
                        'Auto-detected from paths.yaml / project layout if omitted.')
    p.add_argument('--n-speakers', type=int, default=40,
                   help='Number of test speakers (default: 40).')
    p.add_argument('--clip-sec', type=int, default=5,
                   help='Seconds of audio per speaker (default: 5).')
    p.add_argument('--device', default='cpu',
                   help='Torch device, e.g. cpu or cuda (default: cpu).')
    p.add_argument('--n-boot', type=int, default=10_000,
                   help='Bootstrap iterations (default: 10000).')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for bootstrap (default: 42).')
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    SR = 16_000
    rng = np.random.default_rng(args.seed)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    run_date  = datetime.now().strftime('%Y-%m-%d')

    # ── discover test-clean path ──────────────────────────────────────────
    if args.librispeech is not None:
        test_clean = args.librispeech
    else:
        try:
            test_clean = get_dataset_paths()['test_clean']
        except Exception:
            test_clean = PROJECT_ROOT.parent / 'datasets' / 'LibriSpeech' / 'test-clean'

    if not test_clean.exists():
        print(f"ERROR: test-clean directory not found: {test_clean}")
        print("Pass --librispeech /path/to/LibriSpeech/test-clean")
        sys.exit(1)

    print(f"LibriSpeech test-clean: {test_clean}")
    print(f"Loading {args.n_speakers} speakers …")
    speakers = collect_speakers(test_clean, args.n_speakers, args.clip_sec, SR)
    n_actual = len(speakers)
    print(f"  Found {n_actual} speakers: {[s for s, _ in speakers[:5]]} …")

    # ── evaluate each phase ───────────────────────────────────────────────
    # results[phase_label] = list of {speaker, kbps, pesq_wb, stoi}
    results = {}
    for phase_label, ckpt_path in PHASES:
        if not ckpt_path.exists():
            print(f"\n  [SKIP] {phase_label}: checkpoint not found at {ckpt_path}")
            continue
        print(f"\n  Loading {phase_label} …", end=' ', flush=True)
        model, _ = load_model(ckpt_path, device)
        print('ok')
        phase_results = []
        for spk, audio in speakers:
            recon, kbps = encode_decode(model, audio, SR, device, chunk_sec=float(args.clip_sec))
            n_common = min(len(audio), len(recon))
            pesq_wb, stoi = compute_metrics(audio[:n_common], recon[:n_common], SR)
            phase_results.append({'speaker': spk, 'kbps': kbps,
                                  'pesq_wb': pesq_wb, 'stoi': stoi})
            pstr = f'{pesq_wb:.3f}' if pesq_wb is not None else 'n/a '
            print(f'    {spk:>8}  kbps={kbps:.2f}  PESQ={pstr}  STOI={stoi:.3f}')
        results[phase_label] = phase_results
        del model

    # ── bootstrap CIs ─────────────────────────────────────────────────────
    ci_table = {}  # phase -> {pesq: (mean,lo,hi), stoi: (mean,lo,hi)}
    for label, rows in results.items():
        pesq_vals = [r['pesq_wb'] for r in rows]
        stoi_vals = [r['stoi']    for r in rows]
        ci_table[label] = {
            'pesq': bootstrap_ci(pesq_vals, n_boot=args.n_boot, rng=rng),
            'stoi': bootstrap_ci(stoi_vals, n_boot=args.n_boot, rng=rng),
            'kbps': bootstrap_ci([r['kbps'] for r in rows], n_boot=args.n_boot, rng=rng),
        }

    # ── Wilcoxon tests ────────────────────────────────────────────────────
    wilcoxon_rows = []
    for contrast_label, phase_a, phase_b, interpretation in CONTRASTS:
        if phase_a not in results or phase_b not in results:
            continue
        # Match rows by speaker order (both lists are over the same speakers)
        a_pesq = [r['pesq_wb'] for r in results[phase_a]]
        b_pesq = [r['pesq_wb'] for r in results[phase_b]]
        a_stoi = [r['stoi']    for r in results[phase_a]]
        b_stoi = [r['stoi']    for r in results[phase_b]]
        wstat_p, p_p = wilcoxon_test(a_pesq, b_pesq)
        wstat_s, p_s = wilcoxon_test(a_stoi, b_stoi)
        wilcoxon_rows.append({
            'contrast': contrast_label,
            'interpretation': interpretation,
            'pesq_W': wstat_p, 'pesq_p': p_p,
            'stoi_W': wstat_s, 'stoi_p': p_s,
        })

    # ── build report text ─────────────────────────────────────────────────
    SEP  = '=' * 72
    sep  = '-' * 72

    def fmt_ci(triple, w=6):
        m, lo, hi = triple
        if np.isnan(m):
            return f"{'n/a':>{w}}  {'':>20}"
        return f"{m:>{w}.3f}  [{lo:.3f}, {hi:.3f}]"

    def fmt_p(p):
        if p is None:
            return '     n/a'
        return f'{p:.4f}' if p >= 0.0001 else '<0.0001'

    lines = [
        '',
        SEP,
        'CONFIDENCE INTERVAL EVALUATION',
        f'Generated : {timestamp}',
        f'Speakers  : {n_actual}  (first {n_actual} numeric IDs from test-clean)',
        f'Clip      : {args.clip_sec}s  |  SR: 16 kHz mono  |  device: {device}',
        f'Bootstrap : {args.n_boot:,} iterations  |  CI: 95%  |  seed: {args.seed}',
        f'PESQ      : {"available" if any(r["pesq_wb"] is not None for rows in results.values() for r in rows) else "NOT AVAILABLE — run on Windows with: pip install pesq"}',
        f'Wilcoxon  : {"available (scipy)" if SCIPY_OK else "NOT AVAILABLE — pip install scipy"}',
        SEP,
        '',
        'PHASE MEANS WITH 95% BOOTSTRAP CI',
        '',
        f"{'Phase':<8}  {'PESQ-WB mean':>12}  {'95% CI':>16}     {'STOI mean':>9}  {'95% CI':>16}     {'kbps':>6}",
        sep,
    ]
    for label, ckpt in PHASES:
        if label not in ci_table:
            lines.append(f'  {label:<6}  [skipped — checkpoint not found]')
            continue
        ci = ci_table[label]
        pesq_str = fmt_ci(ci['pesq'], w=12)
        stoi_str = fmt_ci(ci['stoi'], w=9)
        kbps_m   = ci['kbps'][0]
        lines.append(f'  {label:<6}  {pesq_str}     {stoi_str}     {kbps_m:>5.2f}k')
    lines += ['', sep, '']

    lines += [
        'WILCOXON SIGNED-RANK TESTS (paired, two-tailed)',
        '  H0: no difference in per-speaker score between the two phases.',
        '',
        f"{'Contrast':<16}  {'Metric':<8}  {'W-stat':>8}  {'p-value':>8}  {'sig':>4}  Interpretation",
        sep,
    ]
    if not SCIPY_OK:
        lines.append('  scipy not installed — install with: pip install scipy')
    elif not wilcoxon_rows:
        lines.append('  No contrasts available (missing checkpoints).')
    else:
        for row in wilcoxon_rows:
            for metric, W, p in [('PESQ-WB', row['pesq_W'], row['pesq_p']),
                                  ('STOI',    row['stoi_W'], row['stoi_p'])]:
                if W is None:
                    Wstr, pstr, sig = '     n/a', '     n/a', 'n/a'
                else:
                    Wstr = f'{W:>8.1f}'
                    pstr = f'{fmt_p(p):>8}'
                    sig  = sig_stars(p)
                lines.append(
                    f"  {row['contrast']:<16}  {metric:<8}  {Wstr}  {pstr}  {sig:>4}  "
                    f"{row['interpretation'] if metric == 'PESQ-WB' else ''}"
                )
        lines += [
            '',
            "  Stars: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
        ]
    lines += ['', sep, '']

    lines += [
        'PER-SPEAKER RAW DATA',
        sep,
    ]
    for label, ckpt in PHASES:
        if label not in results:
            continue
        lines.append(f'\n  Phase {label}')
        lines.append(f"  {'Speaker':>10}  {'kbps':>6}  {'PESQ-WB':>8}  {'STOI':>6}")
        lines.append(f"  {'-'*42}")
        for r in results[label]:
            pstr = f"{r['pesq_wb']:>8.3f}" if r['pesq_wb'] is not None else '     n/a'
            lines.append(f"  {r['speaker']:>10}  {r['kbps']:>6.2f}  {pstr}  {r['stoi']:>6.3f}")

    lines += ['', SEP, '']

    report = '\n'.join(lines)
    print('\n' + report)

    # ── write output ──────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / 'comparisons' / f'{run_date}_confidence_intervals'
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / 'report.txt').write_text(report, encoding='utf-8')
    print(f'\nSaved: {out_dir / "report.txt"}')

    # CSV — per-speaker per-phase
    csv_path = out_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['phase', 'speaker', 'kbps', 'pesq_wb', 'stoi'])
        w.writeheader()
        for label, ckpt in PHASES:
            if label not in results:
                continue
            for r in results[label]:
                w.writerow({'phase': label, **r})
    print(f'Saved: {csv_path}')
    print('\nDone. Copy report.txt back to your main machine.')


if __name__ == '__main__':
    main()
