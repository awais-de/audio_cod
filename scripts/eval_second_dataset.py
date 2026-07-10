#!/usr/bin/env python3
"""
Second-dataset evaluation — Tier 3 OOD validation.

Evaluates phases C, D, D-VAE, G on VCTK (or any directory of .flac/.wav
speech files at any sample rate) to verify:
  1. D-VAE zlib compression ratio is still the highest (ordering preserved OOD)
  2. D-VAE PESQ/STOI still lower than G (quality-entropy tradeoff holds OOD)

Single forward pass per speaker: encode → quantize (compression + entropy) →
decode (PESQ/STOI). No double inference.

Usage:
  python scripts/eval_second_dataset.py --dataset path/to/VCTK-Corpus-0.92/wav48_silence_trimmed

All output is written to a single file:
  comparisons/<date>_second_dataset/report.txt

Copy that one file back from the evaluation machine.

Requirements: pip install pesq pystoi scipy soundfile torch numpy
  (pesq optional — STOI + compression ratio still run without it)
"""

import argparse
import csv
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.codec_utils import load_model, compute_metrics

try:
    from scipy.stats import wilcoxon as _wilcoxon
    from scipy.signal import resample_poly
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# Key phases for the Tier 3 claim — enough to show ordering + quality cost
DEFAULT_PHASES = [
    ('C',     PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt'),
    ('D',     PROJECT_ROOT / 'checkpoints_active/temporal_phaseD/best.pt'),
    ('D-VAE', PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae/best.pt'),
    ('G',     PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt'),
]

NUM_LEVELS = 8
TARGET_SR  = 16_000


# ── audio loading ─────────────────────────────────────────────────────────────

def resample_audio(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio
    # resample_poly handles integer ratios exactly (e.g. 48000→16000 = down 3)
    if SCIPY_OK:
        from math import gcd
        g = gcd(src_sr, tgt_sr)
        return resample_poly(audio, tgt_sr // g, src_sr // g).astype(np.float32)
    # numpy fallback
    n_out = int(len(audio) * tgt_sr / src_sr)
    return np.interp(
        np.linspace(0, len(audio), n_out),
        np.arange(len(audio)), audio
    ).astype(np.float32)


def load_clip(path: Path, clip_sec: int) -> tuple:
    """Returns (audio_float32, original_sr)."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample_audio(audio.astype(np.float32), sr, TARGET_SR)
    n = clip_sec * TARGET_SR
    if len(audio) >= n:
        audio = audio[:n]
    else:
        audio = np.pad(audio, (0, n - len(audio)))
    return np.clip(audio, -1.0, 1.0), sr


# ── VCTK + generic directory walker ──────────────────────────────────────────

def _is_speech_file(p: Path) -> bool:
    return p.suffix.lower() in {'.flac', '.wav', '.ogg'}


def collect_vctk_speakers(root: Path, n: int, clip_sec: int) -> list:
    """
    VCTK structure: root/pXXX/pXXX_YYY_mic2.flac  (prefer mic2 over mic1).
    Falls back to generic walker if no pXXX directories found.
    Returns list of (speaker_id, audio_float32).
    """
    spk_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith('p')],
        key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0
    )

    if not spk_dirs:
        return collect_generic_speakers(root, n, clip_sec)

    speakers = []
    for spk_dir in spk_dirs:
        if len(speakers) >= n:
            break
        # prefer mic2, fall back to mic1, then any speech file
        files_mic2 = sorted(spk_dir.glob('*_mic2.*'))
        files_mic1 = sorted(spk_dir.glob('*_mic1.*'))
        files_any  = sorted(f for f in spk_dir.iterdir() if _is_speech_file(f))
        chosen = next(
            (f for f in files_mic2 if _is_speech_file(f)), None
        ) or next(
            (f for f in files_mic1 if _is_speech_file(f)), None
        ) or (files_any[0] if files_any else None)

        if chosen is None:
            continue
        try:
            audio, _ = load_clip(chosen, clip_sec)
            speakers.append((spk_dir.name, audio))
        except Exception as e:
            print(f'    [warn] {spk_dir.name}: {e}')

    return speakers


def collect_generic_speakers(root: Path, n: int, clip_sec: int) -> list:
    """
    Generic fallback: one file per immediate subdirectory, sorted alphabetically.
    Also handles a flat directory (all files at root level, one per 'speaker').
    """
    by_spk = {}
    for f in sorted(root.rglob('*')):
        if not _is_speech_file(f):
            continue
        spk = f.parent.name if f.parent != root else f.stem
        if spk not in by_spk:
            by_spk[spk] = f

    speakers = []
    for spk in sorted(by_spk)[:n]:
        try:
            audio, _ = load_clip(by_spk[spk], clip_sec)
            speakers.append((spk, audio))
        except Exception as e:
            print(f'    [warn] {spk}: {e}')
    return speakers


# ── single-pass encode → compress → decode ────────────────────────────────────

def evaluate_speaker(model, audio: np.ndarray, device) -> dict:
    """
    Single forward pass:
      encode → quantize (zlib ratio, Shannon H) → decode (PESQ/STOI).
    Returns dict with pesq_wb, stoi, kbps, zlib_ratio, mean_H.
    """
    clip_size  = len(audio)          # already clipped to clip_sec * SR
    raw_bits   = 0
    comp_bits  = 0
    all_q      = []
    recon_list = []

    with torch.no_grad():
        # process as one chunk (audio is already one clip_sec block)
        x   = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
        z   = model.encode(x)
        z_np = z.squeeze(0).cpu().numpy()          # (frames, dims)

        z_min  = float(z_np.min())
        z_max  = float(z_np.max())
        scale  = (z_max - z_min) / (NUM_LEVELS - 1) + 1e-8
        q = np.clip(
            np.round((z_np - z_min) / scale), 0, NUM_LEVELS - 1
        ).astype(np.uint8)

        compressed = zlib.compress(q.tobytes(), level=9)
        raw_bits   = q.size * 3
        comp_bits  = len(compressed) * 8
        all_q.append(q)

        z_rec   = q.astype(np.float32) * scale + z_min
        x_recon = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))
        recon_list.append(x_recon.squeeze().cpu().numpy())

    duration    = clip_size / TARGET_SR
    zlib_ratio  = raw_bits / comp_bits if comp_bits > 0 else 0.0
    kbps        = comp_bits / duration / 1000

    q_all  = np.concatenate(all_q, axis=0)
    n_dims = q_all.shape[1]
    H = np.zeros(n_dims)
    for d in range(n_dims):
        counts  = np.bincount(q_all[:, d], minlength=NUM_LEVELS).astype(float)
        probs   = counts / counts.sum()
        nonzero = probs[probs > 0]
        H[d]    = -np.sum(nonzero * np.log2(nonzero))

    recon = np.concatenate(recon_list)
    recon = recon[:clip_size] if len(recon) >= clip_size else \
            np.pad(recon, (0, clip_size - len(recon)))
    pesq_wb, stoi = compute_metrics(audio, recon, TARGET_SR)

    return {
        'pesq_wb':    pesq_wb,
        'stoi':       stoi,
        'kbps':       kbps,
        'zlib_ratio': zlib_ratio,
        'mean_H':     float(H.mean()),
    }


# ── statistics ────────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=10_000, ci=95, rng=None):
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
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def fmt_p(p):
    if p is None:
        return '     n/a'
    return f'{p:.4f}' if p >= 0.0001 else '<0.0001'


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, required=True,
                   help='Path to dataset root, e.g. VCTK-Corpus-0.92/wav48_silence_trimmed')
    p.add_argument('--dataset-name', default=None,
                   help='Label for the dataset in the report (auto-detected if omitted).')
    p.add_argument('--n-speakers', type=int, default=40,
                   help='Number of speakers to evaluate (default: 40).')
    p.add_argument('--clip-sec', type=int, default=5,
                   help='Seconds of audio per speaker (default: 5).')
    p.add_argument('--device', default='cpu',
                   help='Torch device (default: cpu).')
    p.add_argument('--n-boot', type=int, default=10_000,
                   help='Bootstrap iterations (default: 10000).')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args    = parse_args()
    device  = args.device
    rng     = np.random.default_rng(args.seed)
    run_date = datetime.now().strftime('%Y-%m-%d')
    ts       = datetime.now().strftime('%Y-%m-%d %H:%M')

    dataset_root = args.dataset
    if not dataset_root.exists():
        print(f'ERROR: dataset path not found: {dataset_root}')
        sys.exit(1)

    ds_name = args.dataset_name or dataset_root.parent.name or dataset_root.name

    print(f'Dataset : {dataset_root}  ({ds_name})')
    print(f'Loading {args.n_speakers} speakers …')
    speakers = collect_vctk_speakers(dataset_root, args.n_speakers, args.clip_sec)
    n_actual = len(speakers)
    if n_actual == 0:
        print('ERROR: no speech files found under dataset path.')
        sys.exit(1)
    print(f'  Found {n_actual} speakers: {[s for s, _ in speakers[:5]]} …')

    # ── evaluate each phase ───────────────────────────────────────────────────
    results = {}   # phase_label -> list of per-speaker dicts
    for phase_label, ckpt_path in DEFAULT_PHASES:
        if not ckpt_path.exists():
            print(f'\n  [SKIP] {phase_label}: checkpoint not found')
            continue
        print(f'\n  Loading {phase_label} …', end=' ', flush=True)
        model, _ = load_model(ckpt_path, device)
        print('ok')
        phase_rows = []
        for spk, audio in speakers:
            row = evaluate_speaker(model, audio, device)
            row['speaker'] = spk
            phase_rows.append(row)
            pstr = f"{row['pesq_wb']:.3f}" if row['pesq_wb'] is not None else 'n/a '
            print(f'    {spk:>8}  kbps={row["kbps"]:.2f}  '
                  f'PESQ={pstr}  STOI={row["stoi"]:.3f}  '
                  f'zlib={row["zlib_ratio"]:.3f}×  H̄={row["mean_H"]:.3f}')
        results[phase_label] = phase_rows
        del model

    if not results:
        print('ERROR: no phases could be evaluated (all checkpoints missing?).')
        sys.exit(1)

    # ── bootstrap CIs ────────────────────────────────────────────────────────
    ci = {}
    for label, rows in results.items():
        ci[label] = {
            'pesq':  bootstrap_ci([r['pesq_wb']    for r in rows], args.n_boot, rng=rng),
            'stoi':  bootstrap_ci([r['stoi']        for r in rows], args.n_boot, rng=rng),
            'ratio': bootstrap_ci([r['zlib_ratio']  for r in rows], args.n_boot, rng=rng),
            'H':     bootstrap_ci([r['mean_H']      for r in rows], args.n_boot, rng=rng),
            'kbps':  bootstrap_ci([r['kbps']        for r in rows], args.n_boot, rng=rng),
        }

    # ── Wilcoxon: D-VAE vs D, D-VAE vs G ────────────────────────────────────
    contrasts = [
        ('D-VAE vs D', 'D-VAE', 'D',
         'entropy suppression costs quality (D-VAE lower = KL hurts)'),
        ('D-VAE vs G', 'D-VAE', 'G',
         'D-VAE quality cost vs final codec'),
    ]
    wilcoxon_rows = []
    for label, pa, pb in [(c[0], c[1], c[2]) for c in contrasts]:
        if pa not in results or pb not in results:
            continue
        interp = next(c[3] for c in contrasts if c[0] == label)
        Wp, pp = wilcoxon_test(
            [r['pesq_wb'] for r in results[pa]],
            [r['pesq_wb'] for r in results[pb]]
        )
        Ws, ps = wilcoxon_test(
            [r['stoi'] for r in results[pa]],
            [r['stoi'] for r in results[pb]]
        )
        Wr, pr = wilcoxon_test(
            [r['zlib_ratio'] for r in results[pa]],
            [r['zlib_ratio'] for r in results[pb]]
        )
        wilcoxon_rows.append({
            'label': label, 'interp': interp,
            'pesq_W': Wp, 'pesq_p': pp,
            'stoi_W': Ws, 'stoi_p': ps,
            'ratio_W': Wr, 'ratio_p': pr,
        })

    # ── compression ordering check ────────────────────────────────────────────
    ratio_means = {label: ci[label]['ratio'][0] for label in results}
    ranked = sorted(ratio_means, key=ratio_means.get, reverse=True)
    ordering_str = ' > '.join(ranked)
    dvae_is_highest = ranked[0] == 'D-VAE' if 'D-VAE' in results else None

    # ── build report ──────────────────────────────────────────────────────────
    SEP = '=' * 74
    sep = '-' * 74

    def fc(triple, w=7):
        m, lo, hi = triple
        if np.isnan(m): return f"{'n/a':>{w}}  {'':>18}"
        return f"{m:>{w}.3f}  [{lo:.3f}, {hi:.3f}]"

    has_pesq = any(
        r['pesq_wb'] is not None for rows in results.values() for r in rows
    )

    lines = [
        '',
        SEP,
        'SECOND-DATASET EVALUATION (OOD VALIDATION)',
        f'Generated  : {ts}',
        f'Dataset    : {ds_name}  ({dataset_root})',
        f'Speakers   : {n_actual}  |  clip: {args.clip_sec}s  |  device: {device}',
        f'Bootstrap  : {args.n_boot:,} iterations, 95% CI, seed {args.seed}',
        f'PESQ       : {"available" if has_pesq else "NOT AVAILABLE — run on Windows: pip install pesq"}',
        f'Scipy      : {"available" if SCIPY_OK else "NOT AVAILABLE — pip install scipy"}',
        SEP,
        '',
        '━━  KEY QUESTION  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
        'Does the D-VAE entropy anomaly (highest zlib ratio, lowest quality) hold',
        f'on out-of-distribution speech ({ds_name})?',
        '',
    ]

    # -- Section 1: Compression ordering --
    lines += [
        '1. COMPRESSION RATIO — zlib level 9, 95% bootstrap CI',
        '',
        f"  {'Phase':<8}  {'zlib ratio mean':>15}  {'95% CI':>15}     "
        f"{'Mean H(d)':>9}  {'95% CI':>15}     {'kbps':>6}",
        f'  {sep}',
    ]
    for label, _ in DEFAULT_PHASES:
        if label not in ci:
            lines.append(f'  {label:<8}  [skipped]')
            continue
        lines.append(
            f'  {label:<8}  {fc(ci[label]["ratio"], w=15)}     '
            f'{fc(ci[label]["H"], w=9)}     {ci[label]["kbps"][0]:>5.2f}k'
        )
    lines += [
        f'  {sep}',
        f'  Ordering (highest → lowest ratio): {ordering_str}',
    ]
    if dvae_is_highest is not None:
        verdict = 'YES ✓ — D-VAE is most compressed on this dataset' if dvae_is_highest \
                  else 'NO ✗ — D-VAE does NOT rank highest here'
        lines.append(f'  D-VAE ranks #1 on {ds_name}? {verdict}')
    lines.append('')

    # -- Section 2: Quality metrics --
    lines += [
        '2. QUALITY METRICS — PESQ-WB and STOI, 95% bootstrap CI',
        '',
        f"  {'Phase':<8}  {'PESQ-WB mean':>12}  {'95% CI':>15}     "
        f"{'STOI mean':>9}  {'95% CI':>15}",
        f'  {sep}',
    ]
    for label, _ in DEFAULT_PHASES:
        if label not in ci:
            lines.append(f'  {label:<8}  [skipped]')
            continue
        lines.append(
            f'  {label:<8}  {fc(ci[label]["pesq"], w=12)}     '
            f'{fc(ci[label]["stoi"], w=9)}'
        )
    lines += [f'  {sep}', '']

    # -- Section 3: Wilcoxon --
    lines += [
        '3. WILCOXON SIGNED-RANK TESTS (paired, two-tailed)',
        f'   n = {n_actual} speakers',
        '',
        f"  {'Contrast':<16}  {'Metric':8}  {'W-stat':>8}  {'p-value':>8}  {'sig':>4}",
        f'  {sep}',
    ]
    if not SCIPY_OK:
        lines.append('  scipy not installed — pip install scipy')
    elif not wilcoxon_rows:
        lines.append('  No contrasts available.')
    else:
        for row in wilcoxon_rows:
            first = True
            for metric, W, p in [
                ('PESQ-WB',    row['pesq_W'],  row['pesq_p']),
                ('STOI',       row['stoi_W'],  row['stoi_p']),
                ('zlib ratio', row['ratio_W'], row['ratio_p']),
            ]:
                Wstr = f'{W:>8.1f}' if W is not None else '     n/a'
                pstr = f'{fmt_p(p):>8}'
                sig  = sig_stars(p)
                label_col = row['label'] if first else ''
                lines.append(f'  {label_col:<16}  {metric:<8}  {Wstr}  {pstr}  {sig:>4}')
                first = False
            lines.append(f'    ({row["interp"]})')
            lines.append('')
        lines.append("  Stars: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    lines += ['', sep, '']

    # -- Per-speaker raw data --
    lines += ['PER-SPEAKER RAW DATA', sep]
    for label, _ in DEFAULT_PHASES:
        if label not in results:
            continue
        lines.append(f'\n  Phase {label}')
        lines.append(f"  {'Speaker':>10}  {'kbps':>6}  "
                     f"{'PESQ-WB':>8}  {'STOI':>6}  "
                     f"{'zlib ratio':>10}  {'Mean H':>7}")
        lines.append(f'  {"-"*58}')
        for r in results[label]:
            pstr = f"{r['pesq_wb']:>8.3f}" if r['pesq_wb'] is not None else '     n/a'
            lines.append(
                f"  {r['speaker']:>10}  {r['kbps']:>6.2f}  "
                f"{pstr}  {r['stoi']:>6.3f}  "
                f"{r['zlib_ratio']:>10.3f}×  {r['mean_H']:>7.3f}"
            )
    lines += ['', SEP, '']

    report = '\n'.join(lines)
    print('\n' + report)

    # ── write output ──────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / 'comparisons' / f'{run_date}_second_dataset'
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / 'report.txt').write_text(report, encoding='utf-8')
    print(f'Saved: {out_dir / "report.txt"}')

    csv_path = out_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'phase', 'speaker', 'kbps', 'pesq_wb', 'stoi', 'zlib_ratio', 'mean_H'
        ])
        w.writeheader()
        for label, _ in DEFAULT_PHASES:
            if label not in results:
                continue
            for r in results[label]:
                w.writerow({'phase': label, **r})
    print(f'Saved: {csv_path}')
    print('\nDone. Copy report.txt back to your main machine.')


if __name__ == '__main__':
    main()
