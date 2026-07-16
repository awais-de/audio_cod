#!/usr/bin/env python3
"""
Music evaluation — MS Thesis / Tier 5: modality independence.

Evaluates phases C, D, D-VAE, G on a music dataset (default: MUSDB18-HQ)
to verify the entropy-quality coupling holds beyond speech:
  1. D-VAE zlib compression ratio still highest (ordering preserved)
  2. D-VAE SI-SDR still lowest (quality-entropy tradeoff holds on music)

Quality metric: SI-SDR (scale-invariant SDR) — speech-agnostic, no external
library required.  PESQ/STOI are NOT used here (both are speech-specific).

Usage — MUSDB18-HQ (recommended):
  python scripts/eval_music.py --dataset ../datasets/musdb18-hq --split test

Usage — any flat directory of audio files:
  python scripts/eval_music.py --dataset /path/to/music

Output:
  comparisons/<date>_music_eval/report.txt
  comparisons/<date>_music_eval/metrics.csv

Run on Windows with:
  python scripts/eval_music.py --dataset ../datasets/musdb18-hq --split test --device cuda
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

from src.codec_utils import load_model

try:
    from scipy.signal import resample_poly
    from scipy.stats import wilcoxon as _wilcoxon
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

DEFAULT_PHASES = [
    ('C',     PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt'),
    ('D',     PROJECT_ROOT / 'checkpoints_active/temporal_phaseD/best.pt'),
    ('D-VAE', PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae/best.pt'),
    ('G',     PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt'),
]

NUM_LEVELS = 8
TARGET_SR  = 16_000
AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.mp3', '.aiff', '.aif'}


# ── audio ─────────────────────────────────────────────────────────────────────

def resample_audio(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio
    if SCIPY_OK:
        from math import gcd
        g = gcd(src_sr, TARGET_SR)
        return resample_poly(audio, TARGET_SR // g, src_sr // g).astype(np.float32)
    n_out = int(len(audio) * TARGET_SR / src_sr)
    return np.interp(
        np.linspace(0, len(audio), n_out),
        np.arange(len(audio)), audio
    ).astype(np.float32)


def load_clip(path: Path, clip_sec: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)           # stereo → mono
    audio = resample_audio(audio.astype(np.float32), sr)
    n = clip_sec * TARGET_SR
    if len(audio) >= n:
        # take centre of track — avoids silence at start/end of music files
        mid = len(audio) // 2
        start = max(0, mid - n // 2)
        audio = audio[start:start + n]
    else:
        audio = np.pad(audio, (0, n - len(audio)))
    return np.clip(audio, -1.0, 1.0)


# ── dataset walkers ───────────────────────────────────────────────────────────

def collect_musdb18(root: Path, split: str, n: int, clip_sec: int) -> list:
    """
    MUSDB18-HQ structure: root/{train,test}/Track Name/mixture.wav
    Prefers mixture.wav; falls back to any audio file in the track folder.
    """
    split_dirs = []
    if split in ('test', 'all'):
        td = root / 'test'
        if td.is_dir():
            split_dirs.append(td)
    if split in ('train', 'all'):
        td = root / 'train'
        if td.is_dir():
            split_dirs.append(td)

    if not split_dirs:
        print(f'  [warn] No {split}/ directory found under {root} — falling back to generic walker')
        return collect_generic(root, n, clip_sec)

    track_dirs = []
    for sd in split_dirs:
        track_dirs += sorted(d for d in sd.iterdir() if d.is_dir())

    tracks = []
    for td in track_dirs:
        if len(tracks) >= n:
            break
        mixture = td / 'mixture.wav'
        if not mixture.exists():
            # fall back to first audio file
            candidates = [f for f in sorted(td.iterdir()) if f.suffix.lower() in AUDIO_EXTS]
            if not candidates:
                continue
            mixture = candidates[0]
        try:
            audio = load_clip(mixture, clip_sec)
            tracks.append((td.name[:40], audio))   # truncate long track names
        except Exception as e:
            print(f'  [warn] {td.name}: {e}')

    return tracks


def collect_generic(root: Path, n: int, clip_sec: int) -> list:
    """
    Flat or nested directory: one track per immediate subdirectory,
    or one file per entry if root is flat.
    """
    by_track: dict = {}
    for f in sorted(root.rglob('*')):
        if f.suffix.lower() not in AUDIO_EXTS:
            continue
        key = f.parent.name if f.parent != root else f.stem
        if key not in by_track:
            by_track[key] = f

    tracks = []
    for key in sorted(by_track)[:n]:
        try:
            audio = load_clip(by_track[key], clip_sec)
            tracks.append((key, audio))
        except Exception as e:
            print(f'  [warn] {key}: {e}')
    return tracks


# ── quality metric ────────────────────────────────────────────────────────────

def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio (dB). Higher = better."""
    n = min(len(reference), len(estimate))
    ref = reference[:n] - reference[:n].mean()
    est = estimate[:n] - estimate[:n].mean()
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8)
    target = alpha * ref
    noise  = est - target
    denom  = np.dot(noise, noise)
    if denom < 1e-10:
        return 60.0                          # near-perfect reconstruction
    return float(10 * np.log10(np.dot(target, target) / denom))


# ── single-pass encode → compress → decode ────────────────────────────────────

def evaluate_track(model, audio: np.ndarray, device) -> dict:
    clip_size = len(audio)

    with torch.no_grad():
        x    = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
        z    = model.encode(x)
        z_np = z.squeeze(0).cpu().numpy()          # (frames, dims)

        z_min = float(z_np.min())
        z_max = float(z_np.max())
        scale = (z_max - z_min) / (NUM_LEVELS - 1) + 1e-8
        q = np.clip(
            np.round((z_np - z_min) / scale), 0, NUM_LEVELS - 1
        ).astype(np.uint8)

        compressed = zlib.compress(q.tobytes(), level=9)
        raw_bits   = q.size * 3
        comp_bits  = len(compressed) * 8

        z_rec   = q.astype(np.float32) * scale + z_min
        x_recon = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))
        recon   = x_recon.squeeze().cpu().numpy()

    duration   = clip_size / TARGET_SR
    zlib_ratio = raw_bits / comp_bits if comp_bits > 0 else 0.0
    kbps       = comp_bits / duration / 1000

    # Shannon H per dimension
    H = np.zeros(q.shape[1])
    for d in range(q.shape[1]):
        counts  = np.bincount(q[:, d], minlength=NUM_LEVELS).astype(float)
        probs   = counts / counts.sum()
        nonzero = probs[probs > 0]
        H[d]    = -np.sum(nonzero * np.log2(nonzero))

    recon = recon[:clip_size] if len(recon) >= clip_size else \
            np.pad(recon, (0, clip_size - len(recon)))

    return {
        'si_sdr':     si_sdr(audio, recon),
        'kbps':       kbps,
        'zlib_ratio': zlib_ratio,
        'mean_H':     float(H.mean()),
    }


# ── statistics ────────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=10_000, ci=95, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    vals = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boots, [alpha, 100.0 - alpha])
    return float(vals.mean()), float(lo), float(hi)


def wilcoxon_test(a_vals, b_vals):
    if not SCIPY_OK:
        return None, None
    pairs = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
    if len(pairs) < 5:
        return None, None
    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    if np.all(a_arr == b_arr):
        return 0.0, 1.0
    stat, p = _wilcoxon(a_arr, b_arr, alternative='two-sided')
    return float(stat), float(p)


def sig_stars(p):
    if p is None: return 'n/a'
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def fmt_p(p):
    if p is None: return '     n/a'
    return f'{p:.4f}' if p >= 0.0001 else '<0.0001'


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, required=True,
                   help='Path to musdb18-hq root, or any directory of audio files.')
    p.add_argument('--split', default='test', choices=['test', 'train', 'all'],
                   help='MUSDB18 split to use (default: test).')
    p.add_argument('--dataset-name', default=None,
                   help='Label for the dataset in the report (auto-detected if omitted).')
    p.add_argument('--n-tracks', type=int, default=40,
                   help='Number of tracks to evaluate (default: 40).')
    p.add_argument('--clip-sec', type=int, default=5,
                   help='Seconds of audio per track, taken from centre (default: 5).')
    p.add_argument('--device', default='cpu',
                   help='Torch device (default: cpu).')
    p.add_argument('--n-boot', type=int, default=10_000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args     = parse_args()
    device   = args.device
    rng      = np.random.default_rng(args.seed)
    run_date = datetime.now().strftime('%Y-%m-%d')
    ts       = datetime.now().strftime('%Y-%m-%d %H:%M')

    if not args.dataset.exists():
        print(f'ERROR: dataset path not found: {args.dataset}')
        sys.exit(1)

    ds_name = args.dataset_name or args.dataset.name

    # detect MUSDB18 vs generic
    is_musdb = (args.dataset / 'test').is_dir() or (args.dataset / 'train').is_dir()

    print(f'Dataset : {args.dataset}  ({ds_name})')
    print(f'Format  : {"MUSDB18-HQ" if is_musdb else "generic audio directory"}')
    print(f'Loading {args.n_tracks} tracks …')

    if is_musdb:
        tracks = collect_musdb18(args.dataset, args.split, args.n_tracks, args.clip_sec)
    else:
        tracks = collect_generic(args.dataset, args.n_tracks, args.clip_sec)

    n_actual = len(tracks)
    if n_actual == 0:
        print('ERROR: no audio files found.')
        sys.exit(1)
    print(f'  Found {n_actual} tracks.')

    # ── evaluate each phase ───────────────────────────────────────────────────
    results = {}
    for phase_label, ckpt_path in DEFAULT_PHASES:
        if not ckpt_path.exists():
            print(f'\n  [SKIP] {phase_label}: checkpoint not found')
            continue
        print(f'\n  Loading {phase_label} …', end=' ', flush=True)
        model, _ = load_model(ckpt_path, device)
        print('ok')
        phase_rows = []
        for track_name, audio in tracks:
            row = evaluate_track(model, audio, device)
            row['track'] = track_name
            phase_rows.append(row)
            print(f'    {track_name:<42}  kbps={row["kbps"]:.2f}  '
                  f'SI-SDR={row["si_sdr"]:+.1f}dB  '
                  f'zlib={row["zlib_ratio"]:.3f}×  H̄={row["mean_H"]:.3f}')
        results[phase_label] = phase_rows
        del model

    if not results:
        print('ERROR: no phases evaluated.')
        sys.exit(1)

    # ── bootstrap CIs ────────────────────────────────────────────────────────
    ci = {}
    for label, rows in results.items():
        ci[label] = {
            'si_sdr': bootstrap_ci([r['si_sdr']      for r in rows], args.n_boot, rng=rng),
            'ratio':  bootstrap_ci([r['zlib_ratio']  for r in rows], args.n_boot, rng=rng),
            'H':      bootstrap_ci([r['mean_H']      for r in rows], args.n_boot, rng=rng),
            'kbps':   bootstrap_ci([r['kbps']        for r in rows], args.n_boot, rng=rng),
        }

    # ── Wilcoxon ─────────────────────────────────────────────────────────────
    contrasts = [
        ('D-VAE vs D', 'D-VAE', 'D',  'entropy suppression costs quality on music'),
        ('D-VAE vs G', 'D-VAE', 'G',  'D-VAE quality cost vs best model on music'),
    ]
    wilcoxon_rows = []
    for label, pa, pb, interp in contrasts:
        if pa not in results or pb not in results:
            continue
        Ws, ps = wilcoxon_test(
            [r['si_sdr']      for r in results[pa]],
            [r['si_sdr']      for r in results[pb]]
        )
        Wr, pr = wilcoxon_test(
            [r['zlib_ratio']  for r in results[pa]],
            [r['zlib_ratio']  for r in results[pb]]
        )
        wilcoxon_rows.append({
            'label': label, 'interp': interp,
            'sisdr_W': Ws, 'sisdr_p': ps,
            'ratio_W': Wr, 'ratio_p': pr,
        })

    ratio_means      = {label: ci[label]['ratio'][0] for label in results}
    ranked           = sorted(ratio_means, key=ratio_means.get, reverse=True)
    ordering_str     = ' > '.join(ranked)
    dvae_is_highest  = ranked[0] == 'D-VAE' if 'D-VAE' in results else None

    # ── report ────────────────────────────────────────────────────────────────
    SEP = '=' * 74
    sep = '-' * 74

    def fc(triple, w=7, fmt='.3f'):
        m, lo, hi = triple
        if np.isnan(m): return f"{'n/a':>{w}}  {'':>18}"
        return f'{m:{w}{fmt}}  [{lo:{fmt}}, {hi:{fmt}}]'

    lines = [
        '',
        SEP,
        'MUSIC EVALUATION — MODALITY INDEPENDENCE (MS THESIS)',
        f'Generated  : {ts}',
        f'Dataset    : {ds_name}  ({args.dataset})',
        f'Split      : {args.split}',
        f'Tracks     : {n_actual}  |  clip: {args.clip_sec}s (centre)  |  device: {device}',
        f'Bootstrap  : {args.n_boot:,} iterations, 95% CI, seed {args.seed}',
        f'Metric     : SI-SDR (dB) — speech-agnostic; higher = better',
        SEP,
        '',
        '━━  KEY QUESTION  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
        'Does the entropy-quality coupling observed on speech (LibriSpeech, VCTK)',
        f'hold on music ({ds_name})?',
        'Expected: D-VAE = highest zlib ratio (most compressed) + lowest SI-SDR.',
        '',
    ]

    lines += [
        '1. COMPRESSION RATIO — zlib level 9, 95% bootstrap CI',
        '',
        f"  {'Phase':<8}  {'zlib ratio':>15}  {'95% CI':>15}     "
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
        verdict = 'YES ✓' if dvae_is_highest else 'NO ✗ — ordering not preserved on music'
        lines.append(f'  D-VAE ranks #1? {verdict}')
    lines.append('')

    lines += [
        '2. QUALITY — SI-SDR (dB), 95% bootstrap CI',
        '',
        f"  {'Phase':<8}  {'SI-SDR mean':>11}  {'95% CI':>17}",
        f'  {sep}',
    ]
    for label, _ in DEFAULT_PHASES:
        if label not in ci:
            lines.append(f'  {label:<8}  [skipped]')
            continue
        lines.append(f'  {label:<8}  {fc(ci[label]["si_sdr"], w=11, fmt=".2f")}')
    lines += [f'  {sep}', '']

    lines += [
        '3. WILCOXON SIGNED-RANK TESTS (paired, two-tailed)',
        f'   n = {n_actual} tracks',
        '',
        f"  {'Contrast':<16}  {'Metric':10}  {'W-stat':>8}  {'p-value':>8}  {'sig':>4}",
        f'  {sep}',
    ]
    if not SCIPY_OK:
        lines.append('  scipy not installed — pip install scipy')
    elif not wilcoxon_rows:
        lines.append('  No contrasts available (D-VAE or D/G checkpoint missing).')
    else:
        for row in wilcoxon_rows:
            for metric, W, p in [
                ('SI-SDR',     row['sisdr_W'], row['sisdr_p']),
                ('zlib ratio', row['ratio_W'], row['ratio_p']),
            ]:
                Wstr = f'{W:>8.1f}' if W is not None else '     n/a'
                lines.append(
                    f'  {row["label"]:<16}  {metric:<10}  {Wstr}  '
                    f'{fmt_p(p):>8}  {sig_stars(p):>4}'
                )
            lines.append(f'    ({row["interp"]})')
            lines.append('')
        lines.append('  Stars: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant')
    lines += ['', sep, '']

    lines += ['PER-TRACK RAW DATA', sep]
    for label, _ in DEFAULT_PHASES:
        if label not in results:
            continue
        lines.append(f'\n  Phase {label}')
        lines.append(f"  {'Track':<44}  {'kbps':>6}  {'SI-SDR':>8}  "
                     f"{'zlib ratio':>10}  {'Mean H':>7}")
        lines.append(f'  {"-"*80}')
        for r in results[label]:
            lines.append(
                f"  {r['track']:<44}  {r['kbps']:>6.2f}  "
                f"{r['si_sdr']:>+8.2f}  {r['zlib_ratio']:>10.3f}×  {r['mean_H']:>7.3f}"
            )
    lines += ['', SEP, '']

    report = '\n'.join(lines)
    print('\n' + report)

    out_dir = PROJECT_ROOT / 'comparisons' / f'{run_date}_music_eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'report.txt').write_text(report, encoding='utf-8')
    print(f'Saved: {out_dir / "report.txt"}')

    csv_path = out_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['phase', 'track', 'kbps', 'si_sdr', 'zlib_ratio', 'mean_H'])
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
