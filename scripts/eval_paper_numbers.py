#!/usr/bin/env python3
"""
eval_paper_numbers.py — Combined evaluation for the 20 CP project paper.

Runs two evaluations in sequence and writes all results to a single
dated output directory under comparisons/:

  Part 1 — Phase A / B PESQ
    Targets the canonical 5-speaker set used in the 2026-07-10 phaseAB eval
    (speakers 1089, 1188, 1221, 1284, 1320), so the PESQ column fills in the
    n/a entries in the existing report without changing kbps or STOI.

  Part 2 — R-D sweep on n=40
    Runs the Phase G bit-depth sweep (1–6 bit) on the same 40-speaker set
    used by eval_confidence_intervals.py (first 40 speakers, sorted by numeric
    ID), so the 3-bit anchor point matches the n=40 CI evaluation exactly.

Phase G n=40 kbps is already recorded in comparisons/2026-07-17_confidence_intervals/
report.txt (5.90 kbps).  No additional run is needed for that number.

Output: comparisons/YYYY-MM-DD_paper_numbers/
  report.txt          — human-readable results, ready to copy back
  phaseAB_metrics.csv — per-speaker Phase A/B metrics
  rd_sweep_n40.csv    — per-speaker R-D sweep metrics (n=40)

Usage (run from project root on Windows):
  python scripts/eval_paper_numbers.py
  python scripts/eval_paper_numbers.py --librispeech "C:\\data\\LibriSpeech\\test-clean"
  python scripts/eval_paper_numbers.py --device cuda

Requirements (pip install if missing):
  pesq pystoi scipy soundfile torch numpy
"""

import argparse
import csv
import math
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.codec_utils import load_model, encode_decode, compute_metrics
from src.paths import get_dataset_paths

# ── constants ─────────────────────────────────────────────────────────────────

SR       = 16_000
CLIP_SEC = 5

# Canonical 5-speaker set for Phase A/B (matches 2026-07-10_phaseAB_eval).
PHASE_AB_SPEAKERS = {'1089', '1188', '1221', '1284', '1320'}

N_SPEAKERS_RD = 40   # matches eval_confidence_intervals.py

BIT_DEPTHS = [1, 2, 3, 4, 5, 6]

PHASE_A_CKPT = PROJECT_ROOT / 'checkpoints_active/temporal_phaseA/best.pt'
PHASE_B_CKPT = PROJECT_ROOT / 'checkpoints_active/temporal_phaseB/best.pt'
PHASE_G_CKPT = PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt'

ENCODEC_REF = [
    {'label': 'EnCodec 1.5 kbps', 'kbps': 1.5,  'pesq': 1.611, 'stoi': 0.829},
    {'label': 'EnCodec 3.0 kbps', 'kbps': 3.0,  'pesq': 2.148, 'stoi': 0.880},
    {'label': 'EnCodec 6.0 kbps', 'kbps': 6.0,  'pesq': 2.842, 'stoi': 0.922},
]


# ── audio helpers ─────────────────────────────────────────────────────────────

def load_clip(path: Path) -> np.ndarray:
    audio, file_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        n = int(len(audio) * SR / file_sr)
        audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
    return np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32)


def collect_targeted(test_clean: Path, target_ids: set) -> list:
    """Collect (speaker_id, audio) for a fixed set of speaker IDs."""
    found = {}
    for f in sorted(test_clean.rglob('*.flac')):
        spk = f.parts[-3]
        if spk in target_ids and spk not in found:
            found[spk] = load_clip(f)
        if len(found) == len(target_ids):
            break
    missing = target_ids - set(found)
    if missing:
        print(f"  WARNING: speakers not found in dataset: {sorted(missing)}")
    return [(spk, found[spk]) for spk in sorted(found)]


def collect_n(test_clean: Path, n: int) -> list:
    """Collect the first n speakers by numeric ID — matches eval_confidence_intervals.py."""
    by_speaker = {}
    for f in sorted(test_clean.rglob('*.flac')):
        spk = f.parts[-3]
        if spk not in by_speaker:
            by_speaker[spk] = f

    def spk_key(s):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)

    sorted_ids = sorted(by_speaker, key=spk_key)[:n]
    return [(spk, load_clip(by_speaker[spk])) for spk in sorted_ids]


# ── Part 1: Phase A / B ───────────────────────────────────────────────────────

def _eval_one_phase(name: str, ckpt: Path, speakers: list, device) -> list:
    print(f"\n  Loading {name} …", end=' ', flush=True)
    model, meta = load_model(ckpt, device)
    model.eval()
    print(f"ok  (bottleneck_dim={meta.get('bottleneck_dim')})")
    rows = []
    for spk, audio in speakers:
        recon, kbps = encode_decode(model, audio, SR, device, chunk_sec=float(CLIP_SEC))
        n_common = min(len(audio), len(recon))
        pesq_wb, stoi = compute_metrics(audio[:n_common], recon[:n_common], SR)
        rows.append({'speaker': spk, 'kbps': kbps, 'pesq_wb': pesq_wb, 'stoi': stoi})
        pstr = f"{pesq_wb:.3f}" if pesq_wb is not None else "n/a"
        print(f"    {spk:>8}  kbps={kbps:.2f}  PESQ-WB={pstr}  STOI={stoi:.3f}")
    del model
    return rows


def run_phaseAB(test_clean: Path, device: str) -> dict:
    print(f"\n{'='*68}")
    print("PART 1 — PHASE A / B  (5-speaker canonical set)")
    print(f"  Speakers : {sorted(PHASE_AB_SPEAKERS)}")
    print(f"  Clip     : {CLIP_SEC}s  |  {SR} Hz mono")
    print(f"{'='*68}")

    speakers = collect_targeted(test_clean, PHASE_AB_SPEAKERS)
    if not speakers:
        print("  ERROR: no matching speakers found — check --librispeech path")
        return {}

    phases = [('Phase A', PHASE_A_CKPT), ('Phase B', PHASE_B_CKPT)]
    results = {}
    for name, ckpt in phases:
        if not ckpt.exists():
            print(f"\n  [SKIP] {name}: checkpoint not found at {ckpt}")
            continue
        results[name] = _eval_one_phase(name, ckpt, speakers, device)
    return results


# ── Part 2: R-D sweep on n=40 ─────────────────────────────────────────────────

def _encode_decode_nbits(model, audio: np.ndarray, num_levels: int,
                         device) -> tuple:
    """Encode-decode at an arbitrary quantization depth. Returns (recon, kbps)."""
    chunk_size = CLIP_SEC * SR
    recon_chunks, total_bits = [], 0
    with torch.no_grad():
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            if len(chunk) < 160:
                continue
            x    = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            z    = model.encode(x)
            z_np = z.squeeze(0).cpu().numpy()

            z_min = float(z_np.min())
            z_max = float(z_np.max())
            scale = (z_max - z_min) / (num_levels - 1) + 1e-8
            q     = np.clip(np.round((z_np - z_min) / scale), 0, num_levels - 1)
            dtype = np.uint8 if num_levels <= 256 else np.uint16
            raw   = q.astype(dtype).tobytes()

            compressed  = zlib.compress(raw, level=9)
            total_bits += len(compressed) * 8

            q_dec = np.frombuffer(zlib.decompress(compressed),
                                  dtype=dtype).reshape(z_np.shape)
            z_rec = q_dec.astype(np.float32) * scale + z_min
            x_rec = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))
            recon_chunks.append(x_rec.squeeze().cpu().numpy())

    recon = np.concatenate(recon_chunks) if recon_chunks else np.zeros_like(audio)
    recon = (recon[:len(audio)] if len(recon) >= len(audio)
             else np.pad(recon, (0, len(audio) - len(recon))))
    kbps  = total_bits / (len(audio) / SR) / 1000
    return recon.astype(np.float32), kbps


def run_rd_sweep(test_clean: Path, device: str) -> tuple:
    print(f"\n{'='*68}")
    print(f"PART 2 — R-D SWEEP  (Phase G, n={N_SPEAKERS_RD})")
    print(f"  Speakers : first {N_SPEAKERS_RD} by numeric ID (matches CI eval)")
    print(f"  Clip     : {CLIP_SEC}s  |  {SR} Hz mono")
    print(f"  Bits     : {BIT_DEPTHS}  ({[2**b for b in BIT_DEPTHS]} levels)")
    print(f"{'='*68}")

    if not PHASE_G_CKPT.exists():
        print(f"  ERROR: Phase G checkpoint not found at {PHASE_G_CKPT}")
        return [], []

    speakers = collect_n(test_clean, N_SPEAKERS_RD)
    print(f"  Loaded {len(speakers)} speakers: "
          f"{[s for s, _ in speakers[:5]]} … {[s for s, _ in speakers[-2:]]}")

    print(f"\n  Loading Phase G …", end=' ', flush=True)
    model, _ = load_model(PHASE_G_CKPT, device)
    model.eval()
    print("ok")

    all_rows = []
    for bits in BIT_DEPTHS:
        num_levels = 2 ** bits
        theo_kbps  = 32 * bits * 100 / 1000
        print(f"\n  --- {bits}-bit ({num_levels} levels)  theoretical {theo_kbps:.1f} kbps ---")
        for spk, audio in speakers:
            recon, kbps = _encode_decode_nbits(model, audio, num_levels, device)
            pesq_wb, stoi = compute_metrics(audio, recon, SR)
            pstr = f"{pesq_wb:.3f}" if pesq_wb is not None else "n/a"
            print(f"    {spk:>8}  kbps={kbps:.2f}  PESQ={pstr}  STOI={stoi:.3f}")
            all_rows.append({
                'bits': bits, 'num_levels': num_levels, 'theo_kbps': theo_kbps,
                'speaker': spk, 'kbps': kbps, 'pesq_wb': pesq_wb, 'stoi': stoi,
            })

    del model

    def _mean(b, key):
        vals = [r[key] for r in all_rows if r['bits'] == b and r[key] is not None]
        return float(np.mean(vals)) if vals else float('nan')

    summary = [
        {'bits': b, 'num_levels': 2**b, 'theo_kbps': 32*b*100/1000,
         'kbps': _mean(b, 'kbps'), 'pesq': _mean(b, 'pesq_wb'), 'stoi': _mean(b, 'stoi')}
        for b in BIT_DEPTHS
    ]
    return all_rows, summary


# ── report builder ────────────────────────────────────────────────────────────

def _mean_field(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def build_report(phaseAB: dict, rd_rows: list, rd_summary: list,
                 timestamp: str, device: str) -> str:
    SEP = '=' * 68
    sep = '-' * 68
    n_rd = len({r['speaker'] for r in rd_rows}) if rd_rows else 0

    lines = [
        '', SEP,
        'PAPER NUMBERS EVALUATION — 20 CP Project Report',
        f'Generated : {timestamp}',
        f'Device    : {device}',
        SEP,
    ]

    # ── Part 1 ───────────────────────────────────────────────────────────────
    lines += ['', f'PART 1 — PHASE A / B', f'  Speaker set: {sorted(PHASE_AB_SPEAKERS)}', sep]
    if not phaseAB:
        lines.append('  [no results — checkpoints missing or speakers not found]')
    else:
        lines.append(f"  {'Phase':<12} {'kbps':>6}  {'PESQ-WB':>8}  {'STOI':>6}")
        lines.append(f"  {'-'*42}")
        for name, rows in phaseAB.items():
            kbps_m = _mean_field(rows, 'kbps')
            pesq_m = _mean_field(rows, 'pesq_wb')
            stoi_m = _mean_field(rows, 'stoi')
            pstr   = f"{pesq_m:.3f}" if pesq_m is not None else "n/a"
            lines.append(
                f"  {name:<12} {kbps_m:>5.2f}k  {pstr:>8}  "
                f"{stoi_m:>6.3f}" if stoi_m is not None else f"  {name:<12} n/a"
            )
        lines.append('')
        lines.append('  PER-SPEAKER')
        for name, rows in phaseAB.items():
            lines += [f"\n  {name}",
                      f"  {'Speaker':>10}  {'kbps':>6}  {'PESQ-WB':>8}  {'STOI':>6}",
                      f"  {'-'*42}"]
            for r in rows:
                pstr = f"{r['pesq_wb']:.3f}" if r['pesq_wb'] is not None else "     n/a"
                lines.append(
                    f"  {r['speaker']:>10}  {r['kbps']:>6.2f}  {pstr:>8}  {r['stoi']:>6.3f}"
                )

    # ── Part 2 ───────────────────────────────────────────────────────────────
    lines += [
        '', '', SEP,
        f'PART 2 — R-D SWEEP  (Phase G, n={n_rd})',
        f'  Checkpoint : temporal_phaseG/best.pt  (trained at 3-bit)',
        f'  Speaker set: first {n_rd} numeric IDs from test-clean (matches CI eval)',
        sep,
    ]
    if not rd_summary:
        lines.append('  [no results — Phase G checkpoint not found]')
    else:
        lines.append(
            f"  {'Bits':<6} {'Levels':<8} {'Theo kbps':>10} {'Eff kbps':>10} "
            f"{'PESQ-WB':>9} {'STOI':>7}"
        )
        lines.append(f"  {sep}")
        for s in rd_summary:
            pstr  = f"{s['pesq']:.3f}" if not math.isnan(s['pesq'])  else "    n/a"
            ststr = f"{s['stoi']:.3f}" if not math.isnan(s['stoi'])  else "    n/a"
            mark  = '  <- trained' if s['bits'] == 3 else ''
            lines.append(
                f"  {s['bits']:<6} {s['num_levels']:<8} {s['theo_kbps']:>9.1f}k "
                f"{s['kbps']:>9.2f}k {pstr:>9} {ststr:>7}{mark}"
            )
        lines += ['', '  ENCODEC REFERENCE', sep]
        for ref in ENCODEC_REF:
            lines.append(
                f"  {ref['label']:<28} {ref['kbps']:>9.1f}k "
                f"{ref['pesq']:>9.3f} {ref['stoi']:>7.3f}"
            )

    lines += ['', SEP, '']
    return '\n'.join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--librispeech', type=Path, default=None,
                   help='Path to LibriSpeech test-clean. Auto-detected from '
                        'paths.yaml if omitted.')
    p.add_argument('--device', default='cpu',
                   help='Torch device: cpu or cuda (default: cpu).')
    return p.parse_args()


def main():
    args      = parse_args()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    run_date  = datetime.now().strftime('%Y-%m-%d')

    # Resolve test-clean
    if args.librispeech is not None:
        test_clean = Path(args.librispeech)
    else:
        try:
            test_clean = get_dataset_paths()['test_clean']
        except Exception:
            test_clean = PROJECT_ROOT.parent / 'datasets' / 'LibriSpeech' / 'test-clean'
    if not test_clean.exists():
        print(f"ERROR: test-clean not found: {test_clean}")
        print("Pass --librispeech /path/to/LibriSpeech/test-clean")
        sys.exit(1)
    print(f"LibriSpeech test-clean : {test_clean}")

    out_dir = PROJECT_ROOT / 'comparisons' / f'{run_date}_paper_numbers'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory       : {out_dir}\n")

    phaseAB              = run_phaseAB(test_clean, args.device)
    rd_rows, rd_summary  = run_rd_sweep(test_clean, args.device)

    report = build_report(phaseAB, rd_rows, rd_summary, timestamp, args.device)
    print('\n' + report)
    (out_dir / 'report.txt').write_text(report, encoding='utf-8')
    print(f"Saved : {out_dir / 'report.txt'}")

    if phaseAB:
        ab_csv = out_dir / 'phaseAB_metrics.csv'
        with open(ab_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f,
                               fieldnames=['phase', 'speaker', 'kbps', 'pesq_wb', 'stoi'])
            w.writeheader()
            for name, rows in phaseAB.items():
                for r in rows:
                    w.writerow({'phase': name, **r})
        print(f"Saved : {ab_csv}")

    if rd_rows:
        rd_csv = out_dir / 'rd_sweep_n40.csv'
        with open(rd_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                'bits', 'num_levels', 'theo_kbps', 'speaker', 'kbps', 'pesq_wb', 'stoi',
            ])
            w.writeheader()
            for r in rd_rows:
                w.writerow(r)
        print(f"Saved : {rd_csv}")

    print(f"\nDone. Copy comparisons/{run_date}_paper_numbers/ back to main machine.")


if __name__ == '__main__':
    main()
