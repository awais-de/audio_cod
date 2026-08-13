#!/usr/bin/env python3
"""
R-D sweep (1-6 bit) for a channel-width variant checkpoint, on the same
n=40 speaker set used by eval_confidence_intervals.py / eval_paper_numbers.py.

Usage:
  python scripts/eval_width_rd_sweep.py --checkpoint checkpoints_active/temporal_phaseG_16/best.pt --label 16dim --device cuda
  python scripts/eval_width_rd_sweep.py --checkpoint checkpoints_active/temporal_phaseG_64/best.pt --label 64dim --device cuda

Output: comparisons/YYYY-MM-DD_rd_sweep_<label>/
  report.txt
  rd_sweep_<label>.csv
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.codec_utils import load_model, compute_metrics
from src.paths import get_dataset_paths

SR = 16_000
CLIP_SEC = 5
N_SPEAKERS = 40
BIT_DEPTHS = [1, 2, 3, 4, 5, 6]


def load_clip(path: Path) -> np.ndarray:
    audio, file_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        n = int(len(audio) * SR / file_sr)
        audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
    return np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32)


def collect_n(test_clean: Path, n: int) -> list:
    """First n speakers by numeric ID — matches eval_confidence_intervals.py."""
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


def encode_decode_nbits(model, audio: np.ndarray, num_levels: int, device) -> tuple:
    chunk_size = CLIP_SEC * SR
    recon_chunks, total_bits = [], 0
    with torch.no_grad():
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            if len(chunk) < 160:
                continue
            x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            z = model.encode(x)
            z_np = z.squeeze(0).cpu().numpy()

            z_min = float(z_np.min())
            z_max = float(z_np.max())
            scale = (z_max - z_min) / (num_levels - 1) + 1e-8
            q = np.clip(np.round((z_np - z_min) / scale), 0, num_levels - 1)
            dtype = np.uint8 if num_levels <= 256 else np.uint16
            raw = q.astype(dtype).tobytes()

            compressed = zlib.compress(raw, level=9)
            total_bits += len(compressed) * 8

            q_dec = np.frombuffer(zlib.decompress(compressed), dtype=dtype).reshape(z_np.shape)
            z_rec = q_dec.astype(np.float32) * scale + z_min
            x_rec = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))
            recon_chunks.append(x_rec.squeeze().cpu().numpy())

    recon = np.concatenate(recon_chunks) if recon_chunks else np.zeros_like(audio)
    recon = (recon[:len(audio)] if len(recon) >= len(audio)
             else np.pad(recon, (0, len(audio) - len(recon))))
    kbps = total_bits / (len(audio) / SR) / 1000
    return recon.astype(np.float32), kbps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--label', required=True, help='Short label for output dir/files, e.g. 16dim')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = args.device
    test_clean = get_dataset_paths()['test_clean']
    run_date = datetime.now().strftime('%Y-%m-%d')
    out_dir = PROJECT_ROOT / 'comparisons' / f'{run_date}_rd_sweep_{args.label}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Speakers   : first {N_SPEAKERS} by numeric ID (matches CI eval)")
    speakers = collect_n(test_clean, N_SPEAKERS)
    print(f"  Loaded {len(speakers)} speakers")

    model, meta = load_model(args.checkpoint, device)
    model.eval()
    print(f"  bottleneck_dim={meta.get('bottleneck_dim')}")

    all_rows = []
    for bits in BIT_DEPTHS:
        num_levels = 2 ** bits
        bdim = meta.get('bottleneck_dim', 32)
        theo_kbps = bdim * bits * 100 / 1000
        print(f"\n--- {bits}-bit ({num_levels} levels)  theoretical {theo_kbps:.1f} kbps ---")
        for spk, audio in speakers:
            recon, kbps = encode_decode_nbits(model, audio, num_levels, device)
            pesq_wb, stoi = compute_metrics(audio, recon, SR)
            pstr = f"{pesq_wb:.3f}" if pesq_wb is not None else "n/a"
            print(f"  {spk:>8}  kbps={kbps:.2f}  PESQ={pstr}  STOI={stoi:.3f}")
            all_rows.append({
                'bits': bits, 'num_levels': num_levels, 'theo_kbps': theo_kbps,
                'speaker': spk, 'kbps': kbps, 'pesq_wb': pesq_wb, 'stoi': stoi,
            })

    def _mean(b, key):
        vals = [r[key] for r in all_rows if r['bits'] == b and r[key] is not None]
        return float(np.mean(vals)) if vals else float('nan')

    summary = [
        {'bits': b, 'kbps': _mean(b, 'kbps'), 'pesq': _mean(b, 'pesq_wb'), 'stoi': _mean(b, 'stoi')}
        for b in BIT_DEPTHS
    ]

    lines = ['', '=' * 68, f'R-D SWEEP — {args.label}  (n={len(speakers)})',
             f'Checkpoint: {args.checkpoint}', '=' * 68, '',
             f"{'Bits':<6}{'Eff kbps':>10}{'PESQ-WB':>10}{'STOI':>8}", '-' * 34]
    for s in summary:
        mark = '  <- trained' if s['bits'] == 3 else ''
        lines.append(f"{s['bits']:<6}{s['kbps']:>9.2f}k{s['pesq']:>10.3f}{s['stoi']:>8.3f}{mark}")
    report = '\n'.join(lines)
    print('\n' + report)

    (out_dir / 'report.txt').write_text(report)
    with open(out_dir / f'rd_sweep_{args.label}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['bits', 'num_levels', 'theo_kbps', 'speaker', 'kbps', 'pesq_wb', 'stoi'])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nSaved: {out_dir}")


if __name__ == '__main__':
    main()
