#!/usr/bin/env python3
"""
Bitstream Corruption Robustness Test
=====================================
Measures how the Phase G codec degrades under random byte corruption of
the compressed bitstream — revealing whether it fails gracefully or
catastrophically.

Because zlib is a lossless codec with CRC checksums, byte corruption
almost always causes decompression to fail entirely (CRC error) rather
than producing degraded audio. This gives two modes:
  - Catastrophic failure rate (fraction of corruptions that crash decompression)
  - Graceful degradation curve (PESQ/STOI at successful decompression rates)

Output: comparisons/YYYY-MM-DD_corruption_test/
  report.txt   — per-corruption-rate summary
  metrics.csv  — full raw results
"""

import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import get_dataset_paths
from src.codec_utils import load_model, find_checkpoint, compute_metrics

SR           = 16000
CLIP_SEC     = 5
N_SPEAKERS   = 5
N_TRIALS     = 10      # independent corruption trials per (file, rate)
CORRUPT_RATES = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def corrupt_bytes(data: bytes, rate: float, rng: np.random.Generator) -> bytes:
    """Randomly flip bits in `data` at the given rate (per bit)."""
    if rate == 0.0:
        return data
    arr    = np.frombuffer(data, dtype=np.uint8).copy()
    n_bits = len(arr) * 8
    n_flip = max(1, int(n_bits * rate))
    bit_positions = rng.choice(n_bits, size=n_flip, replace=False)
    for bp in bit_positions:
        byte_idx = bp // 8
        bit_idx  = bp % 8
        arr[byte_idx] ^= (1 << bit_idx)
    return arr.tobytes()


def encode_chunk(model, chunk: np.ndarray, device):
    """Returns (compressed_bytes, z_shape, z_min, z_max, scale)."""
    num_levels = 8
    x   = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.encode(x)
    z_np  = z.squeeze(0).cpu().numpy()
    z_min = float(z_np.min())
    z_max = float(z_np.max())
    scale = (z_max - z_min) / (num_levels - 1) + 1e-8
    q     = np.clip(np.round((z_np - z_min) / scale), 0, num_levels - 1).astype(np.uint8)
    compressed = zlib.compress(q.tobytes(), level=9)
    return compressed, z_np.shape, z_min, z_max, scale


def decode_chunk(model, corrupted: bytes, z_shape, z_min, z_max, scale, device):
    """Returns (audio_chunk, success). success=False on decompression error."""
    try:
        raw = zlib.decompress(corrupted)
    except zlib.error:
        return None, False
    q_dec = np.frombuffer(raw, dtype=np.uint8).reshape(z_shape)
    z_rec = q_dec.astype(np.float32) * scale + z_min
    with torch.no_grad():
        x_recon = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))
    return x_recon.squeeze().cpu().numpy(), True


def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = find_checkpoint(PROJECT_ROOT)

    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir   = PROJECT_ROOT / 'comparisons' / f'{timestamp}_corruption_test'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print("BITSTREAM CORRUPTION ROBUSTNESS TEST")
    print(f"{'='*68}")
    print(f"checkpoint : {ckpt_path.parent.name}")
    print(f"device     : {device}")
    print(f"rates      : {CORRUPT_RATES}")
    print(f"trials/rate: {N_TRIALS}\n")

    model, _ = load_model(ckpt_path, device)
    model.eval()

    # Fixed canonical 5-speaker set (matches eval_phaseAB.py / eval_paper_numbers.py).
    # Do NOT use "first N found by directory traversal" -- adding/removing speaker
    # dirs from the dataset silently changes which N get picked (path string sort,
    # not speaker ID), breaking comparability across runs.
    TARGET_SPEAKERS = {'1089', '1188', '1221', '1284', '1320'}
    paths = get_dataset_paths()
    speakers = {}
    for f in sorted(paths['test_clean'].rglob('*.flac')):
        spk = f.parts[-3]
        if spk in TARGET_SPEAKERS and spk not in speakers:
            speakers[spk] = f
        if len(speakers) == min(N_SPEAKERS, len(TARGET_SPEAKERS)):
            break
    test_files = list(speakers.values())

    rng = np.random.default_rng(42)

    # Results: list of dicts with file, rate, trial, success, pesq, stoi
    all_rows = []

    for fidx, audio_path in enumerate(test_files, 1):
        spk = audio_path.parts[-3]
        audio, fsr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if fsr != SR:
            n     = int(len(audio) * SR / fsr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio = np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32)

        # Pre-encode the clean audio (single chunk)
        compressed, z_shape, z_min, z_max, scale = encode_chunk(model, audio, device)

        print(f"[{fidx}/{N_SPEAKERS}] speaker {spk}  "
              f"compressed={len(compressed)} bytes  "
              f"z_shape={z_shape}")

        for rate in CORRUPT_RATES:
            n_success = 0
            pesq_list, stoi_list = [], []
            for trial in range(N_TRIALS):
                corrupted = corrupt_bytes(compressed, rate, rng)
                recon, ok = decode_chunk(model, corrupted, z_shape, z_min, z_max, scale, device)
                if ok and recon is not None:
                    recon = recon[:len(audio)]
                    if len(recon) < len(audio):
                        recon = np.pad(recon, (0, len(audio) - len(recon)))
                    pesq, stoi = compute_metrics(audio, recon, SR)
                    n_success += 1
                    pesq_list.append(pesq)
                    stoi_list.append(stoi)
                all_rows.append({
                    'speaker': spk, 'corrupt_rate': rate, 'trial': trial,
                    'success': int(ok),
                    'pesq': pesq_list[-1] if ok else None,
                    'stoi': stoi_list[-1] if ok else None,
                })

            mean_pesq = float(np.mean([v for v in pesq_list if v is not None])) \
                        if pesq_list else float('nan')
            mean_stoi = float(np.mean([v for v in stoi_list if v is not None])) \
                        if stoi_list else float('nan')
            print(f"  rate={rate:.3f}  success={n_success}/{N_TRIALS}  "
                  f"pesq={mean_pesq:.3f}  stoi={mean_stoi:.3f}")

    # Aggregate across all speakers per rate
    SEP = '=' * 68
    sep = '-' * 68
    lines = [
        '', SEP,
        'BITSTREAM CORRUPTION ROBUSTNESS TEST',
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Checkpoint: {ckpt_path.parent.name}",
        f"Files: {N_SPEAKERS} speakers x {CLIP_SEC}s  |  {N_TRIALS} trials per rate",
        SEP, '',
        f"{'Corrupt rate':>14} {'Success rate':>14} {'PESQ-WB (ok)':>14} {'STOI (ok)':>11}",
        sep,
    ]

    for rate in CORRUPT_RATES:
        rate_rows = [r for r in all_rows if r['corrupt_rate'] == rate]
        n_total   = len(rate_rows)
        n_ok      = sum(r['success'] for r in rate_rows)
        succ_rate = n_ok / n_total if n_total > 0 else 0.0
        pesq_vals = [r['pesq'] for r in rate_rows if r['pesq'] is not None]
        stoi_vals = [r['stoi'] for r in rate_rows if r['stoi'] is not None]
        mean_p    = float(np.mean(pesq_vals)) if pesq_vals else float('nan')
        mean_s    = float(np.mean(stoi_vals)) if stoi_vals else float('nan')
        lines.append(
            f"  {rate:>12.3f}   {succ_rate:>12.1%}   {mean_p:>12.3f}   {mean_s:>9.3f}"
        )

    lines += [
        sep, '',
        'NOTE: zlib CRC checking means most bit flips cause total decompression',
        'failure rather than graceful audio degradation. Success rate quantifies',
        'the codec\'s tolerance for channel errors.',
        SEP,
    ]
    report = '\n'.join(lines)
    print(report)

    with open(out_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    with open(out_dir / 'metrics.csv', 'w', encoding='utf-8') as f:
        f.write('speaker,corrupt_rate,trial,success,pesq_wb,stoi\n')
        for r in all_rows:
            f.write(f"{r['speaker']},{r['corrupt_rate']:.4f},{r['trial']},"
                    f"{r['success']},{r['pesq'] or ''},{r['stoi'] or ''}\n")

    print(f"\nreport:      {out_dir}/report.txt")
    print(f"metrics csv: {out_dir}/metrics.csv")


if __name__ == '__main__':
    main()
