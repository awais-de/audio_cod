#!/usr/bin/env python3
"""
Rate-Distortion Sweep — Scalar Quantization vs EnCodec (RVQ)
=============================================================
Evaluates Phase G at varying quantization bit depths (1-bit through 6-bit)
to trace an empirical R-D curve for scalar quantization.

Compared against EnCodec reference points (real PESQ/STOI measured on the
same LibriSpeech test-clean clips) to quantify the cost of scalar vs.
residual vector quantization at matched bitrate.

Bit depths tested:
  1-bit  (2  levels)  ~2-3 kbps effective
  2-bit  (4  levels)  ~4-5 kbps effective
  3-bit  (8  levels)  ~5-6 kbps effective  ← trained operating point
  4-bit  (16 levels)  ~8-9 kbps effective
  5-bit  (32 levels)  ~11-12 kbps effective
  6-bit  (64 levels)  ~14-15 kbps effective

Caveat: Phase G was trained for 3-bit only. Other bit depths represent
inference-time variation — valid for characterising the R-D slope but
sub-optimal relative to per-bitrate trained models.

Output: comparisons/YYYY-MM-DD_rd_sweep/
  report.txt      — human-readable table
  rd_curve.csv    — (bit_depth, num_levels, kbps, pesq, stoi) per speaker
  rd_curve.png    — R-D plot (requires matplotlib)
"""

import sys
import math
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

SR         = 16000
CLIP_SEC   = 5
N_SPEAKERS = 5

# Bit depths to sweep
BIT_DEPTHS = [1, 2, 3, 4, 5, 6]   # → num_levels = 2^b

# EnCodec reference points (real PESQ/STOI on LibriSpeech test-clean,
# 5 speakers × 5s, from offline evaluation on macOS with real pesq/pystoi)
ENCODEC_REF = [
    {'label': 'EnCodec 1.5 kbps', 'kbps': 1.5,  'pesq': 1.611, 'stoi': 0.829},
    {'label': 'EnCodec 3.0 kbps', 'kbps': 3.0,  'pesq': 2.148, 'stoi': 0.880},
    {'label': 'EnCodec 6.0 kbps', 'kbps': 6.0,  'pesq': 2.842, 'stoi': 0.922},
]


def encode_decode_nbits(model, audio: np.ndarray, sr: int, device,
                        num_levels: int, chunk_sec: float = 5.0):
    """Encode-decode with arbitrary number of quantization levels."""
    chunk_size  = int(chunk_sec * sr)
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
            q     = np.clip(np.round((z_np - z_min) / scale),
                            0, num_levels - 1).astype(np.uint8)

            # Pack into bytes: use uint16 for num_levels > 256
            if num_levels <= 256:
                raw = q.astype(np.uint8).tobytes()
            else:
                raw = q.astype(np.uint16).tobytes()

            compressed   = zlib.compress(raw, level=9)
            total_bits  += len(compressed) * 8

            q_dec  = np.frombuffer(zlib.decompress(compressed),
                                   dtype=np.uint8 if num_levels <= 256 else np.uint16
                                   ).reshape(z_np.shape)
            z_rec  = q_dec.astype(np.float32) * scale + z_min
            x_rec  = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))
            recon_chunks.append(x_rec.squeeze().cpu().numpy())

    recon    = np.concatenate(recon_chunks) if recon_chunks else np.zeros_like(audio)
    recon    = recon[:len(audio)] if len(recon) >= len(audio) \
               else np.pad(recon, (0, len(audio) - len(recon)))
    duration = len(audio) / sr
    kbps     = total_bits / duration / 1000
    return recon.astype(np.float32), kbps


def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = find_checkpoint(PROJECT_ROOT)

    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir   = PROJECT_ROOT / 'comparisons' / f'{timestamp}_rd_sweep'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print("RATE-DISTORTION SWEEP — Phase G scalar quantization vs EnCodec RVQ")
    print(f"{'='*72}")
    print(f"checkpoint : {ckpt_path.parent.name}")
    print(f"device     : {device}")
    print(f"bit depths : {BIT_DEPTHS}  ({[2**b for b in BIT_DEPTHS]} levels)")
    print(f"speakers   : {N_SPEAKERS} × {CLIP_SEC}s")
    print(f"output     : {out_dir}\n")

    model, _ = load_model(ckpt_path, device)
    model.eval()

    paths = get_dataset_paths()
    speakers = {}
    for f in sorted(paths['test_clean'].rglob('*.flac')):
        spk = f.parts[-3]
        if spk not in speakers:
            speakers[spk] = f
        if len(speakers) == N_SPEAKERS:
            break
    test_files = list(speakers.values())

    # Load audio clips once
    clips = []
    for fpath in test_files:
        audio, fsr = sf.read(fpath)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if fsr != SR:
            n     = int(len(audio) * SR / fsr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio = np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32)
        clips.append((fpath.parts[-3], audio))

    all_rows = []

    for bits in BIT_DEPTHS:
        num_levels = 2 ** bits
        theo_kbps  = 32 * bits * 100 / 1000      # 32 dims × bits × 100 Hz
        print(f"\n--- {bits}-bit ({num_levels} levels)  theoretical cap: {theo_kbps:.1f} kbps ---")

        for spk, audio in clips:
            recon, kbps = encode_decode_nbits(model, audio, SR, device, num_levels)
            pesq, stoi  = compute_metrics(audio, recon, SR)
            p_str = f"{pesq:.3f}" if pesq is not None else "n/a"
            s_str = f"{stoi:.3f}" if stoi is not None else "n/a"
            print(f"  {spk}  kbps={kbps:.2f}  pesq={p_str}  stoi={s_str}")
            all_rows.append({
                'bits': bits, 'num_levels': num_levels,
                'speaker': spk, 'kbps': kbps, 'pesq': pesq, 'stoi': stoi,
            })

    # Aggregate per bit depth
    def mean_by_bits(b, key):
        vals = [r[key] for r in all_rows if r['bits'] == b and r[key] is not None]
        return float(np.mean(vals)) if vals else float('nan')

    sweep_summary = []
    for bits in BIT_DEPTHS:
        sweep_summary.append({
            'bits':       bits,
            'num_levels': 2 ** bits,
            'theo_kbps':  32 * bits * 100 / 1000,
            'kbps':       mean_by_bits(bits, 'kbps'),
            'pesq':       mean_by_bits(bits, 'pesq'),
            'stoi':       mean_by_bits(bits, 'stoi'),
        })

    # Report
    SEP = '=' * 72
    sep = '-' * 72
    lines = [
        '', SEP,
        'RATE-DISTORTION SWEEP — Phase G scalar quantization',
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Checkpoint: {ckpt_path.parent.name}  (trained at 3-bit)",
        f"Test set: {N_SPEAKERS} speakers × {CLIP_SEC}s  |  16 kHz mono",
        SEP, '',
        'OURS (scalar quantization, Phase G weights)',
        f"  {'Bits':<6} {'Levels':<8} {'Theo kbps':>10} {'Eff kbps':>10} "
        f"{'PESQ-WB':>9} {'STOI':>7}",
        sep,
    ]
    for s in sweep_summary:
        p_str = f"{s['pesq']:.3f}" if not math.isnan(s['pesq']) else "n/a"
        st_str = f"{s['stoi']:.3f}" if not math.isnan(s['stoi']) else "n/a"
        marker = '  <- trained' if s['bits'] == 3 else ''
        lines.append(
            f"  {s['bits']:<6} {s['num_levels']:<8} {s['theo_kbps']:>9.1f}k "
            f"{s['kbps']:>9.2f}k {p_str:>9} {st_str:>7}{marker}"
        )

    lines += ['', 'ENCODEC REFERENCE (RVQ, reported values)', sep]
    for ref in ENCODEC_REF:
        lines.append(f"  {ref['label']:<28} {ref['kbps']:>9.1f}k "
                     f"{ref['pesq']:>9.3f} {ref['stoi']:>7.3f}")

    lines += [
        sep, '',
        'NOTE: Phase G was trained at 3-bit. Other operating points show',
        'inference-time bit-depth variation and are sub-optimal relative to',
        'models trained per bitrate. The 3-bit point is the true ceiling.',
        SEP,
    ]
    report = '\n'.join(lines)
    print('\n' + report)

    with open(out_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # CSV — per-speaker rows
    with open(out_dir / 'rd_curve.csv', 'w', encoding='utf-8') as f:
        f.write('bits,num_levels,theo_kbps,speaker,eff_kbps,pesq_wb,stoi\n')
        for r in all_rows:
            theo = 32 * r['bits'] * 100 / 1000
            f.write(f"{r['bits']},{r['num_levels']},{theo:.1f},"
                    f"{r['speaker']},{r['kbps']:.4f},"
                    f"{r['pesq'] or ''},{r['stoi'] or ''}\n")

    # EnCodec reference CSV
    with open(out_dir / 'encodec_ref.csv', 'w', encoding='utf-8') as f:
        f.write('label,kbps,pesq_wb,stoi\n')
        for ref in ENCODEC_REF:
            f.write(f"{ref['label']},{ref['kbps']},{ref['pesq']},{ref['stoi']}\n")

    # Plot
    _plot(sweep_summary, out_dir)

    print(f"\nreport:      {out_dir}/report.txt")
    print(f"rd csv:      {out_dir}/rd_curve.csv")
    print(f"encodec csv: {out_dir}/encodec_ref.csv")
    print(f"plot:        {out_dir}/rd_curve.png  (if matplotlib available)")


def _plot(sweep_summary, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Rate-Distortion: Scalar Quantization (Phase G) vs EnCodec (RVQ)',
                 fontsize=12, fontweight='bold')

    our_kbps  = [s['kbps']  for s in sweep_summary]
    our_pesq  = [s['pesq']  for s in sweep_summary]
    our_stoi  = [s['stoi']  for s in sweep_summary]
    bit_labels = [f"{s['bits']}b" for s in sweep_summary]

    enc_kbps = [r['kbps']  for r in ENCODEC_REF]
    enc_pesq = [r['pesq']  for r in ENCODEC_REF]
    enc_stoi = [r['stoi']  for r in ENCODEC_REF]

    # PESQ plot
    has_pesq = not all(math.isnan(v) for v in our_pesq)
    if has_pesq:
        valid_k = [k for k, p in zip(our_kbps, our_pesq) if not math.isnan(p)]
        valid_p = [p for p in our_pesq if not math.isnan(p)]
        ax1.plot(valid_k, valid_p, 'o-', color='steelblue',
                 label='Ours (scalar)', linewidth=2, markersize=7)
        for k, p, lbl in zip(our_kbps, our_pesq, bit_labels):
            if not math.isnan(p):
                ax1.annotate(lbl, (k, p), textcoords='offset points',
                             xytext=(4, 4), fontsize=8, color='steelblue')
        ax1.plot(enc_kbps, enc_pesq, 's--', color='darkorange',
                 label='EnCodec (RVQ)', linewidth=2, markersize=7)
        ax1.set_xlabel('Effective bitrate (kbps)')
        ax1.set_ylabel('PESQ-WB')
        ax1.set_title('PESQ-WB vs Bitrate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=1.0, top=4.5)
    else:
        ax1.text(0.5, 0.5, 'PESQ not available\n(install pesq + python dev headers)',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=10,
                 color='gray')
        ax1.set_title('PESQ-WB vs Bitrate')

    # STOI plot
    valid_k_s = [k for k, s in zip(our_kbps, our_stoi) if not math.isnan(s)]
    valid_s   = [s for s in our_stoi if not math.isnan(s)]
    ax2.plot(valid_k_s, valid_s, 'o-', color='steelblue',
             label='Ours (scalar)', linewidth=2, markersize=7)
    for k, s, lbl in zip(our_kbps, our_stoi, bit_labels):
        if not math.isnan(s):
            ax2.annotate(lbl, (k, s), textcoords='offset points',
                         xytext=(4, 4), fontsize=8, color='steelblue')
    ax2.plot(enc_kbps, enc_stoi, 's--', color='darkorange',
             label='EnCodec (RVQ)', linewidth=2, markersize=7)
    ax2.set_xlabel('Effective bitrate (kbps)')
    ax2.set_ylabel('STOI')
    ax2.set_title('STOI vs Bitrate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0.0, top=1.0)

    plt.tight_layout()
    plot_path = out_dir / 'rd_curve.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  plot saved: {plot_path}")


if __name__ == '__main__':
    main()
