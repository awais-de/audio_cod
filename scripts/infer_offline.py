#!/usr/bin/env python3
"""
Offline inference: encode a single audio file, decode, and evaluate round-trip quality.

Usage:
  python scripts/infer_offline.py
  python scripts/infer_offline.py --input /path/to/speech.wav
  python scripts/infer_offline.py --input audio.flac --checkpoint checkpoints_active/temporal_phaseC/best.pt

Each run writes to inference_runs/<timestamp>/:
  source.wav          resampled/normalised input (what was fed to the encoder)
  reconstructed.wav   decoded output
  metrics.json        all quantitative results for this run
"""

import sys
import json
import zlib
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.codec_utils import find_checkpoint, load_model
from src.paths import get_dataset_paths

try:
    from pesq import pesq as pesq_fn
    PESQ_OK = True
except ImportError:
    PESQ_OK = False

try:
    from pystoi import stoi as stoi_fn
    STOI_OK = True
except ImportError:
    STOI_OK = False

NUM_LEVELS = 8


def find_default_input() -> Path:
    paths = get_dataset_paths()
    test_clean = paths['test_clean']
    for f in sorted(test_clean.rglob('*.flac')):
        return f
    raise FileNotFoundError(
        f"No .flac files found in {test_clean}. "
        "Pass --input explicitly or check your dataset path."
    )


def run(args):
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_checkpoint(PROJECT_ROOT)
    input_path = Path(args.input) if args.input else find_default_input()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = PROJECT_ROOT / 'inference_runs' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, ckpt = load_model(ckpt_path, device)

    # Load and prepare audio
    audio, sr_orig = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    sr = args.sample_rate
    if sr_orig != sr:
        n = int(len(audio) * sr / sr_orig)
        audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    n_samples = len(audio)
    duration = n_samples / sr

    if n_samples == 0:
        raise ValueError(f"Audio file is empty: {input_path}")

    sf.write(out_dir / 'source.wav', audio, sr)

    # Encode → decode round-trip
    chunk_sz = int(args.chunk_sec * sr)
    recon_chunks = []
    total_bits = 0
    n_skipped = 0
    latent_shape = None

    with torch.no_grad():
        for start in range(0, n_samples, chunk_sz):
            chunk = audio[start:start + chunk_sz]
            if len(chunk) < 160:
                n_skipped += 1
                continue

            x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            z = model.encode(x)
            z_np = z.squeeze(0).cpu().numpy()

            if latent_shape is None:
                latent_shape = list(z_np.shape)

            z_min, z_max = float(z_np.min()), float(z_np.max())
            scale = (z_max - z_min) / (NUM_LEVELS - 1) + 1e-8
            q = np.clip(np.round((z_np - z_min) / scale), 0, NUM_LEVELS - 1).astype(np.uint8)
            comp = zlib.compress(q.tobytes(), level=9)
            total_bits += len(comp) * 8

            q_dec = np.frombuffer(zlib.decompress(comp), dtype=np.uint8).reshape(z_np.shape)
            z_rec = q_dec.astype(np.float32) * scale + z_min
            x_recon = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))
            recon_chunks.append(x_recon.squeeze().cpu().numpy())

    if not recon_chunks:
        raise RuntimeError("No chunks were processed — input may be too short.")

    recon = np.concatenate(recon_chunks)
    if len(recon) >= n_samples:
        recon = recon[:n_samples]
    else:
        recon = np.pad(recon, (0, n_samples - len(recon)))
    recon = recon.astype(np.float32)

    sf.write(out_dir / 'reconstructed.wav', recon, sr)

    # Sanity: output length
    assert len(recon) == n_samples, \
        f"Length mismatch: expected {n_samples}, got {len(recon)}"

    # Metrics
    kbps = total_bits / duration / 1000

    n = min(n_samples, len(recon))
    ref_raw = audio[:n]
    deg_raw = recon[:n]

    # SNR on raw (unscaled) signals
    sig_pwr = np.mean(ref_raw ** 2)
    err_pwr = np.mean((ref_raw - deg_raw) ** 2) + 1e-12
    snr_db = float(10 * np.log10(sig_pwr / err_pwr))

    # PESQ / STOI require amplitude-normalised signals
    ref_n = ref_raw / (np.abs(ref_raw).max() + 1e-8)
    deg_n = deg_raw / (np.abs(deg_raw).max() + 1e-8)
    pesq_score = float(pesq_fn(sr, ref_n, deg_n, 'wb')) if PESQ_OK else None
    stoi_score = float(stoi_fn(ref_n, deg_n, sr, extended=False)) if STOI_OK else None

    metrics = {
        'input': str(input_path),
        'checkpoint': str(ckpt_path),
        'phase': ckpt.get('phase', 'unknown'),
        'd_model': ckpt.get('d_model'),
        'bottleneck_dim': ckpt.get('bottleneck_dim'),
        'temporal_stride': ckpt.get('temporal_stride'),
        'sample_rate': sr,
        'duration_s': round(duration, 3),
        'chunk_sec': args.chunk_sec,
        'n_chunks': len(recon_chunks),
        'n_chunks_skipped': n_skipped,
        'latent_shape_per_chunk': latent_shape,
        'bitrate_kbps': round(kbps, 3),
        'snr_db': round(snr_db, 2),
        'pesq_wb': round(pesq_score, 4) if pesq_score is not None else None,
        'stoi': round(stoi_score, 4) if stoi_score is not None else None,
        'output_dir': str(out_dir),
    }

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Minimal stdout
    print(f"input:       {input_path.name}  ({duration:.2f}s, {sr_orig}Hz → {sr}Hz)")
    print(f"checkpoint:  {ckpt_path.parent.name}/{ckpt_path.name}  "
          f"(phase={metrics['phase']}, d_model={metrics['d_model']}, "
          f"bottleneck={metrics['bottleneck_dim']}, stride={metrics['temporal_stride']})")
    print(f"chunks:      {len(recon_chunks)} processed"
          + (f", {n_skipped} skipped (< 160 samples)" if n_skipped else ""))
    print(f"latent:      {latent_shape}  per chunk  (frames × dims)")
    print(f"bitrate:     {kbps:.2f} kbps")
    print(f"snr:         {snr_db:.1f} dB  [waveform-level; use pesq/stoi for perceptual quality]")
    if pesq_score is not None:
        print(f"pesq_wb:     {pesq_score:.3f}")
    else:
        print("pesq_wb:     n/a  (pip install pesq)")
    if stoi_score is not None:
        print(f"stoi:        {stoi_score:.3f}")
    else:
        print("stoi:        n/a  (pip install pystoi)")
    print(f"output:      {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Offline encode→decode inference with metric evaluation.'
    )
    parser.add_argument('--input', type=str, default=None,
                        help='Audio file to process (default: first .flac in LibriSpeech test-clean)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (default: phaseC — highest PESQ/STOI)')
    parser.add_argument('--chunk-sec', type=float, default=1.0,
                        help='Chunk duration in seconds (default: 1.0)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Target sample rate (default: 16000)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
