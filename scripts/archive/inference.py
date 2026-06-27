#!/usr/bin/env python3
"""
Neural Audio Codec — Inference
================================
Compress and reconstruct any speech audio file using the trained
temporal-bottleneck + 3-bit QAT codec.

Usage:
    python scripts/inference.py --input speech.wav --output reconstructed.wav

All architecture parameters are read directly from the checkpoint —
no config file needed.
"""

import sys
import zlib
import time
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    # Read architecture from checkpoint metadata
    d_model = ckpt.get('d_model')
    if d_model is None:
        k = 'encoder.transformer_blocks.0.attention.qkv.weight'
        d_model = state[k].shape[1] if k in state else 384

    ids = set()
    for k in state:
        if 'encoder.transformer_blocks.' in k:
            p = k.split('.')
            if len(p) > 2 and p[2].isdigit():
                ids.add(int(p[2]))
    n_layers = max(ids) + 1 if ids else 6

    n_heads        = ckpt.get('n_heads', 8)
    window_size    = ckpt.get('window_size', 200)
    bottleneck_dim = ckpt.get('bottleneck_dim')
    temporal_stride = ckpt.get('temporal_stride', 1)

    model = NeuralAudioCodec(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        window_size=window_size, dropout=0.0,
        bottleneck_dim=bottleneck_dim,
        temporal_stride=temporal_stride,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    return model, {
        'd_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads,
        'window_size': window_size, 'bottleneck_dim': bottleneck_dim,
        'temporal_stride': temporal_stride,
        'phase': ckpt.get('phase', '?'),
        'train_loss': ckpt.get('train_loss'),
    }


# ---------------------------------------------------------------------------
# Codec pipeline
# ---------------------------------------------------------------------------

def encode_decode(model: NeuralAudioCodec, audio: np.ndarray, sr: int,
                  device: torch.device, chunk_sec: float = 5.0) -> tuple:
    """
    Full encode → 3-bit quantize → zlib compress → decompress → decode pipeline.
    Returns reconstructed audio, bitrate (kbps), and latency (ms).
    """
    num_levels = 8
    chunk_size = int(chunk_sec * sr)
    recon_chunks, total_bits, latencies = [], 0, []

    with torch.no_grad():
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            if len(chunk) < 160:
                continue
            x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)

            t0 = time.perf_counter()

            # Encode
            z = model.encode(x)
            z_np = z.squeeze(0).cpu().numpy()

            # 3-bit uniform quantization
            z_min, z_max = float(z_np.min()), float(z_np.max())
            scale = (z_max - z_min) / (num_levels - 1) + 1e-8
            q = np.clip(np.round((z_np - z_min) / scale), 0, num_levels - 1).astype(np.uint8)

            # Entropy coding
            compressed = zlib.compress(q.tobytes(), level=9)
            total_bits += len(compressed) * 8

            # Decompress and dequantize
            q_dec = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(z_np.shape)
            z_rec = q_dec.astype(np.float32) * scale + z_min

            # Decode
            x_recon = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))

            latencies.append((time.perf_counter() - t0) * 1000)
            recon_chunks.append(x_recon.squeeze().cpu().numpy())

    recon = np.concatenate(recon_chunks) if recon_chunks else np.zeros_like(audio)
    recon = recon[:len(audio)] if len(recon) >= len(audio) else np.pad(recon, (0, len(audio) - len(recon)))
    kbps = total_bits / (len(audio) / sr) / 1000
    return recon.astype(np.float32), kbps, float(np.mean(latencies))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Neural Audio Codec — compress and reconstruct speech audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inference.py --input speech.wav --output reconstructed.wav
  python scripts/inference.py --input speech.flac --output out.wav --checkpoint checkpoints_active/temporal_phaseC/best.pt
        """
    )
    parser.add_argument('--input',      required=True,  help='Input audio file (.wav, .flac, .mp3)')
    parser.add_argument('--output',     required=True,  help='Output reconstructed .wav file')
    parser.add_argument('--checkpoint', default=str(PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt'),
                        help='Model checkpoint (default: Phase C best)')
    parser.add_argument('--chunk-sec',  type=float, default=5.0,
                        help='Processing chunk size in seconds (default: 5.0)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SR = 16000

    print(f"\n{'='*60}")
    print("Neural Audio Codec — Inference")
    print(f"{'='*60}")
    print(f"Input      : {args.input}")
    print(f"Output     : {args.output}")
    print(f"Checkpoint : {Path(args.checkpoint).name}")
    print(f"Device     : {device}\n")

    # Load model
    print("Loading model...", end=' ', flush=True)
    model, meta = load_model(Path(args.checkpoint), device)
    print(f"OK  (d_model={meta['d_model']}, n_layers={meta['n_layers']}, "
          f"bottleneck={meta['bottleneck_dim']}, stride={meta['temporal_stride']}, "
          f"phase={meta['phase']})")

    latent_hz = 2000 // (meta['temporal_stride'] or 1)
    raw_cap = (meta['bottleneck_dim'] or meta['d_model']) * 3 * latent_hz / 1000
    print(f"  Latent rate : {latent_hz} Hz   Raw bitrate cap : {raw_cap:.1f} kbps (3-bit)\n")

    # Load audio
    print("Loading audio...", end=' ', flush=True)
    audio, sr = sf.read(args.input)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio_t = torch.FloatTensor(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(audio_t, sr, SR).squeeze(0).numpy()
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    duration = len(audio) / SR
    print(f"OK  ({duration:.1f}s, {SR} Hz mono)")

    # Run codec
    print("\nRunning codec pipeline...")
    recon, kbps, lat_ms = encode_decode(model, audio, SR, device, chunk_sec=args.chunk_sec)

    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, recon, SR)

    # Metrics
    try:
        from pesq import pesq as pesq_fn
        n = min(len(audio), len(recon))
        ref = audio[:n] / (np.abs(audio[:n]).max() + 1e-8)
        deg = recon[:n] / (np.abs(recon[:n]).max() + 1e-8)
        pesq_score = pesq_fn(SR, ref, deg, 'wb')
    except Exception:
        pesq_score = None

    try:
        from pystoi import stoi as stoi_fn
        n = min(len(audio), len(recon))
        stoi_score = stoi_fn(audio[:n], recon[:n], SR, extended=False)
    except Exception:
        stoi_score = None

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Bitrate        : {kbps:.1f} kbps  (cap: {raw_cap:.1f} kbps)")
    print(f"Latency        : {lat_ms:.0f} ms per chunk  (algo delay: {meta['window_size']*0.5:.0f} ms)")
    if pesq_score is not None:
        print(f"PESQ (WB)      : {pesq_score:.3f}  (range 1-4.5, higher = better)")
    if stoi_score is not None:
        print(f"STOI           : {stoi_score:.3f}  (range 0-1, higher = better)")
    print(f"Output saved   : {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
