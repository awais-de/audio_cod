#!/usr/bin/env python3
"""
Decoder: compressed binary (.nacodec) → audio file

Usage:
  python scripts/decode.py encoded.nacodec reconstructed.wav
  python scripts/decode.py encoded.nacodec out.wav --checkpoint checkpoints_active/temporal_phaseC/best.pt
"""

import struct
import sys
import zlib
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec

MAGIC      = b'NACODEC1'
HEADER     = struct.Struct('!8sIIII')   # magic, sr, n_samples, chunk_samples, n_chunks
CHUNK_HDR  = struct.Struct('!ffB')      # z_min, z_max, n_dims
NUM_LEVELS = 8


def find_checkpoint():
    for p in [
        PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt',
        PROJECT_ROOT / 'checkpoints_active/temporal_phaseF/best.pt',
        PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt',
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No trained checkpoint found. Pass --checkpoint explicitly."
    )


def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    d     = ckpt.get('d_model', 384)
    ids   = {int(k.split('.')[2]) for k in state
             if 'encoder.transformer_blocks.' in k and k.split('.')[2].isdigit()}
    model = NeuralAudioCodec(
        d_model         = d,
        n_layers        = max(ids) + 1 if ids else 6,
        n_heads         = ckpt.get('n_heads', 8),
        window_size     = ckpt.get('window_size', 200),
        dropout         = 0.0,
        bottleneck_dim  = ckpt.get('bottleneck_dim', 32),
        temporal_stride = ckpt.get('temporal_stride', 20),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Decode a .nacodec binary back to an audio file.'
    )
    parser.add_argument('input',        type=Path, help='Input .nacodec binary')
    parser.add_argument('output',       type=Path, help='Output audio file (.wav)')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='Model checkpoint (default: best available G→F→C)')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ckpt_path = args.checkpoint or find_checkpoint()
    device    = torch.device(args.device)

    print(f"\n{'='*56}")
    print("DECODE")
    print(f"{'='*56}")
    print(f"  Input      : {args.input}")
    print(f"  Output     : {args.output}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")

    model, ckpt = load_model(ckpt_path, device)
    print(f"  Model      : Phase {ckpt.get('phase','?')}  "
          f"d_model={ckpt.get('d_model',384)}  "
          f"bottleneck={ckpt.get('bottleneck_dim',32)}\n")

    with open(args.input, 'rb') as f:
        # Parse header
        magic, sr, n_samples, chunk_samples, n_chunks = HEADER.unpack(f.read(HEADER.size))
        if magic != MAGIC:
            raise ValueError(
                f"Not a valid .nacodec file (expected magic {MAGIC!r}, got {magic!r})"
            )

        duration = n_samples / sr
        print(f"  Audio info : {duration:.2f}s @ {sr} Hz  ({n_chunks} chunks)")

        recon_chunks = []
        total_bits   = 0

        for i in range(n_chunks):
            z_min, z_max, n_dims = CHUNK_HDR.unpack(f.read(CHUNK_HDR.size))
            shape    = tuple(struct.unpack('!I', f.read(4))[0] for _ in range(n_dims))
            comp_len = struct.unpack('!I', f.read(4))[0]
            comp     = f.read(comp_len)
            total_bits += comp_len * 8

            # Dequantise
            q     = np.frombuffer(zlib.decompress(comp), dtype=np.uint8).reshape(shape)
            scale = (z_max - z_min) / (NUM_LEVELS - 1) + 1e-8
            z_rec = q.astype(np.float32) * scale + z_min

            # Decode
            with torch.no_grad():
                x_recon = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))
            recon_chunks.append(x_recon.squeeze().cpu().numpy())

            print(f"  chunk {i+1:3d}/{n_chunks}  z_shape={shape}  "
                  f"{comp_len} bytes  →  {len(recon_chunks[-1])} samples", flush=True)

    # Assemble and trim to original length
    recon = np.concatenate(recon_chunks)
    if len(recon) >= n_samples:
        recon = recon[:n_samples]
    else:
        recon = np.pad(recon, (0, n_samples - len(recon)))
    recon = recon.astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, recon, sr)

    kbps = total_bits / duration / 1000
    print(f"\n{'='*56}")
    print(f"  Bitrate    : {kbps:.1f} kbps")
    print(f"  Samples    : {len(recon)}  ({duration:.2f}s @ {sr} Hz)")
    print(f"{'='*56}")
    print(f"\nDone.  Output written to: {args.output}")


if __name__ == '__main__':
    main()
