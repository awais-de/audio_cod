#!/usr/bin/env python3
"""
Encoder: audio file → compressed binary (.nacodec)

Usage:
  python scripts/encode.py input.wav output.nacodec
  python scripts/encode.py input.flac output.nacodec --checkpoint checkpoints_active/temporal_phaseC/best.pt
  python scripts/encode.py input.wav output.nacodec --chunk-sec 2.0

Binary format (.nacodec):
  [Header - 28 bytes]
    magic         : 8 bytes  "NACODEC1"
    sample_rate   : 4 bytes  uint32
    n_samples     : 4 bytes  uint32  (original length for exact reconstruction)
    chunk_samples : 4 bytes  uint32  (samples per encoded chunk)
    n_chunks      : 4 bytes  uint32

  [Per chunk - variable]
    z_min         : 4 bytes  float32  (dequantisation scale anchor)
    z_max         : 4 bytes  float32
    n_dims        : 1 byte   uint8    (number of latent dimensions)
    shape[i]      : 4 bytes  uint32   × n_dims
    comp_len      : 4 bytes  uint32
    comp_data     : comp_len bytes    (zlib-compressed 3-bit quantised latent)
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

MAGIC          = b'NACODEC1'
HEADER         = struct.Struct('!8sIIII')   # magic, sr, n_samples, chunk_samples, n_chunks
CHUNK_HDR      = struct.Struct('!ffB')      # z_min, z_max, n_dims
NUM_LEVELS     = 8
DEFAULT_SR     = 16000


def find_checkpoint():
    for p in [
        PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt',
        PROJECT_ROOT / 'checkpoints_active/temporal_phaseF/best.pt',
        PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt',
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No trained checkpoint found in checkpoints_active/. "
        "Pass --checkpoint explicitly or run training first."
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
        description='Encode an audio file to a compressed .nacodec binary.'
    )
    parser.add_argument('input',       type=Path, help='Input audio file (.wav / .flac / .mp3 …)')
    parser.add_argument('output',      type=Path, help='Output binary file (.nacodec)')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='Model checkpoint (default: best available G→F→C)')
    parser.add_argument('--chunk-sec', type=float, default=1.0,
                        help='Chunk size in seconds (default: 1.0)')
    parser.add_argument('--sample-rate', type=int, default=DEFAULT_SR,
                        help=f'Target sample rate (default: {DEFAULT_SR})')
    parser.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ckpt_path = args.checkpoint or find_checkpoint()
    device    = torch.device(args.device)

    print(f"\n{'='*56}")
    print("ENCODE")
    print(f"{'='*56}")
    print(f"  Input      : {args.input}")
    print(f"  Output     : {args.output}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")

    model, ckpt = load_model(ckpt_path, device)
    print(f"  Model      : Phase {ckpt.get('phase','?')}  "
          f"d_model={ckpt.get('d_model',384)}  "
          f"bottleneck={ckpt.get('bottleneck_dim',32)}\n")

    # Load + normalise audio
    SR = args.sample_rate
    audio, sr = sf.read(args.input)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        n     = int(len(audio) * SR / sr)
        audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
    audio     = np.clip(audio, -1.0, 1.0).astype(np.float32)
    n_samples = len(audio)
    duration  = n_samples / SR
    chunk_sz  = int(args.chunk_sec * SR)

    print(f"  Audio      : {duration:.2f}s @ {SR} Hz  ({n_samples} samples)")

    # Encode each chunk
    chunks = []
    with torch.no_grad():
        for start in range(0, n_samples, chunk_sz):
            chunk = audio[start:start + chunk_sz]
            if len(chunk) < 160:
                continue
            x    = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            z    = model.encode(x).squeeze(0).cpu().numpy()
            zmin = float(z.min())
            zmax = float(z.max())
            scale = (zmax - zmin) / (NUM_LEVELS - 1) + 1e-8
            q    = np.clip(np.round((z - zmin) / scale), 0, NUM_LEVELS - 1).astype(np.uint8)
            comp = zlib.compress(q.tobytes(), level=9)
            chunks.append((zmin, zmax, q.shape, comp))
            print(f"  chunk {len(chunks):3d}/{(n_samples + chunk_sz - 1)//chunk_sz}"
                  f"  z_shape={q.shape}  compressed={len(comp)} bytes", flush=True)

    # Write binary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(HEADER.pack(MAGIC, SR, n_samples, chunk_sz, len(chunks)))
        for zmin, zmax, shape, comp in chunks:
            f.write(CHUNK_HDR.pack(zmin, zmax, len(shape)))
            for d in shape:
                f.write(struct.pack('!I', d))
            f.write(struct.pack('!I', len(comp)))
            f.write(comp)

    # Stats
    total_bits  = sum(len(c[3]) * 8 for c in chunks)
    kbps        = total_bits / duration / 1000
    file_kb     = args.output.stat().st_size / 1024
    orig_kb     = n_samples * 2 / 1024         # 16-bit PCM equivalent
    compression = orig_kb / file_kb

    print(f"\n{'='*56}")
    print(f"  Chunks     : {len(chunks)}")
    print(f"  Bitrate    : {kbps:.1f} kbps")
    print(f"  File size  : {file_kb:.1f} KB  (vs {orig_kb:.0f} KB uncompressed PCM)")
    print(f"  Compression: {compression:.0f}×")
    print(f"{'='*56}")
    print(f"\nDone.  Decode with:")
    print(f"  python scripts/decode.py {args.output} <output.wav>")


if __name__ == '__main__':
    main()
