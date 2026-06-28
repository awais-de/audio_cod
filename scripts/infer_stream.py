#!/usr/bin/env python3
"""
Real-time teleconferencing inference: Mic → Encode → [binary] → Decode → Speaker

Simulates a full codec pipeline in real time:
  1. Capture a chunk from the microphone
  2. Encode it to a compressed binary payload (what would be transmitted over network)
  3. Decode the binary payload back to audio
  4. Play the decoded audio through the speaker

The binary payload printed per-frame is exactly what encode.py would write to disk —
the only difference is it passes through memory instead of a file/network.

Requirements:
  pip install sounddevice

Usage:
  python scripts/infer_stream.py                      # default 500ms chunks
  python scripts/infer_stream.py --chunk-ms 300       # lower latency
  python scripts/infer_stream.py --list-devices       # show audio device indices
  python scripts/infer_stream.py --input-device 2 --output-device 4
  python scripts/infer_stream.py --checkpoint checkpoints_active/temporal_phaseC/best.pt
"""

import struct
import sys
import time
import zlib
import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec

try:
    import sounddevice as sd
    SD_OK = True
except ImportError:
    SD_OK = False

NUM_LEVELS = 8
SR         = 16000
CHUNK_HDR  = struct.Struct('!ffB')   # z_min, z_max, n_dims


# Model loading

def find_checkpoint():
    for p in [
        PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt',
        PROJECT_ROOT / 'checkpoints_active/temporal_phaseF/best.pt',
        PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt',
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("No trained checkpoint found.")


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


# Per-frame encode / decode  (same logic as encode.py / decode.py)

def encode_frame(model, audio_chunk, device):
    """
    audio_chunk : float32 numpy array, shape (N,)
    returns     : bytes payload  (what would go over the network)
                  int   compressed bit count
    """
    x    = torch.FloatTensor(audio_chunk).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.encode(x).squeeze(0).cpu().numpy()

    zmin  = float(z.min())
    zmax  = float(z.max())
    scale = (zmax - zmin) / (NUM_LEVELS - 1) + 1e-8
    q     = np.clip(np.round((z - zmin) / scale), 0, NUM_LEVELS - 1).astype(np.uint8)
    comp  = zlib.compress(q.tobytes(), level=1)   # level=1 for speed

    # Pack into a self-describing payload (mirrors .nacodec chunk layout)
    hdr = CHUNK_HDR.pack(zmin, zmax, len(q.shape))
    for d in q.shape:
        hdr += struct.pack('!I', d)
    hdr += struct.pack('!I', len(comp))
    return hdr + comp, len(comp) * 8


def decode_frame(model, payload, device):
    """
    payload : bytes  (output of encode_frame)
    returns : float32 numpy array, shape (N,)
    """
    off             = 0
    zmin, zmax, nd  = CHUNK_HDR.unpack_from(payload, off);  off += CHUNK_HDR.size
    shape           = []
    for _ in range(nd):
        shape.append(struct.unpack_from('!I', payload, off)[0]);  off += 4
    comp_len        = struct.unpack_from('!I', payload, off)[0];  off += 4
    comp            = payload[off:off + comp_len]

    q     = np.frombuffer(zlib.decompress(comp), dtype=np.uint8).reshape(shape)
    scale = (zmax - zmin) / (NUM_LEVELS - 1) + 1e-8
    z_rec = q.astype(np.float32) * scale + zmin

    with torch.no_grad():
        x_recon = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))
    return x_recon.squeeze().cpu().numpy().astype(np.float32)


# Streaming loop

def run(args):
    if not SD_OK:
        print("ERROR: sounddevice is not installed.")
        print("       pip install sounddevice")
        sys.exit(1)

    if args.list_devices:
        print(sd.query_devices())
        return

    device = torch.device(args.device)
    ckpt_path = args.checkpoint or find_checkpoint()

    print(f"\n{'='*60}")
    print("NEURAL AUDIO CODEC  —  REAL-TIME TELECONFERENCING")
    print(f"{'='*60}")

    model, ckpt = load_model(ckpt_path, device)
    chunk_samples = int(args.chunk_ms / 1000 * SR)

    print(f"  Model      : Phase {ckpt.get('phase','?')}  "
          f"(loss={ckpt.get('train_loss', float('nan')):.5f})")
    print(f"  Chunk      : {args.chunk_ms} ms  ({chunk_samples} samples)")
    print(f"  Sample rate: {SR} Hz")
    print(f"  Device     : {device}")
    if args.input_device  is not None: print(f"  Mic device : {args.input_device}")
    if args.output_device is not None: print(f"  Spk device : {args.output_device}")

    # Warm-up inference (first forward pass is always slower due to JIT/CUDA init)
    print("\n  Warming up model ...", end=' ', flush=True)
    dummy   = np.zeros(chunk_samples, dtype=np.float32)
    payload, _ = encode_frame(model, dummy, device)
    decode_frame(model, payload, device)
    print("ready.")

    print("\n  Pipeline:")
    print("    Mic → [encode] → binary payload → [decode] → Speaker")
    print(f"\n  Latency breakdown:")
    print(f"    Capture  : {args.chunk_ms} ms  (chunk duration)")
    print(f"    Codec    : ~measured per frame")
    print(f"    Playback : {args.chunk_ms} ms  (chunk duration)")
    print(f"    Total    : ~{args.chunk_ms * 2}ms + codec time")
    print(f"\n  Press Ctrl+C to stop.\n")
    print(f"  {'Frame':>5}  {'Bitrate':>8}  {'Codec ms':>9}  {'Payload B':>10}  {'Latency ms':>11}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*11}")

    frame        = 0
    total_bits   = 0
    total_proc   = 0.0
    rec_kwargs   = dict(samplerate=SR, channels=1, dtype='float32',
                        device=args.input_device)
    play_kwargs  = dict(samplerate=SR, device=args.output_device)

    try:
        while True:
            # --- CAPTURE (blocking) ---
            raw   = sd.rec(chunk_samples, blocking=True, **rec_kwargs)
            chunk = raw.flatten()

            # --- ENCODE ---
            t0      = time.perf_counter()
            payload, bits = encode_frame(model, chunk, device)

            # --- DECODE ---
            recon   = decode_frame(model, payload, device)
            proc_ms = (time.perf_counter() - t0) * 1000

            # Trim/pad to exact chunk length
            if len(recon) >= len(chunk):
                recon = recon[:len(chunk)]
            else:
                recon = np.pad(recon, (0, len(chunk) - len(recon)))

            # --- PLAY (blocking) ---
            sd.play(recon, blocking=True, **play_kwargs)

            frame       += 1
            total_bits  += bits
            total_proc  += proc_ms
            kbps         = bits / (chunk_samples / SR) / 1000
            latency_ms   = args.chunk_ms + proc_ms

            print(f"  {frame:5d}  {kbps:7.1f}k  {proc_ms:8.1f}ms  "
                  f"{len(payload):10d}  {latency_ms:10.0f}ms",
                  flush=True)

    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("Stopped.")
        if frame > 0:
            avg_kbps = total_bits / (frame * chunk_samples / SR) / 1000
            avg_proc = total_proc / frame
            print(f"  Frames processed : {frame}")
            print(f"  Avg bitrate      : {avg_kbps:.1f} kbps")
            print(f"  Avg codec time   : {avg_proc:.1f} ms/frame")
            print(f"  Chunk latency    : {args.chunk_ms} ms")
            print(f"  Total latency    : ~{args.chunk_ms + avg_proc:.0f} ms")
        print(f"{'='*60}")


# CLI

def main():
    parser = argparse.ArgumentParser(
        description='Real-time teleconferencing: mic → codec → speaker.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--chunk-ms',      type=int,  default=500,
                        help='Chunk size in milliseconds (default: 500, min ~200)')
    parser.add_argument('--checkpoint',    type=Path, default=None,
                        help='Checkpoint path (default: best available G→F→C)')
    parser.add_argument('--device',        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--input-device',  type=int,  default=None,
                        help='Sounddevice mic index (default: system default)')
    parser.add_argument('--output-device', type=int,  default=None,
                        help='Sounddevice speaker index (default: system default)')
    parser.add_argument('--list-devices',  action='store_true',
                        help='Print available audio devices and exit')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
