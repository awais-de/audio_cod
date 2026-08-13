#!/usr/bin/env python3
"""
Inspect a .nacodec compressed bitstream file.

Usage:
  python scripts/inspect_nacodec.py compressed.nacodec
  python scripts/inspect_nacodec.py compressed.nacodec --no-chunks
"""

import struct
import zlib
import argparse
from pathlib import Path

MAGIC      = b'NACODEC1'
HEADER     = struct.Struct('!8sIIII')   # magic, sr, n_samples, chunk_samples, n_chunks
CHUNK_HDR  = struct.Struct('!ffB')      # z_min, z_max, n_dims


def inspect(path: Path, show_chunks: bool = True):
    file_size = path.stat().st_size

    with open(path, 'rb') as f:
        magic, sr, n_samples, chunk_samples, n_chunks = HEADER.unpack(f.read(HEADER.size))

        if magic != MAGIC:
            raise ValueError(f"Not a valid .nacodec file (got magic {magic!r}, expected {MAGIC!r})")

        duration   = n_samples / sr
        chunk_sec  = chunk_samples / sr
        pcm_bytes  = n_samples * 2          # 16-bit PCM equivalent
        kbps       = (file_size * 8) / duration / 1000

        print(f"\n=== {path.name} ===")
        print(f"magic:        {magic.decode()}")
        print(f"sample_rate:  {sr} Hz")
        print(f"duration:     {duration:.2f} s  ({n_samples} samples)")
        print(f"chunk_size:   {chunk_samples} samples  ({chunk_sec:.2f} s per chunk)")
        print(f"n_chunks:     {n_chunks}")

        chunks = []
        for i in range(n_chunks):
            z_min, z_max, n_dims = CHUNK_HDR.unpack(f.read(CHUNK_HDR.size))
            shape    = tuple(struct.unpack('!I', f.read(4))[0] for _ in range(n_dims))
            comp_len = struct.unpack('!I', f.read(4))[0]
            comp     = f.read(comp_len)
            raw_len  = len(zlib.decompress(comp))
            chunks.append((z_min, z_max, shape, comp_len, raw_len))

        if show_chunks:
            print(f"\nCHUNKS")
            print(f"  {'#':>3}  {'z_min':>7}  {'z_max':>7}  {'shape':<12}  {'raw B':>6}  {'comp B':>6}  {'ratio':>6}")
            for i, (z_min, z_max, shape, comp_len, raw_len) in enumerate(chunks, 1):
                ratio = raw_len / comp_len
                print(f"  {i:>3}  {z_min:>7.3f}  {z_max:>7.3f}  {str(shape):<12}  {raw_len:>6}  {comp_len:>6}  {ratio:>5.2f}×")

        comp_total = sum(c[3] for c in chunks)
        raw_total  = sum(c[4] for c in chunks)
        avg_ratio  = raw_total / comp_total if comp_total else 0

        print(f"\nSUMMARY")
        print(f"  file size:    {file_size / 1024:.1f} KB")
        print(f"  PCM equiv:    {pcm_bytes / 1024:.0f} KB  (16-bit uncompressed)")
        print(f"  compression:  {pcm_bytes / file_size:.0f}×  vs uncompressed PCM")
        print(f"  zlib ratio:   {avg_ratio:.2f}×  (latent bytes before/after zlib)")
        print(f"  bitrate:      {kbps:.2f} kbps\n")


def main():
    parser = argparse.ArgumentParser(description='Inspect a .nacodec compressed bitstream.')
    parser.add_argument('input', type=Path, help='Path to .nacodec file')
    parser.add_argument('--no-chunks', action='store_true',
                        help='Skip per-chunk table, show summary only')
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"File not found: {args.input}")

    inspect(args.input, show_chunks=not args.no_chunks)


if __name__ == '__main__':
    main()
