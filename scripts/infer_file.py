#!/usr/bin/env python3
"""
File inference: audio → encode → decode → reconstructed audio

Chains encode.py and decode.py in one command.

Usage:
  python scripts/infer_file.py input.wav
  python scripts/infer_file.py input.wav --output reconstructed.wav
  python scripts/infer_file.py input.wav --checkpoint checkpoints_active/temporal_phaseC/best.pt
  python scripts/infer_file.py input.wav --keep-binary   # also keep the .nacodec file
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description='Encode an audio file through the neural codec and decode it back.'
    )
    parser.add_argument('input',        type=Path, help='Input audio file')
    parser.add_argument('--output',     type=Path, default=None,
                        help='Output WAV (default: <input>_reconstructed.wav)')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--chunk-sec',  type=float, default=1.0)
    parser.add_argument('--keep-binary', action='store_true',
                        help='Keep the intermediate .nacodec binary file')
    parser.add_argument('--device',     default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    args = parser.parse_args()

    output = args.output or args.input.with_name(args.input.stem + '_reconstructed.wav')

    # Intermediate binary — temp file unless --keep-binary
    if args.keep_binary:
        binary = args.input.with_suffix('.nacodec')
    else:
        binary = Path(tempfile.mktemp(suffix='.nacodec'))

    python = sys.executable

    def build_cmd(script, *extra):
        cmd = [python, str(SCRIPTS / script)]
        if args.checkpoint:
            cmd += ['--checkpoint', str(args.checkpoint)]
        cmd += ['--device', args.device]
        cmd += list(extra)
        return cmd

    try:
        # --- ENCODE ---
        enc_cmd = build_cmd('encode.py',
                            str(args.input), str(binary),
                            '--chunk-sec', str(args.chunk_sec))
        print(f"[1/2] Encoding  {args.input} → {binary}")
        result = subprocess.run(enc_cmd, check=True)

        # --- DECODE ---
        dec_cmd = build_cmd('decode.py', str(binary), str(output))
        print(f"\n[2/2] Decoding  {binary} → {output}")
        result = subprocess.run(dec_cmd, check=True)

        print(f"\nDone.  Reconstructed audio: {output}")
        if args.keep_binary:
            print(f"       Binary payload:       {binary}")

    finally:
        if not args.keep_binary and binary.exists():
            binary.unlink()


if __name__ == '__main__':
    main()
