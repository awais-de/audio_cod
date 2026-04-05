#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from pesq import pesq
from pystoi import stoi

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths

DATA_ROOTS = []
SR = 16000
DEFAULT_SEG_SEC = 4.0  # longer segment for reliable STOI/PESQ
DEFAULT_N_FILES = 10
DEFAULT_CKPT = Path('checkpoints_emergency/best_emergency.pt')
DEFAULT_OUT = Path('checkpoints_emergency/eval_summary.txt')


def pick_non_silent_segment(audio: np.ndarray, seg_len: int, tries: int = 10) -> np.ndarray:
    if len(audio) <= seg_len:
        if len(audio) < seg_len:
            audio = np.pad(audio, (0, seg_len - len(audio)))
        return audio[:seg_len]
    for _ in range(tries):
        start = random.randint(0, len(audio) - seg_len)
        segment = audio[start:start + seg_len]
        rms = np.sqrt(np.mean(segment**2))
        if rms > 0.01:  # threshold to avoid near-silence
            return segment
    # fallback: center segment
    mid = len(audio) // 2
    start = max(0, mid - seg_len // 2)
    return audio[start:start + seg_len]


def load_files(root: Path, limit: int):
    files = list(root.rglob('*.flac'))
    random.shuffle(files)
    return files[:limit]


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint PESQ/STOI")
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT, help='Path to checkpoint')
    parser.add_argument('--n-files', type=int, default=DEFAULT_N_FILES, help='Number of files to sample')
    parser.add_argument('--seg-sec', type=float, default=DEFAULT_SEG_SEC, help='Segment length in seconds')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT, help='Output summary path')
    parser.add_argument('--data-root', type=Path, default=None, help='Override data root (folder with .flac)')
    args = parser.parse_args()

    seg_samples = int(SR * args.seg_sec)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not args.ckpt.exists():
        print(f"Checkpoint not found: {args.ckpt}")
        return

    model = NeuralAudioCodec(d_model=384, n_layers=6, n_heads=8, window_size=384).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # pick dataset root
    data_root = args.data_root
    if data_root is None:
        paths = get_dataset_paths()
        DATA_ROOTS.extend([paths["test_clean"], paths["train_clean_100"]])
        for root in DATA_ROOTS:
            if root.exists():
                data_root = root
                break
    if data_root is None:
        print("No dataset root found.")
        return

    files = load_files(data_root, args.n_files)
    print(f"Evaluating on {len(files)} files from {data_root}")

    pesq_scores = []
    stoi_scores = []

    with torch.no_grad():
        for i, f in enumerate(files, 1):
            audio, sr = sf.read(str(f))
            if sr != SR:
                # assume mono; if not, select channel 0
                if audio.ndim > 1:
                    audio = audio[:, 0]
                # naive resample if needed (rare for LibriSpeech); skip for speed
                # Here we assume SR==16k for LibriSpeech
            if audio.ndim > 1:
                audio = audio[:, 0]

            segment = pick_non_silent_segment(audio, seg_samples)
            x = torch.from_numpy(segment).float().unsqueeze(0).unsqueeze(0).to(device)

            recon = model(x)
            min_len = min(x.shape[-1], recon.shape[-1])
            x = x[..., :min_len]
            recon = recon[..., :min_len]

            orig = x.squeeze().cpu().numpy()
            rec = recon.squeeze().cpu().numpy()

            try:
                p = pesq(SR, orig, rec, 'wb')
                pesq_scores.append(p)
            except Exception as e:
                print(f"PESQ error on {f}: {e}")
            try:
                s = stoi(orig, rec, SR, extended=False)
                stoi_scores.append(s)
            except Exception as e:
                print(f"STOI error on {f}: {e}")

            if i % 2 == 0:
                print(f"  [{i}/{len(files)}] PESQ={pesq_scores[-1] if pesq_scores else 'n/a'} STOI={stoi_scores[-1] if stoi_scores else 'n/a'}")

    mean_pesq = float(np.mean(pesq_scores)) if pesq_scores else 0.0
    mean_stoi = float(np.mean(stoi_scores)) if stoi_scores else 0.0

    print(f"\nFinal: PESQ={mean_pesq:.3f}, STOI={mean_stoi:.3f}")

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(
        f"checkpoint={args.ckpt}\nfiles={len(files)}\nPESQ={mean_pesq:.3f}\nSTOI={mean_stoi:.3f}\n"
    )
    print(f"Saved summary to {args.out}")


if __name__ == '__main__':
    main()
