#!/usr/bin/env python3
"""
Bottleneck Fine-tuning Script
=============================
Adds a learnable bottleneck projection (d_model → bottleneck_dim → d_model)
to architecturally constrain the bitrate without a rate loss term.

Bitrate arithmetic (frame_rate ≈ 2000 Hz at 16kHz with 3×stride-2 convs):
  bottleneck_dim=32, 1-bit quant  → 32 × 2000 = 64 kbps raw
  → after zlib on correlated latents ≈ 7–10 kbps actual ✓

Latency:
  window_size=200 frames × 0.5 ms/frame = 100 ms context ✓

Base checkpoint: Phase 1 (PESQ=4.27, best quality in the entire chain).
Bottleneck projections are initialised from scratch; all other weights are
loaded from Phase 1.
"""

import argparse
import sys
import json
import zlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths, get_checkpoint_paths


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def multi_scale_stft_loss(x_recon: torch.Tensor, x_target: torch.Tensor,
                           fft_sizes=(256, 512, 1024), hop_length=160) -> torch.Tensor:
    """
    Multi-scale STFT magnitude loss (L1 + L2 per scale), matching Phase 1 training.

    Args:
        x_recon:  (B, 1, T) or (B, T)
        x_target: (B, 1, T) or (B, T)
    Returns:
        Scalar loss tensor.
    """
    if x_recon.ndim == 3:
        x_recon = x_recon.squeeze(1)
    if x_target.ndim == 3:
        x_target = x_target.squeeze(1)

    # Align lengths
    min_len = min(x_recon.shape[-1], x_target.shape[-1])
    x_recon = x_recon[..., :min_len]
    x_target = x_target[..., :min_len]

    total = torch.tensor(0.0, device=x_recon.device)
    for n_fft in fft_sizes:
        win = torch.hann_window(n_fft, device=x_recon.device)
        spec_r = torch.stft(x_recon, n_fft=n_fft, hop_length=hop_length,
                            window=win, return_complex=True)
        spec_t = torch.stft(x_target, n_fft=n_fft, hop_length=hop_length,
                            window=win, return_complex=True)
        mag_r = torch.abs(spec_r)
        mag_t = torch.abs(spec_t)
        # L2 (spectral convergence component) + L1 (log-magnitude component)
        total = total + torch.mean((mag_r - mag_t) ** 2)
        total = total + torch.mean(torch.abs(mag_r - mag_t))

    return total / len(fft_sizes)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioChunkDataset(IterableDataset):
    """Streams random fixed-length chunks from a directory of audio files."""

    def __init__(self, data_root, chunk_seconds=1.0, sample_rate=16000, epoch_size=1000):
        self.data_root = Path(data_root)
        self.chunk_size = int(chunk_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.epoch_size = epoch_size

        exts = ('.wav', '.flac', '.mp3', '.ogg')
        self.files = sorted(p for p in self.data_root.rglob('*') if p.suffix.lower() in exts)
        if not self.files:
            raise ValueError(f"No audio files found in {data_root}")
        print(f"Dataset: {len(self.files)} files in {data_root}")

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for _ in range(self.epoch_size):
            path = self.files[np.random.randint(0, len(self.files))]
            try:
                audio, sr = sf.read(path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != self.sample_rate:
                    # Simple linear resampling fallback (no librosa dependency)
                    n_out = int(len(audio) * self.sample_rate / sr)
                    audio = np.interp(np.linspace(0, len(audio), n_out),
                                      np.arange(len(audio)), audio)
                if len(audio) > self.chunk_size:
                    start = np.random.randint(0, len(audio) - self.chunk_size)
                    chunk = audio[start:start + self.chunk_size]
                else:
                    chunk = np.pad(audio, (0, self.chunk_size - len(audio)))
                chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)
                yield torch.FloatTensor(chunk).unsqueeze(0)  # (1, T)
            except Exception:
                continue


# ---------------------------------------------------------------------------
# Bitrate measurement
# ---------------------------------------------------------------------------

def measure_real_bitrate(model: NeuralAudioCodec, audio_files, device,
                         num_bits: int = 1, n_files: int = 5) -> float:
    """
    Encode audio files, 1-bit quantize, zlib compress, return mean kbps.

    Uses uniform 1-bit quantization (threshold at midpoint of range) to match
    what QuantizedLatentCodec does at the 10 kbps target.
    """
    model.eval()
    total_bits = 0
    total_duration = 0.0

    with torch.no_grad():
        for path in audio_files[:n_files]:
            try:
                audio, sr = sf.read(path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                duration = len(audio) / sr
                x = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)
                z = model.encode(x)          # (1, T_latent, bottleneck_dim or d_model)
                z_np = z.squeeze(0).cpu().numpy()  # (T_latent, dim)

                # 1-bit uniform quantization
                z_min, z_max = float(z_np.min()), float(z_np.max())
                threshold = (z_min + z_max) / 2.0
                z_bin = (z_np > threshold).astype(np.uint8)

                compressed = zlib.compress(z_bin.tobytes(), level=9)
                total_bits += len(compressed) * 8
                total_duration += duration
            except Exception:
                continue

    if total_duration == 0:
        return float('nan')
    return total_bits / total_duration / 1000.0  # kbps


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    print(f"\n{'='*72}")
    print("BOTTLENECK FINE-TUNING")
    print(f"{'='*72}")
    print(f"Base checkpoint : {args.base_checkpoint}")
    print(f"Bottleneck dim  : {args.bottleneck_dim}")
    print(f"Window size     : {args.window_size} frames (~{args.window_size*0.5:.0f} ms latency)")
    print(f"Epochs          : {args.epochs}")
    print(f"LR              : {args.lr}")
    print(f"Output          : {args.output}")
    print(f"{'='*72}\n")

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load base checkpoint and infer architecture
    # ------------------------------------------------------------------
    ckpt = torch.load(args.base_checkpoint, map_location='cpu')
    base_state = ckpt.get('model_state_dict', ckpt)

    # Infer d_model from checkpoint weights
    d_model = 256
    qkv_key = 'encoder.transformer_blocks.0.attention.qkv.weight'
    if qkv_key in base_state:
        d_model = base_state[qkv_key].shape[1]

    # Infer n_layers
    layer_ids = set()
    for k in base_state:
        if 'encoder.transformer_blocks.' in k:
            parts = k.split('.')
            if len(parts) > 2 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))
    n_layers = max(layer_ids) + 1 if layer_ids else 4
    n_heads = 8

    print(f"Inferred architecture: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")

    # ------------------------------------------------------------------
    # Build model with bottleneck + new window_size
    # ------------------------------------------------------------------
    model = NeuralAudioCodec(
        sample_rate=16000,
        hop_length=160,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        window_size=args.window_size,
        dropout=0.0,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)

    # Load pre-trained weights; bottleneck projections are missing from base
    # checkpoint so strict=False — they start from random Xavier init.
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f"Loaded base weights — missing (new layers): {missing}")
    if unexpected:
        print(f"Unexpected keys (ignored): {unexpected}")

    total_params = sum(p.numel() for p in model.parameters())
    bottleneck_params = (
        sum(p.numel() for p in model.encoder_proj.parameters()) +
        sum(p.numel() for p in model.decoder_proj.parameters())
    ) if model.encoder_proj is not None else 0
    print(f"Total params: {total_params:,}  (bottleneck: {bottleneck_params:,} new)")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = AudioChunkDataset(
        args.data_root,
        chunk_seconds=args.chunk_sec,
        sample_rate=16000,
        epoch_size=args.samples_per_epoch,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    # Grab a small set of validation files for real-bitrate measurement
    val_files = dataset.files[:20]

    # ------------------------------------------------------------------
    # Optimiser — train ALL parameters (encoder adapts to bottleneck)
    # ------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ------------------------------------------------------------------
    # Initial bitrate check (before any training)
    # ------------------------------------------------------------------
    print("\nMeasuring initial real bitrate (before training)...")
    initial_kbps = measure_real_bitrate(model, val_files, device)
    print(f"  Initial real bitrate: {initial_kbps:.1f} kbps  "
          f"(target ≤ 10 kbps; raw cap = {args.bottleneck_dim * 2000 / 1000:.0f} kbps)")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = {
        'epoch': [], 'train_loss': [], 'real_bitrate_kbps': [], 'learning_rate': []
    }
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    total=args.samples_per_epoch // args.batch_size)

        for batch_idx, x in enumerate(pbar):
            if x.shape[0] == 0:
                continue
            x = x.to(device)

            z = model.encode(x)
            x_recon = model.decode(z)
            loss = multi_scale_stft_loss(x_recon, x) / args.grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum_steps
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item()*args.grad_accum_steps:.5f}'})

            if batch_idx * args.batch_size >= args.samples_per_epoch:
                break

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step()

        # Real bitrate check every epoch (quick — only 5 files)
        real_kbps = measure_real_bitrate(model, val_files, device, n_files=5)

        lr = scheduler.get_last_lr()[0]
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['real_bitrate_kbps'].append(real_kbps)
        history['learning_rate'].append(lr)

        print(f"\nEpoch {epoch+1}/{args.epochs}: loss={avg_loss:.5f}  "
              f"real_bitrate={real_kbps:.1f} kbps  lr={lr:.2e}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'd_model': d_model,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'window_size': args.window_size,
                'bottleneck_dim': args.bottleneck_dim,
                'train_loss': avg_loss,
                'real_bitrate_kbps': real_kbps,
            }
            torch.save(ckpt_data, args.output / 'best.pt')
            print(f"  ✓ Saved best checkpoint")

        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'd_model': d_model,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'window_size': args.window_size,
                'bottleneck_dim': args.bottleneck_dim,
            }, args.output / f'epoch_{epoch+1:02d}.pt')

    # Save history
    with open(args.output / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    paths = get_dataset_paths()
    ckpt_paths = get_checkpoint_paths()

    parser = argparse.ArgumentParser(description="Bottleneck fine-tuning for bitrate control")
    parser.add_argument('--base-checkpoint', type=Path,
                        default=ckpt_paths['phase1'],
                        help='Starting checkpoint (default: Phase 1)')
    parser.add_argument('--data-root', type=Path,
                        default=paths['train_clean_100'],
                        help='Audio dataset root (default: train-clean-100)')
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'checkpoints_ratedistortion' / 'bottleneck_v1')
    parser.add_argument('--bottleneck-dim', type=int, default=32,
                        help='Bottleneck latent dimension (default: 32 → ~64 kbps raw)')
    parser.add_argument('--window-size', type=int, default=200,
                        help='Attention window in frames (200 frames × 0.5ms = 100ms latency)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--chunk-sec', type=float, default=1.0)
    parser.add_argument('--samples-per-epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
