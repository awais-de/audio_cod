#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) for Bottleneck Neural Codec
=============================================================
Fine-tunes from the bottleneck v1 checkpoint with 1-bit quantization
baked into the forward pass via the Straight-Through Estimator (STE).

The STE trick:
  - Forward pass: apply real 1-bit quantization  → decoder sees quantized latents
  - Backward pass: treat quantization as identity → gradients flow normally

The model learns to pack speech into latents that survive 1-bit rounding,
giving clean audio AND the low bitrate (~10-14 kbps after zlib).

Base: checkpoints_ratedistortion/bottleneck_v1/best.pt  (epoch 28, loss=0.0237)
Out:  checkpoints_ratedistortion/bottleneck_v1_qat/
"""

import sys
import json
import zlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import soundfile as sf
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths


# ---------------------------------------------------------------------------
# STE 1-bit quantization
# ---------------------------------------------------------------------------

def ste_quantize_1bit(z: torch.Tensor) -> torch.Tensor:
    """
    1-bit quantization with Straight-Through Estimator.

    Forward : z → {z_min, z_max} per chunk (real quantization, 2 levels)
    Backward: gradient passes through unchanged (identity approximation)

    Args:
        z: (B, T, dim) float tensor
    Returns:
        z_ste: same shape, quantized in forward / differentiable in backward
    """
    z_min = z.min()
    z_max = z.max()
    threshold = (z_min + z_max) / 2.0

    # Hard threshold: 0 or 1
    z_bin = (z > threshold).float()
    # Map to {z_min, z_max}
    z_quant = z_bin * z_max + (1.0 - z_bin) * z_min
    # STE: add the quantization error as a constant (no gradient)
    z_ste = z + (z_quant - z).detach()
    return z_ste


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def multi_scale_stft_loss(x_recon: torch.Tensor, x_target: torch.Tensor,
                           fft_sizes=(256, 512, 1024), hop_length=160) -> torch.Tensor:
    if x_recon.ndim == 3:
        x_recon = x_recon.squeeze(1)
    if x_target.ndim == 3:
        x_target = x_target.squeeze(1)
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
        total = total + torch.mean((mag_r - mag_t) ** 2)
        total = total + torch.mean(torch.abs(mag_r - mag_t))
    return total / len(fft_sizes)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioChunkDataset(IterableDataset):
    def __init__(self, data_root, chunk_seconds=1.0, sample_rate=16000, epoch_size=1000):
        self.chunk_size = int(chunk_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.epoch_size = epoch_size
        exts = ('.wav', '.flac', '.mp3', '.ogg')
        self.files = sorted(p for p in Path(data_root).rglob('*') if p.suffix.lower() in exts)
        if not self.files:
            raise ValueError(f"No audio files in {data_root}")
        print(f"Dataset: {len(self.files)} files")

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
                    n = int(len(audio) * self.sample_rate / sr)
                    audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
                if len(audio) > self.chunk_size:
                    start = np.random.randint(0, len(audio) - self.chunk_size)
                    chunk = audio[start:start + self.chunk_size]
                else:
                    chunk = np.pad(audio, (0, self.chunk_size - len(audio)))
                yield torch.FloatTensor(np.clip(chunk, -1.0, 1.0).astype(np.float32)).unsqueeze(0)
            except Exception:
                continue


# ---------------------------------------------------------------------------
# Bitrate check (uses quantized latents — matches training)
# ---------------------------------------------------------------------------

def measure_real_bitrate(model, audio_files, device, n_files=5, chunk_samples=16000):
    model.eval()
    total_bits, total_dur = 0, 0.0
    with torch.no_grad():
        for path in audio_files[:n_files]:
            try:
                audio, sr = sf.read(path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
                chunks = []
                for start in range(0, len(audio), chunk_samples):
                    chunk = audio[start:start + chunk_samples]
                    if len(chunk) < 160:
                        continue
                    x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
                    z = model.encode(x).squeeze(0).cpu().numpy()
                    chunks.append(z)
                if not chunks:
                    continue
                z_np = np.concatenate(chunks, axis=0)
                threshold = (z_np.min() + z_np.max()) / 2.0
                z_bin = (z_np > threshold).astype(np.uint8)
                compressed = zlib.compress(z_bin.tobytes(), level=9)
                total_bits += len(compressed) * 8
                total_dur += len(audio) / sr
            except Exception:
                continue
    return total_bits / total_dur / 1000.0 if total_dur > 0 else float('nan')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    print(f"\n{'='*68}")
    print("QUANTIZATION-AWARE TRAINING (QAT)")
    print(f"{'='*68}")
    print(f"Base checkpoint : {args.base_checkpoint}")
    print(f"Output          : {args.output}")
    print(f"Epochs          : {args.epochs}  |  LR: {args.lr}")
    print(f"STE 1-bit quant applied in every forward pass")
    print(f"{'='*68}\n")

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.base_checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    d_model = ckpt.get('d_model', 384)
    n_layers = ckpt.get('n_layers', 6)
    n_heads = ckpt.get('n_heads', 8)
    window_size = ckpt.get('window_size', 200)
    bottleneck_dim = ckpt.get('bottleneck_dim', 32)

    model = NeuralAudioCodec(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        window_size=window_size, dropout=0.0,
        bottleneck_dim=bottleneck_dim,
    ).to(device)
    model.load_state_dict(state)
    print(f"Loaded: d_model={d_model}, n_layers={n_layers}, "
          f"bottleneck={bottleneck_dim}, window={window_size}")

    # Dataset
    dataset = AudioChunkDataset(args.data_root, chunk_seconds=args.chunk_sec,
                                epoch_size=args.samples_per_epoch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    val_files = dataset.files[:20]

    # Optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Initial bitrate (should be ~10-14 kbps from bottleneck v1)
    print("\nInitial real bitrate (before QAT):")
    init_kbps = measure_real_bitrate(model, val_files, device)
    print(f"  {init_kbps:.1f} kbps\n")

    history = {'epoch': [], 'train_loss': [], 'real_bitrate_kbps': [], 'learning_rate': []}
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    total=args.samples_per_epoch // args.batch_size)

        for batch_idx, x in enumerate(pbar):
            if x.shape[0] == 0:
                continue
            x = x.to(device)

            # Forward with STE quantization in the bottleneck
            z = model.encode(x)              # float32 latents
            z_ste = ste_quantize_1bit(z)     # quantized fwd, differentiable bwd
            x_recon = model.decode(z_ste)    # decoder sees 1-bit quantized latents

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
        real_kbps = measure_real_bitrate(model, val_files, device, n_files=5)
        lr = scheduler.get_last_lr()[0]

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['real_bitrate_kbps'].append(real_kbps)
        history['learning_rate'].append(lr)

        print(f"\nEpoch {epoch+1}/{args.epochs}: loss={avg_loss:.5f}  "
              f"real_bitrate={real_kbps:.1f} kbps  lr={lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'd_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads,
                'window_size': window_size, 'bottleneck_dim': bottleneck_dim,
                'train_loss': avg_loss, 'real_bitrate_kbps': real_kbps,
                'qat': True,
            }, args.output / 'best.pt')
            print(f"  ✓ Saved best checkpoint")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'd_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads,
                'window_size': window_size, 'bottleneck_dim': bottleneck_dim,
                'qat': True,
            }, args.output / f'epoch_{epoch+1:02d}.pt')

    with open(args.output / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best loss: {best_loss:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    paths = get_dataset_paths()

    parser = argparse.ArgumentParser(description="QAT fine-tuning for 1-bit bottleneck codec")
    parser.add_argument('--base-checkpoint', type=Path,
                        default=PROJECT_ROOT / 'checkpoints_ratedistortion/bottleneck_v1/best.pt')
    parser.add_argument('--data-root', type=Path, default=paths['train_clean_100'])
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'checkpoints_ratedistortion/bottleneck_v1_qat')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--chunk-sec', type=float, default=1.0)
    parser.add_argument('--samples-per-epoch', type=int, default=1000)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
