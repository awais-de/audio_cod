#!/usr/bin/env python3
"""
Phase B: 3-bit QAT Fine-tuning from Phase A checkpoint
========================================================
Loads the Phase A best checkpoint (temporal bottleneck, float32 trained)
and fine-tunes with 3-bit Straight-Through Estimator quantization.

The model already knows how to compress speech at 100 Hz / 32 dims.
Phase B adapts the weights to survive 3-bit rounding (8 levels).

Bitrate cap: 32 dims × 3 bits × 100 Hz = 9,600 bps = 9.6 kbps

LR: much lower than Phase A (1e-5), cosine annealing.
"""

import sys
import json
import zlib
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


# 3-bit STE quantization

def ste_quantize_3bit(z: torch.Tensor) -> torch.Tensor:
    num_levels = 8  # 2^3
    z_min = z.min()
    z_max = z.max()
    scale = (z_max - z_min) / (num_levels - 1) + 1e-8
    z_norm = (z - z_min) / scale
    z_int = torch.clamp(torch.round(z_norm), 0, num_levels - 1)
    z_quant = z_int * scale + z_min
    return z + (z_quant - z).detach()


# Loss

def multi_scale_stft_loss(x_recon, x_target, fft_sizes=(256, 512, 1024), hop=160):
    if x_recon.ndim == 3:
        x_recon = x_recon.squeeze(1)
    if x_target.ndim == 3:
        x_target = x_target.squeeze(1)
    n = min(x_recon.shape[-1], x_target.shape[-1])
    x_recon, x_target = x_recon[..., :n], x_target[..., :n]

    total = torch.tensor(0.0, device=x_recon.device)
    for n_fft in fft_sizes:
        win = torch.hann_window(n_fft, device=x_recon.device)
        Sr = torch.stft(x_recon, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        St = torch.stft(x_target, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        Mr, Mt = torch.abs(Sr), torch.abs(St)
        total = total + torch.mean((Mr - Mt) ** 2) + torch.mean(torch.abs(Mr - Mt))
    return total / len(fft_sizes)


# Dataset

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


# Bitrate measurement

def measure_real_bitrate(model, audio_files, device, n_files=5, chunk_samples=16000):
    model.eval()
    total_bits, total_dur = 0, 0.0
    num_levels = 8

    with torch.no_grad():
        for path in audio_files[:n_files]:
            try:
                audio, sr = sf.read(path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
                latent_chunks = []
                for start in range(0, len(audio), chunk_samples):
                    chunk = audio[start:start + chunk_samples]
                    if len(chunk) < 160:
                        continue
                    x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
                    z = model.encode(x).squeeze(0).cpu().numpy()
                    latent_chunks.append(z)
                if not latent_chunks:
                    continue
                z_np = np.concatenate(latent_chunks, axis=0)
                z_min, z_max = z_np.min(), z_np.max()
                scale = (z_max - z_min) / (num_levels - 1) + 1e-8
                q = np.clip(np.round((z_np - z_min) / scale), 0, num_levels - 1).astype(np.uint8)
                compressed = zlib.compress(q.tobytes(), level=9)
                total_bits += len(compressed) * 8
                total_dur += len(audio) / sr
            except Exception:
                continue

    return total_bits / total_dur / 1000.0 if total_dur > 0 else float('nan')


# Training

def train(args):
    print(f"\n{'='*68}")
    print("PHASE B: 3-BIT QAT FINE-TUNING")
    print(f"{'='*68}")
    print(f"Phase A checkpoint : {args.base_checkpoint}")
    print(f"Output             : {args.output}")
    print(f"Epochs             : {args.epochs}  |  LR: {args.lr}  |  Cosine annealing")
    print(f"{'='*68}\n")

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    # Load Phase A checkpoint
    ckpt = torch.load(args.base_checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    d_model = ckpt.get('d_model', 384)
    n_layers = ckpt.get('n_layers', 6)
    n_heads = ckpt.get('n_heads', 8)
    window_size = ckpt.get('window_size', 200)
    bottleneck_dim = ckpt.get('bottleneck_dim', 32)
    temporal_stride = ckpt.get('temporal_stride', 20)

    model = NeuralAudioCodec(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        window_size=window_size, dropout=0.0,
        bottleneck_dim=bottleneck_dim,
        temporal_stride=temporal_stride,
    ).to(device)
    model.load_state_dict(state)
    print(f"Loaded Phase A: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    print(f"  bottleneck_dim={bottleneck_dim}, temporal_stride={temporal_stride}, "
          f"window_size={window_size}")

    raw_cap = bottleneck_dim * 3 * (2000 // temporal_stride) / 1000
    print(f"  Raw bitrate cap: {raw_cap:.1f} kbps\n")

    # Dataset
    dataset = AudioChunkDataset(args.data_root, chunk_seconds=args.chunk_sec,
                                epoch_size=args.samples_per_epoch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    val_files = dataset.files[:20]

    # Initial bitrate
    print("Initial bitrate (Phase A, before QAT)...")
    init_kbps = measure_real_bitrate(model, val_files, device)
    print(f"  {init_kbps:.1f} kbps  (cap={raw_cap:.1f} kbps)\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

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

            z = model.encode(x)
            z_ste = ste_quantize_3bit(z)
            x_recon = model.decode(z_ste)

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
                'window_size': window_size,
                'bottleneck_dim': bottleneck_dim,
                'temporal_stride': temporal_stride,
                'train_loss': avg_loss,
                'real_bitrate_kbps': real_kbps,
                'qat': True, 'num_bits': 3,
                'phase': 'B',
            }, args.output / 'best.pt')
            print(f"  Saved best checkpoint")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'd_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads,
                'window_size': window_size,
                'bottleneck_dim': bottleneck_dim,
                'temporal_stride': temporal_stride,
                'qat': True, 'num_bits': 3,
                'phase': 'B',
            }, args.output / f'epoch_{epoch+1:02d}.pt')

    with open(args.output / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nPhase B done. Best loss: {best_loss:.6f}")


# CLI

def main():
    import argparse
    paths = get_dataset_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint', type=Path,
                        default=PROJECT_ROOT / 'checkpoints_ratedistortion/temporal_phaseA/best.pt')
    parser.add_argument('--data-root', type=Path, default=paths['train_clean_100'])
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'checkpoints_ratedistortion/temporal_phaseB')
    parser.add_argument('--epochs', type=int, default=20)
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
