#!/usr/bin/env python3
"""
Phase C: Extended 3-bit QAT with Noise Augmentation
=====================================================
Loads temporal_phaseB/best.pt and continues training with:
  - Noise augmentation (white, pink, babble) at random SNR 5-20 dB
  - Applied at training time only — model learns robust representations
  - Same 3-bit STE quantization as Phase B
  - Lower LR (cosine from 5e-6 → 1e-7) since weights are already adapted

Goal: close the quality gap for female/high-frequency voices and
      improve robustness in noisy conditions.
"""

import sys
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths
from src.losses import (
    ste_quantize_3bit, multi_scale_stft_loss, NoisyAudioDataset, measure_real_bitrate
)


# Training

def train(args):
    print(f"\n{'='*68}")
    print("PHASE C: 3-BIT QAT + NOISE AUGMENTATION")
    print(f"{'='*68}")
    print(f"Base checkpoint  : {args.base_checkpoint}")
    print(f"Output           : {args.output}")
    print(f"Epochs           : {args.epochs}  |  LR: {args.lr}  |  Cosine annealing")
    print(f"Noise prob       : {args.noise_prob}  |  SNR: {args.snr_min}-{args.snr_max} dB")
    print(f"{'='*68}\n")

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

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
        bottleneck_dim=bottleneck_dim, temporal_stride=temporal_stride,
    ).to(device)
    model.load_state_dict(state)
    print(f"Loaded Phase B: d_model={d_model}, bottleneck_dim={bottleneck_dim}, "
          f"temporal_stride={temporal_stride}\n")

    dataset = NoisyAudioDataset(
        args.data_root, chunk_seconds=args.chunk_sec,
        epoch_size=args.samples_per_epoch,
        noise_prob=args.noise_prob,
        snr_range=(args.snr_min, args.snr_max),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    val_files = dataset.files[:20]

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
                'phase': 'C',
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
                'phase': 'C',
            }, args.output / f'epoch_{epoch+1:02d}.pt')

    with open(args.output / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nPhase C done. Best loss: {best_loss:.6f}")


# CLI

def main():
    import argparse
    paths = get_dataset_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint', type=Path,
                        default=PROJECT_ROOT / 'checkpoints_active/temporal_phaseB/best.pt')
    parser.add_argument('--data-root', type=Path, default=paths['train_clean_100'])
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'checkpoints_active/temporal_phaseC')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--chunk-sec', type=float, default=1.0)
    parser.add_argument('--samples-per-epoch', type=int, default=1000)
    parser.add_argument('--noise-prob', type=float, default=0.6,
                        help='Probability of applying noise augmentation per sample')
    parser.add_argument('--snr-min', type=float, default=5.0)
    parser.add_argument('--snr-max', type=float, default=20.0)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
