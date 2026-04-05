#!/usr/bin/env python3
"""
Temporal Bottleneck + 3-bit QAT Training
==========================================
Architecture:
  waveform → AudioEncoder (2000 Hz, d=384)
           → Linear (384 → 32)
           → Conv1d stride=20  (2000 → ~100 Hz)   ← QUANTIZE HERE (3-bit)
           → ConvTranspose1d ×20 (~100 → 2000 Hz)
           → Linear (32 → 384)
           → AudioDecoder → waveform

Bitrate math (hard architectural cap):
  32 dims × 3 bits × 100 Hz = 9,600 bps = 9.6 kbps

3-bit STE quantization (8 levels) is far gentler than 1-bit — no entropy
penalty hacks needed. The model simply learns to use 8 discrete levels
per dimension per 10ms frame.

Base: Phase 1 checkpoint (PESQ=4.27, best quality in chain)
New layers (encoder_proj, decoder_proj, temporal_enc, temporal_dec)
are initialised from scratch via strict=False loading.
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
from src.paths import get_dataset_paths, get_checkpoint_paths


# ---------------------------------------------------------------------------
# 3-bit Straight-Through Estimator
# ---------------------------------------------------------------------------

def ste_quantize_3bit(z: torch.Tensor) -> torch.Tensor:
    """
    3-bit uniform quantization with Straight-Through Estimator.

    Forward : z → 8 discrete levels (uniform grid between z_min and z_max)
    Backward: gradient passes through unchanged

    8 levels (3-bit) produces far less distortion than 2 levels (1-bit),
    enabling the model to converge without any auxiliary loss terms.
    """
    num_levels = 8  # 2^3
    z_min = z.min()
    z_max = z.max()
    scale = (z_max - z_min) / (num_levels - 1) + 1e-8

    z_norm = (z - z_min) / scale
    z_int = torch.clamp(torch.round(z_norm), 0, num_levels - 1)
    z_quant = z_int * scale + z_min

    # STE: real quantization forward, identity backward
    return z + (z_quant - z).detach()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

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
# Bitrate measurement (3-bit quantization + zlib, matches training)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    print(f"\n{'='*68}")
    print("TEMPORAL BOTTLENECK + 3-BIT QAT TRAINING")
    print(f"{'='*68}")
    print(f"Base checkpoint  : {args.base_checkpoint}")
    print(f"Output           : {args.output}")
    print(f"Bottleneck dim   : {args.bottleneck_dim}")
    print(f"Temporal stride  : {args.temporal_stride}  "
          f"(~{2000//args.temporal_stride} Hz frame rate)")
    raw_cap = args.bottleneck_dim * 3 * (2000 // args.temporal_stride) / 1000
    print(f"Raw bitrate cap  : {raw_cap:.1f} kbps  (32 × 3-bit × {2000//args.temporal_stride} Hz)")
    print(f"Epochs           : {args.epochs}  |  LR: {args.lr}")
    print(f"{'='*68}\n")

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    # Load base checkpoint and infer architecture
    ckpt = torch.load(args.base_checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    d_model = ckpt.get('d_model')
    if d_model is None:
        k = 'encoder.transformer_blocks.0.attention.qkv.weight'
        d_model = state[k].shape[1] if k in state else 256

    ids = set()
    for k in state:
        if 'encoder.transformer_blocks.' in k:
            p = k.split('.')
            if len(p) > 2 and p[2].isdigit():
                ids.add(int(p[2]))
    n_layers = max(ids) + 1 if ids else 4
    n_heads = ckpt.get('n_heads', 8)

    model = NeuralAudioCodec(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        window_size=args.window_size, dropout=0.0,
        bottleneck_dim=args.bottleneck_dim,
        temporal_stride=args.temporal_stride,
    ).to(device)

    missing, _ = model.load_state_dict(state, strict=False)
    print(f"Loaded {args.base_checkpoint.name}: d_model={d_model}, "
          f"n_layers={n_layers}, n_heads={n_heads}")
    print(f"New layers (random init): {missing}\n")

    # Dataset
    dataset = AudioChunkDataset(args.data_root, chunk_seconds=args.chunk_sec,
                                epoch_size=args.samples_per_epoch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    val_files = dataset.files[:20]

    # Initial bitrate sanity check
    print("Initial bitrate (before training)...")
    init_kbps = measure_real_bitrate(model, val_files, device)
    print(f"  {init_kbps:.1f} kbps  (cap={raw_cap:.1f} kbps)\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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

            z = model.encode(x)           # (B, T/stride, 32) — compressed latents
            z_ste = ste_quantize_3bit(z)  # 3-bit quantized, STE gradient
            x_recon = model.decode(z_ste) # decoder sees 3-bit quantized latents

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
                'window_size': args.window_size,
                'bottleneck_dim': args.bottleneck_dim,
                'temporal_stride': args.temporal_stride,
                'train_loss': avg_loss,
                'real_bitrate_kbps': real_kbps,
                'qat': True, 'num_bits': 3,
            }, args.output / 'best.pt')
            print(f"  ✓ Saved best checkpoint")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'd_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads,
                'window_size': args.window_size,
                'bottleneck_dim': args.bottleneck_dim,
                'temporal_stride': args.temporal_stride,
                'qat': True, 'num_bits': 3,
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
    ckpt_paths = get_checkpoint_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint', type=Path, default=ckpt_paths['phase1'],
                        help='Phase 1 checkpoint (best quality base)')
    parser.add_argument('--data-root', type=Path, default=paths['train_clean_100'])
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'checkpoints_ratedistortion/temporal_qat_v1')
    parser.add_argument('--bottleneck-dim', type=int, default=32)
    parser.add_argument('--temporal-stride', type=int, default=20,
                        help='Temporal downsampling factor (20 → ~100 Hz → 9.6 kbps cap)')
    parser.add_argument('--window-size', type=int, default=200,
                        help='Attention window frames (200 × 0.5ms = 100ms latency)')
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
