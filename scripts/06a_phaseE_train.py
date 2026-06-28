#!/usr/bin/env python3
"""
Phase E: Log-Magnitude STFT Loss
=================================
Single change from Phase C: linear STFT magnitude → log-magnitude.

  Phase C loss:  mean(|Mr - Mt|²) + mean(|Mr - Mt|)       [linear magnitude]
  Phase E loss:  mean(|log(Mr+1) - log(Mt+1)|²)            [log magnitude]
               + mean( |log(Mr+1) - log(Mt+1)|)

log1p compresses the dynamic range. Quiet frequency components (consonants,
sibilants, high-frequency detail) were getting small gradients under linear
magnitude — the model learned to ignore them and average them out, producing
the "muffled" character. Under log magnitude they get proportionally larger
gradients and the model must learn to reconstruct them.

Everything else unchanged from Phase C:
  - Architecture: identical
  - Quantization: 3-bit STE
  - Noise augmentation: 60%, 5-20 dB SNR
  - Loads from: temporal_phaseC/best.pt

Loads: checkpoints_active/temporal_phaseC/best.pt
Output: checkpoints_active/temporal_phaseE/
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


# Loss — the only thing that changed

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
        # log-magnitude: compresses dynamic range, equalises gradient across frequencies
        Mr = torch.log1p(torch.abs(Sr))
        Mt = torch.log1p(torch.abs(St))
        total = total + torch.mean((Mr - Mt) ** 2) + torch.mean(torch.abs(Mr - Mt))
    return total / len(fft_sizes)


# 3-bit STE (unchanged from Phase C)

def ste_quantize_3bit(z):
    num_levels = 8
    z_min, z_max = z.min(), z.max()
    scale = (z_max - z_min) / (num_levels - 1) + 1e-8
    z_norm = (z - z_min) / scale
    z_int = torch.clamp(torch.round(z_norm), 0, num_levels - 1)
    z_quant = z_int * scale + z_min
    return z + (z_quant - z).detach()


# Noise augmentation (unchanged from Phase C)

def pink_noise(n):
    f = np.fft.rfftfreq(n)
    f[0] = 1.0
    spectrum = (np.random.randn(len(f)) + 1j * np.random.randn(len(f))) / np.sqrt(f)
    spectrum[0] = 0
    noise = np.fft.irfft(spectrum, n=n).astype(np.float32)
    return noise / (np.abs(noise).max() + 1e-8)


def add_noise(clean, noise_type, snr_db):
    signal_power = np.mean(clean ** 2) + 1e-8
    if noise_type == 'white':
        noise = np.random.randn(len(clean)).astype(np.float32)
    elif noise_type == 'pink':
        noise = pink_noise(len(clean))
    else:
        noise = np.random.randn(len(clean)).astype(np.float32)
    noise_power = np.mean(noise ** 2) + 1e-8
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return np.clip(clean + noise, -1.0, 1.0).astype(np.float32)


class NoisyAudioDataset(IterableDataset):
    def __init__(self, data_root, chunk_seconds=1.0, sample_rate=16000,
                 epoch_size=1000, noise_prob=0.6, snr_range=(5, 20)):
        self.chunk_size = int(chunk_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.epoch_size = epoch_size
        self.noise_prob = noise_prob
        self.snr_min, self.snr_max = snr_range
        exts = ('.wav', '.flac', '.mp3', '.ogg')
        self.files = sorted(p for p in Path(data_root).rglob('*') if p.suffix.lower() in exts)
        if not self.files:
            raise ValueError(f"No audio files in {data_root}")
        print(f"Dataset: {len(self.files)} files  |  noise_prob={noise_prob}  "
              f"SNR={snr_range[0]}-{snr_range[1]}dB")

    def _load_chunk(self, path):
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            n = int(len(audio) * self.sample_rate / sr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        if len(audio) > self.chunk_size:
            start = np.random.randint(0, len(audio) - self.chunk_size)
            return audio[start:start + self.chunk_size]
        return np.pad(audio, (0, self.chunk_size - len(audio)))

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        noise_types = ['white', 'pink']
        for _ in range(self.epoch_size):
            path = self.files[np.random.randint(0, len(self.files))]
            try:
                chunk = self._load_chunk(path)
                if np.random.random() < self.noise_prob:
                    snr = np.random.uniform(self.snr_min, self.snr_max)
                    noise_type = np.random.choice(noise_types + ['babble'])
                    if noise_type == 'babble':
                        babble_path = self.files[np.random.randint(0, len(self.files))]
                        try:
                            babble = self._load_chunk(babble_path)
                            chunk = add_noise(chunk, 'white', snr)
                            babble_power = np.mean(babble ** 2) + 1e-8
                            sig_power = np.mean(chunk ** 2) + 1e-8
                            target_babble_power = sig_power / (10 ** (snr / 10))
                            babble_scaled = babble * np.sqrt(target_babble_power / babble_power)
                            chunk = np.clip(chunk + babble_scaled, -1.0, 1.0).astype(np.float32)
                        except Exception:
                            chunk = add_noise(chunk, 'white', snr)
                    else:
                        chunk = add_noise(chunk, noise_type, snr)
                yield torch.FloatTensor(chunk).unsqueeze(0)
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
                for start in range(0, len(audio), chunk_samples):
                    chunk = audio[start:start + chunk_samples]
                    if len(chunk) < 160:
                        continue
                    x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
                    z = model.encode(x).squeeze(0).cpu().numpy()
                    z_min, z_max = z.min(), z.max()
                    scale = (z_max - z_min) / (num_levels - 1) + 1e-8
                    q = np.clip(np.round((z - z_min) / scale), 0, num_levels - 1).astype(np.uint8)
                    total_bits += len(zlib.compress(q.tobytes(), level=9)) * 8
                    total_dur += len(chunk) / sr
            except Exception:
                continue
    return total_bits / total_dur / 1000.0 if total_dur > 0 else float('nan')


# Training

def train(args):
    print(f"\n{'='*68}")
    print("PHASE E: LOG-MAGNITUDE STFT LOSS")
    print(f"{'='*68}")
    print(f"Base checkpoint  : {args.base_checkpoint}")
    print(f"Output           : {args.output}")
    print(f"Epochs           : {args.epochs}  |  LR: {args.lr}  |  Cosine annealing")
    print(f"Loss             : log1p(|STFT|) — L1 + L2  at n_fft=[256,512,1024]")
    print(f"Quantization     : 3-bit STE")
    print(f"Noise prob       : {args.noise_prob}  |  SNR: {args.snr_min}-{args.snr_max} dB")
    print(f"{'='*68}\n")

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.base_checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    d_model         = ckpt.get('d_model', 384)
    n_layers        = ckpt.get('n_layers', 6)
    n_heads         = ckpt.get('n_heads', 8)
    window_size     = ckpt.get('window_size', 200)
    bottleneck_dim  = ckpt.get('bottleneck_dim', 32)
    temporal_stride = ckpt.get('temporal_stride', 20)

    model = NeuralAudioCodec(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        window_size=window_size, dropout=0.0,
        bottleneck_dim=bottleneck_dim, temporal_stride=temporal_stride,
    ).to(device)
    model.load_state_dict(state)
    print(f"Loaded Phase C: d_model={d_model}, bottleneck_dim={bottleneck_dim}, "
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

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
                'phase': 'E',
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
                'phase': 'E',
            }, args.output / f'epoch_{epoch+1:02d}.pt')

    runs_dir = PROJECT_ROOT / 'runs' / 'temporal_phaseE'
    runs_dir.mkdir(parents=True, exist_ok=True)
    with open(runs_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(args.output / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nPhase E done. Best loss: {best_loss:.6f}")


# CLI

def main():
    import argparse
    paths = get_dataset_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint', type=Path,
                        default=PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt')
    parser.add_argument('--data-root',         type=Path, default=paths['train_clean_100'])
    parser.add_argument('--output',            type=Path,
                        default=PROJECT_ROOT / 'checkpoints_active/temporal_phaseE')
    parser.add_argument('--epochs',            type=int,   default=20)
    parser.add_argument('--lr',                type=float, default=5e-6)
    parser.add_argument('--batch-size',        type=int,   default=1)
    parser.add_argument('--grad-accum-steps',  type=int,   default=4)
    parser.add_argument('--chunk-sec',         type=float, default=1.0)
    parser.add_argument('--samples-per-epoch', type=int,   default=1000)
    parser.add_argument('--noise-prob',        type=float, default=0.6)
    parser.add_argument('--snr-min',           type=float, default=5.0)
    parser.add_argument('--snr-max',           type=float, default=20.0)
    parser.add_argument('--device',            type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
