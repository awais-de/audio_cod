"""
Training loop, dataset, and losses for the neural audio codec.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiScaleSpectralLoss(nn.Module):
    """Multi-scale spectral loss for perceptual quality"""
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=None, win_sizes=None):
        super().__init__()
        self.fft_sizes = fft_sizes

        if hop_sizes is None:
            self.hop_sizes = [fs // 4 for fs in fft_sizes]
        else:
            self.hop_sizes = hop_sizes

        if win_sizes is None:
            self.win_sizes = fft_sizes
        else:
            self.win_sizes = win_sizes

    def stft(self, x, fft_size, hop_size, win_size):
        x = x.squeeze(1)
        window = torch.hann_window(win_size).to(x.device)
        spec = torch.stft(
            x, n_fft=fft_size, hop_length=hop_size, win_length=win_size,
            window=window, return_complex=True, center=True, pad_mode='reflect'
        )
        return spec

    def forward(self, pred, target):
        loss = 0.0
        valid_scales = 0

        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            if pred.shape[-1] < fft_size or target.shape[-1] < fft_size:
                continue

            try:
                pred_spec = self.stft(pred, fft_size, hop_size, win_size)
                target_spec = self.stft(target, fft_size, hop_size, win_size)

                pred_mag = torch.abs(pred_spec) + 1e-7
                target_mag = torch.abs(target_spec) + 1e-7

                pred_log_mag = torch.log(pred_mag).clamp(min=-15, max=15)
                target_log_mag = torch.log(target_mag).clamp(min=-15, max=15)

                spectral_loss = F.l1_loss(pred_log_mag, target_log_mag)
                mag_loss = F.l1_loss(pred_mag, target_mag)

                if not (torch.isnan(spectral_loss) or torch.isinf(spectral_loss) or
                        torch.isnan(mag_loss) or torch.isinf(mag_loss)):
                    loss += spectral_loss + mag_loss
                    valid_scales += 1
            except Exception as e:
                continue

        if valid_scales == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return loss / valid_scales


class AudioLoss(nn.Module):
    """Combined time-domain and spectral loss"""
    def __init__(self, l1_weight=1.0, spectral_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight
        self.spectral_loss = MultiScaleSpectralLoss()

    def forward(self, pred, target):
        pred = torch.clamp(pred, -1.0, 1.0)
        target = torch.clamp(target, -1.0, 1.0)

        if torch.isnan(pred).any() or torch.isinf(pred).any():
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.5, neginf=-0.5)

        if torch.isnan(target).any() or torch.isinf(target).any():
            target = torch.nan_to_num(target, nan=0.0, posinf=0.5, neginf=-0.5)

        l1_loss = F.l1_loss(pred, target)
        if torch.isnan(l1_loss) or torch.isinf(l1_loss):
            l1_loss = torch.tensor(0.5, device=pred.device, requires_grad=True)

        spectral_loss = self.spectral_loss(pred, target)
        if torch.isnan(spectral_loss) or torch.isinf(spectral_loss):
            spectral_loss = torch.tensor(0.5, device=pred.device, requires_grad=True)

        total_loss = self.l1_weight * l1_loss + self.spectral_weight * spectral_loss
        total_loss = torch.clamp(total_loss, max=100.0)

        loss_dict = {
            'total': total_loss.item(),
            'l1': l1_loss.item(),
            'spectral': spectral_loss.item(),
        }

        return total_loss, loss_dict


class AudioDataset(Dataset):
    """Audio dataset with on-the-fly resampling and random cropping"""
    def __init__(self, audio_dir, sample_rate=16000, segment_length=6000, extensions=['.wav', '.flac', '.mp3']):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.resamplers = {}

        self.audio_files = []
        for ext in extensions:
            self.audio_files.extend(list(self.audio_dir.rglob(f"*{ext}")))

        logger.info(f"Found {len(self.audio_files)} audio files in {audio_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            audio_path = self.audio_files[idx]

            waveform, sr = torchaudio.load(str(audio_path), backend='ffmpeg')

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.sample_rate:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = self.resamplers[sr](waveform)

            max_val = torch.abs(waveform).max()
            if max_val > 1e-6:
                waveform = waveform / (max_val + 1e-8)

            if waveform.shape[1] < self.segment_length:
                padding = self.segment_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            elif waveform.shape[1] > self.segment_length:
                max_start = waveform.shape[1] - self.segment_length
                start = torch.randint(0, max_start + 1, (1,)).item()
                waveform = waveform[:, start:start + self.segment_length]

            return waveform.float()
        except Exception as e:
            return torch.zeros(1, self.segment_length, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, log_interval=50):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'l1': 0, 'spectral': 0}
    num_batches = 0

    for batch_idx, audio in enumerate(dataloader):
        audio = audio.to(device)

        if torch.isnan(audio).any():
            continue

        optimizer.zero_grad()
        reconstructed = model(audio)

        if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
            reconstructed = torch.nan_to_num(reconstructed, nan=0.0, posinf=0.5, neginf=-0.5)

        if reconstructed.shape[-1] > audio.shape[-1]:
            reconstructed = reconstructed[:, :, :audio.shape[-1]]
        elif reconstructed.shape[-1] < audio.shape[-1]:
            pad_size = audio.shape[-1] - reconstructed.shape[-1]
            reconstructed = F.pad(reconstructed, (0, pad_size))

        loss, loss_dict = criterion(reconstructed, audio)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        num_batches += 1

        if batch_idx % log_interval == 0 and batch_idx > 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | "
                f"L1: {loss_dict['l1']:.4f} | Spectral: {loss_dict['spectral']:.4f}"
            )

    if num_batches == 0:
        return float('inf'), {'l1': float('inf'), 'spectral': float('inf')}

    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    loss_components = {'l1': 0, 'spectral': 0}
    num_batches = 0

    with torch.no_grad():
        for audio in dataloader:
            audio = audio.to(device)

            if torch.isnan(audio).any():
                continue

            reconstructed = model(audio)

            if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                reconstructed = torch.nan_to_num(reconstructed, nan=0.0, posinf=0.5, neginf=-0.5)

            if reconstructed.shape[-1] > audio.shape[-1]:
                reconstructed = reconstructed[:, :, :audio.shape[-1]]
            elif reconstructed.shape[-1] < audio.shape[-1]:
                pad_size = audio.shape[-1] - reconstructed.shape[-1]
                reconstructed = F.pad(reconstructed, (0, pad_size))

            loss, loss_dict = criterion(reconstructed, audio)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            num_batches += 1

    if num_batches == 0:
        return float('inf'), {'l1': float('inf'), 'spectral': float('inf')}

    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def train(model, train_loader, val_loader, epochs, lr, device, checkpoint_dir):
    """Main training loop"""

    if device.type == 'cuda':
        logger.info("Enabling GPU optimizations...")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = AudioLoss(l1_weight=1.0, spectral_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    start_time = datetime.now()

    for epoch in range(1, epochs + 1):
        epoch_start = datetime.now()

        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'='*80}")

        train_loss, train_components = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_components = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = (datetime.now() - epoch_start).total_seconds()

        logger.info(f"\n{'-'*80}")
        logger.info(f"EPOCH {epoch} COMPLETE ({elapsed:.1f}s)")
        logger.info(f"{'-'*80}")
        logger.info(f"Training Loss:    {train_loss:.6f} (L1: {train_components['l1']:.6f}, Spectral: {train_components['spectral']:.6f})")
        logger.info(f"Validation Loss:  {val_loss:.6f} (L1: {val_components['l1']:.6f}, Spectral: {val_components['spectral']:.6f})")
        logger.info(f"Learning Rate:    {scheduler.get_last_lr()[0]:.8f}")
        logger.info(f"{'-'*80}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model (Val Loss: {val_loss:.6f})\n")

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Saved periodic checkpoint (Epoch {epoch})\n")

    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nTraining complete. Total time: {total_time/3600:.2f} hours")


if __name__ == "__main__":
    from src.model import NeuralAudioCodec

    with open('config/training.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB)")
    else:
        logger.info("Using CPU (slow training)")

    model_cfg = config['model']
    model = NeuralAudioCodec(
        sample_rate=model_cfg['sample_rate'],
        hop_length=model_cfg['hop_length'],
        d_model=model_cfg['d_model'],
        n_layers=model_cfg['n_layers'],
        n_heads=model_cfg['n_heads'],
        window_size=model_cfg['window_size'],
        dropout=model_cfg['dropout']
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params:,}")

    train_cfg = config['training']
    data_cfg = config['data']

    train_dataset = AudioDataset(
        data_cfg['train_dir'],
        sample_rate=model_cfg['sample_rate'],
        segment_length=train_cfg['segment_length']
    )

    val_dataset = AudioDataset(
        data_cfg['val_dir'],
        sample_rate=model_cfg['sample_rate'],
        segment_length=train_cfg['segment_length']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Batch size: {train_cfg['batch_size']}")
    logger.info(f"Segment length: {train_cfg['segment_length']} samples ({train_cfg['segment_length']/model_cfg['sample_rate']:.2f}s)")
    logger.info(f"Starting training for {train_cfg['epochs']} epochs...\n")

    train(
        model,
        train_loader,
        val_loader,
        epochs=train_cfg['epochs'],
        lr=train_cfg['learning_rate'],
        device=device,
        checkpoint_dir=config['checkpoint']['dir']
    )
