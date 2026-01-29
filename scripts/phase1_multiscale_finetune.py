#!/usr/bin/env python3
"""
PHASE 1: Multi-Scale Spectral Loss Fine-tuning
- Load V3 checkpoint
- Implement 3-scale STFT loss (256, 512, 1024)
- Train for 20 epochs with conservative LR
- Expected: PESQ 3.0-3.05 (+0.05 to +0.1 improvement)
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
from datetime import datetime
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import model
from src.model import NeuralAudioCodec


class Phase1MultiScaleSpectralLoss(nn.Module):
    """Multi-scale spectral loss optimized for Phase 1 (3 scales: 256, 512, 1024)"""
    def __init__(self):
        super().__init__()
        # Three scales for multi-resolution analysis
        self.fft_configs = [
            {'fft_size': 256, 'hop_size': 64, 'win_size': 256},
            {'fft_size': 512, 'hop_size': 128, 'win_size': 512},
            {'fft_size': 1024, 'hop_size': 256, 'win_size': 1024},
        ]
    
    def stft(self, x, fft_size, hop_size, win_size):
        """Compute STFT"""
        x = x.squeeze(1) if x.dim() == 3 else x
        window = torch.hann_window(win_size, device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x, n_fft=fft_size, hop_length=hop_size, win_length=win_size,
            window=window, return_complex=True, center=True, pad_mode='reflect'
        )
        return spec
    
    def forward(self, pred, target):
        """Compute multi-scale spectral loss"""
        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        valid_scales = 0
        
        for config in self.fft_configs:
            fft_size = config['fft_size']
            
            # Skip if audio is too short
            if pred.shape[-1] < fft_size * 2 or target.shape[-1] < fft_size * 2:
                continue
            
            try:
                # Compute STFTs
                pred_spec = self.stft(pred, **config)
                target_spec = self.stft(target, **config)
                
                # Magnitude spectra
                pred_mag = torch.abs(pred_spec) + 1e-8
                target_mag = torch.abs(target_spec) + 1e-8
                
                # Log magnitude loss (perceptual)
                pred_log_mag = torch.log(pred_mag)
                target_log_mag = torch.log(target_mag)
                log_mag_loss = F.l1_loss(pred_log_mag, target_log_mag)
                
                # Linear magnitude loss (energy)
                lin_mag_loss = F.l1_loss(pred_mag, target_mag)
                
                # Normalized by scale
                scale_loss = (log_mag_loss + lin_mag_loss) / 2.0
                
                if not (torch.isnan(scale_loss) or torch.isinf(scale_loss)):
                    loss = loss + scale_loss
                    valid_scales += 1
                
            except Exception as e:
                logger.debug(f"Skipping FFT size {fft_size}: {e}")
                continue
        
        if valid_scales == 0:
            logger.warning("No valid scales computed!")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        return loss / valid_scales


class Phase1Loss(nn.Module):
    """Combined loss for Phase 1: Time-domain + Multi-scale spectral"""
    def __init__(self, time_weight=1.0, spectral_weight=1.5):
        super().__init__()
        self.time_weight = time_weight
        self.spectral_weight = spectral_weight
        self.spectral_loss_fn = Phase1MultiScaleSpectralLoss()
    
    def forward(self, pred, target):
        # Clamp to valid range
        pred = torch.clamp(pred, -1.0, 1.0)
        target = torch.clamp(target, -1.0, 1.0)
        
        # Time-domain L1 loss
        time_loss = F.l1_loss(pred, target)
        
        # Multi-scale spectral loss
        spectral_loss = self.spectral_loss_fn(pred, target)
        
        # Combined loss with spectral emphasis
        total_loss = self.time_weight * time_loss + self.spectral_weight * spectral_loss
        
        # Clamp to prevent explosion
        total_loss = torch.clamp(total_loss, max=50.0)
        
        loss_dict = {
            'total': total_loss.item(),
            'time': time_loss.item(),
            'spectral': spectral_loss.item(),
        }
        
        return total_loss, loss_dict


class AudioDataset(Dataset):
    """Audio dataset for training"""
    def __init__(self, audio_dir, sample_rate=16000, segment_length=6000):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.resamplers = {}
        
        self.audio_files = []
        for ext in ['.wav', '.flac', '.mp3']:
            self.audio_files.extend(list(self.audio_dir.rglob(f"*{ext}")))
        
        logger.info(f"✓ Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        try:
            audio_path = self.audio_files[idx]
            waveform, sr = torchaudio.load(str(audio_path), backend='ffmpeg')
            
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample
            if sr != self.sample_rate:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = self.resamplers[sr](waveform)
            
            # Normalize
            max_val = torch.abs(waveform).max()
            if max_val > 1e-6:
                waveform = waveform / (max_val + 1e-8)
            
            # Segment
            if waveform.shape[1] < self.segment_length:
                padding = self.segment_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            elif waveform.shape[1] > self.segment_length:
                max_start = waveform.shape[1] - self.segment_length
                start = torch.randint(0, max_start + 1, (1,)).item()
                waveform = waveform[:, start:start + self.segment_length]
            
            return waveform.float()
        except Exception as e:
            logger.debug(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, self.segment_length, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, log_interval=50):
    """Train one epoch"""
    model.train()
    total_loss = 0
    loss_dict_sum = {'time': 0, 'spectral': 0}
    num_batches = 0
    
    for batch_idx, audio in enumerate(dataloader):
        audio = audio.to(device)
        
        if torch.isnan(audio).any():
            continue
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed = model(audio)
        
        # Handle size mismatch
        if reconstructed.shape[-1] > audio.shape[-1]:
            reconstructed = reconstructed[:, :, :audio.shape[-1]]
        elif reconstructed.shape[-1] < audio.shape[-1]:
            pad_size = audio.shape[-1] - reconstructed.shape[-1]
            reconstructed = F.pad(reconstructed, (0, pad_size))
        
        # Compute loss
        loss, loss_dict = criterion(reconstructed, audio)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at batch {batch_idx}")
            optimizer.zero_grad()
            continue
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for key in loss_dict_sum:
            if key in loss_dict:
                loss_dict_sum[key] += loss_dict[key]
        num_batches += 1
        
        if batch_idx % log_interval == 0 and batch_idx > 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f} | Avg: {avg_loss:.6f} | "
                f"Time: {loss_dict['time']:.6f} | Spectral: {loss_dict['spectral']:.6f}"
            )
    
    if num_batches == 0:
        return float('inf'), loss_dict_sum
    
    avg_loss = total_loss / num_batches
    for key in loss_dict_sum:
        loss_dict_sum[key] /= num_batches
    
    return avg_loss, loss_dict_sum


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    loss_dict_sum = {'time': 0, 'spectral': 0}
    num_batches = 0
    
    with torch.no_grad():
        for audio in dataloader:
            audio = audio.to(device)
            
            if torch.isnan(audio).any():
                continue
            
            reconstructed = model(audio)
            
            # Handle size mismatch
            if reconstructed.shape[-1] > audio.shape[-1]:
                reconstructed = reconstructed[:, :, :audio.shape[-1]]
            elif reconstructed.shape[-1] < audio.shape[-1]:
                pad_size = audio.shape[-1] - reconstructed.shape[-1]
                reconstructed = F.pad(reconstructed, (0, pad_size))
            
            loss, loss_dict = criterion(reconstructed, audio)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            total_loss += loss.item()
            for key in loss_dict_sum:
                if key in loss_dict:
                    loss_dict_sum[key] += loss_dict[key]
            num_batches += 1
    
    if num_batches == 0:
        return float('inf'), loss_dict_sum
    
    avg_loss = total_loss / num_batches
    for key in loss_dict_sum:
        loss_dict_sum[key] /= num_batches
    
    return avg_loss, loss_dict_sum


def main():
    logger.info("=" * 80)
    logger.info("PHASE 1: MULTI-SCALE SPECTRAL LOSS FINE-TUNING")
    logger.info("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🎯 Device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Model config
    model_config = {
        'd_model': 384,
        'n_layers': 6,
        'n_heads': 8,
        'window_size': 384,
        'hop_length': 160,
        'sample_rate': 16000,
    }
    
    # Load model
    logger.info("📦 Loading model...")
    model = NeuralAudioCodec(**model_config)
    model = model.to(device)
    
    # Load V3 checkpoint
    v3_checkpoint = Path('checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt')
    if not v3_checkpoint.exists():
        logger.error(f"❌ V3 checkpoint not found: {v3_checkpoint}")
        return
    
    logger.info(f"📂 Loading V3 checkpoint: {v3_checkpoint}")
    try:
        checkpoint = torch.load(v3_checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info("✅ V3 checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load V3 checkpoint: {e}")
        return
    
    # Dataset
    audio_dir = Path('/home/muaw1874/Desktop/ac_proj/datasets/LibriSpeech/train-clean-100')
    if not audio_dir.exists():
        logger.error(f"❌ Dataset not found: {audio_dir}")
        return
    
    logger.info("📊 Loading dataset...")
    dataset = AudioDataset(audio_dir, sample_rate=16000, segment_length=6000)
    
    # Use fewer files to speed up training
    num_files = min(1500, len(dataset))
    logger.info(f"📊 Using {num_files} files for Phase 1")
    
    indices = torch.randperm(len(dataset))[:num_files].tolist()
    from torch.utils.data import Subset
    subset = Subset(dataset, indices)
    
    # DataLoaders
    train_loader = DataLoader(subset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    
    # Loss and optimizer
    criterion = Phase1Loss(time_weight=1.0, spectral_weight=1.5)
    criterion = criterion.to(device)
    
    # Conservative learning rate for fine-tuning
    lr = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    logger.info(f"⚙️ Learning rate: {lr}")
    logger.info(f"⚙️ Epochs: 20")
    logger.info(f"⚙️ Batch size: 4")
    logger.info(f"⚙️ Loss: Time (1.0x) + Multi-Scale Spectral (1.5x)")
    logger.info("")
    
    # Checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f'checkpoints_emergency/phase1_multiscale_{timestamp}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Checkpoint dir: {checkpoint_dir}")
    
    # Training loop
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, 21):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch}/20")
        logger.info(f"{'=' * 80}")
        
        train_loss, train_dict = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        
        logger.info(f"✅ Epoch {epoch} - Train Loss: {train_loss:.6f}")
        logger.info(f"   Time Loss: {train_dict['time']:.6f} | Spectral Loss: {train_dict['spectral']:.6f}")
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f'epoch_{epoch:02d}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save best
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = checkpoint_dir / 'best.pt'
            torch.save(model.state_dict(), best_path)
            logger.info(f"🏆 New best loss: {train_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"⏸️ Early stopping after {epoch} epochs")
            break
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1 TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"✅ Best checkpoint: {checkpoint_dir / 'best.pt'}")
    logger.info(f"✅ Best loss: {best_loss:.6f}")
    logger.info("")
    logger.info("Next step: Run evaluation to check PESQ improvement")
    logger.info("")
    
    # Save metadata
    metadata = {
        'phase': 'Phase 1',
        'strategy': 'Multi-Scale Spectral Loss',
        'v3_checkpoint': str(v3_checkpoint),
        'fft_scales': [256, 512, 1024],
        'learning_rate': lr,
        'epochs': epoch,
        'best_loss': best_loss,
        'model_config': model_config,
        'checkpoint_dir': str(checkpoint_dir),
    }
    
    with open(checkpoint_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
