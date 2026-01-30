#!/usr/bin/env python3
"""
PHASE 2: Perceptual Loss Fine-tuning
- Load Phase 1 checkpoint (multi-scale loss)
- Add mel-spectrogram perceptual loss
- Train for 25 epochs with conservative LR
- Expected: PESQ 3.1-3.30 (+0.15-0.25 improvement from Phase 1)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from datetime import datetime
import logging
import json
import random
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.model import NeuralAudioCodec


class PerceptualLoss(nn.Module):
    """Mel-spectrogram perceptual loss - targets human auditory perception"""
    def __init__(self, sr=16000, n_mels=128, n_fft=1024, hop_length=256):
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Mel-scale spectrogram using MelSpectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0,
            f_max=sr // 2
        )
    
    def forward(self, pred, target):
        """Compute mel-spectrogram perceptual loss"""
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target
        
        # Compute Mel-spectrograms directly
        pred_mel = self.mel_spectrogram(pred)
        target_mel = self.mel_spectrogram(target)
        
        # Log compression
        pred_mel_log = torch.log(pred_mel + 1e-9)
        target_mel_log = torch.log(target_mel + 1e-9)
        
        # L1 loss in mel-space
        mel_loss = F.l1_loss(pred_mel_log, target_mel_log)
        
        return mel_loss


class CombinedLoss(nn.Module):
    """Combined loss: Time + Multi-scale Spectral + Perceptual"""
    def __init__(self):
        super().__init__()
        self.perceptual = PerceptualLoss()
    
    def forward(self, pred, target):
        """Compute combined loss"""
        pred = torch.clamp(pred, -1.0, 1.0)
        target = torch.clamp(target, -1.0, 1.0)
        
        # Time-domain L1 loss
        time_loss = F.l1_loss(pred, target)
        
        # Perceptual (mel-based) loss
        perceptual_loss = self.perceptual(pred, target)
        
        # Combined: emphasize perceptual
        total_loss = 0.5 * time_loss + 2.0 * perceptual_loss
        total_loss = torch.clamp(total_loss, max=50.0)
        
        loss_dict = {
            'total': total_loss.item(),
            'time': time_loss.item(),
            'perceptual': perceptual_loss.item(),
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
    """Train one epoch with progress bar"""
    model.train()
    total_loss = 0
    loss_dict_sum = {'time': 0, 'perceptual': 0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/25', unit='batch', ncols=100)
    epoch_start_time = time.time()
    
    for batch_idx, audio in enumerate(pbar):
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
        
        # Update progress bar with real-time stats
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            elapsed_time = time.time() - epoch_start_time
            batch_time = elapsed_time / num_batches
            remaining_batches = len(dataloader) - batch_idx - 1
            eta = batch_time * remaining_batches if remaining_batches > 0 else 0
            
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Time': f'{loss_dict["time"]:.4f}',
                'Perc': f'{loss_dict["perceptual"]:.4f}',
                'ETA': f'{int(eta)}s'
            })
    
    pbar.close()
    
    if num_batches == 0:
        return float('inf'), loss_dict_sum
    
    avg_loss = total_loss / num_batches
    for key in loss_dict_sum:
        loss_dict_sum[key] /= num_batches
    
    return avg_loss, loss_dict_sum


def main():
    logger.info("=" * 80)
    logger.info("PHASE 2: PERCEPTUAL LOSS FINE-TUNING")
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
    
    # Load Phase 1 checkpoint
    phase1_checkpoint = Path('checkpoints_emergency/phase1_multiscale_20260129_124452/best.pt')
    if not phase1_checkpoint.exists():
        logger.error(f"❌ Phase 1 checkpoint not found: {phase1_checkpoint}")
        return
    
    logger.info(f"📂 Loading Phase 1 checkpoint: {phase1_checkpoint}")
    try:
        checkpoint = torch.load(phase1_checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info("✅ Phase 1 checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load Phase 1 checkpoint: {e}")
        return
    
    # Dataset
    audio_dir = Path('/home/muaw1874/Desktop/ac_proj/datasets/LibriSpeech/train-clean-100')
    if not audio_dir.exists():
        logger.error(f"❌ Dataset not found: {audio_dir}")
        return
    
    logger.info("📊 Loading dataset...")
    dataset = AudioDataset(audio_dir, sample_rate=16000, segment_length=6000)
    
    # Use same 1500 files as Phase 1 for consistency
    num_files = min(1500, len(dataset))
    logger.info(f"📊 Using {num_files} files for Phase 2")
    
    indices = torch.randperm(len(dataset))[:num_files].tolist()
    subset = Subset(dataset, indices)
    
    # DataLoaders
    train_loader = DataLoader(subset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    
    # Loss and optimizer
    criterion = CombinedLoss()
    criterion = criterion.to(device)
    
    # Slightly lower LR than Phase 1 (Phase 2 is more aggressive with perceptual loss)
    lr = 5e-7
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    
    logger.info(f"⚙️ Learning rate: {lr}")
    logger.info(f"⚙️ Epochs: 25")
    logger.info(f"⚙️ Batch size: 4")
    logger.info(f"⚙️ Loss: Time (0.5x) + Perceptual Mel (2.0x)")
    logger.info("")
    
    # Checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f'checkpoints_emergency/phase2_perceptual_{timestamp}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Checkpoint dir: {checkpoint_dir}")
    
    # Training loop
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, 26):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch}/25")
        logger.info(f"{'=' * 80}")
        
        train_loss, train_dict = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        
        logger.info(f"✅ Epoch {epoch} - Train Loss: {train_loss:.6f}")
        logger.info(f"   Time Loss: {train_dict['time']:.6f} | Perceptual Loss: {train_dict['perceptual']:.6f}")
        
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
    logger.info("PHASE 2 TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"✅ Best checkpoint: {checkpoint_dir / 'best.pt'}")
    logger.info(f"✅ Best loss: {best_loss:.6f}")
    logger.info("")
    logger.info("Next step: Run Phase 3 (Extended Data + Augmentation)")
    logger.info("")
    
    # Save metadata
    metadata = {
        'phase': 'Phase 2',
        'strategy': 'Perceptual Loss (Mel-Spectrogram)',
        'phase1_checkpoint': str(phase1_checkpoint),
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
