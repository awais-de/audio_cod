#!/usr/bin/env python3
"""
PHASE 4: Adversarial Fine-tuning
- Load Phase 3 checkpoint
- Train lightweight discriminator
- Adversarial loss + perceptual loss
- Train for 30 epochs
- Expected: PESQ 3.35-3.75 (+0.10-0.20 improvement from Phase 3)
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.model import NeuralAudioCodec


class SimpleDiscriminator(nn.Module):
    """Lightweight discriminator for adversarial training"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, 1)
    
    def forward(self, x):
        # x shape: [batch, channels, length]
        # Average over channels to get [batch, 1, length]
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PerceptualLoss(nn.Module):
    """Mel-spectrogram perceptual loss"""
    def __init__(self, sr=16000, n_mels=128, n_fft=1024, hop_length=256):
        super().__init__()
        self.sr = sr
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0,
            f_max=sr // 2
        )
    
    def forward(self, pred, target):
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target
        
        pred_mel = self.mel_spectrogram(pred)
        target_mel = self.mel_spectrogram(target)
        
        pred_mel_log = torch.log(pred_mel + 1e-9)
        target_mel_log = torch.log(target_mel + 1e-9)
        
        return F.l1_loss(pred_mel_log, target_mel_log)


class AudioDataset(Dataset):
    """Simple audio dataset"""
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
            logger.debug(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, self.segment_length, dtype=torch.float32)


def train_epoch(model, discriminator, dataloader, optimizer_g, optimizer_d, criterion_recon, criterion_adv, device, epoch, log_interval=50):
    """Train one epoch with adversarial loss and progress bar"""
    model.train()
    discriminator.train()
    total_loss_g = 0
    total_loss_d = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/30', unit='batch', ncols=100)
    
    for batch_idx, audio in enumerate(pbar):
        audio = audio.to(device)
        
        if torch.isnan(audio).any():
            continue
        
        # Generator update
        optimizer_g.zero_grad()
        
        reconstructed = model(audio)
        
        if reconstructed.shape[-1] > audio.shape[-1]:
            reconstructed = reconstructed[:, :, :audio.shape[-1]]
        elif reconstructed.shape[-1] < audio.shape[-1]:
            pad_size = audio.shape[-1] - reconstructed.shape[-1]
            reconstructed = F.pad(reconstructed, (0, pad_size))
        
        # Reconstruction loss
        recon_loss = criterion_recon(reconstructed, audio)
        
        # Adversarial loss (fool discriminator)
        fake_pred = discriminator(reconstructed)
        real_labels = torch.ones_like(fake_pred)
        adv_loss = F.binary_cross_entropy_with_logits(fake_pred, real_labels)
        
        # Combined generator loss
        loss_g = 0.8 * recon_loss + 0.2 * adv_loss
        
        if torch.isnan(loss_g) or torch.isinf(loss_g):
            optimizer_g.zero_grad()
            continue
        
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_g.step()
        
        # Discriminator update
        optimizer_d.zero_grad()
        
        # Real data
        real_pred = discriminator(audio)
        real_labels = torch.ones_like(real_pred)
        loss_d_real = F.binary_cross_entropy_with_logits(real_pred, real_labels)
        
        # Fake data
        with torch.no_grad():
            fake_recon = model(audio)
            if fake_recon.shape[-1] > audio.shape[-1]:
                fake_recon = fake_recon[:, :, :audio.shape[-1]]
        
        fake_pred = discriminator(fake_recon.detach())
        fake_labels = torch.zeros_like(fake_pred)
        loss_d_fake = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)
        
        loss_d = (loss_d_real + loss_d_fake) / 2
        
        if torch.isnan(loss_d) or torch.isinf(loss_d):
            optimizer_d.zero_grad()
            continue
        
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_d.step()
        
        total_loss_g += loss_g.item()
        total_loss_d += loss_d.item()
        num_batches += 1
        
        # Update progress bar
        if num_batches > 0:
            avg_g = total_loss_g / num_batches
            pbar.set_postfix({'loss': avg_g})
    
    pbar.close()
    
    if num_batches == 0:
        return float('inf'), float('inf')
    
    avg_loss_g = total_loss_g / num_batches
    avg_loss_d = total_loss_d / num_batches
    
    return avg_loss_g, avg_loss_d


def main():
    logger.info("=" * 80)
    logger.info("PHASE 4: ADVERSARIAL FINE-TUNING")
    logger.info("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🎯 Device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    model_config = {
        'd_model': 384,
        'n_layers': 6,
        'n_heads': 8,
        'window_size': 384,
        'hop_length': 160,
        'sample_rate': 16000,
    }
    
    logger.info("📦 Loading model and discriminator...")
    model = NeuralAudioCodec(**model_config)
    model = model.to(device)
    
    discriminator = SimpleDiscriminator()
    discriminator = discriminator.to(device)
    
    # Find latest Phase 3 checkpoint
    phase3_dirs = sorted(Path('checkpoints_emergency').glob('phase3_extended_data_*'))
    if not phase3_dirs:
        logger.error("❌ No Phase 3 checkpoint found")
        return
    
    phase3_checkpoint = phase3_dirs[-1] / 'best.pt'
    if not phase3_checkpoint.exists():
        logger.error(f"❌ Phase 3 checkpoint not found: {phase3_checkpoint}")
        return
    
    logger.info(f"📂 Loading Phase 3 checkpoint: {phase3_checkpoint}")
    try:
        checkpoint = torch.load(phase3_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("✅ Phase 3 checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load Phase 3 checkpoint: {e}")
        return
    
    audio_dir = Path('/home/muaw1874/Desktop/ac_proj/datasets/LibriSpeech/train-clean-100')
    if not audio_dir.exists():
        logger.error(f"❌ Dataset not found: {audio_dir}")
        return
    
    logger.info("📊 Loading dataset...")
    dataset = AudioDataset(audio_dir, sample_rate=16000, segment_length=6000)
    
    # Use 2000 files for Phase 4 (balance between training time and diversity)
    num_files = min(2000, len(dataset))
    logger.info(f"📊 Using {num_files} files for Phase 4")
    
    indices = torch.randperm(len(dataset))[:num_files].tolist()
    subset = Subset(dataset, indices)
    
    train_loader = DataLoader(subset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    
    criterion_recon = PerceptualLoss()
    criterion_recon = criterion_recon.to(device)
    criterion_adv = nn.BCEWithLogitsLoss()
    
    # Separate optimizers for generator and discriminator
    lr_g = 2e-7  # Very conservative
    lr_d = 1e-6  # Discriminator slightly higher
    
    optimizer_g = torch.optim.Adam(model.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=30)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=30)
    
    logger.info(f"⚙️ Generator LR: {lr_g}")
    logger.info(f"⚙️ Discriminator LR: {lr_d}")
    logger.info(f"⚙️ Epochs: 30")
    logger.info("")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f'checkpoints_emergency/phase4_adversarial_{timestamp}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Checkpoint dir: {checkpoint_dir}")
    
    best_loss_g = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, 31):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Epoch {epoch}/30")
        logger.info(f"{'=' * 80}")
        
        loss_g, loss_d = train_epoch(
            model, discriminator, train_loader,
            optimizer_g, optimizer_d,
            criterion_recon, criterion_adv, device, epoch
        )
        
        scheduler_g.step()
        scheduler_d.step()
        
        logger.info(f"✅ Epoch {epoch} - Loss_G: {loss_g:.6f}, Loss_D: {loss_d:.6f}")
        
        checkpoint_path = checkpoint_dir / f'epoch_{epoch:02d}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        
        if loss_g < best_loss_g:
            best_loss_g = loss_g
            best_path = checkpoint_dir / 'best.pt'
            torch.save(model.state_dict(), best_path)
            logger.info(f"🏆 New best loss: {loss_g:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"⏸️ Early stopping after {epoch} epochs")
            break
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4 TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"✅ Best checkpoint: {checkpoint_dir / 'best.pt'}")
    logger.info(f"✅ Best loss: {best_loss_g:.6f}")
    logger.info("")
    logger.info("Next step: Evaluation and ensemble")
    logger.info("")
    
    metadata = {
        'phase': 'Phase 4',
        'strategy': 'Adversarial Fine-tuning',
        'phase3_checkpoint': str(phase3_checkpoint),
        'num_files': num_files,
        'learning_rate_g': lr_g,
        'learning_rate_d': lr_d,
        'epochs': epoch,
        'best_loss': best_loss_g,
        'checkpoint_dir': str(checkpoint_dir),
    }
    
    with open(checkpoint_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
