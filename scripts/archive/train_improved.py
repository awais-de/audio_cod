"""
Improved Training Script with Quality Monitoring
Addresses quality issues by:
1. Larger model capacity
2. Better loss functions
3. Quality-based checkpointing
4. Early stopping based on PESQ/STOI
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime
import logging
from src.model import NeuralAudioCodec
import soundfile as sf
from tqdm import tqdm

# Quality metrics (optional)
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except:
    STOI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiScaleSpectralLoss(nn.Module):
    """Enhanced multi-scale spectral loss with phase awareness"""
    def __init__(self, fft_sizes=[512, 1024, 2048], include_phase=True):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = [fs // 4 for fs in fft_sizes]
        self.win_sizes = fft_sizes
        self.include_phase = include_phase
    
    def stft(self, x, fft_size, hop_size, win_size):
        x = x.squeeze(1)
        window = torch.hann_window(win_size).to(x.device)
        spec = torch.stft(
            x, n_fft=fft_size, hop_length=hop_size, win_length=win_size,
            window=window, return_complex=True, center=True, pad_mode='reflect'
        )
        return spec
    
    def forward(self, pred, target):
        total_loss = 0.0
        valid_scales = 0
        
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            if pred.shape[-1] < fft_size or target.shape[-1] < fft_size:
                continue
                
            try:
                pred_spec = self.stft(pred, fft_size, hop_size, win_size)
                target_spec = self.stft(target, fft_size, hop_size, win_size)
                
                # Magnitude loss
                pred_mag = torch.abs(pred_spec) + 1e-7
                target_mag = torch.abs(target_spec) + 1e-7
                
                pred_log_mag = torch.log(pred_mag).clamp(min=-15, max=15)
                target_log_mag = torch.log(target_mag).clamp(min=-15, max=15)
                
                mag_loss = F.l1_loss(pred_log_mag, target_log_mag)
                
                # Phase loss (if enabled)
                phase_loss = 0.0
                if self.include_phase:
                    pred_phase = torch.angle(pred_spec)
                    target_phase = torch.angle(target_spec)
                    # Cosine distance for phase
                    phase_loss = 1.0 - torch.cos(pred_phase - target_phase).mean()
                
                if not (torch.isnan(mag_loss) or torch.isinf(mag_loss)):
                    total_loss += mag_loss + 0.1 * phase_loss
                    valid_scales += 1
                    
            except Exception as e:
                continue
        
        if valid_scales == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return total_loss / valid_scales


class ImprovedAudioLoss(nn.Module):
    """Enhanced loss with better perceptual weighting"""
    def __init__(self, l1_weight=1.0, spectral_weight=2.0, phase_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight
        self.phase_weight = phase_weight
        self.spectral_loss = MultiScaleSpectralLoss(include_phase=(phase_weight > 0))
    
    def forward(self, pred, target):
        # Time domain L1
        l1_loss = F.l1_loss(pred, target)
        
        # Spectral loss
        spectral = self.spectral_loss(pred, target)
        
        # Combined
        total = self.l1_weight * l1_loss + self.spectral_weight * spectral
        
        return total, {'l1': l1_loss.item(), 'spectral': spectral.item()}


class AudioDataset(Dataset):
    def __init__(self, audio_dir, segment_length=8000, sample_rate=16000, extensions=['.wav', '.flac']):
        self.audio_dir = Path(audio_dir)
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        # Find all audio files
        self.audio_files = []
        for ext in extensions:
            self.audio_files.extend(list(self.audio_dir.rglob(f'*{ext}')))
        
        logger.info(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            audio, sr = sf.read(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * self.sample_rate / sr)),
                    np.arange(len(audio)),
                    audio
                )
            
            # Convert to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Random segment
            if len(audio) > self.segment_length:
                start = np.random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start + self.segment_length]
            else:
                # Pad if too short
                audio = np.pad(audio, (0, self.segment_length - len(audio)))
            
            # Normalize
            audio = audio / (np.abs(audio).max() + 1e-7)
            
            return torch.FloatTensor(audio).unsqueeze(0)
            
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, self.segment_length)


def evaluate_quality(model, val_loader, device, num_samples=5):
    """Evaluate PESQ/STOI on validation set"""
    if not (PESQ_AVAILABLE and STOI_AVAILABLE):
        return None
    
    model.eval()
    pesq_scores = []
    stoi_scores = []
    
    with torch.no_grad():
        for i, audio in enumerate(val_loader):
            if i >= num_samples:
                break
            
            audio = audio.to(device)
            reconstructed = model(audio)
            
            # Convert to numpy
            orig = audio.squeeze().cpu().numpy()
            recon = reconstructed.squeeze().cpu().numpy()
            
            # Ensure same length
            min_len = min(len(orig), len(recon))
            orig = orig[:min_len]
            recon = recon[:min_len]
            
            try:
                pesq_score = pesq(16000, orig, recon, 'wb')
                stoi_score = stoi(orig, recon, 16000, extended=False)
                pesq_scores.append(pesq_score)
                stoi_scores.append(stoi_score)
            except:
                continue
    
    if not pesq_scores:
        return None
    
    return {
        'pesq': np.mean(pesq_scores),
        'stoi': np.mean(stoi_scores)
    }


def train_improved():
    """Main training loop with quality monitoring"""
    
    # Load config
    config_path = 'config/training_improved.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 80)
    logger.info("IMPROVED TRAINING - QUALITY-FOCUSED")
    logger.info("=" * 80)
    logger.info(f"Model: d_model={config['model']['d_model']}, n_layers={config['model']['n_layers']}")
    logger.info(f"Training: {config['training']['epochs']} epochs, batch_size={config['training']['batch_size']}")
    logger.info("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    ckpt_dir = Path(config['checkpoint']['dir'])
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    # Model
    model = NeuralAudioCodec(
        sample_rate=config['model']['sample_rate'],
        hop_length=config['model']['hop_length'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        window_size=config['model']['window_size'],
        dropout=config['model']['dropout']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,} (~{num_params/1e6:.1f}M)")
    
    # Check if we should resume from previous checkpoint
    resume_from = 'checkpoints/best_model.pt'
    if Path(resume_from).exists():
        logger.info(f"🔄 Resuming from {resume_from}")
        try:
            checkpoint = torch.load(resume_from, map_location=device)
            # Only load weights, not optimizer state (fresh start with new config)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info("✅ Loaded previous model weights")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")
    
    # Loss
    criterion = ImprovedAudioLoss(
        l1_weight=config['training']['l1_weight'],
        spectral_weight=config['training']['spectral_weight'],
        phase_weight=config['training'].get('phase_weight', 0.5)
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler with warmup
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'] - warmup_epochs
    )
    
    # Data
    train_dataset = AudioDataset(
        config['data']['train_dir'],
        segment_length=config['training']['segment_length'],
        sample_rate=config['model']['sample_rate'],
        extensions=config['data']['extensions']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        train_dataset,  # Using same for now
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # Training loop
    best_pesq = 0.0
    no_improve_count = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_spectral = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]}')
        
        for batch_idx, audio in enumerate(pbar):
            audio = audio.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            reconstructed = model(audio)
            
            # Loss
            loss, loss_dict = criterion(reconstructed, audio)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )
            
            optimizer.step()
            
            # Stats
            epoch_loss += loss.item()
            epoch_l1 += loss_dict['l1']
            epoch_spectral += loss_dict['spectral']
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'l1': f'{loss_dict["l1"]:.4f}',
                'spec': f'{loss_dict["spectral"]:.4f}'
            })
        
        # Epoch stats
        avg_loss = epoch_loss / len(train_loader)
        avg_l1 = epoch_l1 / len(train_loader)
        avg_spectral = epoch_spectral / len(train_loader)
        
        logger.info(f'Epoch {epoch}: Loss={avg_loss:.4f}, L1={avg_l1:.4f}, Spectral={avg_spectral:.4f}')
        
        # Learning rate schedule
        if epoch > warmup_epochs:
            scheduler.step()
        
        # Quality evaluation
        if epoch % config['training'].get('eval_every', 10) == 0:
            logger.info("Evaluating quality metrics...")
            quality = evaluate_quality(model, val_loader, device)
            
            if quality:
                logger.info(f"  PESQ: {quality['pesq']:.3f}, STOI: {quality['stoi']:.3f}")
                
                # Save if best
                if quality['pesq'] > best_pesq:
                    best_pesq = quality['pesq']
                    no_improve_count = 0
                    
                    save_path = ckpt_dir / 'best_model_improved.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'pesq': quality['pesq'],
                        'stoi': quality['stoi'],
                        **config['model']
                    }, save_path)
                    logger.info(f"  ✅ New best model saved! PESQ: {best_pesq:.3f}")
                else:
                    no_improve_count += 1
                    logger.info(f"  No improvement ({no_improve_count}/{config['training'].get('early_stopping_patience', 20)})")
        
        # Regular checkpoint
        if epoch % config['checkpoint']['save_every'] == 0:
            save_path = ckpt_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                **config['model']
            }, save_path)
            logger.info(f"  Checkpoint saved: {save_path}")
        
        # Early stopping
        patience = config['training'].get('early_stopping_patience', 20)
        if no_improve_count >= patience:
            logger.info(f"Early stopping after {no_improve_count} epochs without improvement")
            break
    
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best PESQ: {best_pesq:.3f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    train_improved()
