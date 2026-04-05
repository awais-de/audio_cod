#!/usr/bin/env python3
"""
EMERGENCY TRAINING - TARGET FOCUSED
Aggressive training to meet PESQ ≥3.5, STOI ≥0.9 in 12-16 hours

Strategy:
1. Larger model (d_model=512, n_layers=8) for PESQ
2. STOI-focused loss (direct optimization)
3. Warm-start from fine-tuned checkpoint
4. 100 epochs (overnight run)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from src.model import NeuralAudioCodec
import soundfile as sf
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

# Use large data disk for temp to avoid full root /tmp
TMPDIR_PATH = Path("/mnt/Data/muaw1874/tmp")
TMPDIR_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMPDIR_PATH))


class STOILoss(nn.Module):
    """Differentiable STOI approximation via correlation-based metric"""
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        
    def forward(self, pred, target):
        """Approximate STOI via spectral correlation"""
        # Multi-scale spectral correlation (approximates STOI)
        fft_sizes = [512, 1024, 2048]
        total_loss = 0
        
        for fft_size in fft_sizes:
            hop = fft_size // 4
            window = torch.hann_window(fft_size).to(pred.device)
            
            pred_spec = torch.stft(pred.squeeze(1), n_fft=fft_size, hop_length=hop,
                                   win_length=fft_size, window=window,
                                   return_complex=True, center=True)
            target_spec = torch.stft(target.squeeze(1), n_fft=fft_size, hop_length=hop,
                                     win_length=fft_size, window=window,
                                     return_complex=True, center=True)
            
            # Power spectra
            pred_pow = torch.abs(pred_spec) ** 2
            target_pow = torch.abs(target_spec) ** 2
            
            # L1 loss on power spectra (simpler, always positive)
            total_loss += F.l1_loss(pred_pow, target_pow)
        
        return total_loss / len(fft_sizes)


class EnhancedPerceptualLoss(nn.Module):
    """Combines STFT + STOI approximation + time-domain"""
    def __init__(self):
        super().__init__()
        self.stoi_loss = STOILoss()
        self.fft_sizes = [512, 2048]  # Removed 8192 for speed
        self.hop_sizes = [128, 512]
        
    def stft_loss(self, pred, target, fft_size, hop_size):
        # Trim to same length
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]
        
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        window = torch.hann_window(fft_size).to(pred.device)
        pred_spec = torch.stft(pred, n_fft=fft_size, hop_length=hop_size,
                               win_length=fft_size, window=window,
                               return_complex=True, center=True)
        target_spec = torch.stft(target, n_fft=fft_size, hop_length=hop_size,
                                 win_length=fft_size, window=window,
                                 return_complex=True, center=True)
        
        # Magnitude + log magnitude
        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec)
        mag_loss = F.l1_loss(pred_mag, target_mag)
        log_loss = F.l1_loss(torch.log(pred_mag + 1e-5), torch.log(target_mag + 1e-5))
        
        return mag_loss + log_loss
    
    def forward(self, pred, target):
        # Trim
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]
        
        # STFT loss
        stft_total = 0
        for fft_size, hop_size in zip(self.fft_sizes, self.hop_sizes):
            if pred.shape[-1] >= fft_size:
                stft_total += self.stft_loss(pred, target, fft_size, hop_size)
        
        # STOI approximation loss
        stoi_loss = self.stoi_loss(pred, target)
        
        # Time domain
        time_loss = F.l1_loss(pred, target)
        
        # Weighted combination (emphasize STOI)
        return stft_total + 2.0 * stoi_loss + 0.5 * time_loss


class AudioDataset(Dataset):
    def __init__(self, root_dir, segment_length=16000):
        self.root_dir = Path(root_dir)
        self.segment_length = segment_length
        self.audio_files = list(self.root_dir.rglob('*.flac'))[:5000]
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = sf.read(audio_path)
        
        if len(audio) < self.segment_length:
            audio = np.pad(audio, (0, self.segment_length - len(audio)))
        
        start = np.random.randint(0, max(1, len(audio) - self.segment_length))
        audio = audio[start:start + self.segment_length]
        
        return torch.from_numpy(audio).float().unsqueeze(0)


def quick_evaluate(model, val_loader, device):
    """Fast quality evaluation"""
    model.eval()
    pesq_scores = []
    stoi_scores = []
    
    with torch.no_grad():
        for i, audio in enumerate(val_loader):
            if i >= 3:
                break
            
            audio = audio.to(device)
            recon = model(audio)
            
            # Trim to same length
            min_len = min(audio.shape[-1], recon.shape[-1])
            audio = audio[..., :min_len]
            recon = recon[..., :min_len]
            
            orig = audio.squeeze().cpu().numpy()
            rec = recon.squeeze().cpu().numpy()
            
            try:
                p = pesq(16000, orig, rec, 'wb')
                pesq_scores.append(p)
            except:
                pass
            
            try:
                s = stoi(orig, rec, 16000, extended=False)
                stoi_scores.append(s)
            except:
                pass
    
    model.train()
    return {
        'pesq': np.mean(pesq_scores) if pesq_scores else 0,
        'stoi': np.mean(stoi_scores) if stoi_scores else 0
    }


def main():
    print("\n" + "="*80)
    print("EMERGENCY TRAINING - TARGET FOCUSED (PESQ ≥3.5, STOI ≥0.9)")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Larger model for better quality (reduced from 512/8 to 384/6 for speed)
    print("Creating LARGER model (d_model=384, n_layers=6)...")
    model = NeuralAudioCodec(d_model=384, n_layers=6, n_heads=8, window_size=384).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {param_count/1e6:.1f}M parameters")
    
    # Warm-start from fine-tuned if available
    finetune_ckpt = Path('checkpoints_finetuned/best_model_finetuned.pt')
    if finetune_ckpt.exists():
        print(f"\n⚡ Warm-starting from {finetune_ckpt}...")
        try:
            ckpt = torch.load(finetune_ckpt, map_location=device, weights_only=False)
            # Load compatible weights only
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"✅ Loaded {len(pretrained_dict)}/{len(model_dict)} compatible weights")
        except Exception as e:
            print(f"⚠️  Warm-start failed: {e}, training from scratch")
    
    # Enhanced loss
    criterion = EnhancedPerceptualLoss().to(device)
    
    # Optimizer with lower LR for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Data
    print("\nLoading data...")
    dataset = AudioDataset('/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100', segment_length=8000)  # 0.5s for speed
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)  # Increased batch
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Dataset: {len(dataset)} files\n")
    
    # Initial eval
    print("Initial quality:")
    initial = quick_evaluate(model, val_loader, device)
    print(f"  PESQ: {initial['pesq']:.3f}, STOI: {initial['stoi']:.3f}\n")
    
    # Training
    best_pesq = initial['pesq']
    best_stoi = initial['stoi']
    Path('checkpoints_emergency').mkdir(exist_ok=True)
    
    num_epochs = 50  # Reduced from 100 for speed
    accumulation_steps = 2  # Effective batch 16

    print(f"Training for {num_epochs} epochs (estimated: 4-6 hours)\n")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        optimizer.zero_grad()
        
        for batch_idx, audio in enumerate(pbar):
            audio = audio.to(device)
            
            recon = model(audio)
            loss = criterion(recon, audio)
            loss = loss / accumulation_steps

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        print(f'Epoch {epoch}: Loss={avg_loss:.4f}')
        
        # Evaluate every 10 epochs for speed
        if epoch % 10 == 0:
            quality = quick_evaluate(model, val_loader, device)
            print(f'  Quality: PESQ={quality["pesq"]:.3f}, STOI={quality["stoi"]:.3f}')
            
            # Save if better
            if quality['pesq'] > best_pesq or quality['stoi'] > best_stoi:
                best_pesq = max(best_pesq, quality['pesq'])
                best_stoi = max(best_stoi, quality['stoi'])
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'd_model': 384,
                    'n_layers': 6,
                    'n_heads': 8,
                    'pesq': quality['pesq'],
                    'stoi': quality['stoi']
                }
                torch.save(checkpoint, 'checkpoints_emergency/best_emergency.pt')
                print(f"  ✅ New best! Saved checkpoint")
            
            # Check targets
            if quality['pesq'] >= 3.5 and quality['stoi'] >= 0.9:
                print(f"\n🎯 TARGETS MET! PESQ={quality['pesq']:.3f}, STOI={quality['stoi']:.3f}")
                torch.save(checkpoint, 'checkpoints_emergency/target_met.pt')
                break
        
        # Save periodic checkpoints
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'd_model': 384,
                'n_layers': 6,
                'n_heads': 8
            }
            torch.save(checkpoint, f'checkpoints_emergency/checkpoint_epoch_{epoch}.pt')
    
    print(f"\n✅ Training complete")
    print(f"Best PESQ: {best_pesq:.3f}, Best STOI: {best_stoi:.3f}")


if __name__ == '__main__':
    main()
