"""
Quick Fine-Tuning Script (2-Day Solution)
Fine-tunes existing model with better loss function for rapid improvement
Target: Get PESQ/STOI to acceptable levels within 24 hours
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
from src.model import NeuralAudioCodec
import soundfile as sf
from tqdm import tqdm

try:
    from pesq import pesq
    from pystoi import stoi
    METRICS_AVAILABLE = True
except:
    METRICS_AVAILABLE = False
    print("⚠️  Install pesq and pystoi for quality monitoring")


class PerceptualLoss(nn.Module):
    """Perceptual loss emphasizing speech quality"""
    def __init__(self):
        super().__init__()
        self.fft_sizes = [256, 512, 1024, 2048]
        self.hop_sizes = [64, 128, 256, 512]
        
    def stft_loss(self, pred, target, fft_size, hop_size):
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        
        # Handle length mismatch by trimming to minimum length
        min_length = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_length]
        target = target[..., :min_length]
        
        window = torch.hann_window(fft_size).to(pred.device)
        
        pred_spec = torch.stft(pred, n_fft=fft_size, hop_length=hop_size, 
                               win_length=fft_size, window=window, 
                               return_complex=True, center=True)
        target_spec = torch.stft(target, n_fft=fft_size, hop_length=hop_size,
                                 win_length=fft_size, window=window,
                                 return_complex=True, center=True)
        
        # Magnitude loss
        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec)
        mag_loss = F.l1_loss(pred_mag, target_mag)
        
        # Log magnitude for perceptual weighting
        pred_log = torch.log(pred_mag + 1e-5)
        target_log = torch.log(target_mag + 1e-5)
        log_loss = F.l1_loss(pred_log, target_log)
        
        return mag_loss + log_loss
    
    def forward(self, pred, target):
        # Handle length mismatch
        min_length = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_length]
        target = target[..., :min_length]
        
        total = 0.0
        for fft_size, hop_size in zip(self.fft_sizes, self.hop_sizes):
            if pred.shape[-1] >= fft_size:
                total += self.stft_loss(pred, target, fft_size, hop_size)
        
        # Also add time-domain loss
        time_loss = F.l1_loss(pred, target)
        
        return total + 2.0 * time_loss  # Emphasize time-domain accuracy


class AudioDataset(Dataset):
    def __init__(self, audio_dir, segment_length=16000):
        self.audio_dir = Path(audio_dir)
        self.segment_length = segment_length
        self.audio_files = list(self.audio_dir.rglob('*.flac'))[:5000]  # Limit for speed
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        try:
            audio, sr = sf.read(self.audio_files[idx])
            
            if sr != 16000:
                audio = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * 16000 / sr)),
                    np.arange(len(audio)), audio
                )
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Random segment
            if len(audio) > self.segment_length:
                start = np.random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start + self.segment_length]
            else:
                audio = np.pad(audio, (0, self.segment_length - len(audio)))
            
            # Normalize
            if np.abs(audio).max() > 1e-7:
                audio = audio / np.abs(audio).max()
            
            return torch.FloatTensor(audio).unsqueeze(0)
        except:
            return torch.zeros(1, self.segment_length)


def quick_evaluate(model, val_loader, device):
    """Quick quality check"""
    if not METRICS_AVAILABLE:
        return {'pesq': 0, 'stoi': 0}
    
    model.eval()
    pesq_scores, stoi_scores = [], []
    
    with torch.no_grad():
        for i, audio in enumerate(val_loader):
            if i >= 10:  # Just 10 samples for speed
                break
            
            audio = audio.to(device)
            recon = model(audio)
            
            orig = audio.squeeze().cpu().numpy()
            rec = recon.squeeze().cpu().numpy()
            
            min_len = min(len(orig), len(rec))
            try:
                pesq_scores.append(pesq(16000, orig[:min_len], rec[:min_len], 'wb'))
                stoi_scores.append(stoi(orig[:min_len], rec[:min_len], 16000))
            except:
                pass
    
    return {
        'pesq': np.mean(pesq_scores) if pesq_scores else 0,
        'stoi': np.mean(stoi_scores) if stoi_scores else 0
    }


def main():
    print("=" * 80)
    print("QUICK FINE-TUNING - 24 HOUR SOLUTION")
    print("=" * 80)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load existing model
    print("Loading model...")
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    
    model = NeuralAudioCodec(
        d_model=256, n_layers=4, n_heads=8,
        window_size=256, dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Perceptual loss
    criterion = PerceptualLoss()
    
    # Lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Data - reduced batch size and segment length for memory efficiency
    print("Loading data...")
    dataset = AudioDataset('/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100', segment_length=8000)  # 0.5s instead of 1s
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)  # Batch 2 instead of 16
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Dataset: {len(dataset)} files")
    print()
    
    # Initial evaluation
    print("Initial quality:")
    initial_quality = quick_evaluate(model, val_loader, device)
    print(f"  PESQ: {initial_quality['pesq']:.3f}, STOI: {initial_quality['stoi']:.3f}")
    print()
    
    # Fine-tune
    best_pesq = initial_quality['pesq']
    Path('checkpoints_finetuned').mkdir(exist_ok=True)
    
    num_epochs = 50  # Fast fine-tuning
    accumulation_steps = 8  # Accumulate gradients over 8 steps (effective batch size = 2*8 = 16)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, audio in enumerate(pbar):
            audio = audio.to(device)
            
            recon = model(audio)
            loss = criterion(recon, audio)
            loss = loss / accumulation_steps  # Scale loss for accumulation
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps  # Unscale for logging
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
        # Final update if there are remaining gradients
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        print(f'Epoch {epoch}: Loss={avg_loss:.4f}')
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            quality = quick_evaluate(model, val_loader, device)
            print(f'  Quality: PESQ={quality["pesq"]:.3f}, STOI={quality["stoi"]:.3f}')
            
            if quality['pesq'] > best_pesq:
                best_pesq = quality['pesq']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'pesq': quality['pesq'],
                    'stoi': quality['stoi'],
                    'd_model': 256, 'n_layers': 4, 'n_heads': 8, 'window_size': 256
                }, 'checkpoints_finetuned/best_model_finetuned.pt')
                print(f'  ✅ New best! Saved checkpoint')
        
        # Save every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'd_model': 256, 'n_layers': 4, 'n_heads': 8, 'window_size': 256
            }, f'checkpoints_finetuned/checkpoint_epoch_{epoch}.pt')
    
    print()
    print("=" * 80)
    print("FINE-TUNING COMPLETE!")
    print(f"Best PESQ: {best_pesq:.3f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
