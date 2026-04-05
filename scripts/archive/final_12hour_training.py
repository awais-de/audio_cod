#!/usr/bin/env python3
"""
Final 12-hour push: 51M model, 100 epochs
Target: Maximum PESQ toward 3.5, maintain STOI >= 0.9
Save checkpoints every 10 epochs
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import NeuralAudioCodec

TMPDIR_PATH = Path("/mnt/Data/muaw1874/tmp")
TMPDIR_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(TMPDIR_PATH))

def safe_stft_loss(pred, target, fft_size, hop_size):
    min_len = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :min_len]
    target = target[..., :min_len]
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    window = torch.hann_window(fft_size).to(pred.device)
    pred_spec = torch.stft(pred, n_fft=fft_size, hop_length=hop_size, win_length=fft_size,
                           window=window, return_complex=True, center=True)
    target_spec = torch.stft(target, n_fft=fft_size, hop_length=hop_size, win_length=fft_size,
                             window=window, return_complex=True, center=True)
    
    pred_mag = torch.abs(pred_spec)
    target_mag = torch.abs(target_spec)
    
    # L1 + log-magnitude + spectral convergence
    mag_loss = F.l1_loss(pred_mag, target_mag)
    log_loss = F.l1_loss(torch.log(pred_mag + 1e-5), torch.log(target_mag + 1e-5))
    
    # Spectral convergence (normalized Frobenius norm)
    sc_loss = torch.norm(target_mag - pred_mag) / torch.norm(target_mag)
    
    return mag_loss + log_loss + 0.5 * sc_loss

class AdvancedPESQLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_fft = [512, 2048, 4096]  # Multi-scale for better perceptual quality
        self.stft_hop = [128, 512, 1024]

    def forward(self, pred, target):
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        stft_total = 0.0
        for fft_size, hop_size in zip(self.stft_fft, self.stft_hop):
            if pred.shape[-1] >= fft_size:
                stft_total = stft_total + safe_stft_loss(pred, target, fft_size, hop_size)
        
        time_loss = F.l1_loss(pred, target)
        stoi_term = self.stoi_term(pred, target)
        
        # Heavily weight STFT for PESQ
        return 3.0 * stft_total + 0.2 * stoi_term + 0.3 * time_loss

    def stoi_term(self, pred, target):
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        fft_sizes = [512, 1024]
        total = 0.0
        for fft_size in fft_sizes:
            hop = fft_size // 4
            window = torch.hann_window(fft_size).to(pred.device)
            pred_spec = torch.stft(pred, n_fft=fft_size, hop_length=hop, win_length=fft_size,
                                   window=window, return_complex=True, center=True)
            target_spec = torch.stft(target, n_fft=fft_size, hop_length=hop, win_length=fft_size,
                                     window=window, return_complex=True, center=True)
            pred_pow = torch.abs(pred_spec) ** 2
            target_pow = torch.abs(target_spec) ** 2
            total = total + F.l1_loss(pred_pow, target_pow)
        return total / len(fft_sizes)

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
    model.eval()
    pesq_scores = []
    stoi_scores = []
    with torch.no_grad():
        for i, audio in enumerate(val_loader):
            if i >= 5:
                break
            audio = audio.to(device)
            recon = model(audio)
            min_len = min(audio.shape[-1], recon.shape[-1])
            audio = audio[..., :min_len]
            recon = recon[..., :min_len]
            orig = audio.squeeze().cpu().numpy()
            rec = recon.squeeze().cpu().numpy()
            try:
                pesq_scores.append(pesq(16000, orig, rec, 'wb'))
            except Exception:
                pass
            try:
                stoi_scores.append(stoi(orig, rec, 16000, extended=False))
            except Exception:
                pass
    model.train()
    return {
        'pesq': float(np.mean(pesq_scores)) if pesq_scores else 0.0,
        'stoi': float(np.mean(stoi_scores)) if stoi_scores else 0.0,
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    print("=" * 80)
    print("FINAL 12-HOUR TRAINING RUN")
    print("Target: PESQ ≥ 3.5, STOI ≥ 0.9")
    print("=" * 80 + "\n")

    # Create 51M model
    print("Creating model (d_model=512, n_layers=8, ~51M params)...")
    model = NeuralAudioCodec(d_model=512, n_layers=8, n_heads=8, window_size=512).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model: {param_count/1e6:.1f}M parameters\n")

    # Warm-start from best existing checkpoint
    resume_ckpt = Path('checkpoints_emergency/best_pesq_finetune.pt')
    if resume_ckpt.exists():
        print(f"⚡ Warm-starting from {resume_ckpt}...")
        try:
            ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"✅ Loaded {len(pretrained_dict)}/{len(model_dict)} compatible weights\n")
        except Exception as e:
            print(f"⚠️  Warm-start failed: {e}, training from scratch\n")

    criterion = AdvancedPESQLoss().to(device)
    
    # Optimizer with warmup schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999))
    
    # Cosine annealing with warmup
    num_epochs = 35  # Realistic for 12 hours (20.5 min/epoch)
    warmup_epochs = 3
    
    def get_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

    dataset = AudioDataset('/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100', segment_length=16000)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    best_pesq = 0.0
    ckpt_dir = Path('checkpoints_final')
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt_path = ckpt_dir / 'best_final.pt'

    accumulation_steps = 2

    print(f"Training configuration:")
    print(f"  Epochs: {num_epochs} (realistic for 12-hour window)")
    print(f"  Batch size: 8 (effective 16 with gradient accumulation)")
    print(f"  Segments: 1s (16000 samples)")
    print(f"  STFT sizes: [512, 2048, 4096]")
    print(f"  Learning rate: 5e-5 → 1e-5 (warmup + cosine)")
    print(f"  Checkpoints: Every 5 epochs + best")
    print(f"  ETA: ~{num_epochs * 20.5 / 60:.1f} hours (~20.5 min/epoch)\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        optimizer.zero_grad()
        
        for batch_idx, audio in enumerate(pbar):
            audio = audio.to(device)
            recon = model(audio)
            loss = criterion(recon, audio) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        print(f'Epoch {epoch}: Loss={avg_loss:.4f}, LR={optimizer.param_groups[0]["lr"]:.2e}')

        # Eval and save every 5 epochs
        if epoch % 5 == 0:
            quality = quick_evaluate(model, val_loader, device)
            print(f'  Quality: PESQ={quality["pesq"]:.3f}, STOI={quality["stoi"]:.3f}')
            
            # Save checkpoint every 5 epochs
            ckpt_path = ckpt_dir / f'final_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'pesq': quality['pesq'],
                'stoi': quality['stoi'],
                'd_model': 512,
                'n_layers': 8,
                'n_heads': 8,
                'loss': avg_loss
            }, ckpt_path)
            print(f'  ✅ Saved checkpoint: {ckpt_path.name}')
            
            # Update best if improved
            if quality['pesq'] > best_pesq:
                best_pesq = quality['pesq']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'pesq': quality['pesq'],
                    'stoi': quality['stoi'],
                    'd_model': 512,
                    'n_layers': 8,
                    'n_heads': 8,
                    'loss': avg_loss
                }, best_ckpt_path)
                print(f'  🎯 NEW BEST! PESQ={quality["pesq"]:.3f}, STOI={quality["stoi"]:.3f}')

    print(f'\n{"=" * 80}')
    print(f'✅ Training complete')
    print(f'Best PESQ achieved: {best_pesq:.3f}')
    print(f'Checkpoints saved in: {ckpt_dir}')
    print(f'{"=" * 80}')

if __name__ == '__main__':
    main()
