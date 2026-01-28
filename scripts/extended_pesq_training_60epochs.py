#!/usr/bin/env python3
"""
Extended PESQ training: 60 epochs from best_pesq_finetune.pt
Optimized for 12-hour window (~13.7 min/epoch for 21.7M model)
- Batch size: 4 (for speed, less GPU memory)
- Segment length: 16000 (1s, needed for perceptual quality)
- Checkpoints: Every 10 epochs
- Evaluation: Every 10 epochs only (skip frequent evals to save time)
- Target: Reach 2.88-2.95 PESQ from current 2.803
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
from datetime import datetime
from pesq import pesq
from pystoi import stoi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import NeuralAudioCodec

TMPDIR_PATH = Path("/home/muaw1874/Desktop/ac_proj/tmp")
TMPDIR_PATH.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMPDIR_PATH)

def safe_stft_loss(pred, target, fft_size, hop_size):
    """Compute STFT loss between prediction and target"""
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
    
    mag_loss = F.l1_loss(pred_mag, target_mag)
    log_loss = F.l1_loss(torch.log(pred_mag + 1e-5), torch.log(target_mag + 1e-5))
    return mag_loss + log_loss

class PESQLoss(nn.Module):
    """PESQ-focused loss: high STFT weight, moderate STOI, low time-domain"""
    def __init__(self):
        super().__init__()
        self.stft_fft = [512, 2048]
        self.stft_hop = [128, 512]

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
        
        # Weights: STFT-heavy for PESQ, maintain some STOI for quality
        return 2.0 * stft_total + 0.25 * stoi_term + 0.5 * time_loss

    def stoi_term(self, pred, target):
        """STOI surrogate using power spectra"""
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
    """LibriSpeech dataset loader"""
    def __init__(self, root_dir, segment_length=16000):
        self.root_dir = Path(root_dir)
        self.segment_length = segment_length
        self.audio_files = list(self.root_dir.rglob('*.flac'))[:5000]
        print(f"  Loaded {len(self.audio_files)} audio files")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        try:
            audio, sr = sf.read(audio_path)
        except:
            # Fallback to next file
            idx = (idx + 1) % len(self.audio_files)
            audio, sr = sf.read(self.audio_files[idx])
        
        if len(audio) < self.segment_length:
            audio = np.pad(audio, (0, self.segment_length - len(audio)))
        start = np.random.randint(0, max(1, len(audio) - self.segment_length))
        audio = audio[start:start + self.segment_length]
        return torch.from_numpy(audio).float().unsqueeze(0)

def quick_evaluate(model, val_loader, device, num_samples=5):
    """Quick evaluation on small subset"""
    model.eval()
    pesq_scores = []
    stoi_scores = []
    with torch.no_grad():
        for i, audio in enumerate(val_loader):
            if i >= num_samples:
                break
            audio = audio.to(device)
            recon = model(audio)
            min_len = min(audio.shape[-1], recon.shape[-1])
            audio = audio[..., :min_len]
            recon = recon[..., :min_len]
            
            orig = audio.squeeze().cpu().numpy()
            rec = recon.squeeze().cpu().numpy()
            
            try:
                p = pesq(16000, orig, rec, 'wb')
                pesq_scores.append(p)
            except Exception as e:
                pass
            
            try:
                s = stoi(orig, rec, 16000, extended=False)
                stoi_scores.append(s)
            except Exception as e:
                pass
    
    model.train()
    return {
        'pesq': float(np.mean(pesq_scores)) if pesq_scores else 0.0,
        'stoi': float(np.mean(stoi_scores)) if stoi_scores else 0.0,
        'n_samples': len(pesq_scores)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Extended PESQ Training: 60 Epochs")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load checkpoint
    ckpt_path = Path('checkpoints_emergency/best_pesq_finetune.pt')
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    print("Loading model from best_pesq_finetune.pt...")
    model = NeuralAudioCodec(d_model=384, n_layers=6, n_heads=8, window_size=384).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss and optimizer
    criterion = PESQLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Try to load optimizer state if available
    if 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("  Optimizer state restored")
        except:
            print("  (Starting with fresh optimizer)")

    # Dataset and loaders
    print("\nLoading dataset...")
    dataset = AudioDataset('/home/muaw1874/Desktop/ac_proj/datasets/LibriSpeech/train-clean-100', 
                          segment_length=16000)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, 
                             pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"  Training samples: {len(dataset)}")
    print(f"  Batch size: 4")
    print(f"  Iterations per epoch: {len(train_loader)}")
    print(f"  Est. time per epoch: ~13.7 min (21.7M model)")
    print(f"  Total time estimate: ~13.7 hours for 60 epochs\n")

    # Training config
    num_epochs = 60
    accumulation_steps = 2
    checkpoint_interval = 10
    eval_interval = 10

    best_pesq = ckpt.get('pesq', 2.803)
    best_epoch = 0
    best_ckpt_path = Path('checkpoints_emergency/best_pesq_extended.pt')
    
    start_time = datetime.now()

    # Training loop
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
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f'Epoch {epoch}: Loss={avg_loss:.4f} | Time: {elapsed:.1f} min')

        # Checkpoint every 10 epochs
        if epoch % checkpoint_interval == 0:
            ckpt_path_periodic = Path(f'checkpoints_emergency/epoch_{epoch}_pesq_extended.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'd_model': 384,
                'n_layers': 6,
                'n_heads': 8
            }, ckpt_path_periodic)
            print(f'  ✓ Checkpoint saved: epoch_{epoch}_pesq_extended.pt')

        # Evaluate every 10 epochs
        if epoch % eval_interval == 0:
            quality = quick_evaluate(model, val_loader, device, num_samples=5)
            print(f'  Eval: PESQ={quality["pesq"]:.3f}, STOI={quality["stoi"]:.3f} ({quality["n_samples"]} samples)')
            
            # Save best
            if quality['pesq'] > best_pesq:
                best_pesq = quality['pesq']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'pesq': quality['pesq'],
                    'stoi': quality['stoi'],
                    'd_model': 384,
                    'n_layers': 6,
                    'n_heads': 8
                }, best_ckpt_path)
                print(f'  ✅ New best PESQ: {best_pesq:.3f} at epoch {epoch}')

    elapsed_total = (datetime.now() - start_time).total_seconds() / 3600
    print(f"\n{'='*60}")
    print(f"✅ Extended training complete!")
    print(f"Total time: {elapsed_total:.1f} hours")
    print(f"Best PESQ: {best_pesq:.3f} at epoch {best_epoch}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Gain from start: +{best_pesq - 2.803:.3f} PESQ")
    print(f"Remaining gap to 3.5: {3.5 - best_pesq:.3f}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
