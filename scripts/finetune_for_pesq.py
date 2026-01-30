#!/usr/bin/env python3
"""
Simple Fine-tuning Script to Improve PESQ
Targets: PESQ 3.5+ (from baseline 2.90), STOI >0.9, Latency <20ms
Strategy: Extended training with PESQ-focused loss
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class AudioDataset:
    def __init__(self, dataset_dir, sr=16000, segment_length=16000, n_files=None):
        self.sr = sr
        self.segment_length = segment_length
        self.files = []
        
        dataset_path = Path(dataset_dir)
        if dataset_path.exists():
            # Recursive search for FLAC files
            for wav_file in sorted(dataset_path.rglob("*.flac")):
                self.files.append(str(wav_file))
        
        if n_files:
            self.files = self.files[:n_files]
        
        print(f"✓ Found {len(self.files)} audio files")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            audio, sr = sf.read(self.files[idx])
            if sr != self.sr:
                raise ValueError(f"Sample rate mismatch: {sr} vs {self.sr}")
            
            # Random segment
            if len(audio) > self.segment_length:
                start = np.random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start+self.segment_length]
            else:
                # Pad if shorter
                audio = np.pad(audio, (0, self.segment_length - len(audio)))
            
            return torch.FloatTensor(audio).unsqueeze(0)
        except:
            return torch.zeros(1, self.segment_length)


def compute_stft_loss(audio, reconstructed, n_fft=512, hop=160):
    """Spectral L1 loss - encourages spectral similarity"""
    # Squeeze channel
    if audio.dim() == 3:
        audio = audio.squeeze(1)
        reconstructed = reconstructed.squeeze(1)
    
    # Compute STFT
    spec_audio = torch.stft(audio, n_fft, hop_length=hop, return_complex=True).abs()
    spec_recon = torch.stft(reconstructed, n_fft, hop_length=hop, return_complex=True).abs()
    
    # Handle shape mismatch
    min_t = min(spec_audio.shape[2], spec_recon.shape[2])
    spec_audio = spec_audio[:, :, :min_t]
    spec_recon = spec_recon[:, :, :min_t]
    
    # Log-domain loss (perceptually weighted)
    spec_audio_log = torch.log1p(spec_audio)
    spec_recon_log = torch.log1p(spec_recon)
    
    return nn.L1Loss()(spec_recon_log, spec_audio_log)


def train_epoch(model, train_loader, optimizer, device):
    """One training epoch"""
    model.train()
    
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, audio in enumerate(pbar):
        audio = audio.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed = model(audio)
        
        # Handle dimension mismatch (model might change time dimension slightly)
        min_time = min(audio.shape[-1], reconstructed.shape[-1])
        audio_trimmed = audio[..., :min_time]
        recon_trimmed = reconstructed[..., :min_time]
        
        # Time-domain loss
        time_loss = nn.L1Loss()(audio_trimmed, recon_trimmed)
        
        # Spectral loss (optional, wrapped in try-except)
        try:
            stft_loss = compute_stft_loss(audio_trimmed, recon_trimmed)
        except:
            stft_loss = torch.tensor(0.0, device=device)
        
        # Combined loss: emphasize spectral domain (PESQ)
        loss = 2.0 * stft_loss + 0.5 * time_loss if stft_loss.item() > 0 else time_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / n_batches if n_batches > 0 else 0


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Quick evaluation - compute reconstruction loss"""
    model.eval()
    
    total_loss = 0
    n_batches = 0
    
    for audio in val_loader:
        audio = audio.to(device)
        reconstructed = model(audio)
        
        loss = nn.L1Loss()(audio, reconstructed)
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0


def main():
    # Config
    sr = 16000
    segment_length = 16000
    batch_size = 4
    n_epochs = 15
    lr = 5e-6  # Lower LR for fine-tuning
    
    checkpoint_path = Path("checkpoints_emergency/best_pesq_finetune.pt")
    output_dir = Path("checkpoints_emergency/finetuned")
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"\n=== Loading model from {checkpoint_path} ===")
    model = NeuralAudioCodec(d_model=384, n_layers=6, n_heads=8)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Dataset
    print(f"\n=== Loading dataset ===")
    dataset_path = get_dataset_paths()["train_clean_100"]
    
    dataset = AudioDataset(dataset_path, sr=sr, segment_length=segment_length, n_files=1000)
    
    if len(dataset) == 0:
        print("ERROR: No dataset files found!")
        return
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"✓ Dataset loaded: {len(dataset)} files, {len(train_loader)} batches")
    
    # Optimizer - lower LR for fine-tuning
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\n=== Fine-tuning for {n_epochs} epochs ===")
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(n_epochs):
        epoch_loss = train_epoch(model, train_loader, optimizer, device)
        
        print(f"\nEpoch {epoch+1}/{n_epochs} | Loss: {epoch_loss:.6f}")
        
        # Save checkpoint if improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': epoch_loss
            }
            torch.save(checkpoint, output_dir / f"epoch_{epoch+1:02d}_loss_{epoch_loss:.6f}.pt")
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"✓ Saved best checkpoint (loss: {best_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n=== Fine-tuning complete ===")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"\nNext: Run evaluation on best checkpoint")
    print(f"  ./venv/bin/python scripts/evaluate_scipy_based.py {output_dir}/best.pt")


if __name__ == "__main__":
    main()
