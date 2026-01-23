"""
Simple training script to debug the NaN issue
Uses a simpler model for testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import yaml


class SimpleAudioCodec(nn.Module):
    """Simplified audio codec for debugging"""
    def __init__(self):
        super().__init__()
        # Simple encoder: downsample by 8x
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, 7, stride=2, padding=3),
            nn.GroupNorm(16, 128),
            nn.GELU(),
        )
        
        # Simple decoder: upsample by 8x
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 8, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, 8, stride=2, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 1, 8, stride=2, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        # x: (batch, 1, time)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        # Crop to match input size
        return reconstructed[:, :, :x.shape[-1]]


class AudioDataset(Dataset):
    """Dataset for loading audio files."""
    def __init__(
        self,
        audio_dir,
        sample_rate=16000,
        segment_length=8000,
        extensions=['.wav', '.flac', '.mp3'],
        max_files=None  # Limit for testing
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        
        # Find all audio files
        self.audio_files = []
        for ext in extensions:
            self.audio_files.extend(list(self.audio_dir.rglob(f"*{ext}")))
        
        # Limit for testing
        if max_files:
            self.audio_files = self.audio_files[:max_files]
        
        print(f"Found {len(self.audio_files)} audio files in {audio_dir}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        try:
            audio_path = self.audio_files[idx]
            
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Normalize to [-1, 1]
            max_val = torch.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / (max_val + 1e-8)
            
            # Ensure exactly segment_length samples
            if waveform.shape[1] < self.segment_length:
                padding = self.segment_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            elif waveform.shape[1] > self.segment_length:
                max_start = waveform.shape[1] - self.segment_length
                start = torch.randint(0, max_start + 1, (1,)).item()
                waveform = waveform[:, start:start + self.segment_length]
            
            assert waveform.shape == (1, self.segment_length), f"Shape mismatch: {waveform.shape}"
            return waveform.float()
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, self.segment_length, dtype=torch.float32)


def train_simple():
    """Simple training loop"""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create model
    model = SimpleAudioCodec().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Create datasets (use only 1000 files for quick test)
    train_dir = config['data']['train_dir']
    
    train_dataset = AudioDataset(
        train_dir,
        sample_rate=16000,
        segment_length=8000,
        max_files=1000  # Only 1000 files for testing
    )
    
    val_dataset = AudioDataset(
        train_dir,
        sample_rate=16000,
        segment_length=8000,
        max_files=100  # Only 100 files for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Train for just 2 epochs for testing
    for epoch in range(1, 3):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/2")
        print(f"{'='*80}\n")
        
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, audio in enumerate(train_loader):
            audio = audio.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(audio)
            
            # Ensure shapes match
            if reconstructed.shape[-1] != audio.shape[-1]:
                reconstructed = reconstructed[:, :, :audio.shape[-1]]
            
            # Compute loss
            loss = criterion(reconstructed, audio)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Batch {batch_idx}: NaN/Inf loss detected, skipping")
                continue
            
            # Backward
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/~{len(train_loader)}: Loss = {loss.item():.6f}")
        
        if num_batches > 0:
            avg_train_loss = total_loss / num_batches
            print(f"\nTraining Loss: {avg_train_loss:.6f}\n")
        else:
            print("No valid batches in training!")
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for audio in val_loader:
                audio = audio.to(device)
                reconstructed = model(audio)
                
                if reconstructed.shape[-1] != audio.shape[-1]:
                    reconstructed = reconstructed[:, :, :audio.shape[-1]]
                
                loss = criterion(reconstructed, audio)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_val_loss += loss.item()
                    val_batches += 1
        
        if val_batches > 0:
            avg_val_loss = total_val_loss / val_batches
            print(f"Validation Loss: {avg_val_loss:.6f}\n")
        else:
            print("No valid validation batches!")
        
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.8f}")


if __name__ == "__main__":
    train_simple()
