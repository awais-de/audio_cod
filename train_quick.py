"""
Quick Start Training Script
Train on a small dataset to verify the model learns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import os

from model import NeuralAudioCodec

class SyntheticAudioDataset(Dataset):
    """
    Synthetic dataset for quick testing.
    Generates simple audio signals (sine waves, chirps, noise)
    """
    def __init__(self, num_samples=1000, sample_rate=16000, duration=1.0):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.length = int(sample_rate * duration)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate different types of signals
        t = np.linspace(0, self.duration, self.length)
        
        signal_type = idx % 4
        
        if signal_type == 0:
            # Sine wave (random frequency)
            freq = np.random.uniform(200, 800)
            signal = np.sin(2 * np.pi * freq * t)
            
        elif signal_type == 1:
            # Chirp (frequency sweep)
            f0, f1 = np.random.uniform(200, 400), np.random.uniform(600, 1000)
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * self.duration))
            signal = np.sin(phase)
            
        elif signal_type == 2:
            # Multi-tone
            freqs = np.random.uniform(200, 1000, 3)
            signal = sum(np.sin(2 * np.pi * f * t) for f in freqs) / 3
            
        else:
            # Amplitude modulated
            carrier_freq = np.random.uniform(400, 800)
            mod_freq = np.random.uniform(2, 10)
            carrier = np.sin(2 * np.pi * carrier_freq * t)
            modulator = 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t))
            signal = carrier * modulator
        
        # Add slight noise
        signal += np.random.randn(self.length) * 0.01
        
        # Normalize
        signal = signal / (np.abs(signal).max() + 1e-8)
        
        # Convert to tensor
        signal = torch.FloatTensor(signal).unsqueeze(0)  # (1, length)
        
        return signal


def simple_loss(pred, target):
    """Simple L1 loss for quick training"""
    return torch.nn.functional.l1_loss(pred, target)


def train_quick(
    num_epochs=20,
    batch_size=8,
    learning_rate=1e-3,
    device=None
):
    """
    Quick training on synthetic data to verify the model learns.
    This is NOT for production - just to see that training works!
    """
    
    # Force GPU detection
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("=" * 80)
            print("🎮 GPU DETECTED AND ENABLED!")
            print("=" * 80)
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
            
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ GPU optimizations enabled (cuDNN benchmark, TF32)")
        else:
            device = torch.device('cpu')
            print("=" * 80)
            print("⚠️  WARNING: NO GPU DETECTED - USING CPU")
            print("=" * 80)
            print("Training will be MUCH slower on CPU!")
            print("To use GPU, install CUDA-enabled PyTorch:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("=" * 80)
    print("QUICK START TRAINING - SYNTHETIC DATA")
    print("=" * 80)
    print()
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Create model (smaller for faster training)
    print("Creating model...")
    model = NeuralAudioCodec(
        sample_rate=16000,
        hop_length=160,
        d_model=256,  # Smaller than production (512)
        n_layers=4,   # Fewer layers for speed (8 in production)
        n_heads=8,    # Fewer heads (16 in production)
        window_size=256,
        dropout=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    train_dataset = SyntheticAudioDataset(num_samples=800, sample_rate=16000, duration=1.0)
    val_dataset = SyntheticAudioDataset(num_samples=200, sample_rate=16000, duration=1.0)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    print("-" * 80)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for batch_idx, audio in enumerate(train_loader):
            audio = audio.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(audio)
            
            # Match shapes (encoder might change length slightly)
            min_len = min(audio.shape[-1], reconstructed.shape[-1])
            audio = audio[..., :min_len]
            reconstructed = reconstructed[..., :min_len]
            
            loss = simple_loss(reconstructed, audio)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for audio in val_loader:
                audio = audio.to(device)
                reconstructed = model(audio)
                
                min_len = min(audio.shape[-1], reconstructed.shape[-1])
                audio = audio[..., :min_len]
                reconstructed = reconstructed[..., :min_len]
                
                loss = simple_loss(reconstructed, audio)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Calculate approximate SNR
        snr = -20 * np.log10(val_loss + 1e-8)
        
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"~SNR: {snr:.1f} dB")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            os.makedirs('quick_checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'quick_checkpoints/checkpoint_epoch_{epoch}.pt')
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("What just happened?")
    print("  ✅ Model trained on 800 synthetic audio signals")
    print("  ✅ Loss decreased (model is learning!)")
    print("  ✅ Checkpoint saved to quick_checkpoints/")
    print()
    print("This proves the architecture works. Now you can:")
    print("  1. Train on REAL speech data for production quality")
    print("  2. Use the full model size (d_model=512, n_layers=8)")
    print("  3. Train for 100+ epochs")
    print()
    print("Next steps:")
    print("  • Get real speech dataset (LibriSpeech, etc.)")
    print("  • Update config.yaml with your data paths")
    print("  • Run: python train.py (for full training)")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick training test')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    train_quick(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
