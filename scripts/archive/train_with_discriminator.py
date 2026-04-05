#!/usr/bin/env python3
"""
Training with Discriminator Loss for PESQ Improvement
Targets: PESQ 3.5+ (from baseline 2.90), STOI >0.9, Latency <20ms
Strategy: Discriminator encourages realistic spectral features
"""
import os
import sys
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from datetime import datetime

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


class Discriminator(nn.Module):
    """Discriminator for spectral realism - judges if spectrogram looks real or coded"""
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, spectrogram):
        """spectrogram: [B, T, F] -> output: [B]"""
        B, T, F = spectrogram.shape
        x = spectrogram.reshape(B, -1)
        x = x[:, :min(x.shape[1], 1024)]  # Cap input dim for stability
        if x.shape[1] < 1024:
            x = torch.cat([x, torch.zeros(B, 1024-x.shape[1], device=x.device)], dim=1)
        return self.net(x).squeeze(1)


def stft_features(audio, n_fft=512, hop=160):
    """Compute STFT magnitude spectrogram"""
    # Squeeze channel if present
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    return torch.stft(audio, n_fft, hop_length=hop, return_complex=True).abs()


def discriminator_loss(disc, reconstructed, original, device):
    """
    Discriminator vs Reconstructed loss
    Goal: Make reconstructed spectrograms look like original
    """
    spec_recon = stft_features(reconstructed)
    spec_orig = stft_features(original)
    
    # Flatten for discriminator [B*T, F] -> sample 256 channels
    B, T, F = spec_orig.shape
    spec_orig_flat = spec_orig.reshape(B, -1)[:, :256]
    spec_recon_flat = spec_recon.reshape(B, -1)[:, :256]
    
    # Discriminator predictions
    pred_orig = disc(spec_orig)     # Should output ~1 (real)
    pred_recon = disc(spec_recon)   # Should output ~0 (fake) initially
    
    # During generator training: make reconstructed look real
    gen_loss = torch.mean((pred_recon - 1) ** 2)
    
    return gen_loss


def train_epoch(model, disc, train_loader, optim_gen, device, loss_config):
    """One training epoch with discriminator"""
    model.train()
    disc.train()
    
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, audio in enumerate(pbar):
        audio = audio.to(device)
        B = audio.shape[0]
        
        # ===== Generator step =====
        optim_gen.zero_grad()
        
        # Encode-decode
        reconstructed = model(audio)
        
        # Losses
        stft_loss = nn.L1Loss()(
            stft_features(audio),
            stft_features(reconstructed)
        )
        
        time_loss = nn.L1Loss()(audio, reconstructed)
        
        disc_loss = discriminator_loss(disc, reconstructed, audio, device)
        
        # Combined generator loss
        loss_g = (
            loss_config.get('w_stft', 2.0) * stft_loss +
            loss_config.get('w_time', 0.5) * time_loss +
            loss_config.get('w_disc', 0.3) * disc_loss
        )
        
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        optim_gen.step()
        optim_disc.step()
        
        total_loss += loss_g.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss_g.item():.4f}'})
    
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
    n_epochs = 20
    lr = 1e-5
    
    checkpoint_path = Path("checkpoints_emergency/best_pesq_finetune.pt")
    output_dir = Path("checkpoints_emergency/discriminator_training")
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"\n=== Loading model from {checkpoint_path} ===")
    model = NeuralAudioCodec(d_model=384, n_layers=6, n_heads=8)
    state = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model = model.to(device)
    
    # Load discriminator
    disc = Discriminator(input_dim=256, hidden_dim=128).to(device)
    
    # Dataset
    print(f"\n=== Loading dataset ===")
    dataset_path = get_dataset_paths()["train_clean_100"]
    print(f"Dataset path: {dataset_path}")
    print(f"Exists: {dataset_path.exists()}")
    
    dataset = AudioDataset(dataset_path, sr=sr, segment_length=segment_length, n_files=500)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Optimizer
    optim_gen = Adam(list(model.parameters()) + list(disc.parameters()), lr=lr)
    
    loss_config = {
        'w_stft': 2.0,
        'w_time': 0.5,
        'w_disc': 0.3
    }
    
    # Training loop
    print(f"\n=== Training for {n_epochs} epochs (Discriminator Loss) ===")
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        epoch_loss = train_epoch(model, disc, train_loader, optim_gen, device, loss_config)
        
        print(f"\nEpoch {epoch+1}/{n_epochs} | Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'loss': epoch_loss
            }
            torch.save(checkpoint, output_dir / f"epoch_{epoch+1}_loss_{epoch_loss:.4f}.pt")
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"✓ Saved best checkpoint")
    
    print(f"\n=== Training complete ===")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"\nNext: Run evaluation on best checkpoint")
    print(f"  python scripts/evaluate_scipy_based.py {output_dir}/best.pt")


if __name__ == "__main__":
    main()
