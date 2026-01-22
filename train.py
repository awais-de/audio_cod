"""
Training script for Neural Audio Codec
Implements multi-component loss for high-quality audio reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale spectral loss for better perceptual quality.
    Computes spectral distance at multiple FFT sizes.
    """
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=None, win_sizes=None):
        super().__init__()
        self.fft_sizes = fft_sizes
        
        if hop_sizes is None:
            self.hop_sizes = [fs // 4 for fs in fft_sizes]
        else:
            self.hop_sizes = hop_sizes
            
        if win_sizes is None:
            self.win_sizes = fft_sizes
        else:
            self.win_sizes = win_sizes
    
    def stft(self, x, fft_size, hop_size, win_size):
        """Compute STFT"""
        # x: (batch, 1, time)
        x = x.squeeze(1)  # (batch, time)
        
        window = torch.hann_window(win_size).to(x.device)
        spec = torch.stft(
            x, 
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            return_complex=True,
            center=True
        )
        return spec
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 1, time)
            target: (batch, 1, time)
        Returns:
            loss: scalar
        """
        loss = 0.0
        
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            # Compute spectrograms
            pred_spec = self.stft(pred, fft_size, hop_size, win_size)
            target_spec = self.stft(target, fft_size, hop_size, win_size)
            
            # Magnitude
            pred_mag = torch.abs(pred_spec)
            target_mag = torch.abs(target_spec)
            
            # Log-magnitude spectral distance
            pred_log_mag = torch.log(pred_mag + 1e-5)
            target_log_mag = torch.log(target_mag + 1e-5)
            
            spectral_loss = F.l1_loss(pred_log_mag, target_log_mag)
            
            # Magnitude loss
            mag_loss = F.l1_loss(pred_mag, target_mag)
            
            loss += spectral_loss + mag_loss
        
        return loss / len(self.fft_sizes)


class AudioLoss(nn.Module):
    """
    Combined loss function for audio quality:
    - Time-domain reconstruction (L1)
    - Multi-scale spectral loss
    - Optional adversarial loss (for future)
    """
    def __init__(
        self,
        l1_weight=1.0,
        spectral_weight=1.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight
        
        self.spectral_loss = MultiScaleSpectralLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 1, time) predicted audio
            target: (batch, 1, time) target audio
        Returns:
            loss: scalar
            loss_dict: dictionary of individual losses
        """
        # Time-domain L1 loss
        l1_loss = F.l1_loss(pred, target)
        
        # Multi-scale spectral loss
        spectral_loss = self.spectral_loss(pred, target)
        
        # Combined loss
        total_loss = (
            self.l1_weight * l1_loss +
            self.spectral_weight * spectral_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1_loss.item(),
            'spectral': spectral_loss.item(),
        }
        
        return total_loss, loss_dict


class AudioDataset(Dataset):
    """
    Dataset for loading audio files.
    Loads audio, resamples to target sample rate, and chunks into segments.
    """
    def __init__(
        self,
        audio_dir,
        sample_rate=16000,
        segment_length=16000,  # 1 second
        extensions=['.wav', '.flac', '.mp3']
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        
        # Find all audio files
        self.audio_files = []
        for ext in extensions:
            self.audio_files.extend(list(self.audio_dir.rglob(f"*{ext}")))
        
        print(f"Found {len(self.audio_files)} audio files in {audio_dir}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or truncate to segment_length
        if waveform.shape[1] < self.segment_length:
            # Pad with zeros
            padding = self.segment_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        else:
            # Random crop
            start = torch.randint(0, waveform.shape[1] - self.segment_length + 1, (1,)).item()
            waveform = waveform[:, start:start + self.segment_length]
        
        return waveform


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'l1': 0, 'spectral': 0}
    
    for batch_idx, audio in enumerate(dataloader):
        audio = audio.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed = model(audio)
        
        # Compute loss
        loss, loss_dict = criterion(reconstructed, audio)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"L1: {loss_dict['l1']:.4f} "
                  f"Spectral: {loss_dict['spectral']:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] /= len(dataloader)
    
    return avg_loss, loss_components


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    loss_components = {'l1': 0, 'spectral': 0}
    
    with torch.no_grad():
        for audio in dataloader:
            audio = audio.to(device)
            
            # Forward pass
            reconstructed = model(audio)
            
            # Compute loss
            loss, loss_dict = criterion(reconstructed, audio)
            
            # Accumulate losses
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
    
    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] /= len(dataloader)
    
    return avg_loss, loss_components


def train(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-4,
    device='cuda',
    checkpoint_dir='checkpoints'
):
    """Main training loop with GPU optimizations"""
    
    # Enable GPU optimizations
    if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
        print("\n🚀 Enabling GPU optimizations:")
        print("   ✓ CUDNN benchmark mode")
        print("   ✓ Tensor cores (if available)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Loss function and optimizer
    criterion = AudioLoss(l1_weight=1.0, spectral_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_components = validate(model, val_loader, criterion, device)
        
        # Learning rate step
        scheduler.step()
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f} (L1: {train_components['l1']:.4f}, Spectral: {train_components['spectral']:.4f})")
        print(f"Val Loss: {val_loss:.4f} (L1: {val_components['l1']:.4f}, Spectral: {val_components['spectral']:.4f})")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model with val loss: {val_loss:.4f}")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)


if __name__ == "__main__":
    from model import NeuralAudioCodec
    
    # Configuration - GPU detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎮 GPU DETECTED!")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU detected - using CPU (training will be SLOW)")
        print("   Consider installing CUDA-enabled PyTorch for much faster training")
    
    print(f"\nUsing device: {device}")
    
    # Create model
    model = NeuralAudioCodec(
        sample_rate=16000,
        hop_length=160,
        d_model=512,
        n_layers=8,
        n_heads=16,
        window_size=512,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create datasets (you need to provide your audio data directory)
    # Example:
    # train_dataset = AudioDataset('path/to/train/audio', sample_rate=16000)
    # val_dataset = AudioDataset('path/to/val/audio', sample_rate=16000)
    # 
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    # 
    # # Train
    # train(model, train_loader, val_loader, epochs=100, lr=1e-4, device=device)
    
    print("\nTo train the model:")
    print("1. Prepare your audio dataset")
    print("2. Update the dataset paths in this script")
    print("3. Run: python train.py")
