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
import urllib.request
import tarfile
import shutil


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
            center=True,
            pad_mode='reflect'
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
        valid_scales = 0
        
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            # Skip if audio is too short for this FFT size
            if pred.shape[-1] < fft_size or target.shape[-1] < fft_size:
                continue
                
            try:
                # Compute spectrograms
                pred_spec = self.stft(pred, fft_size, hop_size, win_size)
                target_spec = self.stft(target, fft_size, hop_size, win_size)
                
                # Magnitude
                pred_mag = torch.abs(pred_spec) + 1e-7
                target_mag = torch.abs(target_spec) + 1e-7
                
                # Log-magnitude spectral distance
                pred_log_mag = torch.log(pred_mag)
                target_log_mag = torch.log(target_mag)
                
                # Clamp to avoid NaN
                pred_log_mag = torch.clamp(pred_log_mag, min=-15, max=15)
                target_log_mag = torch.clamp(target_log_mag, min=-15, max=15)
                
                spectral_loss = F.l1_loss(pred_log_mag, target_log_mag)
                
                # Magnitude loss
                mag_loss = F.l1_loss(pred_mag, target_mag)
                
                # Skip this scale if loss is NaN or Inf
                if torch.isnan(spectral_loss) or torch.isinf(spectral_loss):
                    continue
                if torch.isnan(mag_loss) or torch.isinf(mag_loss):
                    continue
                
                loss += spectral_loss + mag_loss
                valid_scales += 1
            except Exception as e:
                print(f"Warning: Error computing spectral loss for FFT size {fft_size}: {e}")
                continue
        
        if valid_scales == 0:
            # Fallback: return zero loss if no valid scales
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return loss / valid_scales


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
        # Ensure pred and target are finite
        pred = torch.clamp(pred, -1.0, 1.0)
        target = torch.clamp(target, -1.0, 1.0)
        
        # Check for NaN/Inf in inputs
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print(f"WARNING: Predicted audio contains NaN/Inf values!")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.5, neginf=-0.5)
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"WARNING: Target audio contains NaN/Inf values!")
            target = torch.nan_to_num(target, nan=0.0, posinf=0.5, neginf=-0.5)
        
        # Time-domain L1 loss
        l1_loss = F.l1_loss(pred, target)
        
        # Ensure L1 loss is valid
        if torch.isnan(l1_loss) or torch.isinf(l1_loss):
            l1_loss = torch.tensor(0.5, device=pred.device, requires_grad=True)
        
        # Multi-scale spectral loss
        spectral_loss = self.spectral_loss(pred, target)
        
        # Ensure spectral loss is valid
        if torch.isnan(spectral_loss) or torch.isinf(spectral_loss):
            spectral_loss = torch.tensor(0.5, device=pred.device, requires_grad=True)
        
        # Combined loss
        total_loss = (
            self.l1_weight * l1_loss +
            self.spectral_weight * spectral_loss
        )
        
        # Clamp total loss to prevent overflow
        total_loss = torch.clamp(total_loss, max=100.0)
        
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1_loss.item(),
            'spectral': spectral_loss.item(),
        }
        
        return total_loss, loss_dict


class AudioDataset(Dataset):
    """
    Optimized dataset for loading audio files.
    Caches resamplers and uses ffmpeg backend for faster loading.
    """
    def __init__(
        self,
        audio_dir,
        sample_rate=16000,
        segment_length=8000,
        extensions=['.wav', '.flac', '.mp3']
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.resamplers = {}  # Cache resamplers
        
        # Find all audio files
        self.audio_files = []
        for ext in extensions:
            self.audio_files.extend(list(self.audio_dir.rglob(f"*{ext}")))
        
        print(f"Found {len(self.audio_files)} audio files in {audio_dir}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        try:
            audio_path = self.audio_files[idx]
            
            # Load audio with fast backend
            waveform, sr = torchaudio.load(str(audio_path), backend='ffmpeg')
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if necessary (use cached resampler)
            if sr != self.sample_rate:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
                resampler = self.resamplers[sr]
                waveform = resampler(waveform)
            
            # Normalize audio to [-1, 1]
            max_val = torch.abs(waveform).max()
            if max_val > 1e-6:
                waveform = waveform / (max_val + 1e-8)
            
            # Ensure exactly segment_length samples
            if waveform.shape[1] < self.segment_length:
                padding = self.segment_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            elif waveform.shape[1] > self.segment_length:
                max_start = waveform.shape[1] - self.segment_length
                start = torch.randint(0, max_start + 1, (1,)).item()
                waveform = waveform[:, start:start + self.segment_length]
            
            return waveform.float()
        except Exception as e:
            # Return silence on error
            return torch.zeros(1, self.segment_length, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'l1': 0, 'spectral': 0}
    num_batches = 0
    
    for batch_idx, audio in enumerate(dataloader):
        audio = audio.to(device)
        
        # Check for NaN in input audio
        if torch.isnan(audio).any():
            print(f"WARNING: Input audio contains NaN at batch {batch_idx}")
            continue
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed = model(audio)
        
        # Check reconstructed output
        if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
            print(f"WARNING: Model output contains NaN/Inf at batch {batch_idx}")
            reconstructed = torch.nan_to_num(reconstructed, nan=0.0, posinf=0.5, neginf=-0.5)
        
        # Handle size mismatch - crop to match
        if reconstructed.shape[-1] > audio.shape[-1]:
            reconstructed = reconstructed[:, :, :audio.shape[-1]]
        elif reconstructed.shape[-1] < audio.shape[-1]:
            # Pad reconstructed to match audio
            pad_size = audio.shape[-1] - reconstructed.shape[-1]
            reconstructed = F.pad(reconstructed, (0, pad_size))
        
        # Compute loss
        loss, loss_dict = criterion(reconstructed, audio)
        
        # Skip batch if loss is invalid
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Loss is NaN/Inf at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        num_batches += 1
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_so_far = total_loss / num_batches
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Batch Loss: {loss.item():.4f} | "
                  f"Avg Loss: {avg_so_far:.4f} | "
                  f"L1: {loss_dict['l1']:.4f} | "
                  f"Spectral: {loss_dict['spectral']:.4f}")
    
    if num_batches == 0:
        print(f"WARNING: No valid batches in epoch {epoch}")
        return float('inf'), {'l1': float('inf'), 'spectral': float('inf')}
    
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return avg_loss, loss_components


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    loss_components = {'l1': 0, 'spectral': 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, audio in enumerate(dataloader):
            audio = audio.to(device)
            
            # Check for NaN in input
            if torch.isnan(audio).any():
                continue
            
            # Forward pass
            reconstructed = model(audio)
            
            # Check output
            if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                reconstructed = torch.nan_to_num(reconstructed, nan=0.0, posinf=0.5, neginf=-0.5)
            
            # Handle size mismatch - crop to match
            if reconstructed.shape[-1] > audio.shape[-1]:
                reconstructed = reconstructed[:, :, :audio.shape[-1]]
            elif reconstructed.shape[-1] < audio.shape[-1]:
                # Pad reconstructed to match audio
                pad_size = audio.shape[-1] - reconstructed.shape[-1]
                reconstructed = F.pad(reconstructed, (0, pad_size))
            
            # Compute loss
            loss, loss_dict = criterion(reconstructed, audio)
            
            # Skip if invalid
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # Accumulate losses
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            num_batches += 1
    
    if num_batches == 0:
        return float('inf'), {'l1': float('inf'), 'spectral': float('inf')}
    
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
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
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_components = validate(model, val_loader, criterion, device)
        
        # Learning rate step
        scheduler.step()
        
        print(f"\n{'-'*80}")
        print(f"EPOCH {epoch} COMPLETE - METRICS SUMMARY")
        print(f"{'-'*80}")
        print(f"Training:")
        print(f"  Total Loss:    {train_loss:.6f}")
        print(f"  L1 Loss:       {train_components['l1']:.6f}")
        print(f"  Spectral Loss: {train_components['spectral']:.6f}")
        print(f"\nValidation:")
        print(f"  Total Loss:    {val_loss:.6f}")
        print(f"  L1 Loss:       {val_components['l1']:.6f}")
        print(f"  Spectral Loss: {val_components['spectral']:.6f}")
        print(f"\nLearning Rate: {scheduler.get_last_lr()[0]:.8f}")
        print(f"{'-'*80}\n")
        
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
            print(f"✓ Saved best model (Val Loss: {val_loss:.6f})\n")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ Saved periodic checkpoint (Epoch {epoch})\n")


def ensure_dataset_exists(config):
    """
    Check if dataset exists. If not, download and extract it.
    Updates config.yaml with the correct path.
    """
    data_config = config['data']
    train_dir = Path(data_config['train_dir'])
    
    # Check if dataset already exists
    if train_dir.exists() and len(list(train_dir.glob('**/*.flac'))) > 1000:
        print(f"✓ Dataset already exists at {train_dir}")
        return True
    
    # Dataset doesn't exist or is incomplete, need to download
    print("\n" + "="*80)
    print("DATASET NOT FOUND - DOWNLOADING")
    print("="*80)
    
    download_url = "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz"
    dataset_dir = train_dir.parent
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = dataset_dir / "train-clean-100.tar.gz"
    
    # Download if not already present
    if not tar_path.exists():
        print(f"\n📥 Downloading dataset from {download_url}")
        print("   This may take 10-30 minutes depending on your internet speed...")
        print("   File size: ~5.9 GB")
        
        try:
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Progress: {percent}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end='')
            
            urllib.request.urlretrieve(download_url, tar_path, download_progress)
            print("\n   ✓ Download complete!")
        except Exception as e:
            print(f"\n❌ Error downloading dataset: {e}")
            return False
    else:
        print(f"✓ Archive already exists at {tar_path}")
    
    # Extract dataset
    print(f"\n📦 Extracting dataset...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=dataset_dir)
        print("   ✓ Extraction complete!")
    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        return False
    
    # Find extracted directory and verify
    extracted_dir = dataset_dir / "LibriSpeech" / "train-clean-100"
    if not extracted_dir.exists():
        print(f"❌ Expected directory not found: {extracted_dir}")
        return False
    
    # Count files
    flac_files = list(extracted_dir.glob('**/*.flac'))
    print(f"   Found {len(flac_files)} audio files")
    
    if len(flac_files) < 1000:
        print(f"❌ Not enough audio files found. Expected ~28,539, got {len(flac_files)}")
        return False
    
    # Update config with correct path
    print(f"\n📝 Updating config.yaml with dataset path...")
    config['data']['train_dir'] = str(extracted_dir)
    config['data']['val_dir'] = str(extracted_dir)
    
    with open('config.yaml', 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    print(f"   ✓ Config updated!")
    
    # Save config reference for later
    config['data']['train_dir'] = str(extracted_dir)
    config['data']['val_dir'] = str(extracted_dir)
    
    print("\n✓ Dataset ready for training!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    import yaml
    from model import NeuralAudioCodec
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure dataset exists and is properly configured
    if not ensure_dataset_exists(config):
        print("\n❌ Failed to prepare dataset. Exiting.")
        exit(1)
    
    # Reload config after potential updates
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
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
    
    # Get config values
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    checkpoint_config = config['checkpoint']
    
    # Create model
    model = NeuralAudioCodec(
        sample_rate=model_config['sample_rate'],
        hop_length=model_config['hop_length'],
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        window_size=model_config['window_size'],
        dropout=model_config['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create datasets from config paths
    train_dir = data_config['train_dir']
    val_dir = data_config['val_dir']
    
    print(f"\nLoading training data from: {train_dir}")
    train_dataset = AudioDataset(
        train_dir,
        sample_rate=model_config['sample_rate'],
        segment_length=training_config['segment_length'],
        extensions=data_config['extensions']
    )
    
    print(f"Loading validation data from: {val_dir}")
    val_dataset = AudioDataset(
        val_dir,
        sample_rate=model_config['sample_rate'],
        segment_length=training_config['segment_length'],
        extensions=data_config['extensions']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config['num_workers']
    )
    
    # Train
    print(f"\nStarting training for {training_config['epochs']} epochs...")
    train(
        model,
        train_loader,
        val_loader,
        epochs=training_config['epochs'],
        lr=training_config['learning_rate'],
        device=device,
        checkpoint_dir=checkpoint_config['dir']
    )
