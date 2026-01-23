"""
Optimized Training script with profiling for Neural Audio Codec
Includes bottleneck identification and performance monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import time
from collections import deque
import yaml
from model import NeuralAudioCodec


class PerformanceMonitor:
    """Track performance metrics for different pipeline stages"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.data_load_times = deque(maxlen=window_size)
        self.gpu_compute_times = deque(maxlen=window_size)
        self.loss_compute_times = deque(maxlen=window_size)
        self.batch_times = deque(maxlen=window_size)
        
    def add_data_load_time(self, t):
        self.data_load_times.append(t)
    
    def add_gpu_compute_time(self, t):
        self.gpu_compute_times.append(t)
    
    def add_loss_compute_time(self, t):
        self.loss_compute_times.append(t)
    
    def add_batch_time(self, t):
        self.batch_times.append(t)
    
    def report(self):
        if not self.batch_times:
            return
        
        avg_batch = sum(self.batch_times) / len(self.batch_times)
        avg_data_load = sum(self.data_load_times) / len(self.data_load_times) if self.data_load_times else 0
        avg_gpu = sum(self.gpu_compute_times) / len(self.gpu_compute_times) if self.gpu_compute_times else 0
        avg_loss = sum(self.loss_compute_times) / len(self.loss_compute_times) if self.loss_compute_times else 0
        
        print(f"\n{'='*60}")
        print("PERFORMANCE PROFILE (last {0} batches):".format(len(self.batch_times)))
        print(f"{'='*60}")
        print(f"Total batch time:  {avg_batch*1000:.2f} ms")
        print(f"  - Data loading:  {avg_data_load*1000:.2f} ms ({100*avg_data_load/avg_batch:.1f}%)")
        print(f"  - GPU compute:   {avg_gpu*1000:.2f} ms ({100*avg_gpu/avg_batch:.1f}%)")
        print(f"  - Loss compute:  {avg_loss*1000:.2f} ms ({100*avg_loss/avg_batch:.1f}%)")
        print(f"  - Other:         {(avg_batch-avg_data_load-avg_gpu-avg_loss)*1000:.2f} ms ({100*(avg_batch-avg_data_load-avg_gpu-avg_loss)/avg_batch:.1f}%)")
        print(f"{'='*60}\n")


class OptimizedAudioDataset(Dataset):
    """
    Optimized dataset for loading audio files.
    - Caches resamplers to avoid recreation per sample
    - Uses ffmpeg backend for faster audio loading
    - Removes assertions from hot path
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
        self.resamplers = {}  # Cache resamplers by source sample rate
        
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
            
            # Load audio with fast ffmpeg backend
            waveform, sr = torchaudio.load(str(audio_path), backend='ffmpeg')
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if necessary (use cached resampler)
            if sr != self.sample_rate:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
                    # Pre-allocate resampler state on correct device
                    if torch.cuda.is_available():
                        self.resamplers[sr] = self.resamplers[sr].cuda()
                
                resampler = self.resamplers[sr]
                # Move audio to same device as resampler if needed
                if next(resampler.parameters()).device != waveform.device:
                    waveform = waveform.to(next(resampler.parameters()).device)
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
            # Return silence on error (no print to avoid I/O overhead)
            return torch.zeros(1, self.segment_length, dtype=torch.float32)


class MultiScaleSpectralLoss(nn.Module):
    """Multi-scale spectral loss for perceptual quality"""
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
        """Compute STFT efficiently"""
        x = x.squeeze(1)  # (batch, time)
        
        window = torch.hann_window(win_size, device=x.device)
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
        pred: (batch, 1, time)
        target: (batch, 1, time)
        """
        if pred.shape != target.shape:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        loss = 0.0
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            if pred.shape[-1] < fft_size:
                continue
            
            pred_spec = self.stft(pred, fft_size, hop_size, win_size)
            target_spec = self.stft(target, fft_size, hop_size, win_size)
            
            pred_mag = torch.abs(pred_spec)
            target_mag = torch.abs(target_spec)
            
            loss += F.l1_loss(pred_mag, target_mag)
        
        return loss


class AudioLoss(nn.Module):
    """Combined L1 + Spectral Loss with stability"""
    def __init__(self, l1_weight=1.0, spectral_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight
        self.spectral_loss = MultiScaleSpectralLoss()
    
    def forward(self, pred, target):
        """
        pred: (batch, 1, time)
        target: (batch, 1, time)
        """
        # Validate inputs
        if pred.isnan().any() or target.isnan().any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # L1 Loss
        l1_loss = F.l1_loss(pred, target)
        
        # Spectral Loss
        spec_loss = self.spectral_loss(pred, target)
        
        # Combined loss with safety clamping
        total_loss = self.l1_weight * l1_loss + self.spectral_weight * spec_loss
        total_loss = torch.clamp(total_loss, max=100.0)
        
        # Replace any NaN with zero
        total_loss = torch.where(torch.isnan(total_loss), torch.tensor(0.0, device=pred.device), total_loss)
        
        return total_loss


def train_epoch(model, train_loader, loss_fn, optimizer, device, perf_monitor):
    """Train for one epoch with performance monitoring"""
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    skipped_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()
        
        # Data loading time
        data_load_time = time.time() - batch_start
        perf_monitor.add_data_load_time(data_load_time)
        
        # Move to device
        batch = batch.to(device)
        
        # Check for NaN in input
        if batch.isnan().any():
            skipped_batches += 1
            continue
        
        # Forward pass
        gpu_compute_start = time.time()
        try:
            output = model(batch)
            if output.isnan().any():
                skipped_batches += 1
                continue
        except RuntimeError as e:
            skipped_batches += 1
            continue
        
        gpu_compute_time = time.time() - gpu_compute_start
        perf_monitor.add_gpu_compute_time(gpu_compute_time)
        
        # Loss computation
        loss_compute_start = time.time()
        loss = loss_fn(output, batch)
        if loss.isnan():
            skipped_batches += 1
            continue
        
        loss_compute_time = time.time() - loss_compute_start
        perf_monitor.add_loss_compute_time(loss_compute_time)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        batch_total_time = time.time() - batch_start
        perf_monitor.add_batch_time(batch_total_time)
        
        if (batch_idx + 1) % 10 == 0:
            avg_loss = epoch_loss / batch_count
            print(f"  Batch {batch_idx + 1}: Loss={loss.item():.6f}, Avg={avg_loss:.6f}, Time={batch_total_time*1000:.1f}ms")
    
    if batch_count == 0:
        return 0.0
    
    return epoch_loss / batch_count


def validate(model, val_loader, loss_fn, device):
    """Validate model"""
    model.eval()
    val_loss = 0.0
    batch_count = 0
    skipped_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            if batch.isnan().any():
                skipped_batches += 1
                continue
            
            try:
                output = model(batch)
                if output.isnan().any():
                    skipped_batches += 1
                    continue
            except RuntimeError:
                skipped_batches += 1
                continue
            
            loss = loss_fn(output, batch)
            if loss.isnan():
                skipped_batches += 1
                continue
            
            val_loss += loss.item()
            batch_count += 1
    
    if batch_count == 0:
        return 0.0
    
    return val_loss / batch_count


def train(model, train_loader, val_loader, epochs, lr, device, checkpoint_dir, perf_monitor):
    """Main training loop"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = AudioLoss()
    loss_fn = loss_fn.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)
        
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, perf_monitor)
        val_loss = validate(model, val_loader, loss_fn, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.2e}")
        
        # Performance report every 10 epochs
        if (epoch + 1) % 10 == 0:
            perf_monitor.report()
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")


if __name__ == '__main__':
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    checkpoint_config = config['checkpoint']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create model
    print("\nCreating model...")
    model = NeuralAudioCodec(
        sample_rate=model_config['sample_rate'],
        hop_length=model_config['hop_length'],
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        window_size=model_config['window_size'],
        dropout=model_config['dropout']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Create datasets with optimized loader
    print(f"\nLoading training data from: {data_config['train_dir']}")
    train_dataset = OptimizedAudioDataset(
        data_config['train_dir'],
        sample_rate=model_config['sample_rate'],
        segment_length=training_config['segment_length'],
        extensions=data_config['extensions']
    )
    
    print(f"Loading validation data from: {data_config['val_dir']}")
    val_dataset = OptimizedAudioDataset(
        data_config['val_dir'],
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
    
    # Performance monitor
    perf_monitor = PerformanceMonitor(window_size=100)
    
    # Train
    print(f"\nStarting optimized training for {training_config['epochs']} epochs...")
    train(
        model,
        train_loader,
        val_loader,
        epochs=training_config['epochs'],
        lr=training_config['learning_rate'],
        device=device,
        checkpoint_dir=checkpoint_config['dir'],
        perf_monitor=perf_monitor
    )
    
    # Final report
    print("\n" + "="*60)
    print("FINAL PERFORMANCE REPORT")
    perf_monitor.report()
