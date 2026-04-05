#!/usr/bin/env python3
"""
Phase 3b: Rate-Distortion Training
Fine-tune neural codec with learned entropy model.
Objective: Minimize D + lambda * R, where D=distortion, R=rate
Expected outcome: 141 kbps -> 15-25 kbps while maintaining PESQ >= 2.0
"""

import argparse
import sys
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.entropy_model import EntropyBottleneck
from src.rate_distortion_loss import RateDistortionLoss
from src.paths import get_dataset_paths


class AudioDataset(IterableDataset):
    """Streaming dataset for audio files with random chunks."""
    
    def __init__(self, data_root, chunk_seconds=2.0, sample_rate=16000, 
                 num_workers=4, epoch_size=1000):
        """
        Args:
            data_root: Path to audio files
            chunk_seconds: Duration of audio chunks
            sample_rate: Sample rate
            num_workers: Number of workers (affects prefetching)
            epoch_size: Number of samples per epoch
        """
        self.data_root = Path(data_root)
        self.chunk_seconds = chunk_seconds
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_seconds * sample_rate)
        self.epoch_size = epoch_size
        
        # Find all audio files
        exts = ('.wav', '.flac', '.mp3', '.ogg')
        self.files = [p for p in self.data_root.rglob('*') if p.suffix.lower() in exts]
        self.files = sorted(self.files)
        
        if not self.files:
            raise ValueError(f"No audio files found in {data_root}")
        
        print(f"Found {len(self.files)} audio files")
    
    def __iter__(self):
        """Yield random chunks from dataset."""
        worker_info = torch.utils.data.get_worker_info()
        
        for _ in range(self.epoch_size):
            # Random file
            audio_path = self.files[np.random.randint(0, len(self.files))]
            
            try:
                audio, sr = sf.read(audio_path)
                
                # Ensure mono
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Resample if needed
                if sr != self.sample_rate:
                    audio = np.interp(
                        np.linspace(0, len(audio), int(len(audio) * self.sample_rate / sr)),
                        np.arange(len(audio)),
                        audio
                    )
                
                # Random chunk within bounds
                if len(audio) > self.chunk_size:
                    start = np.random.randint(0, len(audio) - self.chunk_size)
                    chunk = audio[start:start + self.chunk_size]
                else:
                    # Pad if too short
                    chunk = np.pad(audio, (0, self.chunk_size - len(audio)), mode='constant')
                
                # Normalize to [-1, 1]
                chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)
                
                yield torch.FloatTensor(chunk).unsqueeze(0)  # (1, T)
            
            except Exception as e:
                # Skip corrupted files
                continue
    
    def __len__(self):
        return self.epoch_size


def load_entropy_model(entropy_ckpt_path, device='cpu'):
    """Load pre-trained entropy model."""
    checkpoint = torch.load(entropy_ckpt_path, map_location=device)
    
    latent_dim = checkpoint.get('latent_dim')
    num_components = checkpoint.get('num_components', 8)
    model_type = checkpoint.get('model_type', 'global_gmm')
    
    entropy_model = EntropyBottleneck(latent_dim, num_components, model_type).to(device)
    entropy_model.load_state_dict(checkpoint['entropy_model_state_dict'])
    entropy_model.eval()
    
    # Freeze entropy model parameters (we don't train it further)
    for param in entropy_model.parameters():
        param.requires_grad = False
    
    return entropy_model


def finetune_rate_distortion(args):
    """Fine-tune neural codec with R-D loss."""
    
    print(f"\n{'='*80}")
    print(f"PHASE 3b: RATE-DISTORTION TRAINING")
    print(f"{'='*80}")
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Entropy model: {args.entropy_model}")
    print(f"Output dir: {args.output}")
    print(f"Lambda schedule: {args.lambda_start} -> {args.lambda_end}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*80}\n")
    
    device = args.device
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    print(f"Loading base model from {args.base_checkpoint}...")
    base_ckpt = torch.load(args.base_checkpoint, map_location=device)
    base_state = base_ckpt if not isinstance(base_ckpt, dict) or 'model_state_dict' not in base_ckpt else base_ckpt.get('model_state_dict')
    
    # Infer architecture
    d_model = 256
    n_layers = 4
    n_heads = 8
    
    qkv_key = 'encoder.transformer_blocks.0.attention.qkv.weight'
    if qkv_key in base_state:
        d_model = base_state[qkv_key].shape[1]
    
    layer_indices = set()
    for key in base_state.keys():
        if 'encoder.transformer_blocks.' in key:
            parts = key.split('.')
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))
    if layer_indices:
        n_layers = max(layer_indices) + 1
    
    model = NeuralAudioCodec(
        sample_rate=args.sample_rate,
        hop_length=160,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        window_size=256,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(base_state)
    
    print(f"Loaded model: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    
    # Load entropy model
    print(f"Loading entropy model from {args.entropy_model}...")
    entropy_model = load_entropy_model(args.entropy_model, device)
    print(f"Entropy model loaded: latent_dim={entropy_model.latent_dim}")
    
    # Create R-D loss
    rd_loss = RateDistortionLoss(
        distortion_type=args.distortion_type,
        entropy_model=entropy_model,
        lambda_init=args.lambda_start,
        lambda_min=args.lambda_end
    )
    
    # Dataset
    print(f"Loading dataset from {args.data_root}...")
    dataset = AudioDataset(
        args.data_root,
        chunk_seconds=args.chunk_sec,
        sample_rate=args.sample_rate,
        epoch_size=args.samples_per_epoch
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    history = {
        'epoch': [],
        'train_loss': [],
        'distortion': [],
        'rate_penalty': [],
        'lambda': [],
        'learning_rate': [],
    }
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Update lambda schedule
        rd_loss.schedule_lambda(epoch, args.epochs, schedule=args.lambda_schedule)
        
        # Train
        model.train()
        train_loss = 0.0
        train_distortion = 0.0
        train_rate = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", 
                   total=args.samples_per_epoch // args.batch_size)
        
        for batch_idx, x_batch in enumerate(pbar):
            x_batch = x_batch.to(device)  # (B, 1, T) - already has channel dim from dataset
            
            # Don't add another unsqueeze(1) - dataset already yields (1, T) which becomes (B, 1, T)
            
            optimizer.zero_grad()
            
            # Forward pass
            z = model.encoder(x_batch)  # (B, D, T)
            x_recon = model.decoder(z)  # (B, 1, T)
            
            # Compute R-D loss
            loss_dict = rd_loss(x_recon, x_batch, z, return_components=True)
            loss = loss_dict['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss_dict['loss']
            train_distortion += loss_dict['distortion']
            train_rate += loss_dict['rate_penalty']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'D': f"{loss_dict['distortion']:.4f}",
                'λR': f"{loss_dict['rate_penalty']:.4f}"
            })
            
            # Break after epoch_size samples
            if batch_idx * args.batch_size >= args.samples_per_epoch:
                break
        
        # Average losses
        train_loss /= max(num_batches, 1)
        train_distortion /= max(num_batches, 1)
        train_rate /= max(num_batches, 1)
        
        scheduler.step()
        
        # Logging
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['distortion'].append(train_distortion)
        history['rate_penalty'].append(train_rate)
        history['lambda'].append(rd_loss.current_lambda)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Loss: {train_loss:.6f} = D: {train_distortion:.6f} + λR: {train_rate:.6f}")
        print(f"  λ: {rd_loss.current_lambda:.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'd_model': d_model,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'entropy_model_path': str(args.entropy_model),
                'train_loss': train_loss,
            }
            checkpoint_path = args.output / 'best.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved best checkpoint to {checkpoint_path}")
        
        # Also save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            epoch_ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'd_model': d_model,
                'n_layers': n_layers,
                'n_heads': n_heads,
            }
            epoch_ckpt_path = args.output / f'epoch_{epoch+1:02d}.pt'
            torch.save(epoch_ckpt, epoch_ckpt_path)
    
    # Save training history
    history_path = args.output / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nSaved training history to {history_path}")
    print(f"Training complete! Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to {args.output}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3b: Rate-Distortion training with learned entropy model"
    )
    parser.add_argument('--base-checkpoint', type=Path, required=True,
                        help='Path to base neural codec checkpoint (Phase 4)')
    parser.add_argument('--entropy-model', type=Path, required=True,
                        help='Path to trained entropy model checkpoint')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Root directory of audio files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output directory for fine-tuned model')
    parser.add_argument('--chunk-sec', type=float, default=2.0,
                        help='Chunk duration in seconds')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Sample rate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--samples-per-epoch', type=int, default=5000,
                        help='Samples per epoch (default: 5000)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--lambda-start', type=float, default=1.0,
                        help='Initial lambda (rate weight)')
    parser.add_argument('--lambda-end', type=float, default=0.001,
                        help='Final lambda (rate weight)')
    parser.add_argument('--lambda-schedule', type=str, default='linear',
                        choices=['linear', 'exponential', 'step'],
                        help='Lambda schedule type')
    parser.add_argument('--distortion-type', type=str, default='hybrid',
                        choices=['mse', 'l1', 'stft', 'hybrid'],
                        help='Distortion metric')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    if args.data_root is None:
        args.data_root = Path(get_dataset_paths()["test_clean"])
    
    finetune_rate_distortion(args)


if __name__ == '__main__':
    main()
