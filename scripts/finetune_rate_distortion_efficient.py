#!/usr/bin/env python3
"""
Phase 3b: Memory-Efficient Rate-Distortion Training
Uses mixed precision, smaller chunks, and gradient accumulation.
Designed to run on 24GB GPUs (V100/RTX 4090).
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
from torch.cuda.amp import autocast, GradScaler
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
    
    def __init__(self, data_root, chunk_seconds=1.0, sample_rate=16000, epoch_size=1000):
        """
        Args:
            data_root: Path to audio files
            chunk_seconds: Duration of audio chunks (reduce for memory)
            sample_rate: Sample rate
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
            
            except Exception:
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
    
    # Freeze entropy model
    for param in entropy_model.parameters():
        param.requires_grad = False
    
    return entropy_model


def finetune_rate_distortion_efficient(args):
    """Memory-efficient R-D fine-tuning."""
    
    print(f"\n{'='*80}")
    print(f"PHASE 3b: MEMORY-EFFICIENT RATE-DISTORTION TRAINING")
    print(f"{'='*80}")
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Rate penalty: L2 norm (simple bitrate proxy)")
    print(f"Output dir: {args.output}")
    print(f"Chunk duration: {args.chunk_sec}s (reduced for memory)")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.grad_accum_steps}")
    print(f"Mixed precision: {args.use_amp}")
    print(f"Freeze encoder: {args.freeze_encoder}")
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
    
    # Freeze encoder if requested (only train decoder)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen - only fine-tuning decoder")
    
    # Load entropy model (optional - using L2 norm for rate instead)
    print(f"Using L2 norm-based rate penalty (no entropy model required)")
    entropy_model = None
    
    # Note: Keep model in float32 for stable gradients, autocast will handle precision in forward pass
    
    # Create R-D loss with simple L2 rate penalty
    rd_loss = RateDistortionLoss(
        distortion_type=args.distortion_type,
        entropy_model=None,  # Use L2 norm instead
        lambda_init=args.lambda_start,
        lambda_min=args.lambda_end,
        encoder_dim=None  # No projection needed
    ).to(device)
    
    # Dataset
    print(f"Loading dataset from {args.data_root}...")
    dataset = AudioDataset(
        args.data_root,
        chunk_seconds=args.chunk_sec,
        sample_rate=args.sample_rate,
        epoch_size=args.samples_per_epoch
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    
    # Optimizer - only model parameters (no projection layer needed)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Mixed precision scaler
    scaler = None  # GradScaler not needed with simple autocast
    
    # Training loop
    history = {
        'epoch': [],
        'train_loss': [],
        'distortion': [],
        'lambda': [],
        'learning_rate': [],
    }
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Update lambda
        rd_loss.schedule_lambda(epoch, args.epochs, schedule=args.lambda_schedule)
        
        # Train
        model.train()
        train_loss = 0.0
        train_distortion = 0.0
        train_rate = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}",
                   total=args.samples_per_epoch // args.batch_size)
        
        optimizer.zero_grad()
        
        for batch_idx, x_batch in enumerate(pbar):
            if x_batch.shape[0] == 0:  # Skip empty batches
                continue
            
            x_batch = x_batch.to(device)
            
            # Forward pass with mixed precision
            if args.use_amp:
                with autocast(dtype=torch.float16):
                    z = model.encoder(x_batch)
                    x_recon = model.decoder(z)
                    loss_dict = rd_loss(x_recon, x_batch, z, return_components=True)
                    loss = loss_dict['loss'] / args.grad_accum_steps
            else:
                z = model.encoder(x_batch)
                x_recon = model.decoder(z)
                loss_dict = rd_loss(x_recon, x_batch, z, return_components=True)
                loss = loss_dict['loss'] / args.grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss_dict['loss'].item()
            train_distortion += loss_dict['distortion']
            train_rate += loss_dict.get('rate_kbps', 0)
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'D': f"{loss_dict['distortion']:.4f}",
                'R': f"{loss_dict.get('rate_kbps', 0):.2f}kbps",
                'λ': f"{rd_loss.current_lambda:.4f}"
            })
            
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
        history['lambda'].append(rd_loss.current_lambda)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Loss: {train_loss:.6f}, D: {train_distortion:.6f}, R: {train_rate:.2f} kbps")
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
        
        # Periodic checkpoint
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


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3b: Memory-efficient R-D training"
    )
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--entropy-model', type=Path, required=False, default=None,
                        help='Entropy model checkpoint (optional, uses L2 norm if not provided)')
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-sec', type=float, default=1.0,
                        help='Chunk duration (reduce for memory, default: 1.0s)')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum-steps', type=int, default=4,
                        help='Gradient accumulation steps (simulates larger batch)')
    parser.add_argument('--samples-per-epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lambda-start', type=float, default=0.01)
    parser.add_argument('--lambda-end', type=float, default=0.0001)
    parser.add_argument('--lambda-schedule', type=str, default='linear')
    parser.add_argument('--distortion-type', type=str, default='mse')
    parser.add_argument('--use-amp', type=bool, default=True,
                        help='Use automatic mixed precision (float16)')
    parser.add_argument('--freeze-encoder', type=bool, default=False,
                        help='Freeze encoder, only train decoder')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    if args.data_root is None:
        args.data_root = Path(get_dataset_paths()["test_clean"])
    
    finetune_rate_distortion_efficient(args)


if __name__ == '__main__':
    main()
