#!/usr/bin/env python3
"""
Train entropy model on latent distribution from trained neural codec.
Goal: Learn p(z) to enable arithmetic coding instead of zlib.
Expected gain: 2-3x bitrate reduction (141 kbps -> 50-70 kbps).
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
from torch.utils.data import DataLoader, TensorDataset
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.entropy_model import create_entropy_model
from src.paths import get_dataset_paths


class LatentDataset:
    """Extract and cache latent codes from dataset."""
    
    def __init__(self, checkpoint_path, data_root, max_files=None, 
                 chunk_seconds=2.0, sample_rate=16000, device='cpu'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            data_root: Path to dataset directory
            max_files: Maximum files to process
            chunk_seconds: Audio chunk duration for latent extraction
            sample_rate: Sample rate
            device: Device to use
        """
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_size = int(chunk_seconds * sample_rate)
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint else checkpoint.get('model_state_dict')
        
        # Infer architecture from state_dict
        d_model = 256
        n_layers = 4
        n_heads = 8
        
        # Try to infer d_model from qkv weight shape
        qkv_key = 'encoder.transformer_blocks.0.attention.qkv.weight'
        if qkv_key in state_dict:
            qkv_shape = state_dict[qkv_key].shape
            d_model = qkv_shape[1]  # (3*d_model, d_model)
        
        # Try to infer n_layers from transformer_blocks
        layer_indices = set()
        for key in state_dict.keys():
            if 'encoder.transformer_blocks.' in key:
                parts = key.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    layer_indices.add(int(parts[2]))
        if layer_indices:
            n_layers = max(layer_indices) + 1
        
        self.model = NeuralAudioCodec(
            sample_rate=sample_rate,
            hop_length=checkpoint.get('hop_length', 160),
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=checkpoint.get('window_size', 256),
            dropout=0.0,
        ).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Extract latents from dataset
        print(f"Extracting latents from {data_root}...")
        self.latents = self._extract_latents(data_root, max_files)
        
        if len(self.latents) == 0:
            raise ValueError(f"No latents extracted from {data_root}")
        
        print(f"Extracted {len(self.latents)} latent vectors")
        print(f"  Latent shape: {self.latents[0].shape}")
        print(f"  Min: {np.min(self.latents[0]):.4f}, Max: {np.max(self.latents[0]):.4f}")
    
    def _extract_latents(self, data_root, max_files):
        """Extract latents from audio files."""
        exts = ('.wav', '.flac', '.mp3', '.ogg')
        files = [p for p in Path(data_root).rglob('*') if p.suffix.lower() in exts]
        files = sorted(files)
        
        if max_files:
            files = files[:max_files]
        
        latents = []
        
        with torch.no_grad():
            for audio_path in tqdm(files, desc="Extracting latents"):
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
                    
                    # Extract latents from chunks
                    for start in range(0, len(audio) - self.chunk_size, self.chunk_size):
                        chunk = audio[start:start + self.chunk_size]
                        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(self.device)
                        
                        latent = self.model.encoder(chunk_tensor)
                        latents.append(latent.squeeze(0).cpu().numpy())  # (D, T)
                
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}")
                    continue
        
        return latents
    
    def get_batched_tensor(self, batch_size=32):
        """Get all latents as batched tensor."""
        # Flatten latents: (N, D, T) -> (N * T, D)
        latent_list = []
        for lat in self.latents:
            if len(lat.shape) == 2:  # (D, T)
                latent_list.extend(lat.T)  # (T, D)
            else:  # (D,) single frame
                latent_list.append(lat)
        
        latent_tensor = torch.from_numpy(np.array(latent_list, dtype=np.float32))
        return DataLoader(
            TensorDataset(latent_tensor),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )


def train_entropy_model(args):
    """Train entropy model on latent distribution."""
    
    print(f"\n{'='*80}")
    print(f"ENTROPY MODEL TRAINING")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output}")
    print(f"Model type: {args.model_type}")
    print(f"Num components: {args.num_components}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*80}\n")
    
    device = args.device
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load and extract latents
    dataset = LatentDataset(
        args.checkpoint,
        args.data_root,
        max_files=args.max_files,
        chunk_seconds=args.chunk_sec,
        sample_rate=args.sample_rate,
        device=device
    )
    
    dataloader = dataset.get_batched_tensor(batch_size=args.batch_size)
    latent_dim = dataset.latents[0].shape[0] if len(dataset.latents[0].shape) > 1 else 1
    
    print(f"\nLatent dimension: {latent_dim}")
    print(f"Dataset size: {len(dataloader) * args.batch_size} latent vectors")
    
    # Create entropy model
    entropy_model = create_entropy_model(
        latent_dim=latent_dim,
        num_components=args.num_components,
        model_type=args.model_type,
        device=device
    )
    
    print(f"Created entropy model: {args.model_type}")
    print(f"  Components: {args.num_components}")
    print(f"  Latent dim: {latent_dim}")
    
    # Optimizer
    optimizer = optim.Adam(entropy_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    best_loss = float('inf')
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    for epoch in range(args.epochs):
        # Train
        entropy_model.train()
        train_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (z_batch,) in enumerate(pbar):
            z_batch = z_batch.to(device)
            
            optimizer.zero_grad()
            loss = entropy_model.entropy_loss(z_batch, beta=1.0)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        train_loss /= len(dataloader)
        scheduler.step()
        
        # Logging
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        print(f"\nEpoch {epoch+1}/{args.epochs}: loss={train_loss:.6f}, lr={scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                'epoch': epoch + 1,
                'entropy_model_state_dict': entropy_model.state_dict(),
                'latent_dim': latent_dim,
                'num_components': args.num_components,
                'model_type': args.model_type,
                'train_loss': train_loss,
            }
            checkpoint_path = args.output / 'best.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved best checkpoint to {checkpoint_path}")
    
    # Save training history
    history_path = args.output / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nSaved training history to {history_path}")
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    
    return entropy_model


def main():
    parser = argparse.ArgumentParser(
        description="Train entropy model on neural codec latents"
    )
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to trained neural codec checkpoint')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Root directory of audio files (default: LibriSpeech test-clean)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output directory for trained entropy model')
    parser.add_argument('--max-files', type=int, default=50,
                        help='Maximum files to use for training (default: 50)')
    parser.add_argument('--chunk-sec', type=float, default=2.0,
                        help='Chunk duration in seconds (default: 2.0)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Sample rate (default: 16000)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--model-type', type=str, default='global_gmm',
                        choices=['global_gmm', 'factorized_gmm'],
                        help='Entropy model type (default: global_gmm)')
    parser.add_argument('--num-components', type=int, default=8,
                        help='Number of mixture components (default: 8)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.data_root is None:
        args.data_root = Path(get_dataset_paths()["test_clean"])
    
    train_entropy_model(args)


if __name__ == '__main__':
    main()
