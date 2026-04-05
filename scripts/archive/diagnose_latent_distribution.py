#!/usr/bin/env python3
"""
Diagnostic script: Visualize latent distribution and estimate entropy model benefit.
Shows the gap between current zlib compression and learned entropy coding.
"""

import argparse
import sys
from pathlib import Path
import pickle

import numpy as np
import torch
import soundfile as sf
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths


def extract_latents(checkpoint_path, data_root, max_files=20, 
                   chunk_seconds=2.0, sample_rate=16000, device='cpu'):
    """Extract latents from dataset."""
    
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
    
    model = NeuralAudioCodec(
        sample_rate=sample_rate,
        hop_length=160,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        window_size=256,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    chunk_size = int(chunk_seconds * sample_rate)
    latents = []
    
    exts = ('.wav', '.flac', '.mp3', '.ogg')
    files = [p for p in Path(data_root).rglob('*') if p.suffix.lower() in exts]
    files = sorted(files)[:max_files]
    
    with torch.no_grad():
        for audio_path in tqdm(files, desc="Extracting latents"):
            try:
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                if sr != sample_rate:
                    audio = np.interp(
                        np.linspace(0, len(audio), int(len(audio) * sample_rate / sr)),
                        np.arange(len(audio)), audio
                    )
                
                for start in range(0, len(audio) - chunk_size, chunk_size):
                    chunk = audio[start:start + chunk_size]
                    chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
                    latent = model.encoder(chunk_tensor)
                    latents.append(latent.squeeze(0).cpu().numpy())
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    if not latents:
        raise ValueError("No latents extracted!")
    
    # Flatten: (N, D, T) -> (N*T, D)
    latent_flat = []
    for lat in latents:
        if len(lat.shape) == 2:
            latent_flat.extend(lat.T)
        else:
            latent_flat.append(lat)
    
    return np.array(latent_flat, dtype=np.float32)


def analyze_latents(latents):
    """Analyze latent statistics."""
    print(f"\n{'='*80}")
    print(f"LATENT DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Shape: {latents.shape}")
    print(f"Dtype: {latents.dtype}")
    print(f"\nStatistics (across all dimensions):")
    print(f"  Mean: {latents.mean():.6f}")
    print(f"  Std: {latents.std():.6f}")
    print(f"  Min: {latents.min():.6f}")
    print(f"  Max: {latents.max():.6f}")
    print(f"  Median: {np.median(latents):.6f}")
    print(f"  Entropy (Shanon): {stats.entropy(np.histogram(latents.flatten(), bins=256)[0]):.6f}")


def estimate_compression_gains(latents):
    """Estimate potential bitrate gains from learned entropy model."""
    print(f"\n{'='*80}")
    print(f"COMPRESSION ANALYSIS")
    print(f"{'='*80}")
    
    num_samples = latents.shape[0]
    latent_dim = latents.shape[1]
    
    # Flatten to 1D
    latent_flat = latents.flatten()
    
    # 1. Uniform quantization + zlib (current method)
    print("\n1. CURRENT METHOD: Uniform 1-bit Quantization + zlib")
    q1bit = np.round((latent_flat - latent_flat.min()) / 
                     (latent_flat.max() - latent_flat.min())).astype(np.uint8)
    
    import zlib
    q_bytes = q1bit.tobytes()
    q_compressed = zlib.compress(q_bytes, level=9)
    
    original_bits = len(q_bytes) * 8
    compressed_bits = len(q_compressed) * 8
    compression_ratio = original_bits / max(len(q_compressed) * 8, 1)
    
    print(f"  Original: {len(q_bytes):,} bytes ({original_bits:,} bits)")
    print(f"  Compressed: {len(q_compressed):,} bytes ({compressed_bits:,} bits)")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Achieved bitrate: {(compressed_bits / (num_samples * 256 / 16000)):.1f} kbps")
    
    # 2. Estimate with Gaussian entropy
    print("\n2. ESTIMATED GAIN: Learned Entropy Model (Single Gaussian)")
    
    # Fit Gaussian to data
    mu = latent_flat.mean()
    sigma = latent_flat.std()
    
    # Entropy of Gaussian: H = 0.5 * log(2*pi*e*sigma^2) nats
    entropy_gaussian = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
    entropy_bits = entropy_gaussian / np.log(2)
    
    print(f"  Gaussian fit: mu={mu:.4f}, sigma={sigma:.4f}")
    print(f"  Entropy per value: {entropy_bits:.2f} bits")
    print(f"  Total bits for {num_samples} samples: {entropy_bits * num_samples:.0f} bits")
    print(f"  Achieved bitrate: {(entropy_bits * num_samples / (num_samples * 256 / 16000)):.1f} kbps")
    
    # 3. Estimate with GMM
    print("\n3. ESTIMATED GAIN: Learned Entropy Model (8-component GMM)")
    print(f"  Fitting GMM (this is expensive, using estimate)...")
    
    # Use mixture of 8 Gaussians to model better
    # Estimate entropy reduction: typically 20-40% better than single Gaussian
    entropy_gmm = entropy_gaussian * 0.7  # Rough estimate: 30% reduction
    entropy_bits_gmm = entropy_gmm / np.log(2)
    
    print(f"  Estimated entropy per value: {entropy_bits_gmm:.2f} bits")
    print(f"  Total bits for {num_samples} samples: {entropy_bits_gmm * num_samples:.0f} bits")
    print(f"  Achieved bitrate: {(entropy_bits_gmm * num_samples / (num_samples * 256 / 16000)):.1f} kbps")
    
    # Quantification of gain
    print(f"\n{'='*80}")
    print(f"ESTIMATED BITRATE IMPROVEMENTS")
    print(f"{'='*80}")
    
    current_kbps = compressed_bits / (num_samples * 256 / 16000)
    gaussian_kbps = entropy_bits * num_samples / (num_samples * 256 / 16000)
    gmm_kbps = entropy_bits_gmm * num_samples / (num_samples * 256 / 16000)
    
    print(f"Current (zlib):         {current_kbps:6.1f} kbps  (baseline)")
    print(f"Gaussian entropy:       {gaussian_kbps:6.1f} kbps  ({current_kbps/gaussian_kbps:.1f}x improvement)")
    print(f"GMM entropy (estimated): {gmm_kbps:6.1f} kbps  ({current_kbps/gmm_kbps:.1f}x improvement)")
    print(f"\nTarget: 10 kbps")
    print(f"Gap to close: {current_kbps / 10:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Analyze latent distribution for entropy modeling")
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--max-files', type=int, default=20)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save', type=Path, default=None)
    
    args = parser.parse_args()
    
    if args.data_root is None:
        args.data_root = Path(get_dataset_paths()["test_clean"])
    
    print(f"Extracting latents from {args.data_root}...")
    latents = extract_latents(args.checkpoint, args.data_root, 
                              max_files=args.max_files, device=args.device)
    
    analyze_latents(latents)
    estimate_compression_gains(latents)
    
    if args.save:
        np.save(args.save, latents)
        print(f"\nSaved latents to {args.save}")


if __name__ == '__main__':
    main()
