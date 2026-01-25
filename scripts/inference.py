#!/usr/bin/env python3
"""
Inference Script for Neural Audio Codec
Load a trained model and compress/decompress audio
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchaudio
import soundfile as sf
import yaml
from pathlib import Path
from src.model import NeuralAudioCodec
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(checkpoint_path, config_path='config/training.yaml'):
    """Load trained model from checkpoint"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    
    # Create model
    model = NeuralAudioCodec(
        sample_rate=model_cfg['sample_rate'],
        hop_length=model_cfg['hop_length'],
        d_model=model_cfg['d_model'],
        n_layers=model_cfg['n_layers'],
        n_heads=model_cfg['n_heads'],
        window_size=model_cfg['window_size'],
        dropout=0.0  # No dropout for inference
    )
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    
    logger.info(f"✓ Loaded model from {checkpoint_path}")
    return model, device, config

def compress_audio(model, audio_path, device, sample_rate=16000):
    """Encode audio to latent representation"""
    
    # Load audio with soundfile (supports FLAC, WAV, etc.)
    waveform, sr = sf.read(audio_path, dtype='float32', always_2d=False)
    
    # Convert to tensor and handle dimensions
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    
    # Handle stereo -> mono
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Normalize
    max_val = torch.abs(waveform).max()
    if max_val > 1e-6:
        waveform = waveform / (max_val + 1e-8)
    
    # Process in chunks to avoid OOM - use only first 2 seconds
    chunk_samples = min(sample_rate * 2, waveform.shape[1])  # 2 seconds max
    waveform = waveform[:, :chunk_samples]
    
    waveform = waveform.unsqueeze(0).to(device)  # Add batch dimension
    
    # Encode
    with torch.no_grad():
        latent = model.encoder(waveform)
    
    logger.info(f"✓ Compressed: {waveform.shape} -> {latent.shape}")
    return latent, waveform

def decompress_audio(model, latent, device):
    """Decode latent representation to audio"""
    
    with torch.no_grad():
        reconstructed = model.decoder(latent)
    
    logger.info(f"✓ Decompressed: {latent.shape} -> {reconstructed.shape}")
    return reconstructed.squeeze(0)

def full_codec(model, audio_path, output_path, device, sample_rate=16000):
    """Full compression-decompression pipeline"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Audio Codec Processing")
    logger.info(f"{'='*80}\n")
    
    # Compress
    latent, original = compress_audio(model, audio_path, device, sample_rate)
    
    # Decompress
    reconstructed = decompress_audio(model, latent, device)
    
    # Save output using soundfile instead of torchaudio
    reconstructed_np = reconstructed.cpu().numpy()
    if reconstructed_np.ndim == 2:
        reconstructed_np = reconstructed_np[0]
    sf.write(output_path, reconstructed_np, sample_rate)
    logger.info(f"✓ Saved reconstructed audio to {output_path}")
    
    # Calculate metrics
    original = original.squeeze(0).cpu()
    reconstructed_cpu = reconstructed.cpu()
    
    # Ensure same length
    min_len = min(original.shape[1], reconstructed_cpu.shape[1])
    original = original[:, :min_len]
    reconstructed_cpu = reconstructed_cpu[:, :min_len]
    
    # SNR (Signal-to-Noise Ratio)
    noise = original - reconstructed_cpu
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    logger.info(f"\n{'-'*80}")
    logger.info(f"Metrics:")
    logger.info(f"{'-'*80}")
    logger.info(f"SNR: {snr:.2f} dB")
    logger.info(f"Compression ratio: {original.numel() / latent.numel():.2f}x")
    logger.info(f"{'-'*80}\n")

def main():
    parser = argparse.ArgumentParser(description='Neural Audio Codec Inference')
    parser.add_argument('--audio', type=str, required=True, help='Input audio file path')
    parser.add_argument('--output', type=str, required=True, help='Output audio file path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt', 
                        help='Model checkpoint path')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                        help='Config file path')
    
    args = parser.parse_args()
    
    # Load model
    model, device, config = load_model(args.checkpoint, args.config)
    sample_rate = config['model']['sample_rate']
    
    # Process audio
    full_codec(model, args.audio, args.output, device, sample_rate)

if __name__ == "__main__":
    main()
