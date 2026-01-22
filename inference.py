"""
Inference script for Neural Audio Codec
Load a trained model and test on audio files
"""

import torch
import torchaudio
import argparse
from pathlib import Path
from model import NeuralAudioCodec
import time


def save_audio(filepath, waveform, sample_rate):
    """
    Save audio with fallback options for different backends.
    Handles common torchaudio backend issues on Windows.
    """
    try:
        # Try soundfile backend first (most reliable on Windows)
        torchaudio.save(filepath, waveform, sample_rate, backend="soundfile")
    except:
        try:
            # Try sox backend
            torchaudio.save(filepath, waveform, sample_rate, backend="sox")
        except:
            # Fallback to scipy if torchaudio fails
            import scipy.io.wavfile as wavfile
            # Convert to numpy and scale to int16
            audio_np = waveform.squeeze().numpy()
            audio_np = (audio_np * 32767).astype('int16')
            wavfile.write(filepath, sample_rate, audio_np)


def load_audio(audio_path, sample_rate=16000):
    """Load and preprocess audio file"""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform, sample_rate


def process_audio(model, audio, device, chunk_size=16000):
    """
    Process audio through the codec.
    Supports chunked processing for long audio.
    """
    model.eval()
    
    # Add batch dimension if needed
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)  # (1, 1, time)
    
    audio = audio.to(device)
    
    with torch.no_grad():
        if audio.shape[2] <= chunk_size:
            # Process entire audio
            reconstructed = model(audio)
        else:
            # Process in chunks
            chunks = []
            for i in range(0, audio.shape[2], chunk_size):
                chunk = audio[:, :, i:i+chunk_size]
                
                # Pad last chunk if necessary
                if chunk.shape[2] < chunk_size:
                    pad_size = chunk_size - chunk.shape[2]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                
                chunk_reconstructed = model(chunk)
                
                # Remove padding from last chunk
                if i + chunk_size > audio.shape[2]:
                    chunk_reconstructed = chunk_reconstructed[:, :, :-(i + chunk_size - audio.shape[2])]
                
                chunks.append(chunk_reconstructed)
            
            reconstructed = torch.cat(chunks, dim=2)
    
    return reconstructed


def calculate_metrics(original, reconstructed):
    """Calculate basic audio quality metrics"""
    # Ensure same length
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]
    
    # SNR (Signal-to-Noise Ratio)
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(original - reconstructed))
    
    return {
        'SNR (dB)': snr.item(),
        'MAE': mae.item()
    }


def main():
    parser = argparse.ArgumentParser(description='Neural Audio Codec Inference')
    parser.add_argument('--input', type=str, required=True, help='Input audio file')
    parser.add_argument('--output', type=str, default='output.wav', help='Output audio file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = NeuralAudioCodec(
        sample_rate=args.sample_rate,
        hop_length=160,
        d_model=512,
        n_layers=8,
        n_heads=16,
        window_size=512,
        dropout=0.0  # No dropout during inference
    ).to(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load audio
    print(f"Loading audio from {args.input}...")
    audio, sr = load_audio(args.input, args.sample_rate)
    print(f"Audio shape: {audio.shape}, Duration: {audio.shape[1] / sr:.2f}s")
    
    # Process
    print("Processing audio...")
    start_time = time.time()
    reconstructed = process_audio(model, audio, args.device)
    elapsed_time = time.time() - start_time
    
    # Calculate Real-Time Factor (RTF)
    audio_duration = audio.shape[1] / sr
    rtf = elapsed_time / audio_duration
    print(f"Processing time: {elapsed_time:.2f}s for {audio_duration:.2f}s audio")
    print(f"Real-Time Factor (RTF): {rtf:.3f}")
    
    # Calculate metrics
    print("\nQuality Metrics:")
    metrics = calculate_metrics(audio.to(args.device), reconstructed.squeeze(0))
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save output
    print(f"\nSaving output to {args.output}...")
    reconstructed = reconstructed.squeeze(0).cpu()  # Remove batch dimension
    save_audio(args.output, reconstructed, sr)
    print("Done!")


def test_model_without_args():
    """
    Test function for running without command-line arguments.
    Useful for quick testing or notebook usage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = NeuralAudioCodec(
        sample_rate=16000,
        hop_length=160,
        d_model=512,
        n_layers=8,
        n_heads=16,
        window_size=512,
        dropout=0.0
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Estimated latency: {model.get_latency_ms():.2f} ms")
    
    # Test with synthetic audio
    print("\nTesting with synthetic audio (1 second)...")
    sample_rate = 16000
    duration = 1.0
    
    # Generate test signal (440 Hz sine wave)
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).unsqueeze(0)  # (1, 1, time)
    audio = audio.to(device)
    
    # Process
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        
        # Encode
        latent = model.encode(audio)
        encode_time = time.time() - start_time
        
        # Decode
        start_time = time.time()
        reconstructed = model.decode(latent)
        decode_time = time.time() - start_time
    
    print(f"Input shape: {audio.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Output shape: {reconstructed.shape}")
    print(f"\nTiming:")
    print(f"  Encode time: {encode_time*1000:.2f} ms")
    print(f"  Decode time: {decode_time*1000:.2f} ms")
    print(f"  Total time: {(encode_time + decode_time)*1000:.2f} ms")
    print(f"  RTF: {(encode_time + decode_time) / duration:.3f}")
    
    # Calculate metrics
    metrics = calculate_metrics(audio, reconstructed)
    print("\nQuality Metrics (untrained model):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Calculate compression
    original_bits = audio.numel() * 32  # 32-bit float
    latent_bits = latent.numel() * 32
    compression_ratio = original_bits / latent_bits
    
    print(f"\nCompression:")
    print(f"  Original: {original_bits / 8 / 1024:.2f} KB")
    print(f"  Latent: {latent_bits / 8 / 1024:.2f} KB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    # Save test outputs
    print("\nSaving test audio files...")
    save_audio('test_original.wav', audio.squeeze(0).cpu(), sample_rate)
    save_audio('test_reconstructed.wav', reconstructed.squeeze(0).cpu(), sample_rate)
    print("Saved: test_original.wav, test_reconstructed.wav")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run with command-line arguments
        main()
    else:
        # Run test without arguments
        print("Running test mode (no arguments provided)...")
        print("To use with audio file, run: python inference.py --input audio.wav --checkpoint model.pt --output output.wav")
        print("-" * 80)
        test_model_without_args()