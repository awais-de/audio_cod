"""
Create Demonstration Materials for Project Report
Generates audio samples, spectrograms, and visualizations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from src.model import NeuralAudioCodec

def create_spectrogram_comparison(original, reconstructed, sr, output_path):
    """Create side-by-side spectrogram comparison"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original
    D1 = np.abs(np.fft.rfft(original))
    ax1.semilogy(D1[:1000])
    ax1.set_title('Original Audio - Spectrum', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frequency Bin')
    ax1.set_ylabel('Magnitude (log scale)')
    ax1.grid(True, alpha=0.3)
    
    # Reconstructed
    D2 = np.abs(np.fft.rfft(reconstructed))
    ax2.semilogy(D2[:1000])
    ax2.set_title('Reconstructed Audio - Spectrum', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frequency Bin')
    ax2.set_ylabel('Magnitude (log scale)')
    ax2.grid(True, alpha=0.3)
    
    # Difference
    diff = np.abs(D1 - D2)
    ax3.semilogy(diff[:1000])
    ax3.set_title('Reconstruction Error - Spectrum', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Frequency Bin')
    ax3.set_ylabel('Error Magnitude (log scale)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved spectrogram: {output_path}")

def create_waveform_comparison(original, reconstructed, sr, output_path, duration=2.0):
    """Create waveform comparison"""
    samples = int(duration * sr)
    orig_clip = original[:samples]
    recon_clip = reconstructed[:samples]
    time_axis = np.arange(len(orig_clip)) / sr
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8))
    
    # Original
    ax1.plot(time_axis, orig_clip, linewidth=0.5, color='blue', alpha=0.7)
    ax1.set_title('Original Audio Waveform', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim([0, duration])
    ax1.grid(True, alpha=0.3)
    
    # Reconstructed
    ax2.plot(time_axis, recon_clip, linewidth=0.5, color='red', alpha=0.7)
    ax2.set_title('Reconstructed Audio Waveform', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim([0, duration])
    ax2.grid(True, alpha=0.3)
    
    # Overlay
    ax3.plot(time_axis, orig_clip, linewidth=0.5, color='blue', alpha=0.5, label='Original')
    ax3.plot(time_axis, recon_clip, linewidth=0.5, color='red', alpha=0.5, label='Reconstructed')
    ax3.set_title('Overlay Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Amplitude')
    ax3.set_xlim([0, duration])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved waveform: {output_path}")

def generate_demonstrations(checkpoint_path, audio_files, output_dir, device='cuda'):
    """Generate all demonstration materials"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("GENERATING DEMONSTRATION MATERIALS")
    print("=" * 80)
    print()
    
    # Load model
    print("🔧 Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = NeuralAudioCodec(
        d_model=checkpoint.get('d_model', 256),
        n_layers=checkpoint.get('n_layers', 4),
        n_heads=checkpoint.get('n_heads', 8),
        window_size=checkpoint.get('window_size', 256),
        dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded")
    print()
    
    sample_rate = 16000
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"📄 Processing file {i}/{len(audio_files)}: {Path(audio_path).name}")
        
        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != sample_rate:
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * sample_rate / sr)),
                np.arange(len(audio)),
                audio
            )
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Limit to 5 seconds for demo
        audio = audio[:5 * sample_rate]
        
        # Process through model (in chunks)
        chunk_size = int(2.0 * sample_rate)
        reconstructed_chunks = []
        
        with torch.no_grad():
            for j in range(0, len(audio), chunk_size):
                chunk = audio[j:j + chunk_size]
                chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
                recon_chunk = model(chunk_tensor)
                reconstructed_chunks.append(recon_chunk.squeeze().cpu().numpy())
                del chunk_tensor, recon_chunk
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        reconstructed = np.concatenate(reconstructed_chunks)
        
        # Ensure same length
        min_len = min(len(audio), len(reconstructed))
        audio = audio[:min_len]
        reconstructed = reconstructed[:min_len]
        
        # Save audio files
        base_name = Path(audio_path).stem
        sf.write(output_dir / f'{base_name}_original.wav', audio, sample_rate)
        sf.write(output_dir / f'{base_name}_reconstructed.wav', reconstructed, sample_rate)
        print(f"   ✅ Saved audio files")
        
        # Create visualizations
        create_spectrogram_comparison(
            audio, reconstructed, sample_rate,
            output_dir / f'{base_name}_spectrogram.png'
        )
        
        create_waveform_comparison(
            audio, reconstructed, sample_rate,
            output_dir / f'{base_name}_waveform.png'
        )
        
        # Calculate metrics
        noise = audio - reconstructed
        snr = 10 * np.log10(np.mean(audio**2) / (np.mean(noise**2) + 1e-10))
        
        print(f"   📊 SNR: {snr:.2f} dB")
        print()
    
    print("=" * 80)
    print(f"✅ All demonstrations generated in: {output_dir}")
    print("=" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Demonstration Materials')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--audio-dir', type=str,
                       default='/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of audio samples to process')
    parser.add_argument('--output-dir', type=str, default='demo_materials')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Find audio files
    audio_dir = Path(args.audio_dir)
    audio_files = list(audio_dir.rglob('*.flac'))[:args.num_samples]
    
    if not audio_files:
        print(f"❌ No audio files found in {audio_dir}")
        return
    
    generate_demonstrations(
        args.checkpoint,
        audio_files,
        args.output_dir,
        device=args.device
    )

if __name__ == '__main__':
    main()
