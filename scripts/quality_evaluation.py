"""
Audio Quality Evaluation for Neural Audio Codec
Measures PESQ, STOI, SNR, and other perceptual quality metrics

Target Requirements:
- PESQ ≥ 3.5
- STOI ≥ 0.9
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import yaml
from src.model import NeuralAudioCodec

# Audio quality metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("⚠️  Warning: pesq not installed. Install with: pip install pesq")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("⚠️  Warning: pystoi not installed. Install with: pip install pystoi")


class QualityEvaluator:
    def __init__(self, checkpoint_path, device='cuda'):
        """Initialize quality evaluator"""
        self.device = device
        print(f"🔧 Loading model from: {checkpoint_path}")
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model = NeuralAudioCodec(
            d_model=checkpoint.get('d_model', 256),
            n_layers=checkpoint.get('n_layers', 4),
            n_heads=checkpoint.get('n_heads', 8),
            window_size=checkpoint.get('window_size', 256),
            dropout=0.0
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.sample_rate = 16000
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
    
    def process_audio(self, audio_path, max_duration=10.0):
        """Process audio file through codec and return original + reconstructed"""
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * self.sample_rate / sr)),
                np.arange(len(audio)),
                audio
            )
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Limit duration to avoid OOM
        max_samples = int(max_duration * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Process in chunks to avoid OOM
        chunk_size = int(2.0 * self.sample_rate)  # 2 second chunks
        reconstructed_chunks = []
        
        with torch.no_grad():
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(self.device)
                
                recon_chunk = self.model(chunk_tensor)
                reconstructed_chunks.append(recon_chunk.squeeze().cpu().numpy())
                
                # Clear GPU memory
                del chunk_tensor, recon_chunk
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Concatenate chunks
        reconstructed = np.concatenate(reconstructed_chunks)
        
        # Ensure same length
        min_len = min(len(audio), len(reconstructed))
        audio = audio[:min_len]
        reconstructed = reconstructed[:min_len]
        
        return audio, reconstructed
    
    def calculate_snr(self, original, reconstructed):
        """Calculate Signal-to-Noise Ratio (dB)"""
        noise = original - reconstructed
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power < 1e-10:
            return 100.0  # Essentially perfect
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def calculate_pesq(self, original, reconstructed):
        """Calculate PESQ (Perceptual Evaluation of Speech Quality)"""
        if not PESQ_AVAILABLE:
            return None
        
        try:
            # PESQ requires 8kHz or 16kHz
            score = pesq(self.sample_rate, original, reconstructed, 'wb')  # wideband
            return score
        except Exception as e:
            print(f"⚠️  PESQ calculation failed: {e}")
            return None
    
    def calculate_stoi(self, original, reconstructed):
        """Calculate STOI (Short-Time Objective Intelligibility)"""
        if not STOI_AVAILABLE:
            return None
        
        try:
            score = stoi(original, reconstructed, self.sample_rate, extended=False)
            return score
        except Exception as e:
            print(f"⚠️  STOI calculation failed: {e}")
            return None
    
    def calculate_mse(self, original, reconstructed):
        """Calculate Mean Squared Error"""
        return np.mean((original - reconstructed) ** 2)
    
    def calculate_mae(self, original, reconstructed):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(original - reconstructed))
    
    def calculate_spectral_convergence(self, original, reconstructed):
        """Calculate spectral convergence"""
        orig_spec = np.abs(np.fft.rfft(original))
        recon_spec = np.abs(np.fft.rfft(reconstructed))
        
        num = np.linalg.norm(orig_spec - recon_spec)
        den = np.linalg.norm(orig_spec)
        
        if den < 1e-10:
            return 0.0
        
        return num / den
    
    def calculate_lsd(self, original, reconstructed):
        """Calculate Log-Spectral Distance"""
        orig_spec = np.abs(np.fft.rfft(original)) + 1e-10
        recon_spec = np.abs(np.fft.rfft(reconstructed)) + 1e-10
        
        log_diff = np.log10(orig_spec) - np.log10(recon_spec)
        lsd = np.sqrt(np.mean(log_diff ** 2))
        
        return lsd
    
    def evaluate_file(self, audio_path):
        """Evaluate single audio file"""
        print(f"📄 Processing: {Path(audio_path).name}")
        
        # Process audio
        original, reconstructed = self.process_audio(audio_path)
        
        # Calculate all metrics
        results = {
            'file': str(audio_path),
            'duration': len(original) / self.sample_rate,
            'snr': self.calculate_snr(original, reconstructed),
            'pesq': self.calculate_pesq(original, reconstructed),
            'stoi': self.calculate_stoi(original, reconstructed),
            'mse': self.calculate_mse(original, reconstructed),
            'mae': self.calculate_mae(original, reconstructed),
            'spectral_convergence': self.calculate_spectral_convergence(original, reconstructed),
            'lsd': self.calculate_lsd(original, reconstructed)
        }
        
        # Print results
        print(f"   Duration: {results['duration']:.2f}s")
        print(f"   SNR:      {results['snr']:.2f} dB")
        
        if results['pesq'] is not None:
            status = "✅" if results['pesq'] >= 3.5 else "❌"
            print(f"   PESQ:     {results['pesq']:.3f} {status} (target: ≥3.5)")
        else:
            print(f"   PESQ:     N/A (install pesq package)")
        
        if results['stoi'] is not None:
            status = "✅" if results['stoi'] >= 0.9 else "❌"
            print(f"   STOI:     {results['stoi']:.3f} {status} (target: ≥0.9)")
        else:
            print(f"   STOI:     N/A (install pystoi package)")
        
        print(f"   MSE:      {results['mse']:.6f}")
        print(f"   MAE:      {results['mae']:.6f}")
        print(f"   Spec Conv: {results['spectral_convergence']:.4f}")
        print(f"   LSD:      {results['lsd']:.4f}")
        print()
        
        return results
    
    def evaluate_directory(self, audio_dir, max_files=None, pattern="*.flac"):
        """Evaluate all audio files in directory"""
        audio_dir = Path(audio_dir)
        
        # Find all audio files
        audio_files = list(audio_dir.rglob(pattern))
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        if not audio_files:
            print(f"❌ No audio files found in {audio_dir} with pattern {pattern}")
            return None
        
        print(f"📁 Found {len(audio_files)} audio files")
        print(f"   Pattern: {pattern}")
        print(f"   Directory: {audio_dir}")
        print()
        
        results = []
        
        for audio_path in tqdm(audio_files, desc="Evaluating files"):
            try:
                result = self.evaluate_file(audio_path)
                results.append(result)
            except Exception as e:
                print(f"⚠️  Error processing {audio_path}: {e}")
                continue
        
        return results
    
    def print_summary(self, results):
        """Print summary statistics"""
        if not results:
            print("❌ No results to summarize")
            return
        
        print("=" * 80)
        print("QUALITY EVALUATION SUMMARY")
        print("=" * 80)
        print()
        
        # Calculate statistics
        snr_values = [r['snr'] for r in results]
        pesq_values = [r['pesq'] for r in results if r['pesq'] is not None]
        stoi_values = [r['stoi'] for r in results if r['stoi'] is not None]
        mse_values = [r['mse'] for r in results]
        mae_values = [r['mae'] for r in results]
        lsd_values = [r['lsd'] for r in results]
        
        total_duration = sum(r['duration'] for r in results)
        
        print(f"📊 Test Set Statistics:")
        print(f"   Total files: {len(results)}")
        print(f"   Total duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        print(f"   Average file length: {total_duration/len(results):.2f}s")
        print()
        
        # SNR
        print(f"🎵 Signal-to-Noise Ratio (SNR):")
        print(f"   Mean:   {np.mean(snr_values):.2f} dB")
        print(f"   Median: {np.median(snr_values):.2f} dB")
        print(f"   Std:    {np.std(snr_values):.2f} dB")
        print(f"   Min:    {np.min(snr_values):.2f} dB")
        print(f"   Max:    {np.max(snr_values):.2f} dB")
        print()
        
        # PESQ
        if pesq_values:
            mean_pesq = np.mean(pesq_values)
            status = "✅ PASS" if mean_pesq >= 3.5 else "❌ FAIL"
            pass_rate = sum(1 for p in pesq_values if p >= 3.5) / len(pesq_values) * 100
            
            print(f"🎯 PESQ (Perceptual Evaluation of Speech Quality):")
            print(f"   Target:  ≥ 3.5")
            print(f"   Mean:    {mean_pesq:.3f} {status}")
            print(f"   Median:  {np.median(pesq_values):.3f}")
            print(f"   Std:     {np.std(pesq_values):.3f}")
            print(f"   Min:     {np.min(pesq_values):.3f}")
            print(f"   Max:     {np.max(pesq_values):.3f}")
            print(f"   Pass Rate: {pass_rate:.1f}% (≥3.5)")
            print()
        else:
            print(f"⚠️  PESQ: Not available (install pesq package)")
            print()
        
        # STOI
        if stoi_values:
            mean_stoi = np.mean(stoi_values)
            status = "✅ PASS" if mean_stoi >= 0.9 else "❌ FAIL"
            pass_rate = sum(1 for s in stoi_values if s >= 0.9) / len(stoi_values) * 100
            
            print(f"🎯 STOI (Short-Time Objective Intelligibility):")
            print(f"   Target:  ≥ 0.9")
            print(f"   Mean:    {mean_stoi:.3f} {status}")
            print(f"   Median:  {np.median(stoi_values):.3f}")
            print(f"   Std:     {np.std(stoi_values):.3f}")
            print(f"   Min:     {np.min(stoi_values):.3f}")
            print(f"   Max:     {np.max(stoi_values):.3f}")
            print(f"   Pass Rate: {pass_rate:.1f}% (≥0.9)")
            print()
        else:
            print(f"⚠️  STOI: Not available (install pystoi package)")
            print()
        
        # Other metrics
        print(f"📈 Additional Metrics:")
        print(f"   MSE:               {np.mean(mse_values):.6f}")
        print(f"   MAE:               {np.mean(mae_values):.6f}")
        print(f"   Log-Spectral Dist: {np.mean(lsd_values):.4f}")
        print()
        
        # Overall assessment
        print("=" * 80)
        print("OVERALL ASSESSMENT")
        print("=" * 80)
        
        criteria_met = []
        criteria_failed = []
        
        # Check PESQ
        if pesq_values:
            if np.mean(pesq_values) >= 3.5:
                criteria_met.append("PESQ ≥ 3.5")
            else:
                criteria_failed.append(f"PESQ < 3.5 (got {np.mean(pesq_values):.3f})")
        
        # Check STOI
        if stoi_values:
            if np.mean(stoi_values) >= 0.9:
                criteria_met.append("STOI ≥ 0.9")
            else:
                criteria_failed.append(f"STOI < 0.9 (got {np.mean(stoi_values):.3f})")
        
        if criteria_met:
            print("✅ Criteria MET:")
            for criterion in criteria_met:
                print(f"   • {criterion}")
            print()
        
        if criteria_failed:
            print("❌ Criteria NOT MET:")
            for criterion in criteria_failed:
                print(f"   • {criterion}")
            print()
        
        if not pesq_values and not stoi_values:
            print("⚠️  Cannot assess criteria - install required packages:")
            print("   pip install pesq pystoi")
            print()
        
        print("=" * 80)
        
        return {
            'snr': {'mean': np.mean(snr_values), 'std': np.std(snr_values)},
            'pesq': {'mean': np.mean(pesq_values), 'std': np.std(pesq_values)} if pesq_values else None,
            'stoi': {'mean': np.mean(stoi_values), 'std': np.std(stoi_values)} if stoi_values else None,
            'criteria_met': criteria_met,
            'criteria_failed': criteria_failed
        }
    
    def save_results(self, results, output_path):
        """Save detailed results to file"""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            f.write("# Audio Quality Evaluation Results\n\n")
            f.write(f"Total files evaluated: {len(results)}\n\n")
            
            f.write("## Per-File Results\n\n")
            f.write("| File | Duration | SNR | PESQ | STOI | MSE | LSD |\n")
            f.write("|------|----------|-----|------|------|-----|-----|\n")
            
            for r in results:
                pesq_str = f"{r['pesq']:.3f}" if r['pesq'] is not None else "N/A"
                stoi_str = f"{r['stoi']:.3f}" if r['stoi'] is not None else "N/A"
                
                f.write(f"| {Path(r['file']).name} | {r['duration']:.2f}s | "
                       f"{r['snr']:.2f} | {pesq_str} | {stoi_str} | "
                       f"{r['mse']:.6f} | {r['lsd']:.4f} |\n")
            
            f.write("\n")
        
        print(f"💾 Results saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Quality Evaluation')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--audio-dir', type=str,
                       default='/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100',
                       help='Directory containing test audio files')
    parser.add_argument('--audio-file', type=str, default=None,
                       help='Single audio file to test (optional)')
    parser.add_argument('--max-files', type=int, default=50,
                       help='Maximum number of files to evaluate')
    parser.add_argument('--pattern', type=str, default='*.flac',
                       help='File pattern to match (e.g., *.flac, *.wav)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    parser.add_argument('--output', type=str, default='quality_results.md',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Check checkpoint
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Initialize evaluator
    evaluator = QualityEvaluator(args.checkpoint, device=args.device)
    
    print("=" * 80)
    print("NEURAL AUDIO CODEC - QUALITY EVALUATION")
    print("=" * 80)
    print(f"Target: PESQ ≥ 3.5, STOI ≥ 0.9")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()
    
    # Check if required packages are installed
    if not PESQ_AVAILABLE or not STOI_AVAILABLE:
        print("⚠️  WARNING: Some quality metrics are not available")
        if not PESQ_AVAILABLE:
            print("   Missing: pesq - Install with: pip install pesq")
        if not STOI_AVAILABLE:
            print("   Missing: pystoi - Install with: pip install pystoi")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
        print()
    
    # Evaluate
    if args.audio_file:
        # Single file
        if not Path(args.audio_file).exists():
            print(f"❌ Audio file not found: {args.audio_file}")
            return
        
        results = [evaluator.evaluate_file(args.audio_file)]
    else:
        # Directory
        if not Path(args.audio_dir).exists():
            print(f"❌ Audio directory not found: {args.audio_dir}")
            return
        
        results = evaluator.evaluate_directory(
            args.audio_dir,
            max_files=args.max_files,
            pattern=args.pattern
        )
    
    if results:
        # Print summary
        summary = evaluator.print_summary(results)
        
        # Save results
        evaluator.save_results(results, args.output)
        
        print()
        print("✅ Quality evaluation complete!")


if __name__ == '__main__':
    main()
