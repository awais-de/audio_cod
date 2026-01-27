"""
Latency Benchmark for Neural Audio Codec
Measures encoding, decoding, and end-to-end latency for real-time teleconferencing
Target: < 20ms end-to-end latency
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from pathlib import Path
import yaml
from src.model import NeuralAudioCodec
import soundfile as sf
from statistics import mean, median, stdev

class LatencyBenchmark:
    def __init__(self, checkpoint_path, device='cuda'):
        """Initialize latency benchmark"""
        self.device = device
        print(f"🔧 Loading model from: {checkpoint_path}")
        print(f"   Device: {device}")
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model = NeuralAudioCodec(
            d_model=checkpoint.get('d_model', 256),
            n_layers=checkpoint.get('n_layers', 4),
            n_heads=checkpoint.get('n_heads', 8),
            window_size=checkpoint.get('window_size', 256),
            dropout=0.0  # No dropout for inference
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.sample_rate = 16000
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
    
    def warmup(self, iterations=10):
        """Warmup GPU for accurate timing"""
        print(f"🔥 Warming up GPU ({iterations} iterations)...")
        dummy_audio = torch.randn(1, 1, 16000).to(self.device)
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model.encoder(dummy_audio)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
        print("✅ Warmup complete\n")
    
    def measure_encoding_latency(self, audio_length_ms, iterations=100):
        """Measure encoding latency for specific audio length"""
        samples = int(audio_length_ms * self.sample_rate / 1000)
        audio = torch.randn(1, 1, samples).to(self.device)
        
        latencies = []
        
        with torch.no_grad():
            for _ in range(iterations):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model.encoder(audio)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean': mean(latencies),
            'median': median(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'std': stdev(latencies) if len(latencies) > 1 else 0,
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
    
    def measure_decoding_latency(self, audio_length_ms, iterations=100):
        """Measure decoding latency for specific audio length"""
        samples = int(audio_length_ms * self.sample_rate / 1000)
        audio = torch.randn(1, 1, samples).to(self.device)
        
        # Encode first to get latent representation
        with torch.no_grad():
            latent = self.model.encoder(audio)
        
        latencies = []
        
        with torch.no_grad():
            for _ in range(iterations):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model.decoder(latent)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
        
        return {
            'mean': mean(latencies),
            'median': median(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'std': stdev(latencies) if len(latencies) > 1 else 0,
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
    
    def measure_end_to_end_latency(self, audio_length_ms, iterations=100):
        """Measure full encode + decode latency"""
        samples = int(audio_length_ms * self.sample_rate / 1000)
        audio = torch.randn(1, 1, samples).to(self.device)
        
        latencies = []
        
        with torch.no_grad():
            for _ in range(iterations):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                latent = self.model.encoder(audio)
                reconstructed = self.model.decoder(latent)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
        
        return {
            'mean': mean(latencies),
            'median': median(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'std': stdev(latencies) if len(latencies) > 1 else 0,
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive latency benchmark across different chunk sizes"""
        
        # Test different audio chunk sizes (typical for real-time systems)
        chunk_sizes = [10, 20, 30, 40, 50, 100]  # milliseconds
        
        print("=" * 80)
        print("NEURAL AUDIO CODEC - LATENCY BENCHMARK")
        print("=" * 80)
        print(f"Target: < 20ms end-to-end latency")
        print(f"Device: {self.device}")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print("=" * 80)
        print()
        
        # Warmup
        self.warmup()
        
        results = {}
        
        for chunk_ms in chunk_sizes:
            print(f"📊 Testing {chunk_ms}ms audio chunks")
            print("-" * 80)
            
            # Encoding latency
            enc_stats = self.measure_encoding_latency(chunk_ms, iterations=100)
            print(f"   ENCODING:")
            print(f"      Mean:   {enc_stats['mean']:.3f} ms")
            print(f"      Median: {enc_stats['median']:.3f} ms")
            print(f"      Min:    {enc_stats['min']:.3f} ms")
            print(f"      Max:    {enc_stats['max']:.3f} ms")
            print(f"      Std:    {enc_stats['std']:.3f} ms")
            print(f"      P95:    {enc_stats['p95']:.3f} ms")
            print(f"      P99:    {enc_stats['p99']:.3f} ms")
            
            # Decoding latency
            dec_stats = self.measure_decoding_latency(chunk_ms, iterations=100)
            print(f"   DECODING:")
            print(f"      Mean:   {dec_stats['mean']:.3f} ms")
            print(f"      Median: {dec_stats['median']:.3f} ms")
            print(f"      Min:    {dec_stats['min']:.3f} ms")
            print(f"      Max:    {dec_stats['max']:.3f} ms")
            print(f"      Std:    {dec_stats['std']:.3f} ms")
            print(f"      P95:    {dec_stats['p95']:.3f} ms")
            print(f"      P99:    {dec_stats['p99']:.3f} ms")
            
            # End-to-end latency
            e2e_stats = self.measure_end_to_end_latency(chunk_ms, iterations=100)
            print(f"   END-TO-END:")
            print(f"      Mean:   {e2e_stats['mean']:.3f} ms")
            print(f"      Median: {e2e_stats['median']:.3f} ms")
            print(f"      Min:    {e2e_stats['min']:.3f} ms")
            print(f"      Max:    {e2e_stats['max']:.3f} ms")
            print(f"      Std:    {e2e_stats['std']:.3f} ms")
            print(f"      P95:    {e2e_stats['p95']:.3f} ms")
            print(f"      P99:    {e2e_stats['p99']:.3f} ms")
            
            # Real-Time Factor (RTF)
            rtf = e2e_stats['mean'] / chunk_ms
            print(f"   Real-Time Factor (RTF): {rtf:.3f}x")
            
            # Check if meets target
            meets_target = e2e_stats['p99'] < 20.0
            status = "✅ PASS" if meets_target else "❌ FAIL"
            print(f"   Status (<20ms P99): {status}")
            print()
            
            results[chunk_ms] = {
                'encoding': enc_stats,
                'decoding': dec_stats,
                'end_to_end': e2e_stats,
                'rtf': rtf,
                'meets_target': meets_target
            }
        
        # Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        print(f"{'Chunk Size':<12} {'E2E Mean':<12} {'E2E P99':<12} {'RTF':<10} {'Status'}")
        print("-" * 80)
        
        for chunk_ms, stats in results.items():
            e2e = stats['end_to_end']
            rtf = stats['rtf']
            status = "✅ PASS" if stats['meets_target'] else "❌ FAIL"
            print(f"{chunk_ms}ms{'':<8} {e2e['mean']:.3f}ms{'':<5} {e2e['p99']:.3f}ms{'':<5} {rtf:.3f}x{'':<5} {status}")
        
        print()
        print("=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        # Find optimal chunk size
        optimal_chunks = [c for c, s in results.items() if s['meets_target']]
        if optimal_chunks:
            print(f"✅ Model meets <20ms latency requirement!")
            print(f"   Optimal chunk sizes: {', '.join(map(str, optimal_chunks))}ms")
            print(f"   Recommended: {min(optimal_chunks)}ms for minimum buffering delay")
        else:
            print(f"⚠️  Model does not meet <20ms latency requirement")
            print(f"   Best performance: {min(results.keys(), key=lambda k: results[k]['end_to_end']['p99'])}ms chunks")
            print(f"   Consider:")
            print(f"      - Model quantization (INT8/FP16)")
            print(f"      - Further model size reduction")
            print(f"      - GPU optimization (TensorRT, ONNX)")
        
        print()
        
        # Additional metrics
        best_chunk = min(results.keys(), key=lambda k: results[k]['end_to_end']['mean'])
        best_rtf = results[best_chunk]['rtf']
        print(f"Best Real-Time Factor: {best_rtf:.3f}x at {best_chunk}ms chunks")
        print(f"   (RTF < 1.0 means faster than real-time)")
        print()
        
        return results
    
    def test_with_real_audio(self, audio_path):
        """Test latency with real audio file"""
        print("=" * 80)
        print("TESTING WITH REAL AUDIO FILE")
        print("=" * 80)
        print(f"File: {audio_path}")
        print()
        
        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != self.sample_rate:
            print(f"⚠️  Resampling from {sr} Hz to {self.sample_rate} Hz")
            # Simple resampling (in production, use librosa or torchaudio)
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * self.sample_rate / sr)),
                np.arange(len(audio)),
                audio
            )
        
        # Convert to tensor
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        
        audio = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Process in chunks
        chunk_size_samples = int(0.020 * self.sample_rate)  # 20ms chunks
        num_chunks = audio.shape[-1] // chunk_size_samples
        
        print(f"Audio length: {audio.shape[-1] / self.sample_rate:.2f} seconds")
        print(f"Processing in {num_chunks} chunks of 20ms each")
        print()
        
        latencies = []
        
        with torch.no_grad():
            for i in range(num_chunks):
                start_idx = i * chunk_size_samples
                end_idx = start_idx + chunk_size_samples
                chunk = audio[:, :, start_idx:end_idx]
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                latent = self.model.encoder(chunk)
                reconstructed = self.model.decoder(latent)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
        
        print(f"Results for 20ms chunks:")
        print(f"   Mean latency:   {mean(latencies):.3f} ms")
        print(f"   Median latency: {median(latencies):.3f} ms")
        print(f"   P95 latency:    {np.percentile(latencies, 95):.3f} ms")
        print(f"   P99 latency:    {np.percentile(latencies, 99):.3f} ms")
        print(f"   RTF:            {mean(latencies) / 20:.3f}x")
        
        meets_target = np.percentile(latencies, 99) < 20.0
        status = "✅ PASS" if meets_target else "❌ FAIL"
        print(f"   Status:         {status}")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Latency Benchmark for Neural Audio Codec')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--audio', type=str, default=None,
                       help='Optional: path to audio file for real-world testing')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        print(f"   Available checkpoints:")
        ckpt_dir = Path('checkpoints')
        if ckpt_dir.exists():
            for ckpt in sorted(ckpt_dir.glob('*.pt')):
                print(f"      - {ckpt}")
        return
    
    # Initialize benchmark
    benchmark = LatencyBenchmark(args.checkpoint, device=args.device)
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Test with real audio if provided
    if args.audio:
        if Path(args.audio).exists():
            benchmark.test_with_real_audio(args.audio)
        else:
            print(f"⚠️  Audio file not found: {args.audio}")
    
    print("=" * 80)
    print("✅ Latency benchmark complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
