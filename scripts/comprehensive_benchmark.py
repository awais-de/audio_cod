"""
Comprehensive Benchmark: Neural Codec vs Opus Baseline
Compares latency, quality (PESQ/STOI), and bitrate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import time
from src.model import NeuralAudioCodec
from baseline_codec import OpusCodec

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


class ComprehensiveBenchmark:
    def __init__(self, neural_checkpoint, device='cuda'):
        self.device = device
        self.sample_rate = 16000
        
        # Load neural codec
        print("🔧 Loading Neural Audio Codec...")
        checkpoint = torch.load(neural_checkpoint, map_location=device)
        self.neural_model = NeuralAudioCodec(
            d_model=checkpoint.get('d_model', 256),
            n_layers=checkpoint.get('n_layers', 4),
            n_heads=checkpoint.get('n_heads', 8),
            window_size=checkpoint.get('window_size', 256),
            dropout=0.0
        ).to(device)
        self.neural_model.load_state_dict(checkpoint['model_state_dict'])
        self.neural_model.eval()
        print("✅ Neural codec loaded")
        
        # Initialize Opus codec
        print("🔧 Loading Opus Codec...")
        self.opus_codec = OpusCodec(bitrate=16000, sample_rate=16000)
        print("✅ Opus codec loaded")
        print()
    
    def process_neural(self, audio):
        """Process audio through neural codec"""
        # Process in 2s chunks to avoid OOM
        chunk_size = int(2.0 * self.sample_rate)
        reconstructed_chunks = []
        latencies = []
        
        with torch.no_grad():
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(self.device)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                recon_chunk = self.neural_model(chunk_tensor)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                
                reconstructed_chunks.append(recon_chunk.squeeze().cpu().numpy())
                
                del chunk_tensor, recon_chunk
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        reconstructed = np.concatenate(reconstructed_chunks)
        avg_latency = np.mean(latencies)
        
        return reconstructed, avg_latency
    
    def process_opus(self, audio):
        """Process audio through Opus codec"""
        start = time.perf_counter()
        reconstructed, bitrate = self.opus_codec.encode_decode(audio)
        latency = (time.perf_counter() - start) * 1000
        
        return reconstructed, latency, bitrate
    
    def calculate_metrics(self, original, reconstructed):
        """Calculate quality metrics"""
        # Ensure same length
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]
        
        # SNR
        noise = original - reconstructed
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # PESQ
        pesq_score = None
        if PESQ_AVAILABLE:
            try:
                pesq_score = pesq(self.sample_rate, original, reconstructed, 'wb')
            except:
                pass
        
        # STOI
        stoi_score = None
        if STOI_AVAILABLE:
            try:
                stoi_score = stoi(original, reconstructed, self.sample_rate, extended=False)
            except:
                pass
        
        return {
            'snr': snr,
            'pesq': pesq_score,
            'stoi': stoi_score
        }
    
    def benchmark_file(self, audio_path):
        """Benchmark single file"""
        print(f"📄 Processing: {Path(audio_path).name}")
        
        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != self.sample_rate:
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * self.sample_rate / sr)),
                np.arange(len(audio)),
                audio
            )
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Limit to 10 seconds
        audio = audio[:10 * self.sample_rate]
        duration = len(audio) / self.sample_rate
        
        print(f"   Duration: {duration:.2f}s")
        
        # Neural codec
        print("   🧠 Neural Codec...")
        neural_recon, neural_latency = self.process_neural(audio)
        neural_metrics = self.calculate_metrics(audio, neural_recon)
        
        # Opus codec
        print("   🔊 Opus Codec...")
        opus_recon, opus_latency, opus_bitrate = self.process_opus(audio)
        opus_metrics = self.calculate_metrics(audio, opus_recon)
        
        return {
            'file': audio_path,
            'duration': duration,
            'neural': {
                'latency': neural_latency,
                'metrics': neural_metrics
            },
            'opus': {
                'latency': opus_latency,
                'bitrate': opus_bitrate,
                'metrics': opus_metrics
            }
        }
    
    def print_results(self, results):
        """Print benchmark results"""
        print()
        print("=" * 80)
        print("COMPREHENSIVE BENCHMARK RESULTS")
        print("=" * 80)
        print()
        
        print(f"📊 Test File: {Path(results['file']).name}")
        print(f"   Duration: {results['duration']:.2f}s")
        print()
        
        # Comparison table
        print("LATENCY COMPARISON:")
        print("-" * 80)
        print(f"{'Metric':<30} {'Neural Codec':<20} {'Opus Codec':<20} {'Winner'}")
        print("-" * 80)
        
        neural_lat = results['neural']['latency']
        opus_lat = results['opus']['latency']
        lat_winner = "Neural" if neural_lat < opus_lat else "Opus"
        print(f"{'Processing Latency (ms)':<30} {neural_lat:<20.2f} {opus_lat:<20.2f} {lat_winner}")
        
        target_met_neural = "✅" if neural_lat < 20 else "❌"
        target_met_opus = "✅" if opus_lat < 20 else "❌"
        print(f"{'Meets <20ms target':<30} {target_met_neural:<20} {target_met_opus:<20}")
        print()
        
        # Quality comparison
        print("QUALITY COMPARISON:")
        print("-" * 80)
        print(f"{'Metric':<30} {'Neural Codec':<20} {'Opus Codec':<20} {'Target':<15} {'Winner'}")
        print("-" * 80)
        
        # SNR
        neural_snr = results['neural']['metrics']['snr']
        opus_snr = results['opus']['metrics']['snr']
        snr_winner = "Neural" if neural_snr > opus_snr else "Opus"
        print(f"{'SNR (dB)':<30} {neural_snr:<20.2f} {opus_snr:<20.2f} {'>20 dB':<15} {snr_winner}")
        
        # PESQ
        if PESQ_AVAILABLE:
            neural_pesq = results['neural']['metrics']['pesq']
            opus_pesq = results['opus']['metrics']['pesq']
            if neural_pesq and opus_pesq:
                pesq_winner = "Neural" if neural_pesq > opus_pesq else "Opus"
                neural_pass = "✅" if neural_pesq >= 3.5 else "❌"
                opus_pass = "✅" if opus_pesq >= 3.5 else "❌"
                print(f"{'PESQ':<30} {f'{neural_pesq:.3f} {neural_pass}':<20} {f'{opus_pesq:.3f} {opus_pass}':<20} {'≥3.5':<15} {pesq_winner}")
        
        # STOI
        if STOI_AVAILABLE:
            neural_stoi = results['neural']['metrics']['stoi']
            opus_stoi = results['opus']['metrics']['stoi']
            if neural_stoi and opus_stoi:
                stoi_winner = "Neural" if neural_stoi > opus_stoi else "Opus"
                neural_pass = "✅" if neural_stoi >= 0.9 else "❌"
                opus_pass = "✅" if opus_stoi >= 0.9 else "❌"
                print(f"{'STOI':<30} {f'{neural_stoi:.3f} {neural_pass}':<20} {f'{opus_stoi:.3f} {opus_pass}':<20} {'≥0.9':<15} {stoi_winner}")
        
        print()
        
        # Bitrate
        print("BITRATE:")
        print("-" * 80)
        print(f"Opus: {results['opus']['bitrate'] / 1000:.2f} kbps")
        print(f"Neural: Not measured (requires quantization)")
        print()
        
        # Overall assessment
        print("=" * 80)
        print("OVERALL ASSESSMENT")
        print("=" * 80)
        print()
        
        print("Neural Audio Codec:")
        if neural_lat < 20:
            print(f"  ✅ Latency: PASSED ({neural_lat:.2f}ms < 20ms)")
        else:
            print(f"  ❌ Latency: FAILED ({neural_lat:.2f}ms > 20ms)")
        
        if PESQ_AVAILABLE and results['neural']['metrics']['pesq']:
            if results['neural']['metrics']['pesq'] >= 3.5:
                print(f"  ✅ PESQ: PASSED ({results['neural']['metrics']['pesq']:.3f} ≥ 3.5)")
            else:
                print(f"  ❌ PESQ: FAILED ({results['neural']['metrics']['pesq']:.3f} < 3.5)")
        
        if STOI_AVAILABLE and results['neural']['metrics']['stoi']:
            if results['neural']['metrics']['stoi'] >= 0.9:
                print(f"  ✅ STOI: PASSED ({results['neural']['metrics']['stoi']:.3f} ≥ 0.9)")
            else:
                print(f"  ❌ STOI: FAILED ({results['neural']['metrics']['stoi']:.3f} < 0.9)")
        
        print()
        print("Opus Baseline:")
        if opus_lat < 20:
            print(f"  ✅ Latency: PASSED ({opus_lat:.2f}ms < 20ms)")
        else:
            print(f"  ❌ Latency: FAILED ({opus_lat:.2f}ms > 20ms)")
        
        if PESQ_AVAILABLE and results['opus']['metrics']['pesq']:
            if results['opus']['metrics']['pesq'] >= 3.5:
                print(f"  ✅ PESQ: PASSED ({results['opus']['metrics']['pesq']:.3f} ≥ 3.5)")
            else:
                print(f"  ❌ PESQ: FAILED ({results['opus']['metrics']['pesq']:.3f} < 3.5)")
        
        if STOI_AVAILABLE and results['opus']['metrics']['stoi']:
            if results['opus']['metrics']['stoi'] >= 0.9:
                print(f"  ✅ STOI: PASSED ({results['opus']['metrics']['stoi']:.3f} ≥ 0.9)")
            else:
                print(f"  ❌ STOI: FAILED ({results['opus']['metrics']['stoi']:.3f} < 0.9)")
        
        print()
        print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Codec Benchmark')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--audio', type=str,
                       default='/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100/2007/149877/2007-149877-0049.flac')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    benchmark = ComprehensiveBenchmark(args.checkpoint, device=args.device)
    results = benchmark.benchmark_file(args.audio)
    benchmark.print_results(results)


if __name__ == '__main__':
    main()
