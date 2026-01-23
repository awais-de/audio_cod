#!/usr/bin/env python3
"""
Benchmark script to compare old vs new data loading performance
Shows the actual speedup from optimizations
"""

import torch
import torchaudio
import torchaudio.transforms
import time
from pathlib import Path
import torch.nn.functional as F

class OldAudioDataset:
    """Original slow implementation"""
    def __init__(self, audio_path, sample_rate=16000):
        self.audio_path = audio_path
        self.sample_rate = sample_rate
    
    def load_sample(self):
        waveform, sr = torchaudio.load(self.audio_path)
        
        # NEW RESAMPLER EVERY TIME (slow!)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        max_val = torch.abs(waveform).max()
        if max_val > 1e-6:
            waveform = waveform / (max_val + 1e-8)
        
        segment_length = 8000
        if waveform.shape[1] < segment_length:
            padding = segment_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[1] > segment_length:
            max_start = waveform.shape[1] - segment_length
            start = torch.randint(0, max_start + 1, (1,)).item()
            waveform = waveform[:, start:start + segment_length]
        
        assert waveform.shape == (1, segment_length)
        return waveform


class NewAudioDataset:
    """Optimized implementation"""
    def __init__(self, audio_path, sample_rate=16000):
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.resamplers = {}  # CACHE!
    
    def load_sample(self):
        waveform, sr = torchaudio.load(self.audio_path, backend='ffmpeg')  # FAST BACKEND
        
        # REUSE EXISTING RESAMPLER (fast!)
        if sr != self.sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            resampler = self.resamplers[sr]
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        max_val = torch.abs(waveform).max()
        if max_val > 1e-6:
            waveform = waveform / (max_val + 1e-8)
        
        segment_length = 8000
        if waveform.shape[1] < segment_length:
            padding = segment_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[1] > segment_length:
            max_start = waveform.shape[1] - segment_length
            start = torch.randint(0, max_start + 1, (1,)).item()
            waveform = waveform[:, start:start + segment_length]
        
        # NO ASSERTION IN HOT PATH
        return waveform


def benchmark_dataset(dataset_class, audio_path, name, iterations=100):
    """Benchmark dataset loading"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    dataset = dataset_class(audio_path)
    
    # Warmup
    for _ in range(5):
        dataset.load_sample()
    
    # Benchmark
    times = []
    for i in range(iterations):
        start = time.time()
        _ = dataset.load_sample()
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            avg_time = sum(times) / len(times)
            print(f"  Iteration {i+1}/{iterations}: {elapsed*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
    
    total_time = sum(times)
    avg_time = total_time / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n  Results:")
    print(f"    Total time: {total_time:.2f}s")
    print(f"    Avg time:   {avg_time*1000:.2f}ms")
    print(f"    Min time:   {min_time*1000:.2f}ms")
    print(f"    Max time:   {max_time*1000:.2f}ms")
    
    return avg_time


if __name__ == '__main__':
    # Find a sample audio file
    data_dir = Path('/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100')
    audio_files = list(data_dir.rglob('*.flac'))
    
    if not audio_files:
        print("No audio files found! Please download LibriSpeech dataset first.")
        exit(1)
    
    sample_audio = str(audio_files[0])
    print(f"\nUsing sample audio: {sample_audio}")
    
    # Benchmark old implementation
    old_avg = benchmark_dataset(OldAudioDataset, sample_audio, "OLD (New Resampler Each Time)", iterations=50)
    
    # Benchmark new implementation
    new_avg = benchmark_dataset(NewAudioDataset, sample_audio, "NEW (Cached Resampler + ffmpeg)", iterations=50)
    
    # Summary
    speedup = old_avg / new_avg
    improvement_pct = (1 - new_avg / old_avg) * 100
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Old implementation: {old_avg*1000:.2f}ms per sample")
    print(f"New implementation: {new_avg*1000:.2f}ms per sample")
    print(f"{'='*60}")
    print(f"Speedup: {speedup:.2f}x faster")
    print(f"Improvement: {improvement_pct:.1f}% faster")
    print(f"{'='*60}")
    print(f"\nFor 28,539 samples (LibriSpeech train-clean-100):")
    print(f"  Old: {old_avg * 28539 / 60:.1f} minutes")
    print(f"  New: {new_avg * 28539 / 60:.1f} minutes")
    print(f"  Time saved per epoch: {(old_avg - new_avg) * 28539 / 60:.1f} minutes")
