#!/usr/bin/env python3
"""
Detailed profiler to identify actual bottlenecks
"""

import torch
import torchaudio
import time
from pathlib import Path
import torch.nn.functional as F

def profile_data_loading():
    """Profile each step of data loading"""
    data_dir = Path('/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100')
    audio_files = list(data_dir.rglob('*.flac'))
    sample_audio = str(audio_files[0])
    
    iterations = 100
    segment_length = 8000
    sample_rate = 16000
    
    timings = {
        'load': [],
        'resample': [],
        'normalize': [],
        'segment': [],
        'total': []
    }
    
    # Warmup
    for _ in range(5):
        waveform, sr = torchaudio.load(sample_audio, backend='ffmpeg')
    
    resampler = torchaudio.transforms.Resample(48000, 16000)
    
    print("Profiling data loading pipeline...")
    print(f"Audio: {Path(sample_audio).name}")
    print(f"Iterations: {iterations}\n")
    
    for i in range(iterations):
        # Load
        start = time.perf_counter()
        waveform, sr = torchaudio.load(sample_audio, backend='ffmpeg')
        load_time = time.perf_counter() - start
        timings['load'].append(load_time)
        
        # Resample
        start = time.perf_counter()
        if sr != sample_rate:
            waveform = resampler(waveform)
        resample_time = time.perf_counter() - start
        timings['resample'].append(resample_time)
        
        # Mono + Normalize
        start = time.perf_counter()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        max_val = torch.abs(waveform).max()
        if max_val > 1e-6:
            waveform = waveform / (max_val + 1e-8)
        normalize_time = time.perf_counter() - start
        timings['normalize'].append(normalize_time)
        
        # Segment
        start = time.perf_counter()
        if waveform.shape[1] < segment_length:
            padding = segment_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[1] > segment_length:
            max_start = waveform.shape[1] - segment_length
            start_pos = torch.randint(0, max_start + 1, (1,)).item()
            waveform = waveform[:, start_pos:start_pos + segment_length]
        segment_time = time.perf_counter() - start
        timings['segment'].append(segment_time)
        
        total_time = load_time + resample_time + normalize_time + segment_time
        timings['total'].append(total_time)
        
        if (i + 1) % 20 == 0:
            avg_load = sum(timings['load'][-20:]) / 20
            avg_resample = sum(timings['resample'][-20:]) / 20
            avg_norm = sum(timings['normalize'][-20:]) / 20
            avg_seg = sum(timings['segment'][-20:]) / 20
            avg_total = sum(timings['total'][-20:]) / 20
            print(f"Iteration {i+1:3d}: Total={avg_total*1000:.2f}ms (Load={avg_load*1000:.2f}ms, "
                  f"Resample={avg_resample*1000:.2f}ms, Norm={avg_norm*1000:.2f}ms, Seg={avg_seg*1000:.2f}ms)")
    
    print(f"\n{'='*80}")
    print("FINAL PROFILE RESULTS")
    print(f"{'='*80}")
    
    # Calculate averages
    avg_load = sum(timings['load']) / len(timings['load'])
    avg_resample = sum(timings['resample']) / len(timings['resample'])
    avg_normalize = sum(timings['normalize']) / len(timings['normalize'])
    avg_segment = sum(timings['segment']) / len(timings['segment'])
    avg_total = sum(timings['total']) / len(timings['total'])
    
    print(f"\nAverage times per sample:")
    print(f"  Load from disk:    {avg_load*1000:.3f}ms ({100*avg_load/avg_total:.1f}%)")
    print(f"  Resample audio:    {avg_resample*1000:.3f}ms ({100*avg_resample/avg_total:.1f}%)")
    print(f"  Normalize audio:   {avg_normalize*1000:.3f}ms ({100*avg_normalize/avg_total:.1f}%)")
    print(f"  Segment audio:     {avg_segment*1000:.3f}ms ({100*avg_segment/avg_total:.1f}%)")
    print(f"  {'─'*40}")
    print(f"  TOTAL:             {avg_total*1000:.3f}ms")
    
    print(f"\nScaling to LibriSpeech train-clean-100 (28,539 samples):")
    total_seconds = avg_total * 28539
    print(f"  Time for 1 epoch:  {total_seconds/60:.1f} minutes ({total_seconds/3600:.2f} hours)")
    print(f"  Time for 100 epochs: {total_seconds*100/3600:.1f} hours ({total_seconds*100/86400:.1f} days)")
    
    print(f"\n{'='*80}")
    print("BOTTLENECK ANALYSIS")
    print(f"{'='*80}")
    
    max_component = max([
        ('Load', avg_load),
        ('Resample', avg_resample),
        ('Normalize', avg_normalize),
        ('Segment', avg_segment)
    ], key=lambda x: x[1])
    
    print(f"\nTop bottleneck: {max_component[0]} ({max_component[1]*1000:.3f}ms, {100*max_component[1]/avg_total:.1f}%)")
    
    if max_component[0] == 'Load':
        print("\nRecommendations:")
        print("  1. This is I/O bound. Consider:")
        print("     - SSD vs HDD? (Check your storage)")
        print("     - Pre-loading entire dataset to RAM if possible")
        print("     - Caching decoded audio to intermediate format")
    elif max_component[0] == 'Resample':
        print("\nRecommendations:")
        print("  1. Ensure data is already at target sample rate if possible")
        print("  2. Cache resampler objects (current code does this)")
    elif max_component[0] in ['Normalize', 'Segment']:
        print("\nRecommendations:")
        print("  1. These are already optimized")
        print("  2. Bottleneck is likely I/O or system memory bandwidth")


if __name__ == '__main__':
    profile_data_loading()
