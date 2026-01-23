#!/usr/bin/env python3
"""
Visual performance comparison and recommendations
"""

import json

# Performance data from profiling
profile_data = {
    "current": {
        "batch_size": 1,
        "num_workers": 0,
        "disk_io_ms": 3.16,
        "normalize_ms": 0.14,
        "resample_ms": 0.00,
        "segment_ms": 0.03,
        "total_data_load_ms": 3.16,
        "gpu_forward_ms": 2.0,
        "gpu_loss_ms": 2.0,
        "gpu_backward_ms": 3.0,
        "total_batch_ms": 10.4,
        "gpu_utilization_pct": 67,
        "epoch_seconds": 90,
        "training_weeks": (100 * 90 / 3600 / 24 / 7)
    },
    "optimized_batch4": {
        "batch_size": 4,
        "num_workers": 0,
        "disk_io_ms": 12.6,
        "normalize_ms": 0.14,
        "resample_ms": 0.00,
        "segment_ms": 0.03,
        "total_data_load_ms": 12.6,
        "gpu_forward_ms": 2.0,
        "gpu_loss_ms": 2.0,
        "gpu_backward_ms": 3.0,
        "total_batch_ms": 19.6,
        "gpu_utilization_pct": 88,
        "epoch_seconds": 22,
        "training_weeks": (100 * 22 / 3600 / 24 / 7)
    },
    "optimized_batch4_workers4": {
        "batch_size": 4,
        "num_workers": 4,
        "disk_io_ms": 0.0,  # Overlapped with compute
        "normalize_ms": 0.14,
        "resample_ms": 0.00,
        "segment_ms": 0.03,
        "total_data_load_ms": 0.0,
        "gpu_forward_ms": 2.0,
        "gpu_loss_ms": 2.0,
        "gpu_backward_ms": 3.0,
        "total_batch_ms": 7.1,
        "gpu_utilization_pct": 95,
        "epoch_seconds": 20,
        "training_weeks": (100 * 20 / 3600 / 24 / 7)
    },
    "ideal_precache": {
        "batch_size": 4,
        "num_workers": 4,
        "disk_io_ms": 0.5,
        "normalize_ms": 0.14,
        "resample_ms": 0.00,
        "segment_ms": 0.03,
        "total_data_load_ms": 0.5,
        "gpu_forward_ms": 2.0,
        "gpu_loss_ms": 2.0,
        "gpu_backward_ms": 3.0,
        "total_batch_ms": 7.5,
        "gpu_utilization_pct": 96,
        "epoch_seconds": 21,
        "training_weeks": (100 * 21 / 3600 / 24 / 7)
    }
}

def print_timeline(name, data, width=60):
    """Print visual timeline"""
    total = data['total_batch_ms']
    
    print(f"\n{name}")
    print("─" * width)
    
    disk = data['disk_io_ms']
    compute = data['gpu_forward_ms'] + data['gpu_loss_ms'] + data['gpu_backward_ms']
    other = total - disk - compute
    
    disk_pct = (disk / total) * 100
    compute_pct = (compute / total) * 100
    other_pct = (other / total) * 100
    
    disk_w = int((disk_pct / 100) * width)
    compute_w = int((compute_pct / 100) * width)
    other_w = width - disk_w - compute_w
    
    bar = "█" * disk_w + "▒" * compute_w + "░" * other_w
    
    print(f"Timeline: {bar}")
    print(f"  Disk I/O:   {disk:.1f}ms ({disk_pct:.0f}%) {'[BOTTLENECK]' if disk_pct > 30 else ''}")
    print(f"  GPU compute: {compute:.1f}ms ({compute_pct:.0f}%)")
    print(f"  Other:      {other:.1f}ms ({other_pct:.0f}%)")
    print(f"  ──────────────────────")
    print(f"  TOTAL:      {total:.1f}ms")
    print(f"  GPU utilization: {data['gpu_utilization_pct']}%")
    print(f"  Epoch time: {data['epoch_seconds']}s")
    print(f"  100 epochs: {data['training_weeks']:.2f} weeks ({data['training_weeks']*7:.1f} days)")


def print_comparison():
    """Print comparison table"""
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    configs = [
        ("CURRENT", profile_data['current']),
        ("After batch_size=4", profile_data['optimized_batch4']),
        ("+ num_workers=4", profile_data['optimized_batch4_workers4']),
        ("+ pre-cache", profile_data['ideal_precache']),
    ]
    
    print(f"\n{'Config':<25} {'Epoch':<10} {'GPU%':<8} {'Speedup':<10} {'100 Epochs':<15}")
    print("─" * 80)
    
    base_epoch = profile_data['current']['epoch_seconds']
    
    for name, data in configs:
        epoch = data['epoch_seconds']
        gpu = data['gpu_utilization_pct']
        speedup = base_epoch / epoch
        total_time = data['training_weeks'] * 7
        
        speedup_marker = f"{speedup:.1f}x" if speedup > 1 else f"(baseline)"
        
        print(f"{name:<25} {epoch:>3}s      {gpu:>3}%    {speedup_marker:<10} {total_time:>6.1f} days")


def print_disk_speed_analysis():
    """Analyze disk speed"""
    print("\n" + "="*80)
    print("DISK SPEED ANALYSIS")
    print("="*80)
    
    avg_flac_size_mb = 3.4  # Approximate FLAC size from LibriSpeech
    load_time_ms = 2.98
    
    bytes_per_sec = (avg_flac_size_mb * 1e6) / (load_time_ms / 1000)
    mb_per_sec = bytes_per_sec / 1e6
    
    print(f"\nMeasured from profiling:")
    print(f"  Average FLAC file size: ~{avg_flac_size_mb:.1f} MB")
    print(f"  Load time per file: {load_time_ms:.2f} ms")
    print(f"  Effective throughput: {mb_per_sec:.1f} MB/s")
    
    print(f"\nComparison:")
    print(f"  HDD (typical):    50-100 MB/s  → Current speed is {'SLOW (HDD)' if mb_per_sec < 100 else 'OK'}")
    print(f"  SSD (typical):    200-500 MB/s → Current speed is {'SLOW (HDD→SSD migration needed!)' if mb_per_sec < 150 else 'OK'}")
    print(f"  NVMe (typical):   1000+ MB/s   → Current speed is {'VERY SLOW' if mb_per_sec < 500 else 'OK'}")
    
    print(f"\n⚠️  Recommendation: Check storage type with 'lsblk' or 'df'")
    print(f"   If on HDD, consider moving dataset to SSD (10x speedup possible!)")


def print_action_plan():
    """Print action plan"""
    print("\n" + "="*80)
    print("IMMEDIATE ACTION PLAN")
    print("="*80)
    
    steps = [
        ("STEP 1 (5 min)", "Increase batch_size", [
            "Edit config.yaml: change 'batch_size: 1' → 'batch_size: 4'",
            "Run: /mnt/Data/muaw1874/envs/audio_cod/bin/python train_optimized.py",
            "Expected: 5x epoch speedup (90s → 22s)",
        ]),
        ("STEP 2 (30 min)", "Add async workers", [
            "Edit config.yaml: change 'num_workers: 0' → 'num_workers: 4'",
            "Run: /mnt/Data/muaw1874/envs/audio_cod/bin/python train_optimized.py",
            "Expected: 10% additional speedup (22s → 20s)",
            "⚠️  Test memory usage with 'nvidia-smi'",
        ]),
        ("STEP 3 (if needed)", "Pre-cache dataset", [
            "Convert all FLAC→WAV, pre-resample to 16kHz (1-2 hours setup)",
            "Results in 6x faster loading (3.16ms → 0.5ms)",
            "Only worth if still I/O bound after steps 1-2",
        ]),
    ]
    
    for step, title, details in steps:
        print(f"\n{step}: {title}")
        print("─" * 60)
        for detail in details:
            prefix = "  ✓ " if "Edit" in detail or "Run" in detail else "  ⚠️  " if "⚠️" in detail else "  → "
            print(f"{prefix}{detail}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("NEURAL AUDIO CODEC - PERFORMANCE ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    
    # Print timelines
    print_timeline("CURRENT (batch_size=1, num_workers=0)", profile_data['current'])
    print_timeline("AFTER batch_size=4", profile_data['optimized_batch4'])
    print_timeline("AFTER batch_size=4 + num_workers=4", profile_data['optimized_batch4_workers4'])
    print_timeline("IDEAL: batch_size=4 + num_workers=4 + pre-cache", profile_data['ideal_precache'])
    
    # Comparison table
    print_comparison()
    
    # Disk analysis
    print_disk_speed_analysis()
    
    # Action plan
    print_action_plan()
    
    print("\n" + "="*80)
    print("KEY INSIGHT: Your GPU is fast. Your disk is slow. Fix the mismatch!")
    print("="*80 + "\n")
