#!/usr/bin/env python3
"""
Quick script to test different batch sizes and measure performance
"""

import yaml
import subprocess
import time
import sys

def test_batch_size(batch_size, num_samples=1000):
    """Test training with given batch size for num_samples"""
    
    # Read config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify batch size
    config['training']['batch_size'] = batch_size
    config['training']['epochs'] = 1  # Just 1 epoch for testing
    
    # Write modified config
    with open('config_test.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n{'='*60}")
    print(f"Testing with batch_size={batch_size}")
    print(f"{'='*60}")
    
    # Run training with profiler
    cmd = [
        '/mnt/Data/muaw1874/envs/audio_cod/bin/python',
        'train_optimized.py'
    ]
    
    # Modify to use config_test.yaml temporarily
    print(f"Running training with batch_size={batch_size}...")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/mnt/Data/muaw1874/audio_cod')
    elapsed = time.time() - start
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return elapsed


if __name__ == '__main__':
    # Test different batch sizes
    batch_sizes = [1, 2, 4]
    
    for bs in batch_sizes:
        elapsed = test_batch_size(bs, num_samples=1000)
        print(f"\nTime for batch_size={bs}: {elapsed:.1f} seconds")
    
    print("\n" + "="*60)
    print("Recommendation: Choose the largest batch_size that fits in GPU memory")
    print("="*60)
