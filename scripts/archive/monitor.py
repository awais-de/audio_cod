#!/usr/bin/env python3
"""
Monitor training progress in real-time
Shows GPU stats, training metrics, and ETA
"""

import subprocess
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

def get_gpu_stats():
    """Get GPU memory and utilization"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu'], 
                              format='csv', capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                used, total, util = lines[1].split(', ')
                used_mb = int(used.split()[0])
                total_mb = int(total.split()[0])
                util_pct = int(util.split()[0])
                return {
                    'memory_used_mb': used_mb,
                    'memory_total_mb': total_mb,
                    'memory_percent': (used_mb / total_mb) * 100,
                    'utilization': util_pct
                }
    except Exception as e:
        pass
    return None

def parse_training_log():
    """Parse training log file for metrics"""
    log_file = Path('training.log')
    if not log_file.exists():
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Get latest metrics
        metrics = {}
        for line in reversed(lines[-100:]):
            if 'Epoch' in line and '[' in line:
                # Extract batch info
                match = re.search(r'Epoch (\d+)/(\d+).*\[(\d+)/(\d+)\].*Loss: ([\d.]+)', line)
                if match:
                    metrics['current_epoch'] = int(match.group(1))
                    metrics['total_epochs'] = int(match.group(2))
                    metrics['current_batch'] = int(match.group(3))
                    metrics['total_batches'] = int(match.group(4))
                    metrics['loss'] = float(match.group(5))
                    break
        
        return metrics if metrics else None
    except Exception as e:
        return None

def calculate_eta(epoch, batch, total_epochs, batches_per_epoch, time_per_batch):
    """Calculate estimated time to completion"""
    batches_done = (epoch - 1) * batches_per_epoch + batch
    batches_total = total_epochs * batches_per_epoch
    batches_remaining = batches_total - batches_done
    seconds_remaining = batches_remaining * time_per_batch
    return timedelta(seconds=seconds_remaining)

def main():
    """Monitor training progress"""
    print("\n" + "="*80)
    print("Neural Audio Codec - Training Monitor")
    print("="*80 + "\n")
    
    batch_times = []
    last_batch = None
    start_time = time.time()
    
    try:
        while True:
            # Get GPU stats
            gpu_stats = get_gpu_stats()
            
            # Get training metrics
            metrics = parse_training_log()
            
            # Clear screen
            subprocess.run(['clear'] if Path('/bin/clear').exists() else ['cls'], 
                         capture_output=True)
            
            print("="*80)
            print(f"Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
            
            if gpu_stats:
                print(f"GPU Memory: {gpu_stats['memory_used_mb']}MB / {gpu_stats['memory_total_mb']}MB "
                      f"({gpu_stats['memory_percent']:.1f}%)")
                print(f"GPU Utilization: {gpu_stats['utilization']}%\n")
            
            if metrics:
                print(f"Epoch: {metrics['current_epoch']}/{metrics['total_epochs']}")
                print(f"Batch: {metrics['current_batch']}/{metrics['total_batches']} "
                      f"({100*metrics['current_batch']/metrics['total_batches']:.1f}%)")
                print(f"Loss: {metrics['loss']:.4f}\n")
                
                # Calculate ETA if we have batch timing
                if last_batch != metrics['current_batch']:
                    batch_times.append(time.time())
                    if len(batch_times) > 1:
                        time_per_batch = batch_times[-1] - batch_times[-2]
                        if len(batch_times) > 10:
                            # Use rolling average of last 10 batches
                            time_per_batch = (batch_times[-1] - batch_times[-10]) / 9
                        
                        eta = calculate_eta(
                            metrics['current_epoch'],
                            metrics['current_batch'],
                            metrics['total_epochs'],
                            metrics['total_batches'],
                            time_per_batch
                        )
                        
                        hours, remainder = divmod(int(eta.total_seconds()), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        
                        print(f"Time per batch: {time_per_batch:.2f}s")
                        print(f"ETA: {hours}h {minutes}m {seconds}s")
                    
                    last_batch = metrics['current_batch']
            else:
                print("Waiting for training to start...\n")
            
            print("="*80)
            print("Press Ctrl+C to exit")
            print("="*80)
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n✓ Monitor stopped")

if __name__ == "__main__":
    main()
