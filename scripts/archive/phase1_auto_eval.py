#!/usr/bin/env python3
"""
Auto-evaluate Phase 1 results once training completes
"""

import subprocess
import time
import json
from pathlib import Path
import sys

def check_training_complete():
    """Check if Phase 1 training has completed"""
    log_file = Path('/home/muaw1874/Desktop/ac_proj/audio_cod/phase1_training.log')
    
    if not log_file.exists():
        return False
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Check for completion marker
    if 'PHASE 1 TRAINING COMPLETE' in content:
        return True
    
    return False

def extract_best_loss():
    """Extract best loss from log"""
    log_file = Path('/home/muaw1874/Desktop/ac_proj/audio_cod/phase1_training.log')
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    best_loss = None
    for line in reversed(lines):
        if '🏆' in line and 'New best loss' in line:
            try:
                # Extract number from "New best loss: X.XXXXXX"
                loss_str = line.split('New best loss:')[1].strip()
                best_loss = float(loss_str)
                break
            except:
                pass
    
    return best_loss

def get_latest_checkpoint():
    """Get the latest Phase 1 checkpoint directory"""
    checkpoint_base = Path('/home/muaw1874/Desktop/ac_proj/audio_cod/checkpoints_emergency')
    phase1_dirs = sorted(checkpoint_base.glob('phase1_multiscale_*'))
    
    if not phase1_dirs:
        return None
    
    return phase1_dirs[-1]

def main():
    print("\n" + "=" * 80)
    print("PHASE 1 AUTO-EVALUATION MONITOR")
    print("=" * 80)
    
    check_count = 0
    max_checks = 200  # ~100 minutes with 30-second intervals
    
    while check_count < max_checks:
        if check_training_complete():
            print(f"\n✅ Phase 1 training completed!")
            
            best_loss = extract_best_loss()
            checkpoint_dir = get_latest_checkpoint()
            
            print(f"\n📊 Phase 1 Results:")
            print(f"  • Best Loss: {best_loss}")
            print(f"  • Checkpoint: {checkpoint_dir / 'best.pt'}")
            
            print(f"\n🔍 Next: Evaluating Phase 1 checkpoint for PESQ/STOI...")
            
            # Run evaluation
            eval_cmd = [
                '/home/muaw1874/Desktop/ac_proj/audio_cod/venv/bin/python',
                '/home/muaw1874/Desktop/ac_proj/audio_cod/scripts/quick_decision.py',
                str(checkpoint_dir / 'best.pt'),
                '--name', 'Phase1_MultiScale'
            ]
            
            try:
                result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=300)
                print(result.stdout)
                if result.stderr:
                    print("Errors:", result.stderr)
            except Exception as e:
                print(f"❌ Evaluation error: {e}")
            
            return 0
        
        check_count += 1
        remaining_time = (max_checks - check_count) * 30 / 60
        print(f"[{check_count}] Waiting for Phase 1 to complete... (~{remaining_time:.1f} min remaining)", flush=True)
        time.sleep(30)
    
    print(f"\n❌ Timeout: Phase 1 did not complete within {max_checks * 30 / 60:.0f} minutes")
    return 1

if __name__ == '__main__':
    sys.exit(main())
