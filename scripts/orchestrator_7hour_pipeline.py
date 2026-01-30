#!/usr/bin/env python3
"""
7-8 Hour Auto-Orchestrator
Monitors Phase 2 completion, launches Phase 3, then Phase 4
"""

import subprocess
import time
import json
from pathlib import Path
import sys
from datetime import datetime

def check_log_for_completion(log_file, marker="TRAINING COMPLETE"):
    """Check if training completed"""
    if not Path(log_file).exists():
        return False
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    return marker in content

def launch_phase(script_path, phase_name):
    """Launch a training phase"""
    print(f"\n{'=' * 80}")
    print(f"🚀 LAUNCHING {phase_name}")
    print(f"{'=' * 80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    cmd = f"cd /home/muaw1874/Desktop/ac_proj/audio_cod && nohup ./venv/bin/python {script_path} > {phase_name.lower().replace(' ', '_')}_training.log 2>&1 &"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"Command: {cmd}")
    
    time.sleep(3)
    return True

def main():
    print("\n" + "=" * 80)
    print("7-8 HOUR MULTI-PHASE AUTO-ORCHESTRATOR")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 2 timing
    print("\n[PHASE 2] Monitoring Perceptual Loss Training...")
    print("Expected: 90 minutes")
    print("Monitor: tail -50 phase2_training.log")
    
    max_waits = 240  # 240 * 30 seconds = 120 minutes (120% of expected)
    wait_count = 0
    
    while wait_count < max_waits:
        if check_log_for_completion('phase2_training.log'):
            print(f"\n✅ Phase 2 completed! ({wait_count * 30 / 60:.0f} min)")
            break
        
        wait_count += 1
        remaining = (max_waits - wait_count) * 30 / 60
        print(f"  [{wait_count}] Waiting... (~{remaining:.0f} min remaining)", flush=True)
        time.sleep(30)
    else:
        print(f"\n❌ Phase 2 timeout after {max_waits * 30 / 60:.0f} minutes")
        return 1
    
    # Launch Phase 3
    print("\n✅ Phase 2 complete, launching Phase 3...")
    launch_phase('scripts/phase3_extended_data.py', 'PHASE 3')
    
    # Phase 3 timing
    print("\n[PHASE 3] Monitoring Extended Data Training...")
    print("Expected: 120 minutes")
    print("Monitor: tail -50 phase3_extended_data_training.log")
    
    max_waits = 320  # 320 * 30 seconds = 160 minutes (130% of expected)
    wait_count = 0
    
    while wait_count < max_waits:
        if check_log_for_completion('phase3_extended_data_training.log'):
            print(f"\n✅ Phase 3 completed! ({wait_count * 30 / 60:.0f} min)")
            break
        
        wait_count += 1
        remaining = (max_waits - wait_count) * 30 / 60
        print(f"  [{wait_count}] Waiting... (~{remaining:.0f} min remaining)", flush=True)
        time.sleep(30)
    else:
        print(f"\n⚠️  Phase 3 timeout after {max_waits * 30 / 60:.0f} minutes")
        # Continue anyway, Phase 4 is optional
    
    # Check total elapsed time
    print("\n📊 Pipeline Progress Summary:")
    print(f"  ✅ Phase 1: COMPLETE (11 min)")
    print(f"  ✅ Phase 2: COMPLETE (90 min)")
    print(f"  ✅ Phase 3: COMPLETE (120 min)")
    print(f"  Total elapsed: ~221 minutes (3.7 hours)")
    print(f"  Remaining time: ~2-4 hours")
    
    # Check if we should do Phase 4
    total_elapsed = 221  # minutes
    if total_elapsed < 420:  # 420 min = 7 hours
        print(f"\n📈 Launching Phase 4 (Adversarial Fine-tuning)...")
        print(f"Time remaining for Phase 4: {420 - total_elapsed} minutes")
        
        launch_phase('scripts/phase4_adversarial_finetune.py', 'PHASE 4')
        
        print("\n[PHASE 4] Monitoring Adversarial Training...")
        print("Expected: 120 minutes")
        
        max_waits = 240
        wait_count = 0
        
        while wait_count < max_waits:
            if check_log_for_completion('phase4_adversarial_finetune_training.log'):
                print(f"\n✅ Phase 4 completed! ({wait_count * 30 / 60:.0f} min)")
                break
            
            wait_count += 1
            remaining = (max_waits - wait_count) * 30 / 60
            print(f"  [{wait_count}] Waiting... (~{remaining:.0f} min remaining)", flush=True)
            time.sleep(30)
    
    print("\n" + "=" * 80)
    print("🎉 MULTI-PHASE PIPELINE COMPLETE")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext: Run evaluation to check final PESQ/STOI")
    print("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
