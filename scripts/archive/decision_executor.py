#!/usr/bin/env python3
"""
Decision Executor - Automatically runs next phase based on monitoring results
Checks V3 monitoring output and decides: DEPLOY or LAUNCH_V4
"""
import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

def check_monitoring_complete():
    """Check if monitoring has finished and produced results"""
    results_file = Path("V3_EVALUATION_RESULTS.json")
    return results_file.exists()

def load_monitoring_results():
    """Load the decision from monitoring"""
    results_file = Path("V3_EVALUATION_RESULTS.json")
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None

def execute_decision():
    """Execute the decision from monitoring"""
    
    print(f"\n{'='*80}")
    print(f"DECISION EXECUTOR - AWAITING V3 MONITORING COMPLETION")
    print(f"{'='*80}\n")
    
    max_wait = 3600  # 1 hour max wait
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if check_monitoring_complete():
            results = load_monitoring_results()
            print(f"✓ Monitoring complete!")
            print(f"\nResults from V3 Evaluation:")
            print(json.dumps(results, indent=2))
            
            decision = results.get('decision')
            best_version = results.get('best_version')
            
            print(f"\n{'='*80}")
            print(f"DECISION: {decision}")
            print(f"BEST VERSION: {best_version}")
            print(f"{'='*80}\n")
            
            if decision == "DEPLOY":
                print(f"✅ TARGET ACHIEVED!")
                print(f"\nDeploying {best_version}")
                print(f"Next step: Copy checkpoint to production")
                print(f"\nCommand:")
                print(f"  cp {results['results'][best_version]['path']} ./best_model.pt")
                return "DEPLOY", best_version
            
            elif decision == "NEAR_TARGET":
                print(f"⚠️  NEAR TARGET - DECISION REQUIRED")
                print(f"\nOptions:")
                print(f"  1. DEPLOY: Accept current {best_version} ({results['results'][best_version]['pesq']:.3f} PESQ)")
                print(f"  2. CONTINUE: Launch V4 for final push")
                print(f"\nDefault: CONTINUE (launching V4 for final optimization)")
                return "CONTINUE", best_version
            
            else:  # CONTINUE_TRAINING or AGGRESSIVE_RETRAIN
                print(f"🚀 CONTINUE WITH V4")
                print(f"\nLaunching V4 training with adaptive strategy")
                print(f"Init from: {best_version}")
                print(f"Strategy: {decision}")
                return "CONTINUE", best_version
        
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for V3 monitoring... ({elapsed/60:.0f} min elapsed)", end='\r')
        time.sleep(30)
    
    print(f"\n❌ Monitoring did not complete within 1 hour")
    return None, None

def launch_v4():
    """Launch V4 training"""
    print(f"\n{'='*80}")
    print(f"LAUNCHING V4 TRAINING")
    print(f"{'='*80}\n")
    
    cmd = "cd /home/muaw1874/Desktop/ac_proj/audio_cod && ./venv/bin/python scripts/finetune_v4.py"
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    decision, best_version = execute_decision()
    
    if decision == "CONTINUE":
        print(f"\n{'='*80}")
        print(f"Preparing to launch V4...")
        print(f"{'='*80}\n")
        
        success = launch_v4()
        if success:
            print(f"\n✓ V4 training completed successfully")
            print(f"Next step: Evaluate V4 results")
        else:
            print(f"\n❌ V4 training failed")
    
    elif decision == "DEPLOY":
        print(f"\nDeployment ready. Manual next step required.")
    
    else:
        print(f"\n❌ No decision could be made")

if __name__ == "__main__":
    main()
