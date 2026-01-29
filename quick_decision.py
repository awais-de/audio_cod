#!/usr/bin/env python3
"""
Quick Decision Maker - Evaluates V3 results and launches next phase
Run this AFTER V3 training completes
"""
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime

def wait_for_v3_completion():
    """Wait for V3 training to complete"""
    print(f"\n{'='*80}")
    print(f"Waiting for V3 training to complete...")
    print(f"{'='*80}\n")
    
    while True:
        result = subprocess.run(
            "ps aux | grep finetune_balanced_v3 | grep -v grep",
            shell=True,
            capture_output=True
        )
        if not result.stdout:
            print("✓ V3 training completed!")
            return True
        print(".", end="", flush=True)

def evaluate_v3():
    """Evaluate V3 best checkpoint"""
    print(f"\n\n{'='*80}")
    print(f"EVALUATING V3 RESULTS")
    print(f"{'='*80}\n")
    
    v3_dir = sorted(Path("checkpoints_emergency").glob("pesq_balanced_v3_*"))
    if not v3_dir:
        print("❌ V3 checkpoint directory not found")
        return None, None
    
    v3_checkpoint = v3_dir[-1] / "best.pt"
    if not v3_checkpoint.exists():
        print(f"❌ V3 best checkpoint not found: {v3_checkpoint}")
        return None, None
    
    print(f"Checkpoint: {v3_checkpoint}")
    cmd = f"./venv/bin/python scripts/evaluate_scipy_based.py {v3_checkpoint}"
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output)
    
    # Parse results
    pesq = None
    stoi = None
    for line in output.split('\n'):
        if 'PESQ:' in line:
            try:
                pesq = float(line.split('PESQ:')[1].split()[0])
            except:
                pass
        if 'STOI:' in line:
            try:
                stoi = float(line.split('STOI:')[1].split()[0])
            except:
                pass
    
    return pesq, stoi

def make_decision(v3_pesq, v3_stoi):
    """Make decision based on V3 results"""
    print(f"\n{'='*80}")
    print(f"DECISION ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"V3 Results:")
    print(f"  PESQ: {v3_pesq:.3f} (Target: 3.5, Gap: {3.5 - v3_pesq:.3f})")
    print(f"  STOI: {v3_stoi:.3f} (Target: >0.9, Status: {'✓' if v3_stoi >= 0.9 else '✗'})")
    
    # Compare with all versions
    versions = {
        'Baseline': {'pesq': 2.803, 'stoi': 0.981},
        'V1': {'pesq': 2.941, 'stoi': 0.950},
        'V2': {'pesq': 2.927, 'stoi': 0.967},
        'V3': {'pesq': v3_pesq, 'stoi': v3_stoi}
    }
    
    print(f"\n{'Version':<12} {'PESQ':<10} {'STOI':<10} {'Status':<30}")
    print(f"{'-'*62}")
    for v, data in versions.items():
        pesq_ok = "✓" if data['pesq'] >= 3.5 else f"{(data['pesq']/3.5)*100:.0f}%"
        stoi_ok = "✓" if data['stoi'] >= 0.9 else "✗"
        status = f"PESQ: {pesq_ok:<8} STOI: {stoi_ok}"
        print(f"{v:<12} {data['pesq']:<10.3f} {data['stoi']:<10.3f} {status:<30}")
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}\n")
    
    if v3_pesq >= 3.5:
        print(f"✅ TARGET ACHIEVED!")
        print(f"\n🚀 DEPLOY V3")
        print(f"   PESQ {v3_pesq:.3f} >= 3.5 ✓")
        print(f"   STOI {v3_stoi:.3f} >= 0.9 ✓")
        return "DEPLOY_V3"
    
    elif v3_pesq >= 3.4:
        print(f"⚠️  NEAR TARGET!")
        print(f"\n📊 V3 Performance: PESQ {v3_pesq:.3f} ({(v3_pesq/3.5)*100:.1f}% of target)")
        print(f"\n🔄 OPTIONS:")
        print(f"   1. DEPLOY V3 (Accept 97% of target)")
        print(f"   2. CONTINUE with V4 (Push for full target)")
        print(f"\n📌 Default: CONTINUE (automated V4 launch)")
        return "CONTINUE_V4"
    
    elif v3_pesq >= 3.2:
        print(f"📈 GOOD PROGRESS!")
        print(f"\n📊 V3 Performance: PESQ {v3_pesq:.3f} (91% of target, gap: {3.5-v3_pesq:.3f})")
        print(f"\n🚀 LAUNCH V4")
        print(f"   Strategy: CONTINUED_TRAINING")
        print(f"   Config: lr=1.5e-6, 25 epochs, 2500 files")
        return "CONTINUE_V4"
    
    elif v3_pesq >= 3.0:
        print(f"⚡ PROMISING!")
        print(f"\n📊 V3 Performance: PESQ {v3_pesq:.3f} (86% of target)")
        print(f"\n🚀 LAUNCH V4")
        print(f"   Strategy: AGGRESSIVE_TRAINING")
        print(f"   Config: lr=1e-6, 30 epochs, 3000 files")
        return "CONTINUE_V4"
    
    else:
        print(f"⚠️  Below expectations")
        print(f"\n📊 V3 Performance: PESQ {v3_pesq:.3f} (only {(v3_pesq/3.5)*100:.1f}% of target)")
        print(f"\n🚀 LAUNCH V4 with aggressive strategy")
        print(f"   Reset from V1, larger dataset, lower LR")
        return "AGGRESSIVE_V4"

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
    # Wait for V3
    # (Uncomment if running standalone)
    # wait_for_v3_completion()
    
    # Evaluate V3
    pesq, stoi = evaluate_v3()
    
    if pesq is None or stoi is None:
        print(f"\n❌ Could not evaluate V3 results")
        return
    
    # Make decision
    decision = make_decision(pesq, stoi)
    
    # Save decision
    results = {
        'timestamp': datetime.now().isoformat(),
        'v3_pesq': pesq,
        'v3_stoi': stoi,
        'decision': decision
    }
    
    with open("V3_DECISION.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Decision saved to: V3_DECISION.json")
    
    # Execute if needed
    if decision != "DEPLOY_V3":
        print(f"\n{'='*80}")
        response = input("Launch V4? (y/n): ").strip().lower()
        if response == 'y':
            print(f"{'='*80}\n")
            success = launch_v4()
            if success:
                print(f"\n✓ V4 completed!")
            else:
                print(f"\n❌ V4 failed")
        else:
            print(f"Skipped V4 launch")

if __name__ == "__main__":
    main()
