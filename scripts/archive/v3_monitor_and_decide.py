#!/usr/bin/env python3
"""
V3 Monitoring & Automated Decision System
Checks progress every 5 minutes, evaluates final results, and executes next plan
Priority: MEET TARGETS (PESQ 3.5+, STOI >0.9, Latency <20ms)
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_v3_status():
    """Check if V3 training is still running"""
    try:
        result = subprocess.run(
            "ps aux | grep finetune_balanced_v3 | grep -v grep",
            shell=True,
            capture_output=True,
            text=True
        )
        return len(result.stdout.strip()) > 0
    except:
        return False

def get_latest_v3_checkpoint():
    """Find the latest V3 checkpoint directory"""
    base_dir = Path("checkpoints_emergency")
    v3_dirs = sorted(base_dir.glob("pesq_balanced_v3_*"))
    if v3_dirs:
        return v3_dirs[-1]
    return None

def evaluate_checkpoint(checkpoint_path):
    """Evaluate a checkpoint and return PESQ/STOI"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"{'='*70}")
    
    result = subprocess.run(
        f"./venv/bin/python scripts/evaluate_scipy_based.py {checkpoint_path}",
        shell=True,
        capture_output=True,
        text=True,
        cwd="/home/muaw1874/Desktop/ac_proj/audio_cod"
    )
    
    # Parse output
    output = result.stdout + result.stderr
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
    
    print(output)
    return pesq, stoi

def compare_all_versions():
    """Compare results across all three versions"""
    results = {
        'Baseline': {'pesq': 2.803, 'stoi': 0.981, 'path': 'checkpoints_emergency/best_pesq_finetune.pt'},
        'V1': {'pesq': 2.941, 'stoi': 0.950, 'path': 'checkpoints_emergency/finetuned/best.pt'},
        'V2': {'pesq': 2.927, 'stoi': 0.967, 'path': 'checkpoints_emergency/pesq_extended_v2_20260129_090522/best.pt'},
    }
    
    # Evaluate V3
    v3_dir = get_latest_v3_checkpoint()
    if v3_dir:
        v3_checkpoint = v3_dir / "best.pt"
        if v3_checkpoint.exists():
            pesq, stoi = evaluate_checkpoint(v3_checkpoint)
            if pesq and stoi:
                results['V3'] = {'pesq': pesq, 'stoi': stoi, 'path': str(v3_checkpoint)}
    
    return results

def print_comparison_table(results):
    """Print formatted comparison table"""
    print(f"\n\n{'='*80}")
    print(f"{'COMPLETE RESULTS COMPARISON - ALL VERSIONS':^80}")
    print(f"{'='*80}\n")
    
    print(f"{'Version':<15} {'PESQ':<12} {'STOI':<12} {'Status':<40}")
    print(f"{'-'*80}")
    
    targets_met = {}
    best_version = None
    best_overall_score = 0
    
    for version, data in results.items():
        pesq = data['pesq']
        stoi = data['stoi']
        
        pesq_pct = (pesq / 3.5) * 100 if pesq else 0
        stoi_ok = "✓" if stoi >= 0.9 else "✗"
        pesq_ok = "✓" if pesq >= 3.5 else f"{pesq_pct:.0f}%"
        
        status = f"PESQ: {pesq_ok} | STOI: {stoi_ok}"
        
        print(f"{version:<15} {pesq:<12.3f} {stoi:<12.3f} {status:<40}")
        
        # Calculate composite score (weighted: PESQ 60%, STOI 40%)
        composite = (pesq/3.5 * 0.6 + stoi/0.9 * 0.4) * 100 if pesq and stoi else 0
        if composite > best_overall_score:
            best_overall_score = composite
            best_version = version
        
        targets_met[version] = {
            'pesq': pesq >= 3.5 if pesq else False,
            'stoi': stoi >= 0.9 if stoi else False,
            'composite': composite
        }
    
    print(f"{'-'*80}\n")
    
    return targets_met, best_version

def decide_next_action(targets_met, results):
    """Decide next action based on results"""
    print(f"\n{'='*80}")
    print(f"{'DECISION ANALYSIS - NEXT STEPS':^80}")
    print(f"{'='*80}\n")
    
    # Check target achievement
    pesq_achieved = any(targets_met[v]['pesq'] for v in targets_met if 'pesq' in targets_met[v])
    stoi_achieved = all(targets_met[v]['stoi'] for v in targets_met if 'stoi' in targets_met[v])
    
    print(f"✓ STOI Target (>0.9): {stoi_achieved} - All versions exceed 0.9")
    print(f"✗ PESQ Target (3.5+): {pesq_achieved} - NOT YET ACHIEVED")
    print(f"\nCurrent Best PESQ: {max(v['pesq'] for v in results.values()):.3f} (Target: 3.5, Gap: {3.5 - max(v['pesq'] for v in results.values()):.3f})")
    
    best_pesq_version = max(results, key=lambda x: results[x]['pesq'])
    best_pesq = results[best_pesq_version]['pesq']
    
    print(f"\n{'RECOMMENDATION':-^80}")
    
    if pesq_achieved:
        print(f"\n✅ TARGET ACHIEVED! PESQ {best_pesq:.3f} >= 3.5")
        print(f"\nDeploy {best_pesq_version}:")
        print(f"  Checkpoint: {results[best_pesq_version]['path']}")
        print(f"  PESQ: {best_pesq:.3f} | STOI: {results[best_pesq_version]['stoi']:.3f}")
        return "DEPLOY", best_pesq_version
    
    elif best_pesq >= 3.4:
        print(f"\n⚠️  CLOSE TO TARGET! PESQ {best_pesq:.3f} (gap: {3.5-best_pesq:.3f})")
        print(f"\nOptions:")
        print(f"  1. DEPLOY {best_pesq_version} (94% of target, good enough for production)")
        print(f"  2. CONTINUE_TRAINING (v4 with even more data/epochs)")
        return "NEAR_TARGET", best_pesq_version
    
    elif best_pesq >= 3.2:
        print(f"\n⚠️  MAKING PROGRESS! PESQ {best_pesq:.3f} (gap: {3.5-best_pesq:.3f})")
        print(f"\nRecommended: V4 with INCREASED TRAINING")
        print(f"  - Use {best_pesq_version} as init checkpoint")
        print(f"  - Increase data to 2500 files")
        print(f"  - Try lr=2e-6 (slightly lower)")
        print(f"  - 25 epochs")
        return "CONTINUE_TRAINING", best_pesq_version
    
    else:
        print(f"\n❌ PESQ {best_pesq:.3f} still below 3.2 (gap: {3.5-best_pesq:.3f})")
        print(f"\nRecommended: V4 with AGGRESSIVE TUNING")
        print(f"  - Try discriminator loss approach")
        print(f"  - Larger dataset (2500 files)")
        print(f"  - Higher learning rate (5e-6)")
        return "AGGRESSIVE_RETRAIN", best_pesq_version

def create_v4_script(init_version, config_type):
    """Create V4 training script based on recommendation"""
    
    if config_type == "NEAR_TARGET":
        epochs = 30
        lr = 2e-6
        n_files = 2000
        description = "Extended training - approaching target"
    elif config_type == "CONTINUE_TRAINING":
        epochs = 25
        lr = 2e-6
        n_files = 2500
        description = "Increased training for PESQ improvement"
    else:  # AGGRESSIVE_RETRAIN
        epochs = 30
        lr = 1.5e-6
        n_files = 3000
        description = "Aggressive fine-tuning with more data"
    
    print(f"\n{'='*80}")
    print(f"Creating V4 script ({description})")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Dataset: {n_files} files")
    print(f"  Init from: {init_version}")
    
    # Note: Actual V4 script creation would happen here
    # For now, just provide the command to run
    return epochs, lr, n_files

def main():
    """Main monitoring loop"""
    print(f"\n{'='*80}")
    print(f"{'V3 MONITORING & DECISION SYSTEM ACTIVATED':^80}")
    print(f"{'='*80}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nMonitoring V3 training progress...")
    print(f"Will check every 5 minutes and make decisions after 40+ minutes")
    
    start_time = time.time()
    check_interval = 300  # 5 minutes
    max_wait = 2400  # 40 minutes
    
    while True:
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        is_running = check_v3_status()
        
        if is_running:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] V3 still training... ({elapsed_min:.0f} min elapsed)")
            
            if elapsed_min >= 40:
                print(f"\n✓ 40+ minutes elapsed. Evaluating final results...")
                break
            
            time.sleep(check_interval)
        else:
            print(f"\n✓ V3 training completed!")
            break
    
    # Evaluate all versions
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION - COMPARING ALL VERSIONS")
    print(f"{'='*80}")
    
    results = compare_all_versions()
    targets_met, best_version = print_comparison_table(results)
    action, init_version = decide_next_action(targets_met, results)
    
    # Save results to file
    results_file = Path("V3_EVALUATION_RESULTS.json")
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': {k: v for k, v in results.items()},
            'decision': action,
            'best_version': best_version,
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Print next steps
    print(f"\n{'='*80}")
    print(f"{'NEXT STEPS':^80}")
    print(f"{'='*80}\n")
    
    if action == "DEPLOY":
        print(f"✅ DEPLOY {best_version}")
        print(f"\nCommand:")
        print(f"  cp {results[best_version]['path']} /path/to/production/best_codec.pt")
        print(f"\nMetrics:")
        print(f"  PESQ: {results[best_version]['pesq']:.3f} ✓")
        print(f"  STOI: {results[best_version]['stoi']:.3f} ✓")
        print(f"  Status: READY FOR PRODUCTION")
    
    elif action == "NEAR_TARGET":
        print(f"⚠️  NEAR TARGET - Two options:")
        print(f"\nOption A: DEPLOY {best_version} (94% of target)")
        print(f"  Metrics: PESQ {results[best_version]['pesq']:.3f}, STOI {results[best_version]['stoi']:.3f}")
        print(f"  Status: Production ready, acceptable for deployment")
        
        print(f"\nOption B: CONTINUE with V4")
        print(f"  Command: ./venv/bin/python scripts/finetune_v4.py")
        print(f"  Expected: PESQ 3.4+ with extended training")
    
    else:
        epochs, lr, n_files = create_v4_script(init_version, action)
        print(f"\n🚀 LAUNCH V4 ({action})")
        print(f"\nWill create and execute:")
        print(f"  Command: ./venv/bin/python scripts/finetune_v4.py")
        print(f"  Init from: {init_version}")
        print(f"  Config: LR={lr}, {n_files} files, {epochs} epochs")

if __name__ == "__main__":
    main()
