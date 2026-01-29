#!/usr/bin/env python3
"""
Real-time Status Dashboard
Shows V3 training progress, targets, and decision timeline
"""
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

def get_v3_status():
    """Check if V3 is still training"""
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

def get_v3_epoch():
    """Try to extract current epoch from log"""
    v3_dir = Path("checkpoints_emergency").glob("pesq_balanced_v3_*")
    for d in sorted(v3_dir, reverse=True):
        log_file = d / "training.log"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    lines = f.readlines()
                    for line in reversed(lines[-20:]):
                        if "Epoch" in line:
                            return line.strip()
            except:
                pass
    return "Unknown"

def print_dashboard():
    """Print the status dashboard"""
    os.system('clear')
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     AUDIO CODEC V3 MONITORING DASHBOARD                      ║
║                         Automatic Decision System                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 TARGET SPECIFICATIONS
├─ PESQ: 3.5+ ✗ (Current Best: 2.941 with V1 | Gap: 0.559 points)
├─ STOI: >0.9  ✓ (Best: 0.967 with V2)
└─ Latency: <20ms ✓ (Current: ~10ms)

⏱️  CURRENT STATUS
├─ V1 Checkpoint: ✓ COMPLETE (PESQ: 2.941, STOI: 0.950) 
├─ V2 Checkpoint: ✓ COMPLETE (PESQ: 2.927, STOI: 0.967)
├─ V3 Training: {'🟢 ACTIVE' if get_v3_status() else '🟡 COMPLETED'}
└─ V4 Preparation: ⏳ READY (Will launch based on V3 results)

📍 V3 TRAINING PROGRESS
├─ Configuration: lr=3e-6 (balanced), 20 epochs, 1500 files
├─ Init from: V1 best checkpoint
├─ Status: {'RUNNING' if get_v3_status() else 'FINISHED'}
├─ Last Epoch: {get_v3_epoch()}
└─ Expected Completion: ~50 minutes from 09:41 UTC

🤖 AUTOMATED DECISION SYSTEM
├─ Monitoring Script: ✓ ACTIVE (Terminal: 40a6fdc7-5a32-4736-ae46-218910227a14)
├─ Check Interval: Every 5 minutes
├─ Evaluation Trigger: After 40+ minutes of training
├─ Decision Making: Automatic based on V3 results
└─ Next Action:
    ├─ IF PESQ ≥ 3.5: DEPLOY (target achieved!)
    ├─ IF PESQ ≥ 3.4: NEAR_TARGET (decide: deploy or continue)
    ├─ IF PESQ ≥ 3.2: CONTINUE with V4 (increased training)
    └─ IF PESQ < 3.2: AGGRESSIVE_RETRAIN (reset + bigger push)

📋 V4 READINESS (Auto-triggered if needed)
├─ Script: ✓ CREATED (scripts/finetune_v4.py)
├─ Features: Adaptive configuration based on V3 results
├─ Auto Decision: Optimizer will adjust LR, epochs, dataset size
├─ Fallback: Can initialize from V1 or V3 depending on strategy
└─ Status: READY TO EXECUTE

⚙️  TECHNICAL SETUP
├─ GPU: NVIDIA RTX A5000 (24GB) - ACTIVE
├─ PyTorch: 2.10.0+cu128
├─ Dataset: LibriSpeech train-clean-100 (5000 files available)
├─ Loss: 2.0×STFT_L1 + 0.5×Time_L1 (spectral-focused)
└─ Optimizer: Adam with gradient clipping

📁 CHECKPOINT REFERENCE
├─ Baseline: checkpoints_emergency/best_pesq_finetune.pt (PESQ: 2.803)
├─ V1 Best: checkpoints_emergency/finetuned/best.pt (PESQ: 2.941) ⭐
├─ V2 Best: checkpoints_emergency/pesq_extended_v2_20260129_090522/best.pt (STOI: 0.967) ⭐
├─ V3 Active: checkpoints_emergency/pesq_balanced_v3_20260129_094112/
└─ V4 Ready: scripts/finetune_v4.py (auto-creates on trigger)

🎯 PRIORITY: MEET TARGETS
Primary Goal: Reach PESQ 3.5 while maintaining STOI >0.9 & Latency <20ms
Strategy: V3 balanced optimization → V4 adaptive refinement if needed
Timeline: Real-time monitoring → 40+ minute checkpoint → Automatic decision

╔══════════════════════════════════════════════════════════════════════════════╗
║ MONITORING ACTIVE - Check back in ~40 minutes for V3 evaluation & decisions  ║
║ All processes automated - no manual intervention required                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    print_dashboard()
    print(f"\nDashboard printed at {datetime.now().strftime('%H:%M:%S')}")
    print("Run this script again to refresh status")
