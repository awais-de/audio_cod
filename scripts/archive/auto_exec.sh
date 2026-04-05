#!/bin/bash
# Simple 2-step executor: Wait 40 min, evaluate, decide, execute
# Run this: ./scripts/auto_exec.sh

set -e

cd /home/muaw1874/Desktop/ac_proj/audio_cod

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    V3 → V4 AUTOMATED EXECUTOR                               ║"
echo "║          Monitoring V3 for 40 minutes, then automatic decision              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "Phase 1: Waiting for V3 training completion..."
echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if V3 is running
if ! pgrep -f "finetune_balanced_v3" > /dev/null; then
    echo "⚠️  V3 training not detected as running!"
    echo "Assuming it already completed..."
else
    echo "✓ V3 training detected (PID: $(pgrep -f finetune_balanced_v3 | head -1))"
    echo ""
    
    # Wait for completion (check every 10 seconds)
    TIMEOUT=2400  # 40 minutes
    ELAPSED=0
    
    while [ $ELAPSED -lt $TIMEOUT ]; do
        if pgrep -f "finetune_balanced_v3" > /dev/null; then
            echo -ne "\r[$(date '+%H:%M:%S')] V3 training in progress... ($((ELAPSED/60)) min elapsed)"
            sleep 10
            ELAPSED=$((ELAPSED + 10))
        else
            echo ""
            echo "✓ V3 training completed at $(date '+%H:%M:%S')"
            break
        fi
    done
    
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo ""
        echo "⏰ 40 minutes elapsed. Proceeding with decision even if still running..."
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "Phase 2: Evaluating V3 results and making decision..."
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Run decision script
./venv/bin/python quick_decision.py

echo ""
echo "✓ Decision complete!"
echo ""
echo "Next steps:"
echo "1. Check V3_DECISION.json for results"
echo "2. If PESQ < 3.5, V4 may have been launched automatically"
echo "3. Monitor V4 training in the background"
echo ""
