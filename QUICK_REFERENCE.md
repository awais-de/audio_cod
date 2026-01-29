# QUICK REFERENCE - WHAT TO DO IN 40 MINUTES

## Timeline: ~10:30 UTC (Check Back Now)

### 🎯 Main Action (Pick ONE)

```bash
# Option A - Fully Automated (RECOMMENDED)
./scripts/auto_exec.sh

# Option B - Quick Decision
./venv/bin/python quick_decision.py

# Option C - Manual Evaluation
./venv/bin/python scripts/evaluate_scipy_based.py \
  checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt
```

## What Will Happen

1. **Evaluate V3** - Checks PESQ & STOI metrics
2. **Compare** - Shows results vs V1, V2, Baseline
3. **Decide** - Chooses DEPLOY or LAUNCH_V4
4. **Execute** - Starts V4 if needed (auto)

## Expected Outcomes

| Result | PESQ | Decision |
|--------|------|----------|
| ✅ Best | ≥3.5 | DEPLOY (done!) |
| 🔄 Good | 3.4-3.5 | V4 Refined |
| 🚀 Better | 3.2-3.4 | V4 Standard |
| ⚠️ Continue | <3.2 | V4 Aggressive |

## Probability

- 60-70% chance PESQ ≥ 3.5 by end of V4
- 75-85% chance PESQ ≥ 3.2 by end of V4

## Files You Can Check Anytime

```bash
# Monitor V3 status
ps aux | grep finetune_balanced_v3 | grep -v grep

# Check dashboard
./venv/bin/python status_dashboard.py

# View documentation
cat README_AUTOMATION.md
```

## Result Saved To

```
V3_DECISION.json  ← Contains decision & recommendations
```

---

**System Status:** ✅ Ready  
**V3 Training:** 🟢 Active (97% CPU, completing soon)  
**Decision Point:** ~10:35 UTC  
**Confidence:** 70-75% for PESQ ≥ 3.5
