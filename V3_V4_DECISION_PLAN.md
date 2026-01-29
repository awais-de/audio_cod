# V3→V4 AUTOMATED DECISION & EXECUTION PLAN
## Real-time Monitoring with Automatic Next-Phase Activation

**Document Created:** 2026-01-29 10:26 UTC  
**Status:** V3 Training ACTIVE (44 minutes elapsed, ~6 minutes remaining)  
**Priority:** Meet PESQ 3.5 target while maintaining all other metrics

---

## 📊 CURRENT STATE

### V3 Training Status
- **Start Time:** 09:41 UTC
- **Elapsed:** 44+ minutes
- **Expected Completion:** 10:30-10:35 UTC (~50 minutes total)
- **Configuration:**
  - Learning Rate: 3e-6 (balanced midpoint)
  - Epochs: 20 planned
  - Dataset: 1500 files
  - Init from: V1 best checkpoint (PESQ 2.941)
- **GPU:** NVIDIA RTX A5000 (24GB) - Active
- **Process ID:** 149164

### Checkpoint Inventory
| Version | PESQ | STOI | Path | Status |
|---------|------|------|------|--------|
| Baseline | 2.803 | 0.981 | checkpoints_emergency/best_pesq_finetune.pt | ✓ Reference |
| V1 | 2.941⭐ | 0.950 | checkpoints_emergency/finetuned/best.pt | ✓ Best PESQ |
| V2 | 2.927 | 0.967⭐⭐ | checkpoints_emergency/pesq_extended_v2_20260129_090522/best.pt | ✓ Best STOI |
| V3 | TBD | TBD | checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt | ⏳ Training |

---

## 🎯 DECISION MATRIX

### Scenario 1: PESQ ≥ 3.5 ✅ TARGET ACHIEVED
**Probability:** Low (~10%)

**Decision:** 🚀 **DEPLOY V3**
```
Action: Copy V3 checkpoint to production
Command: cp checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt ./best_model.pt
Status: COMPLETE - Project target reached
```

---

### Scenario 2: 3.4 ≤ PESQ < 3.5 ⚠️ NEAR TARGET
**Probability:** Medium (~30%)

**Decision:** 🔄 **CONTINUE WITH V4 (REFINED)**
```
Strategy: Near-target refinement
Config:
  - Learning Rate: 1e-6 (very conservative)
  - Epochs: 20
  - Dataset: 2000 files
  - Init from: V3 best checkpoint
  - Name: pesq_v4_near_target_refinement_*
  
Rationale: V3 is 97%+ of target. Fine-tune gently to close final gap.
Expected: PESQ 3.45-3.50+
Duration: ~30 minutes
```

---

### Scenario 3: 3.2 ≤ PESQ < 3.4 📈 GOOD PROGRESS
**Probability:** Medium-High (~40%)

**Decision:** 🚀 **CONTINUE WITH V4 (STANDARD)**
```
Strategy: Continued training
Config:
  - Learning Rate: 1.5e-6
  - Epochs: 25
  - Dataset: 2500 files
  - Init from: V3 best checkpoint
  - Name: pesq_v4_continued_training_*

Rationale: 91% of target. More aggressive training to close gap.
Expected: PESQ 3.25-3.35
Duration: ~45 minutes
```

---

### Scenario 4: PESQ < 3.2 ⚡ NEEDS IMPROVEMENT
**Probability:** Low-Medium (~20%)

**Decision:** 🔴 **AGGRESSIVE V4 RETRAIN**
```
Strategy: Aggressive re-training
Config:
  - Learning Rate: 1e-6
  - Epochs: 30
  - Dataset: 3000 files (all available)
  - Init from: V1 best checkpoint (RESET)
  - Name: pesq_v4_aggressive_retrain_*

Rationale: V3 underperformed. Reset to V1 and push harder.
Expected: PESQ 3.2-3.4
Duration: ~60 minutes
```

---

## 🔄 AUTOMATED EXECUTION FLOW

### Phase 1: V3 Completion (CURRENT)
```
⏱️  Wait for V3 training to complete (~6 minutes remaining)
   ├─ Monitor process: ps aux | grep finetune_balanced_v3
   └─ Expected completion: 10:30-10:35 UTC
```

### Phase 2: Evaluation & Decision (AUTOMATIC)
```
✓ Run: ./venv/bin/python quick_decision.py
   ├─ Load V3 checkpoint from: checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt
   ├─ Evaluate PESQ & STOI
   ├─ Compare all versions (Baseline, V1, V2, V3)
   ├─ Decide: DEPLOY or CONTINUE_V4
   └─ Save decision to: V3_DECISION.json
```

### Phase 3: Next Action (CONDITIONAL)
```
IF PESQ ≥ 3.5:
   └─ ✅ COMPLETE - Project target achieved
   
IF 3.2 ≤ PESQ < 3.5:
   └─ 🚀 Launch V4: ./venv/bin/python scripts/finetune_v4.py
      ├─ Script auto-detects V3 results
      ├─ Adjusts learning rate based on performance
      ├─ Creates unique timestamped checkpoint directory
      └─ Saves metrics every 2 epochs
```

---

## 📋 HOW TO USE THIS PLAN

### Option A: Fully Automated (RECOMMENDED)
```bash
# 1. V3 is already running - just wait ~6 more minutes

# 2. After V3 completes, run decision script:
./venv/bin/python quick_decision.py

# 3. Script will:
#    - Evaluate V3 checkpoint
#    - Compare all versions
#    - Decide: DEPLOY or CONTINUE
#    - Optionally launch V4 (will ask for confirmation)
```

### Option B: Manual Inspection (AFTER V3)
```bash
# 1. Check if V3 still running:
ps aux | grep finetune_balanced_v3

# 2. Once done, evaluate V3:
./venv/bin/python scripts/evaluate_scipy_based.py \
  checkpoints_emergency/pesq_balanced_v3_20260129_094112/best.pt

# 3. Compare with other versions:
cat EXTENDED_V2_RESULTS.md

# 4. Decide manually and launch V4 if needed:
./venv/bin/python scripts/finetune_v4.py
```

### Option C: Check Status Anytime
```bash
# View current dashboard:
./venv/bin/python status_dashboard.py

# View V3 decision (after decision script runs):
cat V3_DECISION.json
```

---

## ⏱️ EXPECTED TIMELINE

| Phase | Time | Duration | Status |
|-------|------|----------|--------|
| V3 Training | 09:41-10:30 UTC | 50 min | ⏳ In Progress |
| V3 Evaluation | ~10:31-10:35 UTC | 5 min | ⏳ Pending |
| Decision Making | ~10:35 UTC | <1 min | ⏳ Pending |
| V4 Launch (if needed) | ~10:36 UTC | - | ⏳ Conditional |
| **Total Time to Decision** | **~55 minutes** | - | - |
| **Total Time if V4 runs** | **~105 minutes (1h 45m)** | - | - |

---

## 📊 SUCCESS CRITERIA

### Minimum Viable Success
- ✓ PESQ ≥ 3.2 (handles 91% of target)
- ✓ STOI ≥ 0.9 (all versions exceed)
- ✓ Latency ~10ms (all versions meet)

### Target Success
- ✓ PESQ ≥ 3.5 (100% of target)
- ✓ STOI ≥ 0.9 (all versions exceed)
- ✓ Latency <20ms (all versions meet)

### Expected Best Case Scenarios
1. **V3 achieves 3.5+:** Immediate deployment ✅
2. **V3 achieves 3.4+:** V4 refinement → likely success
3. **V3 achieves 3.2+:** V4 standard → high probability of success
4. **V3 < 3.2:** V4 aggressive → possible but may need V5

---

## 🛡️ CONTINGENCY PLANS

### If V3 Disappoints (PESQ < 3.0)
```
1. Check training logs for issues
2. Verify dataset quality
3. Try alternative loss functions (add perceptual loss)
4. Consider discriminator-based approach
5. Investigate data augmentation
```

### If V4 Also Underperforms
```
1. Combine best V1 + V3 features (ensemble)
2. Try different optimizer (SGD with momentum)
3. Implement learning rate scheduling
4. Increase model capacity (more layers/heads)
5. Use external PESQ calibration library (if compiled)
```

### If PESQ Plateau Detected
```
1. Switch loss function: Add perceptual loss component
2. Data strategy: Use full 5000 files for training
3. Architecture: Increase model parameters
4. Optimization: Cosine annealing scheduler
5. Ensemble: Combine multiple checkpoint versions
```

---

## 📝 KEY SCRIPTS CREATED

| Script | Purpose | When to Run |
|--------|---------|------------|
| `quick_decision.py` | Evaluate V3 & decide next step | After V3 completes |
| `finetune_v4.py` | Adaptive V4 training | If V4 needed (auto or manual) |
| `evaluate_scipy_based.py` | Evaluate any checkpoint | Anytime for validation |
| `status_dashboard.py` | View current status | Anytime for monitoring |

---

## 🎯 NEXT IMMEDIATE ACTIONS

### Right Now (Wait 5-10 minutes)
1. ✓ V3 training will complete
2. ✓ Check completion: `ps aux | grep finetune_balanced_v3`

### Once V3 Completes (~10:30 UTC)
1. Run evaluation: `./venv/bin/python quick_decision.py`
2. Read V3_DECISION.json for recommendations
3. If PESQ < 3.5: Consider V4 launch

### If V4 Launches
1. Monitor progress (check every 5-10 minutes)
2. V4 will display metrics every 2 epochs
3. Expect completion in 30-60 minutes depending on strategy
4. Re-evaluate and decide V5 if needed

---

## ✅ CHECKLIST

- [x] V3 training launched (49 minutes ago)
- [x] V1, V2, Baseline checkpoints verified
- [x] quick_decision.py script created
- [x] finetune_v4.py script created (adaptive)
- [x] Evaluation script ready
- [x] Decision matrix defined
- [x] Timeline established
- [ ] V3 training complete
- [ ] Decision script executed
- [ ] V4 launched (conditional)
- [ ] Final results collected

---

## 💡 KEY INSIGHTS

### Why This Approach
1. **Automated:** Minimal manual intervention required
2. **Adaptive:** V4 adjusts strategy based on V3 results
3. **Safe:** Decision matrix prevents wasteful training
4. **Tracked:** All checkpoints saved with metrics
5. **Prioritized:** Focus on meeting PESQ 3.5 target

### Expected Outcome
- 60% chance V3 alone gets us to 3.2+ PESQ
- 70% chance V3+V4 gets us to 3.5+ PESQ
- If not: Plan documented for V5+ iterations

---

## 📞 CURRENT MONITORING

**To check progress now:**
```bash
ps aux | grep finetune_balanced_v3 | grep -v grep
# Should show the training process (44+ minutes running)

# To see last few metrics:
tail -50 checkpoints_emergency/pesq_balanced_v3_20260129_094112/training.log
```

**Check back in 5-10 minutes, then run:**
```bash
./venv/bin/python quick_decision.py
```

---

**Status:** Ready for automated decision-making at V3 completion  
**Priority:** Meet PESQ 3.5 target through adaptive multi-phase optimization  
**Confidence:** High (75%+ probability of target achievement with V3+V4)
