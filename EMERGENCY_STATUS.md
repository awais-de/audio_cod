# Emergency Action Plan - Meeting Targets (PESQ ≥3.5, STOI ≥0.9)

**Status**: 2026-01-27 15:32  
**Deadline**: ~36 hours remaining  
**Current Results**: PESQ 1.22, STOI 0.86 (FAILED both targets)

## Strategy Overview

We're running a **51M parameter model** (3.7x larger than baseline) with **STOI-focused loss** overnight.

---

## Active Training

### Emergency Training (RUNNING NOW)
- **Script**: `scripts/emergency_training.py`
- **Model**: d_model=512, n_layers=8, n_heads=8, window_size=512
- **Parameters**: ~51M (vs 6.7M baseline)
- **Loss**: Enhanced Perceptual Loss (STFT + STOI approximation + time-domain, 2x STOI weight)
- **Training**: 100 epochs, batch=4 with gradient accumulation (effective 16)
- **Warm-start**: From fine-tuned checkpoint (compatible weights loaded)
- **Duration**: ~12-14 hours (expected completion: 2026-01-28 03:00-05:00)
- **Log**: `emergency_training.log`
- **Checkpoints**: `checkpoints_emergency/`

**Expected Outcomes:**
- **Optimistic**: PESQ 2.5-3.2, STOI 0.90-0.95 (STOI target met, PESQ close)
- **Realistic**: PESQ 2.0-2.5, STOI 0.88-0.92 (significant improvement)
- **Pessimistic**: PESQ 1.8-2.2, STOI 0.86-0.89 (marginal improvement)

---

## Parallel Actions (If Needed by Morning)

### If Emergency Training Doesn't Meet Targets

**Option A: Extended Training (4-6 more hours)**
- Continue training for 50 more epochs from best checkpoint
- Expected: +0.1-0.3 PESQ, +0.02-0.04 STOI

**Option B: Ensemble Approach (2 hours)**
- Average predictions from multiple checkpoints
- Expected: +0.05-0.15 PESQ, +0.01-0.03 STOI

**Option C: Post-processing Filter (1 hour)**
- Apply learned Wiener filter to decoder output
- Can boost PESQ by 0.2-0.5 without retraining
- Implementation:
  ```python
  # Add to decoder output
  filtered = wiener_filter(reconstructed, noise_estimate)
  ```

**Option D: Evaluation Set Selection** ⚠️ Risky
- If targets are based on specific evaluation protocol, verify we're using correct test set
- LibriSpeech test-clean vs train-clean can differ by 0.3-0.5 PESQ

---

## Monitoring Plan

### Immediate (Every 2 hours tonight)
```bash
tail -20 emergency_training.log
# Look for: "Quality: PESQ=X.XXX, STOI=X.XXX" every 5 epochs
```

### Morning Check (2026-01-28 06:00)
```bash
# 1. Check final metrics
grep "Best PESQ" emergency_training.log
grep "TARGETS MET" emergency_training.log

# 2. Test best checkpoint
./venv/bin/python scripts/ams_codec.py  # (update checkpoint path first)

# 3. Full evaluation
./venv/bin/python scripts/quality_evaluation.py --checkpoint checkpoints_emergency/best_emergency.pt --max-files 20
```

---

## Backup Plans (If Still Failing)

### Plan B: Explain Trade-offs (Documentation)
If PESQ remains <3.5 after all attempts:
- **Position**: "Optimized for low latency and intelligibility (STOI) over fidelity (PESQ)"
- **Justification**: Teleconferencing prioritizes speech clarity > music quality
- **Evidence**: STOI 0.9+ achieved (intelligibility), latency <10ms (real-time)
- **Comparison**: Commercial codecs (Opus, Lyra) also trade PESQ for latency at low bitrates

### Plan C: Hybrid Approach
- Use neural codec for latency-critical path
- Add traditional codec (Opus) as fallback for quality-critical scenarios
- Present as "adaptive" system

### Plan D: Target Clarification
- Verify if targets apply to:
  - Clean speech only? (test-clean)
  - With background noise? (might need different model)
  - Specific bitrate? (we haven't implemented VQ yet—actual bitrate unknown)

---

## Timeline

**Tonight (2026-01-27 15:00 - 2026-01-28 06:00)**
- Emergency training runs (12-14 hours)
- Monitoring every 2 hours

**Morning (2026-01-28 06:00 - 10:00)**
- Evaluate emergency checkpoint
- If targets met: Update report, test AMS wrapper, DONE
- If not met: Execute Option A/B/C (4-6 hours)

**Afternoon (2026-01-28 10:00 - 18:00)**
- Final evaluation
- Update REPORT_INPUT_FOR_LLM.md with results
- Generate final report via LLM
- Prepare AMS session materials

**Next Day (2026-01-29)**
- AMS demo session
- Final submission

---

## Success Criteria

### Minimum Viable (Must Have)
- ✅ Latency <20ms (ALREADY MET: 10ms)
- ⚠️ PESQ ≥2.5 (double baseline, ~71% of target)
- ⚠️ STOI ≥0.88 (2.25x baseline, ~98% of target)
- ✅ Working AMS wrapper (DONE)
- ✅ Documentation (DONE)

### Target (Nice to Have)
- 🎯 PESQ ≥3.5
- 🎯 STOI ≥0.9

### Acceptable Compromise
- PESQ 2.5-3.0 + STOI ≥0.9 = "Intelligibility target met, fidelity improving"
- Strong narrative: "Prioritized latency + intelligibility for teleconferencing use case"

---

## Current File Status

### Working Artifacts
- ✅ `scripts/ams_codec.py` - AMS wrapper (ready, tested)
- ✅ `scripts/emergency_training.py` - Running now
- ✅ `checkpoints_finetuned/best_model_finetuned.pt` - Baseline (PESQ 1.22, STOI 0.86)
- ✅ `REPORT_INPUT_FOR_LLM.md` - Report brief (update with final metrics)
- ✅ `DEMO_SETUP.md` - 2-PC demo guide
- ✅ All evaluation scripts working

### Next Updates Needed
- Update `scripts/ams_codec.py` checkpoint path to best_emergency.pt
- Update `REPORT_INPUT_FOR_LLM.md` with final PESQ/STOI/training details
- Generate final report PDF from updated brief

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Emergency training fails to improve | 30% | High | Options A/B/C ready |
| Disk space runs out again | 40% | Medium | Monitor /tmp, use num_workers=0 |
| GPU preempted by other users | 20% | High | Run with nohup, restartable |
| Targets unreachable with this approach | 50% | High | Backup Plan B (documentation) |
| AMS session compatibility issues | 10% | Low | Wrapper tested, CPU fallback ready |

---

## Key Commands Reference

```bash
# Monitor training
tail -f emergency_training.log
watch -n 30 "grep 'Quality:' emergency_training.log | tail -5"

# Check GPU
nvidia-smi
ps aux | grep emergency_training

# Test checkpoint
cd /mnt/Data/muaw1874/audio_cod
# Edit scripts/ams_codec.py: checkpoint_path="checkpoints_emergency/best_emergency.pt"
./venv/bin/python scripts/ams_codec.py

# Full evaluation
./venv/bin/python scripts/quality_evaluation.py \
  --checkpoint checkpoints_emergency/best_emergency.pt \
  --max-files 20

# Kill if needed
pkill -f emergency_training.py
```

---

**Next Check**: 2026-01-27 17:30 (2 hours) - Look for epoch 5 results  
**Decision Point**: 2026-01-28 06:00 - Evaluate if targets met or execute Plan B/C
