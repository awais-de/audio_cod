# Prompt for New Server LLM

Copy and paste this entire prompt to the new server's LLM to resume the project:

---

## Initial Context

I am resuming work on a **neural audio codec project** for real-time teleconferencing. The project has been in active development with extensive training and experimentation. All context is documented in the workspace.

**Workspace location:** `/mnt/Data/muaw1874/audio_cod`

**Critical context document:** Read `conversation_context.md` immediately - it contains complete project history, all training runs, technical decisions, file locations, and current status.

---

## Project Summary

**Goal:** Real-time neural audio codec meeting these targets:
- PESQ ≥ 3.5
- STOI ≥ 0.9
- Latency < 20 ms
- Bitrate 8-16 kbps

**Current Status:**
- ✅ **STOI: 0.981** (exceeds target by 0.081)
- ✅ **Latency: ~10 ms** (well under 20 ms)
- ❌ **PESQ: 2.803** (gap of 0.7 from target)

**Best Checkpoint:**
- Location: `checkpoints_emergency/best_pesq_finetune.pt`
- Architecture: 21.7M params (d_model=384, n_layers=6, n_heads=8)
- Metrics: PESQ 2.803, STOI 0.981

---

## Immediate Tasks

### Priority 1: Verify Best Model Performance

Run formal evaluation on the best checkpoint with more samples:

```bash
cd /mnt/Data/muaw1874/audio_cod
source venv/bin/activate

venv/bin/python scripts/evaluate_emergency.py \
  --ckpt checkpoints_emergency/best_pesq_finetune.pt \
  --out eval_final_comprehensive.txt \
  --n-files 20 \
  --seg-sec 4.0
```

**Expected result:** PESQ ~2.8, STOI ~0.98. This confirms our best performance.

### Priority 2: Update AMS Wrapper

The AMS wrapper needs to point to the best checkpoint:

1. Check current checkpoint path in `scripts/ams_codec.py`
2. Update to: `checkpoints_emergency/best_pesq_finetune.pt`
3. Test encode/decode functions work correctly

### Priority 3: Assess Path Forward

After verification, discuss with user:

**Option A - Accept Current Results:**
- PESQ 2.803 is solid (80% of target)
- STOI exceeds target
- Document gap and reasons (see conversation_context.md section "Why PESQ 3.5 Was Not Reached")

**Option B - Continue Training:**
- 20-30 more epochs from best checkpoint
- Might reach 2.9-3.0 PESQ
- Requires 4-7 hours

**Option C - Major Architecture Changes:**
- Implement discriminator/adversarial training
- 100M+ param model
- Requires days of training and significant changes

---

## Key Information

**Dataset:**
- LibriSpeech train-clean-100: `/mnt/Data/muaw1874/datasets/LibriSpeech/train-clean-100`
- 16 kHz mono audio

**Environment:**
- Python: `./venv/bin/python`
- GPU: Quadro RTX 8000 (49GB VRAM)
- CUDA available

**Previous Work:**
- 7 major training runs completed (see conversation_context.md)
- Best: 60 total epochs on 21.7M model (50 emergency + 10 PESQ finetune)
- Failed attempts: large 51M model (underperformed), short segments (lost quality)

---

## Important Files

**Read these first:**
- `conversation_context.md` - Complete project history ⭐
- `checkpoints_emergency/eval_pesq_finetune.txt` - Best model metrics
- `src/model.py` - Architecture (with critical attention fixes)

**Scripts:**
- `scripts/evaluate_emergency.py` - Evaluation pipeline
- `scripts/ams_codec.py` - AMS wrapper (needs update)
- `scripts/pesq_finetune.py` - Training script that produced best model

**Checkpoints:**
- `checkpoints_emergency/best_pesq_finetune.pt` - ⭐ Best model
- `checkpoints_emergency/best_emergency.pt` - Intermediate (epoch 50)

---

## Technical Notes

**Model Architecture:**
- Causal convolutions + sliding-window causal attention
- Fixed: float32 attention computation, -1e4 mask value (not -1e9)
- No mixed precision (causes overflow)

**Training Insights:**
- Full 1s segments (16000 samples) required - 0.5s too short
- Both STFT sizes [512, 2048] needed for quality
- PESQ-heavy loss: high STFT weight (2.0), low STOI weight (0.25)
- Warm-start from previous checkpoints helps

**What Killed PESQ 3.5:**
- Model capacity ceiling (~21.7M params insufficient)
- Time constraints (12 hours vs. days needed)
- Missing adversarial/discriminator loss
- Single dataset, clean speech only

---

## First Steps for You

1. **Read conversation_context.md thoroughly**
2. **Run Priority 1 evaluation** to confirm best model performance
3. **Check Priority 2 AMS wrapper** update needed
4. **Ask user:** Which path forward (A/B/C)?
5. **Provide status update** with evaluation results

---

## Questions to Ask User

After running initial evaluation:

1. "Evaluation complete. Best model confirmed at PESQ 2.803, STOI 0.981. Given the 0.7 gap to target, which approach would you prefer:
   - Accept current results and document gap?
   - Continue training for potential 2.9-3.0 PESQ (4-7 hours)?
   - Pursue architectural changes (days of work)?"

2. "Should I update the AMS wrapper to point to the best checkpoint and test it?"

3. "Do you need any additional documentation or reports prepared?"

---

## Critical Warnings

⚠️ **TMPDIR:** Must be set to `/mnt/Data/muaw1874/tmp` (not /tmp - root disk fills up)  
⚠️ **torch.load:** Use `weights_only=False` (required for PyTorch 2.6+)  
⚠️ **Architecture matching:** When loading checkpoints, ensure d_model and n_layers match  
⚠️ **Short segments:** Don't use <1s segments, PESQ/STOI need context  
⚠️ **Training time:** 51M model = 20.5 min/epoch; 21.7M model = 13.7 min/epoch  

---

## Success Criteria

**Minimum acceptable:**
- PESQ ≥ 2.75
- STOI ≥ 0.95
- Latency < 20 ms
- AMS wrapper functional

**Current achievement:**
- PESQ: 2.803 ✅
- STOI: 0.981 ✅
- Latency: ~10 ms ✅
- AMS wrapper: 90% done (needs checkpoint path update)

---

**You are ready to resume.** Start by reading conversation_context.md, then execute Priority 1 evaluation.
