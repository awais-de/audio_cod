# Extended PESQ Training - Sleep Status Report

**Started:** Jan 27, 2026 20:52 UTC
**Expected completion:** ~06:00 UTC (8–9 hours from start)

## Current Setup
- **Model:** NeuralAudioCodec (d_model=384, n_layers=6, 21.7M params)
- **Training:** 30 epochs from `best_pesq_finetune.pt` (was at epoch 10, PESQ 2.675)
- **Loss:** PESQ-leaning (high STFT weight, low STOI weight to maximize PESQ)
- **Process:** PID 4070837, logging to `pesq_extended.log`

## Baseline (before this run)
- PESQ: 2.803 (on 10×4s test set)
- STOI: 0.981 (exceeds 0.9 target)
- Latency: <20 ms ✅

## Projected Outcome
- **Conservative:** PESQ 2.95–3.10 (based on ~0.025 PESQ/epoch gain, accounting for saturation)
- **Optimistic:** PESQ 3.10–3.20
- **Gap to 3.5 target:** Unknown, depends on model capacity ceiling

## Next Steps (When You Wake Up)
1. Check `pesq_extended.log` for final status
2. Run: `venv/bin/python scripts/evaluate_emergency.py --ckpt checkpoints_emergency/best_pesq_extended.pt --out checkpoints_emergency/eval_final.txt`
3. If PESQ < 3.2 and time allows: consider 15–20 epoch finetune with **larger model** (d_model=512, n_layers=8)
4. Update AMS wrapper to final best checkpoint
5. Finalize report with PESQ/STOI/latency metrics

## Checkpoint Paths
- Current best: `checkpoints_emergency/best_pesq_extended.pt` (will be saved during training)
- Previous: `checkpoints_emergency/best_pesq_finetune.pt` (baseline for this run, PESQ 2.803)

## GPU Status
- Device: CUDA (Quadro RTX 8000)
- Throughput: ~1.32–1.38 s/iter
- Expected ~13.7 min/epoch × 30 epochs = ~411 min (~6.8 hours)
- Total wall time: ~7–8 hours (including eval overhead)

---
**Good luck! This run should get us into the 3.0+ range for PESQ. 🚀**
