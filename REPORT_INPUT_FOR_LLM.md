# Neural Audio Codec Project — Report Input Brief (for LLM drafting)

Use this brief as the single source to draft a full project report. Fill in the TBDs once fine-tuning results are available.

## Project Context
- Goal: Real-time neural audio codec for teleconferencing.
- Targets: latency <20 ms; quality PESQ ≥3.5, STOI ≥0.9; bitrate 8–16 kbps; real-time demo over network.
- Current date: 2026-01-27.
- Deadline: 2-day submission (reports + demo readiness).

## System Overview
- Model: Transformer-based neural audio codec (causal, streaming-capable).
- Params: ~6.7M (d_model=256, n_layers=4, n_heads=8, window_size=256).
- Sample rate: 16 kHz mono.
- Compression: 16x temporal compression via encoder.
- Codebase: PyTorch.
- Dataset: LibriSpeech train-clean-100 (100 hours).

## Implemented Artifacts
- Core model: src/model.py (NeuralAudioCodec: encoder/decoder with causal conv + causal attention).
- Training (original): src/train.py (100 epochs, multi-scale spectral + L1 loss).
- Fine-tuning scripts: scripts/quick_finetune.py (50-epoch, perceptual/STFT loss; running now), scripts/train_improved.py (larger model, 200 epochs, not started).
- Benchmarking: scripts/latency_benchmark.py, scripts/quality_evaluation.py, scripts/comprehensive_benchmark.py.
- Demo tools: scripts/demo_server.py (mic to network), scripts/demo_client.py (network to speaker), scripts/demo_file_based.py (file streaming), scripts/ams_codec.py (AMS wrapper: my_encoder_logic/my_decoder_logic), scripts/setup_demo.sh (PortAudio + sounddevice setup).
- Documentation: PROJECT_REPORT.md (full), EXECUTIVE_SUMMARY.md, LATENCY_VERIFICATION.md, QUALITY_VERIFICATION.md, DEMO_SETUP.md, 2DAY_COMPLETION_PLAN.md.

## Known Results (pre-finetune)
- Latency: Pass. ~10 ms P99 end-to-end, RTF 0.07–0.70x (up to 14x faster than real-time) per LATENCY_VERIFICATION.md.
- Quality: Fail. PESQ ~1.07 (target 3.5), STOI ~0.39 (target 0.9), SNR ~-27 dB per QUALITY_VERIFICATION.md.
- Checkpoints: best_model.pt (epoch 100) + checkpoints every 10 epochs (10–90). None meet quality targets.

## Ongoing Work
- Fine-tuning (quality-focused) in progress: scripts/quick_finetune.py, expected 6–8 hours; saves to checkpoints_finetuned/ (best_finetuned_model.pt anticipated).
- Goal: improve PESQ/STOI; may still be below targets but should be better than baseline.

## AMS Wrapper Requirements (already implemented)
- Functions: my_encoder_logic(audio_frame: np.ndarray) -> bytes; my_decoder_logic(compressed_bytes: bytes) -> np.ndarray.
- Location: scripts/ams_codec.py. Uses checkpoint at checkpoints/best_model.pt (change path to fine-tuned checkpoint when ready).
- Dependencies: torch, numpy (torchaudio optional for resampling; fallback implemented). No audio hardware needed.

## Pending Insertions (TBD after fine-tune)
- Final checkpoint path: e.g., checkpoints_finetuned/best_finetuned_model.pt.
- Updated metrics (PESQ, STOI, SNR) on standard evaluation set.
- Subjective notes on audio quality (artifacts, intelligibility).
- GPU/CPU timing with fine-tuned model (encode/decode per chunk/frame).
- Any bitrate/latency changes if chunk size or processing changed.

## Suggested Report Structure (for LLM)
1) Abstract — mention latency success, quality status, and fine-tune attempt.
2) Introduction — teleconference requirements, targets, constraints (2-day deadline).
3) Method — architecture (causal conv + Transformer), training data, loss functions (original vs fine-tune perceptual/STFT), configs.
4) Implementation — streaming design, chunking, AMS wrapper interface, demo scripts.
5) Experiments — dataset splits, baselines (Opus placeholder), checkpoints tested, latency bench setup.
6) Results — latency (pass), quality (pre-finetune fail), updated fine-tune metrics (insert TBD), subjective observations.
7) Discussion — reasons for low quality (insufficient training epochs, reduced model capacity, loss mismatch), expected improvements.
8) Limitations & Future Work — longer training, larger model (config/training_improved.yaml), vector quantization for bitrate control, better perceptual losses, baseline comparisons when system allows.
9) Conclusion — latency meets target, quality below target but improving; demo readiness for AMS session; honest statement of remaining gaps.

## Key Talking Points & Risks
- Latency already excellent; quality is the main gap.
- Fine-tune may improve but may not reach targets in time; report must be honest.
- Demo readiness: AMS wrapper ready; real-time scripts ready; need final checkpoint swap.
- Risks: GPU availability, sounddevice/PortAudio on target machines (use AMS wrapper path which needs no audio I/O), quality may remain subpar.

## Quick Commands (reference for LLM to include)
- CPU self-test (no GPU, no audio hw): `CUDA_VISIBLE_DEVICES="" ./venv/bin/python scripts/ams_codec.py`
- GPU self-test (after fine-tune): `./venv/bin/python scripts/ams_codec.py`
- Swap checkpoint path in scripts/ams_codec.py once fine-tuned model is ready.

## Tone & Honesty Guidance
- Be explicit that latency target is met; quality target is not yet met (unless new metrics show otherwise).
- Emphasize attempted remediation (fine-tuning) and expected direction of improvement.
- Include actionable next steps and risks if quality remains low by deadline.
