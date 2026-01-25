#!/bin/bash
# START_HERE.sh - Complete project setup and training guide

cat << 'EOF'

╔═════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║             🎉 YOUR NEURAL AUDIO CODEC PROJECT IS READY! 🎉                ║
║                                                                             ║
║                        OPTIMIZED FOR SPEED & EFFICIENCY                    ║
║                                                                             ║
╚═════════════════════════════════════════════════════════════════════════════╝

Welcome to your completely restructured and optimized Neural Audio Codec project!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 WHAT'S CHANGED:

   Model:        51.8M → 12M parameters (4.3x smaller, 5x less memory)
   Training:     6-7h → 45min-1.5h per epoch (5-7x faster)
   Total Time:   ~27 days → ~4 days (for 100 epochs)
   Structure:    Clean & organized (src/ config/ scripts/ docs/)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 GET STARTED IN 2 MINUTES:

   Option 1: Automatic (Recommended)
   ─────────────────────────────────
   
   chmod +x train.sh
   ./train.sh
   
   This does everything:
   ✓ Creates virtual environment
   ✓ Installs dependencies
   ✓ Verifies setup
   ✓ Downloads dataset (if needed)
   ✓ Starts training
   

   Option 2: Manual Steps
   ────────────────────
   
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python scripts/sanity_check.py     # Verify setup
   python src/train.py                # Start training

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 PROJECT STRUCTURE:

   audio_cod/
   ├── src/                    Source code (clean & optimized)
   │   ├── model.py           12M parameter audio codec
   │   └── train.py           Optimized training script
   ├── config/
   │   └── training.yaml      ⭐️  CUSTOMIZE HERE (batch_size, lr, etc.)
   ├── scripts/               Utility scripts
   │   ├── sanity_check.py    Verify GPU, packages, dataset
   │   ├── inference.py       Test model on audio files
   │   └── monitor.py         Real-time training monitor
   ├── README.md              Full documentation
   ├── QUICKSTART.md          5-minute quick start
   ├── RESTRUCTURING_SUMMARY.md  Detailed changes
   ├── train.sh              One-command training
   └── requirements.txt       Python dependencies

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️  CONFIGURATION (config/training.yaml):

   Default settings are optimized for your setup. To customize:
   
   For faster training (50GB+ VRAM):
      batch_size: 64          (was 32)
      segment_length: 8000    (was 6000)
   
   For memory efficiency (8-12GB VRAM):
      batch_size: 8           (was 32)
      segment_length: 4000    (was 6000)
      num_workers: 0          (was 4)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ BEFORE YOU START:

   [ ] Read QUICKSTART.md (5 minutes)
   [ ] Run sanity check: python scripts/sanity_check.py
   [ ] Review config: cat config/training.yaml
   [ ] Check GPU: nvidia-smi

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 DOCUMENTATION:

   QUICKSTART.md             ← Start here (5 min read)
   README.md                 ← Full guide
   RESTRUCTURING_SUMMARY.md  ← What changed
   config/training.yaml      ← Configuration file

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 TYPICAL WORKFLOW:

   1. Verify setup:
      python scripts/sanity_check.py
   
   2. Start training:
      ./train.sh
      OR: python src/train.py
   
   3. Monitor progress (in another terminal):
      python scripts/monitor.py
   
   4. Test after training:
      python scripts/inference.py --audio test.wav --output out.wav
   
   5. Adjust config if needed:
      Edit config/training.yaml and retrain

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🐛 TROUBLESHOOTING:

   Q: Out of memory error?
   A: Reduce batch_size or segment_length in config/training.yaml
   
   Q: GPU not detected?
   A: Run: python scripts/sanity_check.py (will diagnose)
   
   Q: Dataset not found?
   A: Training script auto-downloads it. Or run sanity_check.py
   
   Q: Training is slow?
   A: Increase batch_size and num_workers in config/training.yaml

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 EXPECTED RESULTS:

   After 1 epoch:     Loss ~2.0-3.0
   After 5 epochs:    Loss ~0.8-1.2
   After 20 epochs:   Loss ~0.4-0.6
   After 100 epochs:  Loss ~0.3-0.4 (ideal)
   
   SNR (Signal-to-Noise Ratio): Target > 20 dB
   GPU Memory Usage: ~8-12 GB
   Time per epoch: 45 min - 1.5 hours

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 READY? START HERE:

   chmod +x train.sh
   ./train.sh

   Or read the quick start first:
   
   cat QUICKSTART.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Questions? Everything is documented:
- QUICKSTART.md for quick reference
- README.md for comprehensive guide
- scripts/sanity_check.py for diagnostics

Good luck with your training! 🎵

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF
