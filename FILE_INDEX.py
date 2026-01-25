#!/usr/bin/env python3
"""
Project File Index & Guide
Lists all important files and their purposes
"""

PROJECT_FILES = {
    "Entry Points": {
        "train.sh": "🚀 One-command training startup (RECOMMENDED)",
        "src/train.py": "📚 Main training script",
        "scripts/sanity_check.py": "🔍 Setup verification",
        "scripts/inference.py": "🎵 Audio compression/reconstruction",
    },
    
    "Model & Architecture": {
        "src/model.py": "🧠 Neural Audio Codec (12M parameters)",
        "src/__init__.py": "📦 Package initialization",
    },
    
    "Configuration": {
        "config/training.yaml": "⚙️  Hyperparameters (EDIT THIS TO TUNE TRAINING)",
    },
    
    "Utility Scripts": {
        "scripts/monitor.py": "📊 Real-time training monitor",
        "scripts/sanity_check.py": "✅ Pre-training verification",
        "scripts/inference.py": "🎙️  Test trained model on audio",
    },
    
    "Documentation": {
        "README.md": "📖 Full project documentation",
        "QUICKSTART.md": "⚡ 5-minute quick start guide",
        "RESTRUCTURING_SUMMARY.md": "📋 What changed & why",
    },
    
    "Dependencies": {
        "requirements.txt": "📦 Python package requirements",
    },
    
    "Directories": {
        "src/": "Source code",
        "config/": "Configuration files",
        "scripts/": "Utility scripts",
        "checkpoints/": "Saved model checkpoints",
        "data/": "Dataset location (auto-downloaded)",
        "docs/": "Additional documentation",
    },
}

FILE_DESCRIPTIONS = {
    "train.sh": """
    ONE-COMMAND STARTUP SCRIPT
    
    Does everything in one command:
    1. Creates virtual environment (if needed)
    2. Installs dependencies
    3. Runs sanity checks
    4. Downloads dataset (if needed)
    5. Starts training
    
    USAGE:
        chmod +x train.sh
        ./train.sh
    """,
    
    "src/train.py": """
    MAIN TRAINING SCRIPT
    
    Features:
    - Automatic GPU detection & optimization
    - Multi-scale spectral loss
    - Automatic checkpointing
    - Comprehensive logging
    - NaN/Inf protection
    - Gradient clipping
    
    USAGE:
        python src/train.py
    
    OUTPUTS:
    - checkpoints/best_model.pt (best validation loss)
    - checkpoints/checkpoint_epoch_N.pt (periodic saves)
    - Console logs with training metrics
    """,
    
    "src/model.py": """
    NEURAL AUDIO CODEC MODEL
    
    Architecture:
    - Total: 12M parameters (reduced from 51.8M)
    - Encoder: 4 causal conv + 4 transformer blocks (6M params)
    - Decoder: 4 transformer blocks + 4 deconv (6M params)
    - Optimized for streaming with causal attention
    
    Key Features:
    - Causal design (no future context)
    - Sliding-window attention (efficient)
    - GroupNorm (faster than LayerNorm)
    - Tanh output activation
    
    USAGE (Python):
        from src.model import NeuralAudioCodec
        model = NeuralAudioCodec()
        output = model(input_audio)  # input: (batch, 1, samples)
    """,
    
    "config/training.yaml": """
    TRAINING CONFIGURATION
    
    IMPORTANT: Edit this file to customize training!
    
    Key Settings:
    - batch_size: 32 (increase to 64 for speed, decrease for memory)
    - segment_length: 6000 (increase to 8000 for quality)
    - num_workers: 4 (data loading parallelism)
    - learning_rate: 0.0001
    - epochs: 100
    
    Model Settings:
    - d_model: 256 (embedding dimension)
    - n_layers: 4 (transformer layers)
    - n_heads: 8 (attention heads)
    
    Data:
    - train_dir: LibriSpeech train-clean-100 path
    - val_dir: Validation data path
    """,
    
    "scripts/sanity_check.py": """
    SETUP VERIFICATION SCRIPT
    
    Checks:
    ✓ Python version (>= 3.8)
    ✓ Required packages installed
    ✓ GPU/CUDA available
    ✓ Configuration files valid
    ✓ Model can be loaded
    ✓ Dataset available (or provides download info)
    
    RUN BEFORE TRAINING:
        python scripts/sanity_check.py
    
    If any checks fail, script shows how to fix them.
    """,
    
    "scripts/inference.py": """
    INFERENCE SCRIPT
    
    Test trained model on audio files.
    
    USAGE:
        python scripts/inference.py \\
          --audio input.wav \\
          --output output.wav \\
          --checkpoint checkpoints/best_model.pt
    
    OUTPUTS:
    - Reconstructed audio file
    - SNR (Signal-to-Noise Ratio)
    - Compression ratio statistics
    """,
    
    "scripts/monitor.py": """
    TRAINING MONITOR
    
    Real-time monitoring while training.
    
    Shows:
    - GPU memory usage & utilization
    - Current epoch & batch
    - Training loss
    - Time per batch
    - Estimated time to completion
    
    USAGE (in another terminal):
        python scripts/monitor.py
    """,
    
    "README.md": """
    COMPREHENSIVE PROJECT DOCUMENTATION
    
    Sections:
    1. What's new (optimizations)
    2. Quick start (5 minutes)
    3. Model architecture
    4. Configuration guide
    5. Dataset information
    6. Training progress
    7. Inference
    8. Troubleshooting
    9. Performance summary
    
    START HERE for full details.
    """,
    
    "QUICKSTART.md": """
    5-MINUTE QUICK START GUIDE
    
    Steps:
    1. Create environment: python3 -m venv venv
    2. Activate: source venv/bin/activate
    3. Install: pip install -r requirements.txt
    4. Verify: python scripts/sanity_check.py
    5. Train: python src/train.py
    
    Common commands and troubleshooting.
    """,
    
    "RESTRUCTURING_SUMMARY.md": """
    PROJECT RESTRUCTURING & OPTIMIZATION DETAILS
    
    Explains:
    - What changed and why
    - Performance improvements (5-7x faster)
    - Model optimization (51.8M -> 12M params)
    - Data loading optimization
    - Files added/removed/modified
    - Expected results
    """,
}

def print_section(title, items):
    """Print a formatted section"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")
    
    for file_or_dir, desc in items.items():
        print(f"📄 {file_or_dir:30} → {desc}")

def main():
    """Print file index"""
    print(f"\n{chr(27)}[1;94m")  # Bold blue
    print("""
╔═════════════════════════════════════════════════════════════════════════════╗
║         Neural Audio Codec - Project File Index                            ║
╚═════════════════════════════════════════════════════════════════════════════╝
    """)
    print(f"{chr(27)}[0m")  # Reset
    
    for category, files in PROJECT_FILES.items():
        print_section(category, files)
    
    print(f"\n{'='*80}")
    print("QUICKSTART")
    print(f"{'='*80}\n")
    
    print("1️⃣  First Time Setup:")
    print("   python scripts/sanity_check.py\n")
    
    print("2️⃣  Start Training (Recommended):")
    print("   ./train.sh\n")
    
    print("   OR manually:")
    print("   python src/train.py\n")
    
    print("3️⃣  Test Inference:")
    print("   python scripts/inference.py --audio input.wav --output output.wav\n")
    
    print("4️⃣  Monitor Training (in separate terminal):")
    print("   python scripts/monitor.py\n")
    
    print(f"{'='*80}\n")
    
    # Get file details if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in FILE_DESCRIPTIONS:
        print(FILE_DESCRIPTIONS[sys.argv[1]])

if __name__ == "__main__":
    main()
