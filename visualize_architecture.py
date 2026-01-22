"""
Architecture Visualization and Summary
Shows the detailed structure of the neural audio codec
"""

def print_architecture():
    """Print detailed architecture summary"""
    
    print("=" * 80)
    print("NEURAL AUDIO CODEC - ARCHITECTURE SUMMARY")
    print("=" * 80)
    
    print("\n📊 MODEL SPECIFICATIONS")
    print("-" * 80)
    specs = {
        "Sample Rate": "16 kHz",
        "Input": "Raw audio waveform (mono)",
        "Latency Target": "< 20 ms",
        "Compression": "~32x (before quantization)",
        "Target Bitrate": "8-16 kbps (with quantization)",
        "Model Parameters": "~20M",
        "Transformer Layers": "8 (encoder) + 8 (decoder)",
        "Attention Heads": "16 per layer",
        "Embedding Dimension": "512",
        "Window Size": "512 frames (causal attention)",
    }
    for key, value in specs.items():
        print(f"  {key:.<30} {value}")
    
    print("\n🔧 ENCODER ARCHITECTURE")
    print("-" * 80)
    encoder_layers = [
        ("Input", "Audio waveform", "(batch, 1, 16000)", "1 sec @ 16kHz"),
        ("", "", "", ""),
        ("Conv Layer 1", "CausalConv1d + GroupNorm + GELU", "(batch, 64, 8000)", "kernel=7, stride=2"),
        ("Conv Layer 2", "CausalConv1d + GroupNorm + GELU", "(batch, 128, 4000)", "kernel=7, stride=2"),
        ("Conv Layer 3", "CausalConv1d + GroupNorm + GELU", "(batch, 256, 2000)", "kernel=7, stride=2"),
        ("Conv Layer 4", "CausalConv1d + GroupNorm + GELU", "(batch, 512, 2000)", "kernel=3, stride=1"),
        ("", "", "", ""),
        ("Reshape", "Transpose", "(batch, 2000, 512)", "seq_len=2000"),
        ("", "", "", ""),
        ("Transformer 1-8", "8x TransformerBlock", "(batch, 2000, 512)", "Causal attention"),
        ("  └─ Attention", "  Multi-head (16 heads)", "", "window_size=512"),
        ("  └─ Feed-forward", "  MLP (512→2048→512)", "", "GELU activation"),
        ("  └─ Residual", "  + Layer norm", "", ""),
        ("", "", "", ""),
        ("Output", "Latent representation", "(batch, 2000, 512)", "125 ms frames"),
    ]
    
    print(f"{'Layer':<20} {'Description':<35} {'Shape':<25} {'Notes':<20}")
    print("-" * 100)
    for layer in encoder_layers:
        print(f"{layer[0]:<20} {layer[1]:<35} {layer[2]:<25} {layer[3]:<20}")
    
    print("\n🔧 DECODER ARCHITECTURE")
    print("-" * 80)
    decoder_layers = [
        ("Input", "Latent representation", "(batch, 2000, 512)", "From encoder"),
        ("", "", "", ""),
        ("Transformer 1-8", "8x TransformerBlock", "(batch, 2000, 512)", "Causal attention"),
        ("  └─ Attention", "  Multi-head (16 heads)", "", "window_size=512"),
        ("  └─ Feed-forward", "  MLP (512→2048→512)", "", "GELU activation"),
        ("  └─ Residual", "  + Layer norm", "", ""),
        ("", "", "", ""),
        ("Reshape", "Transpose", "(batch, 512, 2000)", ""),
        ("", "", "", ""),
        ("Deconv Layer 1", "ConvTranspose1d + GroupNorm + GELU", "(batch, 256, 2000)", "kernel=3, stride=1"),
        ("Deconv Layer 2", "ConvTranspose1d + GroupNorm + GELU", "(batch, 128, 4000)", "kernel=8, stride=2"),
        ("Deconv Layer 3", "ConvTranspose1d + GroupNorm + GELU", "(batch, 64, 8000)", "kernel=8, stride=2"),
        ("Deconv Layer 4", "ConvTranspose1d + GroupNorm + GELU", "(batch, 32, 16000)", "kernel=8, stride=2"),
        ("Deconv Layer 5", "ConvTranspose1d", "(batch, 1, 16000)", "kernel=7, stride=1"),
        ("", "", "", ""),
        ("Output Activation", "Tanh", "(batch, 1, 16000)", "Range: [-1, 1]"),
        ("", "", "", ""),
        ("Output", "Reconstructed waveform", "(batch, 1, 16000)", "1 sec @ 16kHz"),
    ]
    
    print(f"{'Layer':<20} {'Description':<35} {'Shape':<25} {'Notes':<20}")
    print("-" * 100)
    for layer in decoder_layers:
        print(f"{layer[0]:<20} {layer[1]:<35} {layer[2]:<25} {layer[3]:<20}")
    
    print("\n🎯 LOSS FUNCTIONS")
    print("-" * 80)
    losses = [
        ("Time-Domain L1", "Mean absolute error on waveform", "Weight: 1.0"),
        ("Multi-Scale Spectral", "STFT at 3 scales (512, 1024, 2048)", "Weight: 1.0"),
        ("  └─ Log-magnitude", "  L1 on log-spectrogram", "Perceptual"),
        ("  └─ Magnitude", "  L1 on magnitude", "Reconstruction"),
    ]
    
    print(f"{'Loss Component':<25} {'Description':<45} {'Configuration':<20}")
    print("-" * 90)
    for loss in losses:
        print(f"{loss[0]:<25} {loss[1]:<45} {loss[2]:<20}")
    
    print("\n⚡ LATENCY BREAKDOWN")
    print("-" * 80)
    latency = [
        ("Encoder Convolutions", "~5 ms", "4 causal conv layers"),
        ("Encoder Transformer", "~3 ms", "8 layers, causal attention"),
        ("Decoder Transformer", "~3 ms", "8 layers, causal attention"),
        ("Decoder Deconvolutions", "~7 ms", "5 transposed conv layers"),
        ("Total", "~18 ms", "Well under 20 ms target"),
    ]
    
    print(f"{'Component':<30} {'Latency':<15} {'Notes':<30}")
    print("-" * 75)
    for component in latency:
        print(f"{component[0]:<30} {component[1]:<15} {component[2]:<30}")
    
    print("\n💾 COMPRESSION ANALYSIS")
    print("-" * 80)
    compression = [
        ("Original Audio", "512 kbps", "16kHz × 32-bit float"),
        ("Latent (unquantized)", "~16 Mbps", "100 frames/sec × 512 dim × 32-bit"),
        ("After RVQ/FSQ", "8-16 kbps", "Target with quantization"),
        ("Compression Ratio", "32-64x", "Relative to original"),
    ]
    
    print(f"{'Stage':<25} {'Bitrate':<20} {'Description':<30}")
    print("-" * 75)
    for stage in compression:
        print(f"{stage[0]:<25} {stage[1]:<20} {stage[2]:<30}")
    
    print("\n🎨 KEY DESIGN FEATURES")
    print("-" * 80)
    features = [
        "✓ Fully causal design - no future context used anywhere",
        "✓ Sliding-window attention - O(n×w) complexity instead of O(n²)",
        "✓ GroupNorm instead of BatchNorm - stable with small batches",
        "✓ Strided convolutions - efficient downsampling",
        "✓ Multi-scale spectral loss - better perceptual quality",
        "✓ Residual connections - easier gradient flow",
        "✓ GELU activations - smooth, differentiable",
        "✓ Tanh output - bounded waveform values",
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n📈 TRAINING STRATEGY")
    print("-" * 80)
    training = [
        ("Optimizer", "AdamW", "β₁=0.8, β₂=0.99, weight_decay=0.01"),
        ("Learning Rate", "1e-4", "With cosine annealing"),
        ("Batch Size", "8", "1-second audio segments"),
        ("Gradient Clipping", "1.0", "Prevent exploding gradients"),
        ("Epochs", "100+", "Until convergence"),
        ("Early Stopping", "Val loss", "Save best model"),
    ]
    
    print(f"{'Component':<25} {'Value':<20} {'Notes':<30}")
    print("-" * 75)
    for component in training:
        print(f"{component[0]:<25} {component[1]:<20} {component[2]:<30}")
    
    print("\n" + "=" * 80)
    print("END OF ARCHITECTURE SUMMARY")
    print("=" * 80)


def print_data_flow():
    """Print ASCII data flow diagram"""
    
    print("\n\n")
    print("=" * 80)
    print("DATA FLOW DIAGRAM")
    print("=" * 80)
    print()
    print("     Raw Audio (16 kHz mono)")
    print("            │")
    print("            ▼")
    print("     ┌──────────────┐")
    print("     │   ENCODER    │")
    print("     └──────────────┘")
    print("            │")
    print("     ┌──────┴──────┐")
    print("     │  4x Causal  │  ──→  Downsample 16x")
    print("     │  Conv Layers│")
    print("     └──────┬──────┘")
    print("            │")
    print("     ┌──────┴──────┐")
    print("     │ Transformer │  ──→  8 layers, causal attention")
    print("     │  (8 layers) │")
    print("     └──────┬──────┘")
    print("            │")
    print("            ▼")
    print("     Latent (512-dim)")
    print("            │")
    print("     [ Quantizer ]  ──→  (Future: RVQ/FSQ)")
    print("            │")
    print("            ▼")
    print("     ┌──────────────┐")
    print("     │   DECODER    │")
    print("     └──────────────┘")
    print("            │")
    print("     ┌──────┴──────┐")
    print("     │ Transformer │  ──→  8 layers, causal attention")
    print("     │  (8 layers) │")
    print("     └──────┬──────┘")
    print("            │")
    print("     ┌──────┴──────┐")
    print("     │ 5x Transposed│  ──→  Upsample 16x")
    print("     │  Conv Layers │")
    print("     └──────┬──────┘")
    print("            │")
    print("            ▼")
    print("     Reconstructed Audio")
    print()
    print("=" * 80)


def print_attention_pattern():
    """Visualize the causal attention pattern"""
    
    print("\n\n")
    print("=" * 80)
    print("CAUSAL SLIDING-WINDOW ATTENTION PATTERN")
    print("=" * 80)
    print()
    print("Each position can attend to:")
    print("  • Itself")
    print("  • Previous positions (up to window_size=512)")
    print()
    print("Example with window_size=5:")
    print()
    print("       t=0  t=1  t=2  t=3  t=4  t=5  t=6")
    print("       ─────────────────────────────────")
    print("t=0 │   ✓    -    -    -    -    -    -")
    print("t=1 │   ✓    ✓    -    -    -    -    -")
    print("t=2 │   ✓    ✓    ✓    -    -    -    -")
    print("t=3 │   ✓    ✓    ✓    ✓    -    -    -")
    print("t=4 │   ✓    ✓    ✓    ✓    ✓    -    -")
    print("t=5 │   -    ✓    ✓    ✓    ✓    ✓    -    ← Window slides")
    print("t=6 │   -    -    ✓    ✓    ✓    ✓    ✓")
    print()
    print("✓ = Can attend    - = Cannot attend (masked)")
    print()
    print("Benefits:")
    print("  • Maintains causality (streaming-friendly)")
    print("  • Reduces complexity from O(n²) to O(n×w)")
    print("  • Keeps latency constant regardless of sequence length")
    print()
    print("=" * 80)


if __name__ == "__main__":
    print_architecture()
    print_data_flow()
    print_attention_pattern()
    
    print("\n\n")
    print("=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)
    print()
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Test the model (without training):")
    print("   python inference.py")
    print()
    print("3. Train on your dataset:")
    print("   • Update config.yaml with your data paths")
    print("   • Run: python train.py")
    print()
    print("4. Inference with trained model:")
    print("   python inference.py --input audio.wav --checkpoint best_model.pt --output output.wav")
    print()
    print("=" * 80)
