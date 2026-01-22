"""
GPU Diagnostic Script
Check if your GPU is properly detected and working with PyTorch
"""

import torch
import sys

def check_gpu():
    """Comprehensive GPU check"""
    
    print("=" * 80)
    print("GPU DIAGNOSTIC REPORT")
    print("=" * 80)
    print()
    
    # 1. CUDA Availability
    print("1️⃣  CUDA AVAILABILITY")
    print("-" * 80)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("✅ CUDA is AVAILABLE")
        print(f"   PyTorch built with CUDA: {torch.version.cuda}")
    else:
        print("❌ CUDA is NOT AVAILABLE")
        print("   PyTorch is using CPU-only version")
        print()
        print("To fix this, install CUDA-enabled PyTorch:")
        print("   pip uninstall torch torchaudio")
        print("   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print()
        return False
    print()
    
    # 2. GPU Device Info
    print("2️⃣  GPU DEVICE INFORMATION")
    print("-" * 80)
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        print()
        
        for i in range(num_gpus):
            print(f"GPU {i}:")
            print(f"   Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"   Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Multi-Processors: {props.multi_processor_count}")
            
            # Memory usage
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"   Memory Allocated: {allocated:.2f} GB")
            print(f"   Memory Reserved: {reserved:.2f} GB")
            print(f"   Memory Free: {(props.total_memory / 1e9) - reserved:.2f} GB")
            print()
    print()
    
    # 3. cuDNN Status
    print("3️⃣  cuDNN STATUS")
    print("-" * 80)
    cudnn_available = torch.backends.cudnn.is_available()
    if cudnn_available:
        print(f"✅ cuDNN is available")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"   cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    else:
        print("❌ cuDNN is NOT available")
    print()
    
    # 4. Test GPU Performance
    print("4️⃣  GPU PERFORMANCE TEST")
    print("-" * 80)
    if cuda_available:
        print("Running performance test...")
        
        # Create test tensors
        size = 5000
        device = torch.device('cuda')
        
        import time
        
        # CPU test
        print("\nCPU Test:")
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start
        print(f"   Matrix multiplication ({size}x{size}): {cpu_time:.4f} seconds")
        
        # GPU test
        print("\nGPU Test:")
        a_gpu = torch.randn(size, size, device=device)
        b_gpu = torch.randn(size, size, device=device)
        
        # Warm up
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"   Matrix multiplication ({size}x{size}): {gpu_time:.4f} seconds")
        
        speedup = cpu_time / gpu_time
        print(f"\n🚀 GPU Speedup: {speedup:.2f}x faster than CPU")
        
        if speedup < 5:
            print("   ⚠️  Warning: GPU speedup is lower than expected.")
            print("      This might indicate:")
            print("      - GPU is not being fully utilized")
            print("      - Data transfer overhead")
            print("      - Try larger batch sizes during training")
    print()
    
    # 5. Recommendations
    print("5️⃣  RECOMMENDATIONS FOR TRAINING")
    print("-" * 80)
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        memory_gb = props.total_memory / 1e9
        
        print("Based on your GPU:")
        
        if memory_gb >= 24:
            print("   • Batch size: 16-32")
            print("   • Model: Full size (d_model=512, n_layers=8)")
            print("   • Can train large datasets efficiently")
        elif memory_gb >= 12:
            print("   • Batch size: 8-16")
            print("   • Model: Full size (d_model=512, n_layers=8)")
            print("   • Good for most datasets")
        elif memory_gb >= 8:
            print("   • Batch size: 4-8")
            print("   • Model: Full size or slightly smaller")
            print("   • Reduce segment_length if needed")
        elif memory_gb >= 6:
            print("   • Batch size: 2-4")
            print("   • Model: Consider smaller (d_model=256, n_layers=6)")
            print("   • May need to reduce segment_length")
        else:
            print("   • Batch size: 1-2")
            print("   • Model: Smaller (d_model=256, n_layers=4)")
            print("   • Limited by GPU memory")
        
        print("\n   GPU Optimizations to enable:")
        print("   • torch.backends.cudnn.benchmark = True")
        print("   • torch.backends.cuda.matmul.allow_tf32 = True")
        print("   • Mixed precision training (future enhancement)")
    print()
    
    # 6. Quick Test
    print("6️⃣  QUICK TENSOR TEST")
    print("-" * 80)
    if cuda_available:
        try:
            # Test moving tensors to GPU
            x = torch.randn(100, 100)
            x_gpu = x.to('cuda')
            y_gpu = x_gpu * 2
            y_cpu = y_gpu.cpu()
            print("✅ Successfully created tensor on GPU")
            print("✅ Successfully performed computation on GPU")
            print("✅ Successfully moved tensor back to CPU")
            print("\n🎉 Your GPU is ready for training!")
        except Exception as e:
            print(f"❌ Error during tensor test: {e}")
    print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if cuda_available:
        print("✅ GPU is properly configured and ready to use!")
        print("✅ You can start training with GPU acceleration")
        print()
        print("To train with GPU:")
        print("   python train_quick.py --epochs 20")
        print("   python train.py")
    else:
        print("❌ GPU is NOT available")
        print("❌ Training will use CPU (much slower)")
        print()
        print("To enable GPU:")
        print("1. Make sure you have an NVIDIA GPU")
        print("2. Install NVIDIA drivers")
        print("3. Install CUDA toolkit (optional, can use bundled)")
        print("4. Reinstall PyTorch with CUDA support:")
        print("   pip uninstall torch torchaudio")
        print("   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 80)
    
    return cuda_available


if __name__ == "__main__":
    gpu_available = check_gpu()
    sys.exit(0 if gpu_available else 1)
