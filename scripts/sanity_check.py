#!/usr/bin/env python3
"""
Sanity Check Script for Neural Audio Codec
Verifies:
  - All required packages are installed
  - GPU/CUDA availability
  - Dataset is downloaded and accessible
  - Configuration files are valid
  - Model can be loaded
  - Training can start
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import subprocess

class bcolors:
    """Terminal colors"""
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    OKBLUE = '\033[94m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    """Print a section header"""
    print(f"\n{bcolors.BOLD}{'='*80}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{title:^80}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{'='*80}{bcolors.ENDC}\n")

def check_mark(passed, message):
    """Print a check or X mark"""
    symbol = f"{bcolors.OKGREEN}✓{bcolors.ENDC}" if passed else f"{bcolors.FAIL}✗{bcolors.ENDC}"
    print(f"{symbol} {message}")
    return passed

def check_python_version():
    """Check Python version"""
    print_section("Python Version Check")
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 8
    check_mark(passed, f"Python {version.major}.{version.minor}.{version.micro} (required >= 3.8)")
    return passed

def check_packages():
    """Check required packages"""
    print_section("Package Installation Check")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchaudio': 'torchaudio',
        'yaml': 'PyYAML',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'soundfile': 'SoundFile',
    }
    
    all_passed = True
    for import_name, display_name in required_packages.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            check_mark(True, f"{display_name}: {version}")
        except ImportError:
            check_mark(False, f"{display_name}: NOT INSTALLED")
            all_passed = False
    
    return all_passed

def check_gpu():
    """Check GPU availability"""
    print_section("GPU/CUDA Check")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            check_mark(True, f"CUDA available with {device_count} GPU(s)")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
            check_mark(True, f"CUDA Version: {torch.version.cuda}")
            check_mark(True, f"cuDNN Version: {torch.backends.cudnn.version()}")
            return True
        else:
            check_mark(False, "No CUDA-capable GPU found")
            return False
    except Exception as e:
        check_mark(False, f"Error checking GPU: {e}")
        return False

def check_config_files():
    """Check configuration files"""
    print_section("Configuration Files Check")
    
    config_file = Path('config/training.yaml')
    passed = True
    
    if config_file.exists():
        check_mark(True, f"Config file found: {config_file}")
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required keys
            required_keys = ['model', 'training', 'data', 'checkpoint']
            for key in required_keys:
                if key in config:
                    check_mark(True, f"Config section '{key}' present")
                else:
                    check_mark(False, f"Config section '{key}' missing")
                    passed = False
            
            return passed
        except Exception as e:
            check_mark(False, f"Error parsing config: {e}")
            return False
    else:
        check_mark(False, f"Config file not found: {config_file}")
        return False

def check_model():
    """Check if model can be loaded"""
    print_section("Model Check")
    
    try:
        import torch
        import yaml
        from src.model import NeuralAudioCodec
        
        check_mark(True, "Model module imported successfully")
        
        # Load config
        with open('config/training.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        model_cfg = config['model']
        
        # Try to instantiate model
        model = NeuralAudioCodec(
            sample_rate=model_cfg['sample_rate'],
            hop_length=model_cfg['hop_length'],
            d_model=model_cfg['d_model'],
            n_layers=model_cfg['n_layers'],
            n_heads=model_cfg['n_heads'],
            window_size=model_cfg['window_size'],
            dropout=model_cfg['dropout']
        )
        
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        check_mark(True, f"Model instantiated successfully ({params:,} parameters)")
        
        # Try forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        dummy_input = torch.randn(1, 1, 6000).to(device)
        output = model(dummy_input)
        
        check_mark(True, f"Forward pass successful: {dummy_input.shape} -> {output.shape}")
        return True
        
    except Exception as e:
        check_mark(False, f"Error with model: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dataset():
    """Check if dataset is available"""
    print_section("Dataset Check")
    
    import yaml
    
    with open('config/training.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_dir = Path(config['data']['train_dir'])
    
    if train_dir.exists():
        flac_files = list(train_dir.glob('**/*.flac'))
        wav_files = list(train_dir.glob('**/*.wav'))
        mp3_files = list(train_dir.glob('**/*.mp3'))
        
        total_files = len(flac_files) + len(wav_files) + len(mp3_files)
        
        check_mark(True, f"Dataset directory exists: {train_dir}")
        check_mark(True, f"Found {total_files} audio files")
        check_mark(True, f"  - FLAC: {len(flac_files)}")
        check_mark(True, f"  - WAV: {len(wav_files)}")
        check_mark(True, f"  - MP3: {len(mp3_files)}")
        
        if total_files < 100:
            check_mark(False, f"⚠️  Dataset too small ({total_files} files, expected > 1000)")
            return False
        
        return True
    else:
        check_mark(False, f"Dataset directory not found: {train_dir}")
        return False

def suggest_dataset_download():
    """Suggest downloading dataset"""
    print_section("Dataset Download Information")
    
    print("""
To download the LibriSpeech train-clean-100 dataset (28,539 files, ~5.9GB):

Option 1: Using the training script (automatic)
  python src/train.py
  The script will detect missing dataset and download it automatically.

Option 2: Manual download
  mkdir -p /mnt/Data/muaw1874/datasets
  cd /mnt/Data/muaw1874/datasets
  wget https://openslr.trmal.net/resources/12/train-clean-100.tar.gz
  tar -xzf train-clean-100.tar.gz
  # Update config/training.yaml with the extracted path

Option 3: Update training.yaml manually
  Edit config/training.yaml and set train_dir and val_dir to your dataset location.
    """)

def main():
    """Run all sanity checks"""
    print(f"\n{bcolors.BOLD}{bcolors.OKBLUE}")
    print("""
╔═════════════════════════════════════════════════════════════════════════════╗
║          Neural Audio Codec - Project Sanity Check                          ║
╚═════════════════════════════════════════════════════════════════════════════╝
    """)
    print(f"{bcolors.ENDC}")
    
    results = {
        'Python Version': check_python_version(),
        'Packages': check_packages(),
        'GPU/CUDA': check_gpu(),
        'Config Files': check_config_files(),
        'Model': check_model(),
        'Dataset': check_dataset(),
    }
    
    # Print summary
    print_section("Summary")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        symbol = f"{bcolors.OKGREEN}✓{bcolors.ENDC}" if passed else f"{bcolors.FAIL}✗{bcolors.ENDC}"
        print(f"{symbol} {check}")
    
    print(f"\n{'='*80}\n")
    
    if all_passed:
        print(f"{bcolors.OKGREEN}{bcolors.BOLD}✓ All checks passed! Ready to train!{bcolors.ENDC}\n")
        print("Start training with:")
        print(f"{bcolors.OKBLUE}  python src/train.py{bcolors.ENDC}\n")
        return 0
    else:
        print(f"{bcolors.FAIL}{bcolors.BOLD}✗ Some checks failed!{bcolors.ENDC}\n")
        
        if not results['Dataset']:
            suggest_dataset_download()
        
        print(f"\n{bcolors.WARNING}Please fix the issues above and run this script again.{bcolors.ENDC}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
