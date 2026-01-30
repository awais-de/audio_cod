#!/usr/bin/env python3
"""
Comprehensive Setup Script for Neural Audio Codec Project
Handles: venv creation, dependencies, FFmpeg, dataset/model checks, sanity tests
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
import json

class SetupManager:
    def __init__(self):
        self.os_type = platform.system()  # 'Linux', 'Darwin' (macOS), 'Windows'
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / 'venv'
        self.python_exe = sys.executable
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []

    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80 + "\n")

    def print_success(self, text):
        """Print success message"""
        print(f"✅ {text}")
        self.checks_passed.append(text)

    def print_error(self, text):
        """Print error message"""
        print(f"❌ {text}")
        self.checks_failed.append(text)

    def print_warning(self, text):
        """Print warning message"""
        print(f"⚠️  {text}")
        self.warnings.append(text)

    def print_info(self, text):
        """Print info message"""
        print(f"ℹ️  {text}")

    def run_command(self, cmd, shell=False, capture=False):
        """Run shell command safely"""
        try:
            if capture:
                result = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE, timeout=30)
                stdout = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ''
                stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ''
                return result.returncode == 0, stdout + stderr
            else:
                result = subprocess.run(cmd, shell=shell, timeout=30)
                return result.returncode == 0, ""
        except subprocess.TimeoutExpired:
            return False, "Command timeout"
        except Exception as e:
            return False, str(e)

    # =========================================================================
    # STEP 1: Virtual Environment
    # =========================================================================
    def setup_venv(self):
        """Create and activate virtual environment"""
        self.print_header("STEP 1: Virtual Environment Setup")

        if self.venv_path.exists():
            self.print_info(f"Virtual environment already exists: {self.venv_path}")
            return True

        print(f"Creating virtual environment at {self.venv_path}...")
        success, output = self.run_command(
            [sys.executable, "-m", "venv", str(self.venv_path)]
        )

        if success:
            self.print_success(f"Virtual environment created: {self.venv_path}")
            if self.os_type == 'Windows':
                self.print_info(f"Activate with: {self.venv_path}\\Scripts\\activate")
            else:
                self.print_info(f"Activate with: source {self.venv_path}/bin/activate")
            return True
        else:
            self.print_error(f"Failed to create virtual environment: {output}")
            return False

    def get_pip_exe(self):
        """Get pip executable path"""
        if self.os_type == 'Windows':
            return str(self.venv_path / 'Scripts' / 'pip.exe')
        else:
            return str(self.venv_path / 'bin' / 'pip')

    def get_python_exe(self):
        """Get python executable path"""
        if self.os_type == 'Windows':
            return str(self.venv_path / 'Scripts' / 'python.exe')
        else:
            return str(self.venv_path / 'bin' / 'python')

    # =========================================================================
    # STEP 2: Dependencies
    # =========================================================================
    def install_dependencies(self):
        """Install all Python dependencies from requirements.txt"""
        self.print_header("STEP 2: Installing Python Dependencies")

        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            self.print_error(f"requirements.txt not found at {requirements_file}")
            return False

        pip_exe = self.get_pip_exe()
        print(f"Installing packages from {requirements_file}...")
        print("This may take several minutes...\n")

        success, output = self.run_command(
            [pip_exe, 'install', '--upgrade', 'pip'],
            capture=True
        )
        if success:
            self.print_success("Pip upgraded")
        else:
            self.print_warning(f"Failed to upgrade pip: {output[:100]}")

        success, output = self.run_command(
            [pip_exe, 'install', '-r', str(requirements_file)],
            capture=True
        )

        if success:
            self.print_success("All dependencies installed from requirements.txt")
            return True
        else:
            self.print_error(f"Failed to install dependencies: {output[:200]}")
            return False

    # =========================================================================
    # STEP 3: FFmpeg Support
    # =========================================================================
    def check_ffmpeg(self):
        """Check if FFmpeg is installed"""
        self.print_header("STEP 3: FFmpeg Support Check")

        success, output = self.run_command('ffmpeg -version', shell=True, capture=True)
        
        if success:
            version = output.split('\n')[0]
            self.print_success(f"FFmpeg installed: {version}")
            return True
        else:
            self.print_warning("FFmpeg not installed. Audio codec support may be limited.")
            self.print_install_ffmpeg_instructions()
            return False

    def print_install_ffmpeg_instructions(self):
        """Print OS-specific FFmpeg installation instructions"""
        print("\n" + "-"*80)
        print("To enable full FFmpeg support, install with your package manager:")
        print("-"*80)

        if self.os_type == 'Darwin':  # macOS
            print("\nmacOS (using Homebrew):")
            print("  brew install ffmpeg")
            print("\nThen install Python codec packages:")
            print("  pip install torchcodec audioread")

        elif self.os_type == 'Linux':
            print("\nLinux (Ubuntu/Debian):")
            print("  sudo apt-get update")
            print("  sudo apt-get install -y ffmpeg libavformat-dev libavcodec-dev")
            print("\nThen install Python codec packages:")
            print("  pip install torchcodec audioread")

        elif self.os_type == 'Windows':
            print("\nWindows (using Chocolatey):")
            print("  choco install ffmpeg -y")
            print("\nOr download from: https://ffmpeg.org/download.html")
            print("\nThen install Python codec packages:")
            print("  pip install torchcodec audioread")

        print("-"*80 + "\n")

    def install_ffmpeg_codecs(self):
        """Attempt to install FFmpeg codec packages"""
        self.print_info("Attempting to install FFmpeg codec packages...")
        pip_exe = self.get_pip_exe()
        
        packages = ['torchcodec', 'audioread']
        for pkg in packages:
            success, _ = self.run_command(
                [pip_exe, 'install', pkg],
                capture=True
            )
            if success:
                self.print_success(f"Installed {pkg}")
            else:
                self.print_warning(f"Could not install {pkg} (may need FFmpeg system libs)")

    # =========================================================================
    # STEP 4: Dataset Check
    # =========================================================================
    def check_datasets(self):
        """Check if required datasets exist"""
        self.print_header("STEP 4: Dataset Verification")

        datasets_root = self.project_root / 'datasets' / 'LibriSpeech'
        datasets_found = {
            'train-clean-100': datasets_root / 'train-clean-100',
            'test-clean': datasets_root / 'test-clean',
        }

        found_count = 0
        for name, path in datasets_found.items():
            if path.exists():
                # Count files
                audio_files = list(path.glob('**/*.flac')) + list(path.glob('**/*.wav'))
                self.print_success(f"{name}: {path} ({len(audio_files)} files)")
                found_count += 1
            else:
                self.print_warning(f"{name}: Not found at {path}")

        if found_count == 0:
            print("\nTo download LibriSpeech datasets:")
            print("  mkdir -p datasets && cd datasets")
            print("  wget https://www.openslr.org/resources/12/train-clean-100.tar.gz")
            print("  wget https://www.openslr.org/resources/12/test-clean.tar.gz")
            print("  tar -xzf train-clean-100.tar.gz")
            print("  tar -xzf test-clean.tar.gz")

        return found_count > 0

    # =========================================================================
    # STEP 5: Model Checkpoints
    # =========================================================================
    def check_models(self):
        """Check if trained models exist"""
        self.print_header("STEP 5: Model Checkpoint Verification")

        checkpoints_dir = self.project_root / 'checkpoints_emergency'
        models = {
            'Phase 1 (Multi-scale Spectral)': 'phase1_multiscale_*',
            'Phase 2 (Perceptual)': 'phase2_perceptual_*',
            'Phase 3 (Extended Data)': 'phase3_extended_data_*',
            'Phase 4 (Adversarial)': 'phase4_adversarial_*',
        }

        found_count = 0
        for model_name, pattern in models.items():
            matches = list(checkpoints_dir.glob(pattern))
            if matches:
                checkpoint = matches[0] / 'best.pt'
                if checkpoint.exists():
                    size_mb = checkpoint.stat().st_size / (1024*1024)
                    self.print_success(f"{model_name}: {checkpoint} ({size_mb:.1f} MB)")
                    found_count += 1
                else:
                    self.print_warning(f"{model_name}: Directory found but best.pt missing")
            else:
                self.print_warning(f"{model_name}: Not found")

        return found_count > 0

    # =========================================================================
    # STEP 6: Sanity Checks
    # =========================================================================
    def run_sanity_checks(self):
        """Run sanity checks on installed packages and model functionality"""
        self.print_header("STEP 6: Sanity Checks")

        python_exe = self.get_python_exe()

        # Check core imports
        print("Checking core imports...")
        core_imports = [
            ('torch', 'PyTorch'),
            ('torchaudio', 'Torchaudio'),
            ('numpy', 'NumPy'),
            ('scipy', 'SciPy'),
            ('pesq', 'PESQ'),
            ('pystoi', 'STOI'),
            ('tqdm', 'tqdm'),
        ]

        imports_ok = 0
        for module, name in core_imports:
            code = f"import {module}; print('OK')"
            success, _ = self.run_command(
                [python_exe, '-c', code],
                capture=True
            )
            if success:
                self.print_success(f"{name} imported successfully")
                imports_ok += 1
            else:
                self.print_error(f"Failed to import {name}")

        # Check model loading
        print("\nChecking model architecture...")
        code = """
import sys
sys.path.insert(0, 'src')
from model import NeuralAudioCodec
model = NeuralAudioCodec(d_model=384, n_layers=6)
print('OK')
"""
        success, output = self.run_command(
            [python_exe, '-c', code],
            capture=True
        )
        if success:
            self.print_success("Model architecture loads correctly")
        else:
            self.print_error(f"Failed to load model: {output[:200]}")

        # Check GPU/Device
        print("\nChecking compute device...")
        code = """
import torch
if torch.cuda.is_available():
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("Apple Metal Performance Shaders (MPS)")
else:
    print("CPU only")
"""
        success, output = self.run_command(
            [python_exe, '-c', code],
            capture=True
        )
        if success:
            device_info = output.strip()
            self.print_info(f"Device: {device_info}")

        return imports_ok >= 6

    # =========================================================================
    # STEP 7: Demo Scripts Check
    # =========================================================================
    def check_demo_scripts(self):
        """Check if evaluation/demo scripts exist"""
        self.print_header("STEP 7: Evaluation/Demo Scripts")

        scripts = {
            'Test-Clean Evaluation': 'scripts/eval_testclean.py',
            'Synthetic Evaluation': 'scripts/eval_synthetic.py',
            'AMS Codec': 'scripts/ams_codec.py',
            'Demo Server': 'scripts/demo_server.py',
            'Demo Client': 'scripts/demo_client.py',
        }

        found_count = 0
        for script_name, script_path in scripts.items():
            full_path = self.project_root / script_path
            if full_path.exists():
                self.print_success(f"{script_name}: {script_path}")
                found_count += 1
            else:
                self.print_warning(f"{script_name}: {script_path} not found")

        return found_count >= 2

    # =========================================================================
    # STEP 8: Final Report
    # =========================================================================
    def print_final_report(self):
        """Print final setup report"""
        self.print_header("SETUP REPORT SUMMARY")

        print(f"✅ Checks Passed: {len(self.checks_passed)}")
        for check in self.checks_passed[:5]:
            print(f"   • {check}")
        if len(self.checks_passed) > 5:
            print(f"   ... and {len(self.checks_passed)-5} more")

        if self.checks_failed:
            print(f"\n❌ Checks Failed: {len(self.checks_failed)}")
            for check in self.checks_failed:
                print(f"   • {check}")

        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings[:3]:
                print(f"   • {warning}")

        print("\n" + "="*80)
        if len(self.checks_failed) == 0:
            print("✅ SETUP COMPLETE - Ready for inference and evaluation!")
        else:
            print("⚠️  SETUP PARTIAL - Some optional features may be unavailable")
        print("="*80)

        self.print_next_steps()

    def print_next_steps(self):
        """Print recommended next steps"""
        self.print_header("NEXT STEPS")

        print("1. Activate Virtual Environment:")
        if self.os_type == 'Windows':
            print(f"   {self.venv_path}\\Scripts\\activate")
        else:
            print(f"   source {self.venv_path}/bin/activate")

        print("\n2. Run Evaluation:")
        print("   python scripts/eval_testclean.py    # Real audio (requires test-clean)")
        print("   python scripts/eval_synthetic.py    # Synthetic audio (no dataset needed)")

        print("\n3. Start Demo Server:")
        print("   python scripts/demo_server.py")
        print("   python scripts/demo_client.py")

        print("\n4. View Documentation:")
        print("   - FINAL_EVALUATION_REPORT.md")
        print("   - LOCAL_EVALUATION_GUIDE.md")
        print("   - FFMPEG_INSTALL.md")

        print("\n" + "="*80)

    # =========================================================================
    # Main Execution
    # =========================================================================
    def run_full_setup(self):
        """Run complete setup workflow"""
        print("\n" + "="*80)
        print("  NEURAL AUDIO CODEC - COMPREHENSIVE SETUP")
        print("="*80)
        print(f"  OS: {self.os_type}")
        print(f"  Project Root: {self.project_root}")
        print("="*80)

        # Step 1: venv
        if not self.setup_venv():
            return False

        # Step 2: Dependencies
        if not self.install_dependencies():
            self.print_warning("Some dependencies failed to install. Continuing...")

        # Step 3: FFmpeg
        has_ffmpeg = self.check_ffmpeg()
        if not has_ffmpeg:
            self.install_ffmpeg_codecs()

        # Step 4: Datasets
        has_datasets = self.check_datasets()

        # Step 5: Models
        has_models = self.check_models()

        # Step 6: Sanity checks
        checks_ok = self.run_sanity_checks()

        # Step 7: Demo scripts
        has_scripts = self.check_demo_scripts()

        # Step 8: Report
        self.print_final_report()

        return checks_ok and has_models and has_scripts


def main():
    try:
        manager = SetupManager()
        success = manager.run_full_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
