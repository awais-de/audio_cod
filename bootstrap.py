#!/usr/bin/env python3
"""
bootstrap.py — Post-clone environment setup for the Neural Audio Codec project.

Run this once after cloning to install dependencies, verify the dataset,
download checkpoints, and confirm everything is in a runnable state.

Usage:
  python bootstrap.py
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent
PROJECT_PARENT = PROJECT_ROOT.parent
PYTHON         = sys.executable

# ── Expected structure ────────────────────────────────────────────────────────

DATASET_TEST_CLEAN    = PROJECT_PARENT / "datasets" / "LibriSpeech" / "test-clean"
DATASET_TRAIN_CLEAN   = PROJECT_PARENT / "datasets" / "LibriSpeech" / "train-clean-100"

CHECKPOINTS = {
    "phaseC":    PROJECT_ROOT / "checkpoints_active" / "temporal_phaseC"    / "best.pt",
    "phaseD":    PROJECT_ROOT / "checkpoints_active" / "temporal_phaseD"    / "best.pt",
    "phaseDvae": PROJECT_ROOT / "checkpoints_active" / "temporal_phaseD_vae"/ "best.pt",
    "phaseE":    PROJECT_ROOT / "checkpoints_active" / "temporal_phaseE"    / "best.pt",
    "phaseF":    PROJECT_ROOT / "checkpoints_active" / "temporal_phaseF"    / "best.pt",
    "phaseG":    PROJECT_ROOT / "checkpoints_active" / "temporal_phaseG"    / "best.pt",
}

REQUIRED_SRC_FILES = [
    "src/model.py",
    "src/losses.py",
    "src/codec_utils.py",
    "src/paths.py",
]

REQUIRED_SCRIPTS = [
    "scripts/infer_offline.py",
    "scripts/download_checkpoints.py",
    "scripts/encode.py",
    "scripts/decode.py",
]

# ── Output helpers ────────────────────────────────────────────────────────────

_step_index = 0
_results    = []    # (label, status, note)

def step(label):
    global _step_index
    _step_index += 1
    print(f"\n[step {_step_index}] {label}")
    print("-" * 60)

def ok(label, note=""):
    msg = f"  ok      {label}"
    if note:
        msg += f"  ({note})"
    print(msg)
    _results.append((label, "ok", note))

def warn(label, note=""):
    msg = f"  warn    {label}"
    if note:
        msg += f"  — {note}"
    print(msg)
    _results.append((label, "warn", note))

def fail(label, note=""):
    msg = f"  FAIL    {label}"
    if note:
        msg += f"  — {note}"
    print(msg)
    _results.append((label, "fail", note))

def info(msg):
    print(f"          {msg}")

# ── Steps ─────────────────────────────────────────────────────────────────────

def check_python():
    step("Python version")
    major, minor = sys.version_info[:2]
    version_str  = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) >= (3, 8):
        ok(f"Python {version_str}")
    else:
        fail(f"Python {version_str}", "3.8+ required")
        sys.exit(1)


def check_venv():
    step("Virtual environment")
    in_venv  = sys.prefix != sys.base_prefix
    in_conda = os.environ.get("CONDA_DEFAULT_ENV") is not None
    if in_venv or in_conda:
        env_name = Path(sys.prefix).name
        ok(f"active  ({env_name})")
    else:
        warn(
            "no virtual environment detected",
            "dependencies will be installed to the system Python"
        )
        info("recommended: python -m venv venv && source venv/bin/activate")


def install_dependencies():
    step("Python dependencies")

    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        fail("requirements.txt not found")
        return

    # Upgrade pip silently first
    subprocess.run(
        [PYTHON, "-m", "pip", "install", "--upgrade", "pip", "-q"],
        check=False
    )

    # Install all requirements; capture output to detect partial failures
    result = subprocess.run(
        [PYTHON, "-m", "pip", "install", "-r", str(req_file), "-q"],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        ok("all packages from requirements.txt")
    else:
        stderr = result.stderr.lower()
        # Known optional packages that require system libraries to build from source.
        # Inference works without them — degrade gracefully.
        known_optional = {
            "pyaudio":   "PortAudio not found — streaming inference unavailable",
            "portaudio": "PortAudio not found — streaming inference unavailable",
            "pesq":      "Python dev headers missing (python3-dev) — PESQ metric unavailable",
        }
        hit = next((msg for key, msg in known_optional.items() if key in stderr), None)
        if hit:
            warn("optional package could not be built", hit)
            ok("all other packages installed")
        else:
            fail("dependency installation had errors")
            info(result.stderr[:300].strip())


def check_dataset():
    step("Dataset  (LibriSpeech)")

    # test-clean — required for inference
    if DATASET_TEST_CLEAN.exists():
        flac_files = list(DATASET_TEST_CLEAN.rglob("*.flac"))
        if flac_files:
            ok(f"test-clean", f"{len(flac_files)} .flac files at {DATASET_TEST_CLEAN}")
        else:
            warn("test-clean directory exists but contains no .flac files")
    else:
        fail(
            "test-clean not found",
            f"expected at {DATASET_TEST_CLEAN}"
        )
        info("to download:")
        info(f"  mkdir -p {DATASET_TEST_CLEAN.parent}")
        info(f"  cd {DATASET_TEST_CLEAN.parent}")
        info( "  wget https://www.openslr.org/resources/12/test-clean.tar.gz")
        info( "  tar -xzf test-clean.tar.gz")

    # train-clean-100 — only needed for training
    if DATASET_TRAIN_CLEAN.exists():
        ok("train-clean-100", "training data present")
    else:
        warn(
            "train-clean-100 not found",
            "not required for inference — only needed to re-run training scripts"
        )


def check_checkpoints():
    step("Model checkpoints")

    present = {name: path for name, path in CHECKPOINTS.items() if path.exists()}
    missing = {name: path for name, path in CHECKPOINTS.items() if not path.exists()}

    for name in present:
        mb = present[name].stat().st_size / 1024 / 1024
        ok(name, f"{mb:.0f} MB")

    if not missing:
        return

    info(f"missing: {', '.join(missing)}")
    info("downloading from Google Drive ...")
    sys.stdout.flush()

    result = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "scripts" / "download_checkpoints.py")]
    )

    print()
    if result.returncode == 0:
        for name in missing:
            path = CHECKPOINTS[name]
            if path.exists():
                mb = path.stat().st_size / 1024 / 1024
                ok(name, f"{mb:.0f} MB  (downloaded)")
            else:
                fail(name, "still missing after download")
    else:
        for name in missing:
            fail(name, "download failed — run scripts/download_checkpoints.py manually")


def check_project_structure():
    step("Project structure")

    for rel in REQUIRED_SRC_FILES + REQUIRED_SCRIPTS:
        path = PROJECT_ROOT / rel
        if path.exists():
            ok(rel)
        else:
            fail(rel, "file missing")


def verify_imports():
    step("Package imports")

    # Required
    required = [
        ("torch",     "PyTorch"),
        ("numpy",     "NumPy"),
        ("soundfile", "soundfile"),
        ("yaml",      "PyYAML"),
    ]
    # Optional (degrade gracefully)
    optional = [
        ("pesq",    "pesq   (PESQ metric)"),
        ("pystoi",  "pystoi (STOI metric)"),
        ("tqdm",    "tqdm"),
    ]

    def try_import(module):
        result = subprocess.run(
            [PYTHON, "-c", f"import {module}; print({module}.__version__ if hasattr({module}, '__version__') else 'ok')"],
            capture_output=True, text=True
        )
        return result.returncode == 0, result.stdout.strip()

    for module, label in required:
        success, version = try_import(module)
        if success:
            ok(label, version)
        else:
            fail(label, "import failed — re-run: pip install -r requirements.txt")

    for module, label in optional:
        success, version = try_import(module)
        if success:
            ok(label, version)
        else:
            warn(label, "not available — metrics will show n/a during inference")

    # Project modules
    sys.path.insert(0, str(PROJECT_ROOT))
    project_modules = [
        ("src.model",       "src.model"),
        ("src.losses",      "src.losses"),
        ("src.codec_utils", "src.codec_utils"),
        ("src.paths",       "src.paths"),
    ]
    for module, label in project_modules:
        result = subprocess.run(
            [PYTHON, "-c", f"import sys; sys.path.insert(0, '{PROJECT_ROOT}'); import {module}"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            ok(label)
        else:
            fail(label, result.stderr.split("\\n")[-2] if result.stderr else "import error")


def smoke_test():
    step("Smoke test  (model forward pass on random noise)")

    code = f"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '{PROJECT_ROOT}')
import torch
from src.model import NeuralAudioCodec

model = NeuralAudioCodec(d_model=384, bottleneck_dim=32, temporal_stride=20).eval()
x = torch.randn(1, 1, 16000)   # 1 second of noise at 16 kHz
with torch.no_grad():
    z     = model.encode(x)
    recon = model.decode(z)

assert recon.shape[2] > 0, "decode produced empty output"
z_shape    = tuple(z.shape)
recon_len  = recon.shape[2]
print(f"encode output: {{z_shape}}")
print(f"decode output: (1, 1, {{recon_len}})")
"""
    result = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True)
    if result.returncode == 0:
        lines = [l for l in result.stdout.strip().splitlines() if l]
        for line in lines:
            info(line)
        ok("encode → decode  (random noise, no checkpoint)")
    else:
        fail("forward pass failed")
        info(result.stderr.strip().splitlines()[-1] if result.stderr else "unknown error")


def print_summary():
    n_ok   = sum(1 for _, s, _ in _results if s == "ok")
    n_warn = sum(1 for _, s, _ in _results if s == "warn")
    n_fail = sum(1 for _, s, _ in _results if s == "fail")

    print("\n" + "=" * 60)
    print("summary")
    print("=" * 60)
    print(f"  ok:       {n_ok}")
    print(f"  warnings: {n_warn}")
    print(f"  failures: {n_fail}")

    failures = [(l, n) for l, s, n in _results if s == "fail"]
    if failures:
        print("\nfailed checks:")
        for label, note in failures:
            print(f"  {label}" + (f"  — {note}" if note else ""))

    print()
    ckpt_present = [name for name, path in CHECKPOINTS.items() if path.exists()]
    test_clean_ok = DATASET_TEST_CLEAN.exists() and bool(list(DATASET_TEST_CLEAN.rglob("*.flac")))

    if ckpt_present and test_clean_ok:
        print("ready to run:")
        print("  python scripts/infer_offline.py")
    else:
        print("before running inference:")
        n = 1
        if not ckpt_present:
            print(f"  {n}. python scripts/download_checkpoints.py")
            n += 1
        if not test_clean_ok:
            print(f"  {n}. download LibriSpeech test-clean  (see dataset step above)")
            print(f"     or use your own audio file:")
            print(f"     python scripts/infer_offline.py --input /path/to/audio.wav")
        else:
            print()
            print("then:")
            print("  python scripts/infer_offline.py")

    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Neural Audio Codec — bootstrap")
    print(f"platform:  {platform.system()} {platform.release()}")
    print(f"python:    {sys.version.split()[0]}  ({PYTHON})")
    print(f"root:      {PROJECT_ROOT}")
    print("=" * 60)

    check_python()
    check_venv()
    install_dependencies()
    check_dataset()
    check_checkpoints()
    check_project_structure()
    verify_imports()
    smoke_test()
    print_summary()


if __name__ == "__main__":
    main()
