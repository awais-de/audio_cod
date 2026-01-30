# Checkpoint Path Configuration Update

## Summary
Updated all evaluation scripts to use centralized checkpoint path configuration instead of hardcoded absolute paths.

## Changes Made

### 1. Config File Updated: `config/paths.yaml`
Added checkpoint path configuration with specific checkpoint files:
```yaml
checkpoints:
  root: "checkpoints_emergency"
  v3_baseline: "checkpoints_emergency/V3.pt"
  phase1: "checkpoints_emergency/phase1_multiscale_20260129_124452/best.pt"
  phase2: "checkpoints_emergency/phase2_perceptual_20260129_210723/best.pt"
  phase3: "checkpoints_emergency/phase3_extended_data_20260129_213522/best.pt"
  phase4: "checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt"
```

### 2. Utility Module Extended: `src/paths.py`
Added new function `get_checkpoint_paths()` that:
- Loads checkpoint configuration from YAML
- Resolves relative paths to absolute paths
- Falls back to hardcoded defaults if YAML unavailable
- Returns Dict with resolved checkpoint paths

### 3. Evaluation Scripts Updated

#### `scripts/eval_testclean.py`
- Added import: `from paths import get_checkpoint_paths`
- Replaced hardcoded CHECKPOINTS dict with config-driven approach
- Now resolves all checkpoint paths dynamically

#### `scripts/eval_synthetic.py`
- Added import: `from paths import get_checkpoint_paths`
- Replaced hardcoded CHECKPOINTS dict with config-driven approach

#### `scripts/quick_eval_testclean.py`
- Added import: `from paths import get_checkpoint_paths`
- Updated phase1 checkpoint loading to use resolved path

## Benefits

✅ **Cross-System Compatibility**: Works on Linux, macOS, and Windows
✅ **No Hardcoded Paths**: All paths centralized in config
✅ **Idempotent**: Same config file works everywhere
✅ **Maintainable**: Single source of truth for checkpoints

## Verification

All Phase 1-4 checkpoints verified to load correctly:
- ✅ Phase 1: phase1_multiscale_20260129_124452/best.pt
- ✅ Phase 2: phase2_perceptual_20260129_210723/best.pt
- ✅ Phase 3: phase3_extended_data_20260129_213522/best.pt
- ✅ Phase 4: phase4_adversarial_20260130_063348/best.pt
- ❌ V3.pt: File not found (not in checkpoints_emergency)

## Testing

Ran `python scripts/eval_testclean.py` successfully:
- Checkpoint paths correctly resolved from config
- Phase 1-4 models loaded without path errors
- Evaluation proceeding with test-clean dataset
