from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_parent() -> Path:
    return _project_root().parent


def _resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_paths_config() -> Dict:
    config_file = _project_root() / "config" / "paths.yaml"
    if yaml and config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_dataset_paths() -> Dict[str, Path]:
    config = load_paths_config()
    parent = _project_parent()

    if not config:
        datasets_root = parent / "datasets"
        train_clean_100 = datasets_root / "LibriSpeech" / "train-clean-100"
        test_clean = datasets_root / "LibriSpeech" / "test-clean"
        return {
            "datasets_root": datasets_root.resolve(),
            "train_clean_100": train_clean_100.resolve(),
            "test_clean": test_clean.resolve(),
        }

    datasets_cfg = config.get("datasets", {})
    librispeech_cfg = datasets_cfg.get("librispeech", {})

    datasets_root = _resolve_path(datasets_cfg.get("root", "datasets"), parent)
    train_clean_100 = _resolve_path(
        librispeech_cfg.get("train_clean_100", "datasets/LibriSpeech/train-clean-100"),
        parent,
    )
    test_clean = _resolve_path(
        librispeech_cfg.get("test_clean", "datasets/LibriSpeech/test-clean"),
        parent,
    )

    return {
        "datasets_root": datasets_root,
        "train_clean_100": train_clean_100,
        "test_clean": test_clean,
    }


def get_checkpoint_paths() -> Dict[str, Path]:
    config = load_paths_config()
    root = _project_root()

    if not config:
        checkpoints_root = root / "checkpoints_emergency"
        return {
            "root": checkpoints_root.resolve(),
            "v3_baseline": (checkpoints_root / "V3.pt").resolve(),
            "phase1": (checkpoints_root / "phase1_multiscale_20260129_124452" / "best.pt").resolve(),
            "phase2": (checkpoints_root / "phase2_perceptual_20260129_210723" / "best.pt").resolve(),
            "phase3": (checkpoints_root / "phase3_extended_data_20260129_213522" / "best.pt").resolve(),
            "phase4": (checkpoints_root / "phase4_adversarial_20260130_063348" / "best.pt").resolve(),
        }

    checkpoints_cfg = config.get("checkpoints", {})

    return {
        "root": _resolve_path(checkpoints_cfg.get("root", "checkpoints_emergency"), root),
        "v3_baseline": _resolve_path(checkpoints_cfg.get("v3_baseline", "checkpoints_emergency/V3.pt"), root),
        "phase1": _resolve_path(checkpoints_cfg.get("phase1", "checkpoints_emergency/phase1_multiscale_20260129_124452/best.pt"), root),
        "phase2": _resolve_path(checkpoints_cfg.get("phase2", "checkpoints_emergency/phase2_perceptual_20260129_210723/best.pt"), root),
        "phase3": _resolve_path(checkpoints_cfg.get("phase3", "checkpoints_emergency/phase3_extended_data_20260129_213522/best.pt"), root),
        "phase4": _resolve_path(checkpoints_cfg.get("phase4", "checkpoints_emergency/phase4_adversarial_20260130_063348/best.pt"), root),
    }
