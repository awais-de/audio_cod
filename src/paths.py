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
