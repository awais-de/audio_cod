#!/usr/bin/env python3
"""
Download pre-trained checkpoints from the GitHub release and place them
at their expected paths under checkpoints_active/.

Usage:
  python scripts/download_checkpoints.py
  python scripts/download_checkpoints.py --only phaseG   # single checkpoint
"""

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RELEASE_BASE = (
    "https://github.com/awais-de/audio_cod/releases/download/v1.0-checkpoints"
)

CHECKPOINTS = {
    "phaseC":    ("temporal_phaseC",     f"{RELEASE_BASE}/phaseC_best.pt"),
    "phaseD":    ("temporal_phaseD",     f"{RELEASE_BASE}/phaseD_best.pt"),
    "phaseDvae": ("temporal_phaseD_vae", f"{RELEASE_BASE}/phaseDvae_best.pt"),
    "phaseE":    ("temporal_phaseE",     f"{RELEASE_BASE}/phaseE_best.pt"),
    "phaseF":    ("temporal_phaseF",     f"{RELEASE_BASE}/phaseF_best.pt"),
    "phaseG":    ("temporal_phaseG",     f"{RELEASE_BASE}/phaseG_best.pt"),
}


def download(name, dest_dir, url):
    dest = PROJECT_ROOT / "checkpoints_active" / dest_dir / "best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  {name:<12} skip  ({size_mb:.0f}MB already at {dest.relative_to(PROJECT_ROOT)})")
        return

    tmp = dest.with_suffix(".tmp")

    def progress(count, block_sz, total):
        if total > 0:
            pct = min(count * block_sz * 100 // total, 100)
            mb_done = count * block_sz / 1024 / 1024
            sys.stdout.write(f"\r  {name:<12} {pct:3d}%  {mb_done:.1f}/{total/1024/1024:.0f}MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=progress)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"\r  {name:<12} done  ({size_mb:.0f}MB → {dest.relative_to(PROJECT_ROOT)})")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"\r  {name:<12} FAILED: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained checkpoints from the GitHub release."
    )
    parser.add_argument(
        "--only", type=str, default=None,
        metavar="PHASE",
        help=f"Download a single checkpoint. One of: {', '.join(CHECKPOINTS)}"
    )
    args = parser.parse_args()

    targets = CHECKPOINTS
    if args.only:
        if args.only not in CHECKPOINTS:
            print(f"Unknown phase '{args.only}'. Valid: {', '.join(CHECKPOINTS)}")
            sys.exit(1)
        targets = {args.only: CHECKPOINTS[args.only]}

    total = len(targets)
    print(f"Downloading {total} checkpoint(s) from GitHub release v1.0-checkpoints\n")

    failed = []
    for i, (name, (dest_dir, url)) in enumerate(targets.items(), 1):
        print(f"[{i}/{total}] {name}")
        try:
            download(name, dest_dir, url)
        except Exception:
            failed.append(name)

    print()
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All checkpoints ready.")
        print(f"Run inference with: python scripts/infer_offline.py")


if __name__ == "__main__":
    main()
