#!/usr/bin/env python3
"""
Download pre-trained checkpoints from Google Drive and extract them
to their expected paths under checkpoints_active/.

No access token required.

Usage:
  python scripts/download_checkpoints.py
"""

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GDRIVE_FILE_ID = "1XfjavyOjTYZ5-0oDaLT5_33UdhPWvZ8l"
GDRIVE_URL     = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
ZIP_NAME       = "checkpoints_all.zip"

CHECKPOINTS = {
    "phaseC":    PROJECT_ROOT / "checkpoints_active" / "temporal_phaseC"     / "best.pt",
    "phaseD":    PROJECT_ROOT / "checkpoints_active" / "temporal_phaseD"     / "best.pt",
    "phaseDvae": PROJECT_ROOT / "checkpoints_active" / "temporal_phaseD_vae" / "best.pt",
    "phaseE":    PROJECT_ROOT / "checkpoints_active" / "temporal_phaseE"     / "best.pt",
    "phaseF":    PROJECT_ROOT / "checkpoints_active" / "temporal_phaseF"     / "best.pt",
    "phaseG":    PROJECT_ROOT / "checkpoints_active" / "temporal_phaseG"     / "best.pt",
}


def check_existing():
    present = {k: v for k, v in CHECKPOINTS.items() if v.exists()}
    missing = {k: v for k, v in CHECKPOINTS.items() if not v.exists()}
    return present, missing


def main():
    try:
        import gdown
    except ImportError:
        print("gdown is required:  pip install gdown")
        sys.exit(1)

    present, missing = check_existing()

    if not missing:
        print("all checkpoints already present:")
        for name, path in present.items():
            mb = path.stat().st_size / 1024 / 1024
            print(f"  {name:<12}  {path.relative_to(PROJECT_ROOT)}  ({mb:.0f} MB)")
        print("\nnothing to do.")
        return

    if present:
        print(f"already present:  {', '.join(present)}")
        print(f"missing:          {', '.join(missing)}\n")

    print("downloading checkpoints_all.zip from Google Drive ...")
    zip_path = PROJECT_ROOT / ZIP_NAME

    try:
        gdown.download(url=GDRIVE_URL, output=str(zip_path), quiet=False)
    except Exception as e:
        print(f"\ndownload failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        sys.exit(1)

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print("download produced an empty file — check the Google Drive link.")
        zip_path.unlink(missing_ok=True)
        sys.exit(1)

    print(f"\nextracting ...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            for i, member in enumerate(members, 1):
                zf.extract(member, PROJECT_ROOT)
                print(f"  [{i}/{len(members)}]  {member}")
    except zipfile.BadZipFile:
        print("zip is corrupted — delete it and re-run.")
        zip_path.unlink(missing_ok=True)
        sys.exit(1)
    finally:
        if zip_path.exists():
            zip_path.unlink()

    print("\nverifying ...")
    all_ok = True
    for name, path in CHECKPOINTS.items():
        if path.exists():
            mb = path.stat().st_size / 1024 / 1024
            print(f"  ok      {name:<12}  ({mb:.0f} MB)")
        else:
            print(f"  FAIL    {name:<12}  not found after extraction")
            all_ok = False

    print()
    if all_ok:
        print("all checkpoints ready.")
        print("run:  python scripts/infer_offline.py")
    else:
        print("some checkpoints missing — check zip contents and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
