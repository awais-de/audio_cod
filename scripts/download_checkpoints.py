#!/usr/bin/env python3
"""
Download pre-trained checkpoints from the GitLab Generic Package Registry
and place them at their expected paths under checkpoints_active/.

Requires a GitLab PAT with read_api scope (or read_registry on self-hosted):
  gitlab.tu-ilmenau.de → User Settings → Access Tokens → scope: read_api

Usage:
  export GITLAB_TOKEN=<your-pat>
  python scripts/download_checkpoints.py
  python scripts/download_checkpoints.py --only phaseG
"""

import argparse
import http.client
import os
import ssl
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GITLAB_HOST  = "gitlab.tu-ilmenau.de"
PROJECT_PATH = "muaw1874%2Faudio_cod"
PACKAGE_NAME = "checkpoints"
PACKAGE_VER  = "1.0.0"

CHECKPOINTS = {
    "phaseC":    ("temporal_phaseC",     "phaseC_best.pt"),
    "phaseD":    ("temporal_phaseD",     "phaseD_best.pt"),
    "phaseDvae": ("temporal_phaseD_vae", "phaseDvae_best.pt"),
    "phaseE":    ("temporal_phaseE",     "phaseE_best.pt"),
    "phaseF":    ("temporal_phaseF",     "phaseF_best.pt"),
    "phaseG":    ("temporal_phaseG",     "phaseG_best.pt"),
}

CHUNK_SIZE = 1024 * 1024   # 1 MB read chunks
BAR_WIDTH  = 32


def get_token():
    token = os.environ.get("GITLAB_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "GITLAB_TOKEN not set.\n"
            "Generate a PAT at: https://gitlab.tu-ilmenau.de/-/user_settings/personal_access_tokens\n"
            "Required scope: read_api\n"
            "Then run: export GITLAB_TOKEN=<your-pat>"
        )
    return token


def _progress_bar(received, total):
    if total <= 0:
        sys.stdout.write(f"\r  {received / 1024 / 1024:.1f} MB received")
    else:
        pct    = received * 100 // total
        filled = BAR_WIDTH * received // total
        bar    = "█" * filled + "░" * (BAR_WIDTH - filled)
        mb_r   = received / 1024 / 1024
        mb_t   = total    / 1024 / 1024
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  {mb_r:5.1f} / {mb_t:.0f} MB")
    sys.stdout.flush()


def download_file(token, file_name, dest):
    path = (
        f"/api/v4/projects/{PROJECT_PATH}/packages/generic"
        f"/{PACKAGE_NAME}/{PACKAGE_VER}/{file_name}"
    )

    ctx  = ssl.create_default_context()
    conn = http.client.HTTPSConnection(GITLAB_HOST, context=ctx)
    conn.request("GET", path, headers={
        "PRIVATE-TOKEN": token,
        "User-Agent":    "audio-codec-downloader",
    })
    resp = conn.getresponse()

    if resp.status == 401:
        raise RuntimeError("Unauthorised — check your GITLAB_TOKEN and its scope (read_api).")
    if resp.status == 404:
        raise RuntimeError(f"File not found in package registry: {file_name}")
    if resp.status != 200:
        raise RuntimeError(f"Unexpected status {resp.status}")

    total    = int(resp.getheader("Content-Length", 0))
    received = 0
    tmp      = dest.with_suffix(".tmp")

    try:
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
                _progress_bar(received, total)

        _progress_bar(received, total)
        print()
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def run(args):
    token = get_token()

    targets = CHECKPOINTS
    if args.only:
        if args.only not in CHECKPOINTS:
            print(f"Unknown phase '{args.only}'. Valid: {', '.join(CHECKPOINTS)}")
            sys.exit(1)
        targets = {args.only: CHECKPOINTS[args.only]}

    total  = len(targets)
    failed = []

    print(f"Downloading {total} checkpoint(s) from {GITLAB_HOST}\n")

    for i, (phase, (ckpt_dir, file_name)) in enumerate(targets.items(), 1):
        dest = PROJECT_ROOT / "checkpoints_active" / ckpt_dir / "best.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{total}] {phase:<12}  {file_name}")

        if dest.exists():
            mb = dest.stat().st_size / 1024 / 1024
            print(f"  skip  ({mb:.0f} MB already at {dest.relative_to(PROJECT_ROOT)})")
            continue

        try:
            download_file(token, file_name, dest)
            mb = dest.stat().st_size / 1024 / 1024
            print(f"  done  ({mb:.0f} MB → {dest.relative_to(PROJECT_ROOT)})")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(phase)

    print()
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)

    print("All checkpoints ready.")
    print("Run inference with: python scripts/infer_offline.py")


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained checkpoints from GitLab Package Registry."
    )
    parser.add_argument(
        "--only", type=str, default=None, metavar="PHASE",
        help=f"Download a single checkpoint. One of: {', '.join(CHECKPOINTS)}"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
