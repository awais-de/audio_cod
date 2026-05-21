#!/usr/bin/env python3
"""
Overnight Chain: Phase F → eval_phaseF → Phase G → eval_phaseG
===============================================================
Runs all four steps in order. Aborts on any failure.

Usage:
  python scripts/overnight_chain.py --dry-run   # sanity check only
  python scripts/overnight_chain.py             # real run (nohup recommended)

Logs: runs/overnight_chain.log  (also printed to stdout)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / 'scripts'
CHECKPOINTS = PROJECT_ROOT / 'checkpoints_active'
LOG_DIR = PROJECT_ROOT / 'runs'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg, file=None):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if file:
        file.write(line + "\n")
        file.flush()


def check_file(path, label, dry_run, log_file):
    if path.exists():
        log(f"  OK   {label}: {path}", log_file)
        return True
    else:
        if dry_run:
            log(f"  MISS {label}: {path}  (will be created by chain)", log_file)
            return True  # OK for dry-run — downstream steps produce it
        log(f"  FAIL {label}: {path} not found", log_file)
        return False


def run_step(name, cmd, dry_run, log_file):
    log(f"\n{'='*64}", log_file)
    log(f"STEP: {name}", log_file)
    log(f"CMD : {' '.join(str(c) for c in cmd)}", log_file)
    log(f"{'='*64}", log_file)

    if dry_run:
        log(f"[DRY-RUN] would execute: {' '.join(str(c) for c in cmd)}", log_file)
        return True

    t0 = time.time()
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()
    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        log(f"FAILED (exit {proc.returncode}) after {elapsed/60:.1f} min", log_file)
        return False

    log(f"DONE  ({elapsed/60:.1f} min)", log_file)
    return True


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def preflight(dry_run, log_file):
    log("\nPRE-FLIGHT CHECKS", log_file)
    ok = True

    # Required inputs
    ok &= check_file(CHECKPOINTS / 'temporal_phaseC/best.pt', 'Phase C checkpoint', dry_run, log_file)

    # Scripts
    for script in ['finetune_temporal_phaseF.py', 'finetune_temporal_phaseG.py',
                   'eval_phaseF.py', 'eval_phaseG.py']:
        ok &= check_file(SCRIPTS / script, f'script {script}', dry_run, log_file)

    # Python can import project modules
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("model", PROJECT_ROOT / 'src/model.py')
        assert spec is not None
        log("  OK   src/model.py importable", log_file)
    except Exception as e:
        log(f"  FAIL src/model.py: {e}", log_file)
        ok = False

    # torchaudio available (required for Phase F/G mel loss)
    try:
        import torchaudio  # noqa: F401
        log("  OK   torchaudio importable", log_file)
    except ImportError:
        log("  FAIL torchaudio not installed — Phase F/G require it", log_file)
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Chain definition
# ---------------------------------------------------------------------------

def build_steps(python):
    return [
        {
            'name': 'Phase F training  (40 epochs, triple loss, from Phase C)',
            'cmd':  [python, SCRIPTS / 'finetune_temporal_phaseF.py'],
            'produces': CHECKPOINTS / 'temporal_phaseF/best.pt',
        },
        {
            'name': 'Eval: Phase C vs Phase F',
            'cmd':  [python, SCRIPTS / 'eval_phaseF.py'],
            'produces': None,  # timestamped folder — checked differently
        },
        {
            'name': 'Phase G training  (20 epochs, fine-polish, from Phase F)',
            'cmd':  [python, SCRIPTS / 'finetune_temporal_phaseG.py'],
            'produces': CHECKPOINTS / 'temporal_phaseG/best.pt',
        },
        {
            'name': 'Eval: Phase C vs Phase F vs Phase G',
            'cmd':  [python, SCRIPTS / 'eval_phaseG.py'],
            'produces': None,
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overnight training + eval chain")
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be executed without running anything')
    parser.add_argument('--skip-preflight', action='store_true',
                        help='Skip pre-flight checks (for resuming a partial run)')
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / 'overnight_chain.log'

    python = sys.executable

    with open(log_path, 'a') as log_file:
        log_file.write("\n" + "=" * 64 + "\n")
        mode = "DRY-RUN" if args.dry_run else "REAL RUN"
        log(f"OVERNIGHT CHAIN  [{mode}]  started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        log(f"Log file: {log_path}", log_file)

        if not args.skip_preflight:
            ok = preflight(args.dry_run, log_file)
            if not ok:
                log("\nPre-flight FAILED. Fix above issues before running.", log_file)
                sys.exit(1)
            log("\nAll pre-flight checks passed.\n", log_file)

        steps = build_steps(python)

        for i, step in enumerate(steps, 1):
            log(f"\n[{i}/{len(steps)}] {step['name']}", log_file)
            success = run_step(step['name'], step['cmd'], args.dry_run, log_file)
            if not success:
                log(f"\nChain ABORTED at step {i}: {step['name']}", log_file)
                sys.exit(1)

            if not args.dry_run and step['produces'] and not step['produces'].exists():
                log(f"WARNING: expected output not found: {step['produces']}", log_file)

        log(f"\n{'='*64}", log_file)
        if args.dry_run:
            log("DRY-RUN complete. All steps would execute in the above order.", log_file)
            log("Re-run without --dry-run to start training.", log_file)
        else:
            log("OVERNIGHT CHAIN COMPLETE.", log_file)
            log(f"Results in: {PROJECT_ROOT / 'comparisons'}/", log_file)
            log(f"Checkpoints: {CHECKPOINTS}/temporal_phaseF/  and  temporal_phaseG/", log_file)
        log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)


if __name__ == '__main__':
    main()
