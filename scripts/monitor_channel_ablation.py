#!/usr/bin/env python3
"""
Channel ablation training monitor.

Tracks the 6-phase sequential training chain (A→B→C for width 16, then 64).
Run with:
  watch -n 3 "python scripts/monitor_channel_ablation.py"
"""

import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (phase_label, width, log_path, total_epochs)
CHAIN = [
    ("A", "16", PROJECT_ROOT / "logs/phaseA_16.log", 30),
    ("B", "16", PROJECT_ROOT / "logs/phaseB_16.log", 20),
    ("C", "16", PROJECT_ROOT / "logs/phaseC_16.log", 30),
    ("D", "16", PROJECT_ROOT / "logs/phaseD_16.log", 20),
    ("E", "16", PROJECT_ROOT / "logs/phaseE_16.log", 30),
    ("F", "16", PROJECT_ROOT / "logs/phaseF_16.log", 30),
    ("G", "16", PROJECT_ROOT / "logs/phaseG_16.log", 20),
    ("A", "64", PROJECT_ROOT / "logs/phaseA_64.log", 30),
    ("B", "64", PROJECT_ROOT / "logs/phaseB_64.log", 20),
    ("C", "64", PROJECT_ROOT / "logs/phaseC_64.log", 30),
    ("D", "64", PROJECT_ROOT / "logs/phaseD_64.log", 20),
    ("E", "64", PROJECT_ROOT / "logs/phaseE_64.log", 30),
    ("F", "64", PROJECT_ROOT / "logs/phaseF_64.log", 30),
    ("G", "64", PROJECT_ROOT / "logs/phaseG_64.log", 20),
]

RE_EPOCH = re.compile(
    r'Epoch\s+(\d+)/(\d+):\s+loss=([0-9.]+)'
    r'(?:\s+real_bitrate=([0-9.]+)\s+kbps)?'
    r'.*?lr=([0-9.e+\-]+)'
)
RE_DONE  = re.compile(r'Phase [A-Z] done\. Best loss:\s+([0-9.]+)')
RE_BATCH = re.compile(r'Epoch \d+/\d+:\s+\d+%\|')
RE_EARLY = re.compile(r'Target loss .* reached at epoch (\d+)\. Stopping\.')


def read_lines(path: Path):
    if not path.exists():
        return []
    with open(path, 'rb') as f:
        raw = f.read()
    return raw.replace(b'\r', b'\n').decode('utf-8', errors='replace').splitlines()


def parse_log(log_path: Path, total_epochs: int) -> dict:
    lines = read_lines(log_path)

    epochs    = []
    done      = False
    done_loss = None
    batch_line = None

    for line in lines:
        m = RE_EPOCH.search(line)
        if m:
            epochs.append({
                'epoch':   int(m.group(1)),
                'total':   int(m.group(2)),
                'loss':    float(m.group(3)),
                'bitrate': float(m.group(4)) if m.group(4) else None,
                'lr':      m.group(5),
            })

        m = RE_DONE.search(line)
        if m:
            done      = True
            done_loss = float(m.group(1))

        if RE_EARLY.search(line):
            done = True

        if RE_BATCH.search(line):
            batch_line = line.strip()

    best_loss = min((e['loss'] for e in epochs), default=None)
    cur_epoch = epochs[-1]['epoch'] if epochs else 0
    last      = epochs[-1] if epochs else None

    return {
        'started':    bool(lines),
        'done':       done,
        'done_loss':  done_loss,
        'cur_epoch':  cur_epoch,
        'total':      total_epochs,
        'best_loss':  best_loss,
        'last':       last,
        'batch_line': batch_line,
    }


def bar(current, total, width=16) -> str:
    if total == 0:
        return '░' * width
    filled = round(width * current / total)
    return '█' * filled + '░' * (width - filled)


def fmt_loss(v) -> str:
    return f'{v:.5f}' if v is not None else '  —    '


def main():
    W = 60
    now = datetime.now().strftime('%a %d %b  %H:%M:%S')

    states = []
    for phase, width, log, epochs in CHAIN:
        s = parse_log(log, epochs)
        s.update({'phase': phase, 'width': width, 'log': log})
        states.append(s)

    done_count = sum(1 for s in states if s['done'])
    active     = next((s for s in states if s['started'] and not s['done']), None)

    # ── header ────────────────────────────────────────────────────────────────
    overall_bar = bar(done_count, 14, width=12)
    print('═' * W)
    print(f"  CHANNEL ABLATION  {now}")
    print(f"  Overall  [{overall_bar}]  {done_count} / 14 phases complete")
    print('═' * W)

    # ── per-width blocks ───────────────────────────────────────────────────────
    for width_label in ('16', '64'):
        print(f"\n  WIDTH {width_label}  (bottleneck_dim={width_label})")
        for s in states:
            if s['width'] != width_label:
                continue

            b    = bar(s['cur_epoch'], s['total'])
            ep   = f"{s['cur_epoch']:2d}/{s['total']}"

            if s['done']:
                icon   = '✓'
                detail = f"loss={fmt_loss(s['done_loss'])}  done"
            elif s['started']:
                icon = '►'
                lr   = s['last']['lr'] if s['last'] else '—'
                kbps = (f"  {s['last']['bitrate']:.1f} kbps"
                        if s['last'] and s['last']['bitrate'] else '')
                detail = f"loss={fmt_loss(s['best_loss'])}{kbps}  lr={lr}"
            else:
                icon   = '·'
                detail = '(not started)'

            print(f"    {icon} Phase {s['phase']}  [{b}] {ep}   {detail}")

    # ── live batch line ────────────────────────────────────────────────────────
    print(f"\n  {'─' * (W - 2)}")
    if active and active['batch_line']:
        tag   = f"[{active['phase']}-{active['width']}]"
        # Trim tqdm line to fit terminal width
        line  = active['batch_line']
        avail = W - len(tag) - 4
        if len(line) > avail:
            line = line[:avail]
        print(f"  {tag}  {line}")
    elif done_count == 14:
        print("  All phases complete.")
        print("  Run: python scripts/quick_entropy_eval.py  (update PHASES for 16/32/64)")
    else:
        print("  (waiting — training not yet started)")

    print('═' * W)


if __name__ == '__main__':
    main()
