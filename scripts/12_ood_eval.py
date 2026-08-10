#!/usr/bin/env python3
"""
Out-of-Distribution (OOD) Evaluation
======================================
Tests Phase G on signal types outside its training distribution
(which was clean English speech at 16 kHz).

OOD categories:
  1. Clean speech (test-clean)           — in-distribution baseline
  2. Speech + heavy noise (0 dB SNR)     — severe noise degradation
  3. Speech + simulated reverb           — channel distortion
  4. Pure 440 Hz sine tone              — single-frequency non-speech
  5. Frequency sweep (200-4000 Hz)       — broadband synthetic
  6. Pink noise                          — 1/f noise, spectrally rich
  7. White noise                         — flat-spectrum random signal

PESQ/STOI are computed relative to the original signal (not meaningful
for synthetic non-speech inputs but included for completeness).
Bitrate is always meaningful.

Output: comparisons/YYYY-MM-DD_ood_eval/
  report.txt   — per-category summary
  metrics.csv  — per-file numbers
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import get_dataset_paths
from src.codec_utils import load_model, find_checkpoint, encode_decode, compute_metrics

SR       = 16000
DUR_SEC  = 5
N_SPKRS  = 5


# ── Synthetic signal generators ───────────────────────────────────────────────

def gen_sine(freq=440.0, duration=DUR_SEC, sr=SR) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def gen_sweep(f0=200.0, f1=4000.0, duration=DUR_SEC, sr=SR) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)
    return (0.5 * np.sin(phase)).astype(np.float32)


def gen_pink_noise(duration=DUR_SEC, sr=SR, rng=None) -> np.ndarray:
    rng   = rng or np.random.default_rng(42)
    n     = int(duration * sr)
    white = rng.standard_normal(n)
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    fft  /= np.sqrt(freqs)
    pink  = np.fft.irfft(fft, n=n).astype(np.float32)
    pink /= np.abs(pink).max() + 1e-8
    return (pink * 0.5).astype(np.float32)


def gen_white_noise(duration=DUR_SEC, sr=SR, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    w   = rng.standard_normal(int(duration * sr)).astype(np.float32)
    return (w / (np.abs(w).max() + 1e-8) * 0.5).astype(np.float32)


def add_noise(speech: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    rng   = rng or np.random.default_rng(42)
    noise = rng.standard_normal(len(speech)).astype(np.float32)
    sig_p = np.mean(speech ** 2) + 1e-12
    ns_p  = np.mean(noise  ** 2) + 1e-12
    target_ns_p = sig_p / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_ns_p / ns_p)
    return np.clip(speech + noise, -1.0, 1.0)


def add_reverb(speech: np.ndarray, sr: int = SR) -> np.ndarray:
    """Simple exponential decay reverb impulse response."""
    from scipy import signal as scipy_signal
    t_ir  = np.arange(int(0.3 * sr)) / sr
    ir    = np.exp(-5.0 * t_ir).astype(np.float32)
    ir   /= ir.sum() + 1e-8
    out   = scipy_signal.fftconvolve(speech, ir)[:len(speech)].astype(np.float32)
    return np.clip(out, -1.0, 1.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = find_checkpoint(PROJECT_ROOT)

    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir   = PROJECT_ROOT / 'comparisons' / f'{timestamp}_ood_eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print("OOD EVALUATION — Phase G on signal types outside training distribution")
    print(f"{'='*68}")
    print(f"checkpoint: {ckpt_path.parent.name}")
    print(f"device    : {device}\n")

    model, _ = load_model(ckpt_path, device)
    model.eval()

    # Fixed canonical 5-speaker set (matches eval_phaseAB.py / eval_paper_numbers.py).
    # Do NOT use "first N found by directory traversal" -- adding/removing speaker
    # dirs from the dataset silently changes which N get picked (path string sort,
    # not speaker ID), breaking comparability across runs.
    TARGET_SPEAKERS = {'1089', '1188', '1221', '1284', '1320'}
    paths = get_dataset_paths()
    speakers = {}
    for f in sorted(paths['test_clean'].rglob('*.flac')):
        spk = f.parts[-3]
        if spk in TARGET_SPEAKERS and spk not in speakers:
            speakers[spk] = f
        if len(speakers) == min(N_SPKRS, len(TARGET_SPEAKERS)):
            break
    speech_files = list(speakers.values())

    rng = np.random.default_rng(42)

    # Load speech clips
    speech_clips = []
    for fpath in speech_files:
        audio, fsr = sf.read(fpath)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if fsr != SR:
            n = int(len(audio) * SR / fsr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio = np.clip(audio[:DUR_SEC * SR], -1.0, 1.0).astype(np.float32)
        speech_clips.append(audio)

    def run_category(name, signals, label=None):
        rows = []
        for i, sig in enumerate(signals):
            try:
                recon, kbps = encode_decode(model, sig, SR, device)
                pesq, stoi = compute_metrics(sig, recon, SR)
            except Exception as e:
                print(f"  ERROR on {name}[{i}]: {e}")
                kbps, pesq, stoi = float('nan'), None, None
            lbl = label[i] if label else str(i)
            rows.append({'category': name, 'id': lbl, 'kbps': kbps,
                         'pesq': pesq, 'stoi': stoi})
            p_str = f"{pesq:.3f}" if pesq is not None else "n/a"
            s_str = f"{stoi:.3f}" if stoi is not None else "n/a"
            print(f"  [{i+1}] {lbl:<30} kbps={kbps:.2f}  pesq={p_str}  stoi={s_str}")
        return rows

    spk_ids = [p.parts[-3] for p in speech_files]
    all_rows = []

    print("1. Clean speech (in-distribution baseline)")
    all_rows += run_category('clean_speech', speech_clips, spk_ids)

    print("\n2. Speech + heavy noise (0 dB SNR)")
    noisy = [add_noise(s, snr_db=0.0, rng=rng) for s in speech_clips]
    all_rows += run_category('speech_0dB_snr', noisy, spk_ids)

    print("\n3. Speech + simulated reverb")
    try:
        reverb = [add_reverb(s) for s in speech_clips]
        all_rows += run_category('speech_reverb', reverb, spk_ids)
    except ImportError:
        print("  scipy not available — skipping reverb")

    print("\n4. Pure 440 Hz sine tone")
    all_rows += run_category('sine_440hz', [gen_sine(440)], ['440hz'])

    print("\n5. Frequency sweep (200-4000 Hz)")
    all_rows += run_category('freq_sweep', [gen_sweep()], ['200-4000hz'])

    print("\n6. Pink noise")
    all_rows += run_category('pink_noise', [gen_pink_noise(rng=rng)], ['pink'])

    print("\n7. White noise")
    all_rows += run_category('white_noise', [gen_white_noise(rng=rng)], ['white'])

    # Aggregate per category
    def cat_avg(cat, key):
        vals = [r[key] for r in all_rows if r['category'] == cat and r[key] is not None]
        return float(np.mean(vals)) if vals else float('nan')

    categories = [
        ('clean_speech',   'Clean speech (in-distribution)'),
        ('speech_0dB_snr', 'Speech + noise @ 0 dB SNR'),
        ('speech_reverb',  'Speech + simulated reverb'),
        ('sine_440hz',     'Pure sine 440 Hz'),
        ('freq_sweep',     'Frequency sweep 200-4000 Hz'),
        ('pink_noise',     'Pink noise (1/f)'),
        ('white_noise',    'White noise'),
    ]

    SEP = '=' * 68
    sep = '-' * 68
    lines = [
        '', SEP,
        'OOD EVALUATION — Phase G (non-causal codec vs signal type)',
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Checkpoint: {ckpt_path.parent.name}",
        SEP, '',
        f"{'Category':<38} {'kbps':>7} {'PESQ-WB':>9} {'STOI':>7}",
        sep,
    ]
    for cat, label in categories:
        k = cat_avg(cat, 'kbps')
        p = cat_avg(cat, 'pesq')
        s = cat_avg(cat, 'stoi')
        p_str = f"{p:.3f}" if not np.isnan(p) else "n/a"
        s_str = f"{s:.3f}" if not np.isnan(s) else "n/a"
        lines.append(f"  {label:<36}  {k:>6.2f}k {p_str:>9} {s_str:>7}")
    lines += [
        sep, '',
        'NOTE: PESQ/STOI were designed for speech and may not be meaningful',
        'for synthetic non-speech signals (categories 4-7). Bitrate is always',
        'interpretable: signals with lower entropy are compressed more heavily.',
        SEP,
    ]
    report = '\n'.join(lines)
    print('\n' + report)

    with open(out_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    with open(out_dir / 'metrics.csv', 'w', encoding='utf-8') as f:
        f.write('category,id,kbps,pesq_wb,stoi\n')
        for r in all_rows:
            f.write(f"{r['category']},{r['id']},{r['kbps']:.4f},"
                    f"{r['pesq'] or ''},{r['stoi'] or ''}\n")

    print(f"\nreport:      {out_dir}/report.txt")
    print(f"metrics csv: {out_dir}/metrics.csv")


if __name__ == '__main__':
    main()
