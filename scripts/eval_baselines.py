#!/usr/bin/env python3
"""
Reference-codec baselines: AAC and EnCodec, measured locally.

Why this exists: the AAC and EnCodec numbers previously quoted in the README and
in scripts/13_rd_sweep.py were hardcoded constants with no stored evaluation run
behind them, and the surviving notes disagreed on whether they had been measured
locally or taken from published figures. This script produces them on the same
speaker set, clip length and metric code as eval_confidence_intervals.py, so the
reference rows and the EntroCodec rows are a matched comparison.

AAC encoder note: FFmpeg's native `aac` encoder (AAC-LC), via PyAV. libfdk_aac is
not present in any pip or conda binary distribution because its licence is
GPL-incompatible, so it is not available here without building FFmpeg from source.
The encoder actually used is recorded in the report header.

Output: comparisons/YYYY-MM-DD_baselines/
  report.txt   — summary with 95% bootstrap CIs
  metrics.csv  — per-speaker raw numbers
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import av

from src.paths import get_dataset_paths
from src.codec_utils import compute_metrics

# Reuse the AAC path already used for every previous comparison in this project
# rather than reimplementing it, so a changed number means a changed measurement
# and not a changed implementation. That module guards its own main().
aac_encode_decode = __import__('03b_phaseC_eval').aac_encode_decode

from encodec import EncodecModel

SR = 16000
ENCODEC_SR = 24000


def collect_speakers(test_clean_path: Path, n: int, clip_sec: int, sr: int):
    """Identical selection to eval_confidence_intervals.py: first `n` speakers by
    numeric ID, one utterance each. Deliberately not 'first n by directory
    traversal' — that silently changes which speakers are picked whenever the
    dataset changes, which is what issue #43 was."""
    by_speaker = {}
    for f in sorted(test_clean_path.rglob('*.flac')):
        spk = f.parts[-3]
        if spk not in by_speaker:
            by_speaker[spk] = f

    def spk_key(s):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)

    speakers = []
    for spk in sorted(by_speaker, key=spk_key)[:n]:
        audio, file_sr = sf.read(by_speaker[spk])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            n_samples = int(len(audio) * sr / file_sr)
            audio = np.interp(np.linspace(0, len(audio), n_samples),
                              np.arange(len(audio)), audio)
        audio = np.clip(audio[:clip_sec * sr], -1.0, 1.0).astype(np.float32)
        speakers.append((spk, audio))
    return speakers


def bootstrap_ci(values, n_boot=10_000, ci=95, rng=None):
    """Bootstrap CI on the mean. Matches eval_confidence_intervals.py."""
    if rng is None:
        rng = np.random.default_rng(42)
    vals = np.array([v for v in values if v is not None], dtype=float)
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boots, [alpha, 100.0 - alpha])
    return float(vals.mean()), float(lo), float(hi)


class EncodecRunner:
    """Loads each bandwidth's model once. The helper in 03b_phaseC_eval.py rebuilds
    the model on every call, which is fine for 5 clips and wasteful for 40."""

    def __init__(self, device):
        self.device = device
        self._cache = {}

    def _model(self, bandwidth_kbps):
        if bandwidth_kbps not in self._cache:
            m = EncodecModel.encodec_model_24khz()
            m.set_target_bandwidth(bandwidth_kbps)
            self._cache[bandwidth_kbps] = m.to(self.device).eval()
        return self._cache[bandwidth_kbps]

    def __call__(self, audio_np, bandwidth_kbps):
        model = self._model(bandwidth_kbps)
        audio_t = torch.FloatTensor(audio_np).unsqueeze(0)
        audio_24k = torchaudio.functional.resample(audio_t, SR, ENCODEC_SR)
        audio_24k = audio_24k.unsqueeze(0).to(self.device)

        with torch.no_grad():
            frames = model.encode(audio_24k)
            decoded_24k = model.decode(frames)

        total_codes = sum(codes.numel() for codes, _ in frames)
        bits_per_code = float(np.log2(model.quantizer.bins))
        duration = len(audio_np) / SR
        actual_kbps = (total_codes * bits_per_code) / duration / 1000

        decoded = torchaudio.functional.resample(
            decoded_24k.squeeze(0).cpu(), ENCODEC_SR, SR).squeeze(0).numpy()
        if len(decoded) >= len(audio_np):
            decoded = decoded[:len(audio_np)]
        else:
            decoded = np.pad(decoded, (0, len(audio_np) - len(decoded)))
        return decoded.astype(np.float32), actual_kbps


def aac_encoder_description() -> str:
    codec = av.codec.Codec('aac', 'w')
    libavcodec = '.'.join(str(x) for x in av.library_versions.get('libavcodec', ()))
    return (f"FFmpeg native '{codec.name}' ({codec.long_name}), "
            f"libavcodec {libavcodec}, PyAV {av.__version__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-speakers', type=int, default=40)
    ap.add_argument('--clip-sec', type=int, default=5)
    ap.add_argument('--aac-kbps', type=int, default=10,
                    help='requested AAC bitrate; AAC-LC floors well above this at '
                         '16 kHz mono, and the achieved rate is what gets reported')
    ap.add_argument('--encodec-bw', type=float, nargs='+', default=[1.5, 3.0, 6.0])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / 'comparisons' / f"{datetime.now():%Y-%m-%d}_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = get_dataset_paths()
    speakers = collect_speakers(paths['test_clean'], args.n_speakers,
                                args.clip_sec, SR)

    print(f"\n{'='*72}\nREFERENCE CODEC BASELINES\n{'='*72}")
    print(f"speakers : {len(speakers)}  |  clip: {args.clip_sec}s  |  device: {args.device}")
    print(f"AAC      : {aac_encoder_description()}\n")

    encodec = EncodecRunner(args.device)
    systems = [f'AAC@{args.aac_kbps}kbps'] + [f'EnCodec@{bw}kbps' for bw in args.encodec_bw]
    rows = []

    for i, (spk, audio) in enumerate(speakers, 1):
        row = {'speaker': spk}

        dec, kbps = aac_encode_decode(audio, SR, target_kbps=args.aac_kbps)
        row[f'AAC@{args.aac_kbps}kbps'] = (kbps, *compute_metrics(audio, dec, SR))

        for bw in args.encodec_bw:
            dec, kbps = encodec(audio, bw)
            row[f'EnCodec@{bw}kbps'] = (kbps, *compute_metrics(audio, dec, SR))

        rows.append(row)
        shown = '  '.join(
            f"{name}={row[name][1]:.3f}" if row[name][1] is not None else f"{name}=n/a"
            for name in systems)
        print(f"[{i:>2}/{len(speakers)}] spk {spk:<6} {shown}")

    SEP, sep = '=' * 72, '-' * 72
    lines = [
        '', SEP, 'REFERENCE CODEC BASELINES — AAC and EnCodec',
        f"Generated : {datetime.now():%Y-%m-%d %H:%M}",
        f"Speakers  : {len(speakers)}  (first {args.n_speakers} numeric IDs from test-clean)",
        f"Clip      : {args.clip_sec}s  |  SR: {SR} Hz mono  |  device: {args.device}",
        "Bootstrap : 10,000 iterations  |  CI: 95%  |  seed: 42",
        f"AAC       : {aac_encoder_description()}",
        "EnCodec   : encodec_model_24khz, 16k->24k resample in, 24k->16k out",
        SEP, '',
        f"{'System':<20} {'kbps':>7}  {'PESQ-WB':>9} {'95% CI':>18}"
        f"  {'STOI':>7} {'95% CI':>18}",
        sep,
    ]
    for name in systems:
        kb = [r[name][0] for r in rows]
        pm, plo, phi = bootstrap_ci([r[name][1] for r in rows])
        sm, slo, shi = bootstrap_ci([r[name][2] for r in rows])
        lines.append(
            f"  {name:<18} {np.mean(kb):>6.2f}k  {pm:>9.3f} "
            f"{f'[{plo:.3f}, {phi:.3f}]':>18}  {sm:>7.3f} "
            f"{f'[{slo:.3f}, {shi:.3f}]':>18}")
    lines += [
        sep, '',
        'NOTE: AAC-LC at 16 kHz mono floors well above the requested bitrate; cite the',
        '      achieved rate above, not the request. libfdk_aac (which supports HE-AAC',
        '      and would reach lower rates) is not distributable in pip/conda builds.',
        SEP, '',
    ]

    (out_dir / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')
    with open(out_dir / 'metrics.csv', 'w', encoding='utf-8') as f:
        f.write('system,speaker,kbps,pesq_wb,stoi\n')
        for r in rows:
            for name in systems:
                kb, p, s = r[name]
                f.write(f"{name},{r['speaker']},{kb},"
                        f"{'' if p is None else p},{'' if s is None else s}\n")

    print('\n'.join(lines))
    print(f"report:      {out_dir / 'report.txt'}")
    print(f"metrics csv: {out_dir / 'metrics.csv'}")


if __name__ == '__main__':
    main()
