#!/usr/bin/env python3
"""
Phase F Evaluation: Phase C vs Phase F (Triple Combined Spectral Loss)
=======================================================================
Saves audio samples, metrics CSV, and report to a timestamped folder.

Output: comparisons/YYYY-MM-DD_phaseF_vs_phaseC/
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
from src.codec_utils import load_model, encode_decode, compute_metrics, avg


def main():
    SR = 16000
    CLIP_SEC = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoints = {
        'phaseC': PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt',
        'phaseF': PROJECT_ROOT / 'checkpoints_active/temporal_phaseF/best.pt',
    }

    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir = PROJECT_ROOT / 'comparisons' / f'{timestamp}_phaseF_vs_phaseC'
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = get_dataset_paths()
    speakers = {}
    for f in sorted(paths['test_clean'].rglob('*.flac')):
        spk = f.parts[-3]
        if spk not in speakers:
            speakers[spk] = f
        if len(speakers) == 5:
            break
    test_files = list(speakers.values())

    print(f"\n{'='*68}")
    print("EVAL: Phase C (Linear STFT)  vs  Phase F (Triple Combined Loss)")
    print(f"{'='*68}")
    print(f"Device : {device}  |  Clips: 5 speakers × {CLIP_SEC}s  |  16kHz mono")
    print(f"Output : {out_dir}\n")

    models = {}
    for name, path in checkpoints.items():
        print(f"Loading {name}...", end=' ', flush=True)
        m, meta = load_model(path, device)
        models[name] = m
        print(f"OK  (phase={meta.get('phase')}, loss={meta.get('train_loss', 0):.5f})")
    print()

    all_results = []

    for idx, audio_path in enumerate(test_files, 1):
        spk_id = audio_path.parts[-3]
        sample_dir = out_dir / f'sample_{idx:02d}_spk{spk_id}'
        sample_dir.mkdir(exist_ok=True)

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            n = int(len(audio) * SR / sr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio = np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32)
        sf.write(sample_dir / 'source.wav', audio, SR)

        print(f"[{idx}/5] speaker {spk_id}")
        row = {'speaker': spk_id, 'file': audio_path.name}

        for name, model in models.items():
            recon, kbps = encode_decode(model, audio, SR, device)
            pesq, stoi = compute_metrics(audio, recon, SR)
            sf.write(sample_dir / f'{name}_{kbps:.1f}kbps.wav', recon, SR)
            row[f'{name}_kbps'] = kbps
            row[f'{name}_pesq'] = pesq
            row[f'{name}_stoi'] = stoi
            label = {'phaseC': 'Phase C (Linear STFT)   ', 'phaseF': 'Phase F (Triple Loss)   '}[name]
            print(f"  {label} → {kbps:.1f} kbps  PESQ={pesq:.3f}  STOI={stoi:.3f}")

        all_results.append(row)

    SEP = '=' * 68
    sep = '-' * 68

    summary = (
        f"\n{SEP}\n"
        f"EVAL: Phase C vs Phase F\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"{SEP}\n\n"
        f"{'':28} {'Bitrate':>10} {'PESQ (WB)':>12} {'STOI':>8}\n"
        f"{sep}\n"
        f"{'Phase C  (Linear STFT)':28} {avg(all_results,'phaseC_kbps'):>9.1f}k"
        f" {avg(all_results,'phaseC_pesq'):>12.3f} {avg(all_results,'phaseC_stoi'):>8.3f}\n"
        f"{'Phase F  (Triple Loss)':28} {avg(all_results,'phaseF_kbps'):>9.1f}k"
        f" {avg(all_results,'phaseF_pesq'):>12.3f} {avg(all_results,'phaseF_stoi'):>8.3f}\n"
        f"{sep}\n"
    )

    dp = avg(all_results, 'phaseF_pesq') - avg(all_results, 'phaseC_pesq')
    ds = avg(all_results, 'phaseF_stoi') - avg(all_results, 'phaseC_stoi')
    dk = avg(all_results, 'phaseF_kbps') - avg(all_results, 'phaseC_kbps')
    summary += f"  Δ F vs C   pesq={dp:+.3f}  stoi={ds:+.3f}  bitrate={dk:+.1f} kbps\n"
    summary += f"\n{SEP}\n\nPER-SPEAKER\n{sep}\n"
    summary += (
        f"{'Speaker':<10}"
        f" {'C kbps':>8} {'C PESQ':>8} {'C STOI':>8}"
        f" {'F kbps':>8} {'F PESQ':>8} {'F STOI':>8}\n"
    )
    summary += sep + "\n"
    for r in all_results:
        summary += (
            f"  {r['speaker']:<8}"
            f" {r['phaseC_kbps']:>8.1f} {r['phaseC_pesq']:>8.3f} {r['phaseC_stoi']:>8.3f}"
            f" {r['phaseF_kbps']:>8.1f} {r['phaseF_pesq']:>8.3f} {r['phaseF_stoi']:>8.3f}\n"
        )

    print(summary)

    with open(out_dir / 'report.txt', 'w') as f:
        f.write(summary)
    with open(out_dir / 'metrics.csv', 'w') as f:
        f.write('speaker,file,phaseC_kbps,phaseC_pesq,phaseC_stoi,phaseF_kbps,phaseF_pesq,phaseF_stoi\n')
        for r in all_results:
            f.write(
                f"{r['speaker']},{r['file']},"
                f"{r['phaseC_kbps']:.2f},{r['phaseC_pesq']:.4f},{r['phaseC_stoi']:.4f},"
                f"{r['phaseF_kbps']:.2f},{r['phaseF_pesq']:.4f},{r['phaseF_stoi']:.4f}\n"
            )

    print(f"Audio  : {out_dir}/")
    print(f"Report : {out_dir}/report.txt")
    print(f"CSV    : {out_dir}/metrics.csv")


if __name__ == '__main__':
    main()
