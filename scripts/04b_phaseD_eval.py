#!/usr/bin/env python3
"""
Phase D Evaluation: Phase C vs Phase D
=======================================
Runs both checkpoints on the same 5-speaker LibriSpeech test-clean clips.
Saves audio samples, metrics CSV, and a text report to a timestamped folder.

Output: comparisons/YYYY-MM-DD_phaseD_vs_phaseC/
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

    ckpt_c = PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt'
    ckpt_d = PROJECT_ROOT / 'checkpoints_active/temporal_phaseD/best.pt'

    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir = PROJECT_ROOT / 'comparisons' / f'{timestamp}_phaseD_vs_phaseC'
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
    print("EVAL: Phase C  vs  Phase D (Uniform Noise Proxy)")
    print(f"{'='*68}")
    print(f"Device : {device}  |  Clips: 5 speakers × {CLIP_SEC}s  |  16kHz mono")
    print(f"Output : {out_dir}\n")

    print("Loading Phase C...", end=' ', flush=True)
    model_c, meta_c = load_model(ckpt_c, device)
    print(f"OK  (phase={meta_c.get('phase')}, loss={meta_c.get('train_loss', 0):.4f})")

    print("Loading Phase D...", end=' ', flush=True)
    model_d, meta_d = load_model(ckpt_d, device)
    print(f"OK  (phase={meta_d.get('phase')}, loss={meta_d.get('train_loss', 0):.4f})\n")

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

        recon_c, kbps_c = encode_decode(model_c, audio, SR, device)
        pesq_c, stoi_c = compute_metrics(audio, recon_c, SR)
        sf.write(sample_dir / f'phaseC_{kbps_c:.1f}kbps.wav', recon_c, SR)
        print(f"  Phase C → {kbps_c:.1f} kbps  PESQ={pesq_c:.3f}  STOI={stoi_c:.3f}")

        recon_d, kbps_d = encode_decode(model_d, audio, SR, device)
        pesq_d, stoi_d = compute_metrics(audio, recon_d, SR)
        sf.write(sample_dir / f'phaseD_{kbps_d:.1f}kbps.wav', recon_d, SR)
        print(f"  Phase D → {kbps_d:.1f} kbps  PESQ={pesq_d:.3f}  STOI={stoi_d:.3f}")

        all_results.append({
            'speaker': spk_id, 'file': audio_path.name,
            'phaseC_kbps': kbps_c, 'phaseC_pesq': pesq_c, 'phaseC_stoi': stoi_c,
            'phaseD_kbps': kbps_d, 'phaseD_pesq': pesq_d, 'phaseD_stoi': stoi_d,
        })

    SEP = '=' * 68
    sep = '-' * 68

    summary = (
        f"\n{SEP}\n"
        f"EVAL: Phase C vs Phase D\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"{SEP}\n\n"
        f"{'':22} {'Bitrate':>10} {'PESQ (WB)':>12} {'STOI':>8}\n"
        f"{sep}\n"
        f"{'Phase C (STE)':22} {avg(all_results,'phaseC_kbps'):>9.1f}k {avg(all_results,'phaseC_pesq'):>12.3f} {avg(all_results,'phaseC_stoi'):>8.3f}\n"
        f"{'Phase D (Noise Proxy)':22} {avg(all_results,'phaseD_kbps'):>9.1f}k {avg(all_results,'phaseD_pesq'):>12.3f} {avg(all_results,'phaseD_stoi'):>8.3f}\n"
        f"{sep}\n"
    )

    delta_pesq = avg(all_results, 'phaseD_pesq') - avg(all_results, 'phaseC_pesq')
    delta_stoi = avg(all_results, 'phaseD_stoi') - avg(all_results, 'phaseC_stoi')
    sign_p = '+' if delta_pesq >= 0 else ''
    sign_s = '+' if delta_stoi >= 0 else ''
    summary += f"{'Delta':22} {'':>10}  {sign_p}{delta_pesq:>11.3f}  {sign_s}{delta_stoi:>7.3f}\n"
    summary += f"{SEP}\n\n"
    summary += "PER-SPEAKER\n" + sep + "\n"
    summary += f"{'Speaker':<10} {'C kbps':>8} {'C PESQ':>8} {'C STOI':>8} {'D kbps':>8} {'D PESQ':>8} {'D STOI':>8}\n"
    summary += sep + "\n"
    for r in all_results:
        summary += (
            f"  {r['speaker']:<8} {r['phaseC_kbps']:>8.1f} {r['phaseC_pesq']:>8.3f} {r['phaseC_stoi']:>8.3f}"
            f" {r['phaseD_kbps']:>8.1f} {r['phaseD_pesq']:>8.3f} {r['phaseD_stoi']:>8.3f}\n"
        )

    print(summary)

    with open(out_dir / 'report.txt', 'w') as f:
        f.write(summary)

    with open(out_dir / 'metrics.csv', 'w') as f:
        f.write('speaker,file,phaseC_kbps,phaseC_pesq,phaseC_stoi,phaseD_kbps,phaseD_pesq,phaseD_stoi\n')
        for r in all_results:
            f.write(
                f"{r['speaker']},{r['file']},"
                f"{r['phaseC_kbps']:.2f},{r['phaseC_pesq']:.4f},{r['phaseC_stoi']:.4f},"
                f"{r['phaseD_kbps']:.2f},{r['phaseD_pesq']:.4f},{r['phaseD_stoi']:.4f}\n"
            )

    print(f"Audio  : {out_dir}/")
    print(f"Report : {out_dir}/report.txt")
    print(f"CSV    : {out_dir}/metrics.csv")


if __name__ == '__main__':
    main()
