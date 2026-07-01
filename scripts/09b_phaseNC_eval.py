#!/usr/bin/env python3
"""
Eval: Non-Causal (Phase NC) vs Causal (Phase G)
=================================================
Quantifies the quality ceiling that bidirectional context enables
over the streaming-constrained causal model.

Output: comparisons/YYYY-MM-DD_phaseNC_vs_phaseG/
  report.txt   — summary + per-speaker table
  metrics.csv  — raw per-file numbers
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
from src.model_noncausal import NonCausalNeuralAudioCodec

CLIP_SEC = 5
SR       = 16000
N_SPEAKERS = 5


def load_nc_model(ckpt_path: Path, device):
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    model = NonCausalNeuralAudioCodec(
        d_model        = ckpt.get('d_model', 384),
        n_layers       = ckpt.get('n_layers', 6),
        n_heads        = ckpt.get('n_heads', 8),
        window_size    = ckpt.get('window_size', 200),
        dropout        = 0.0,
        bottleneck_dim = ckpt.get('bottleneck_dim', 32),
        temporal_stride= ckpt.get('temporal_stride', 20),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_g  = PROJECT_ROOT / 'checkpoints_active/temporal_phaseG/best.pt'
    ckpt_nc = PROJECT_ROOT / 'checkpoints_active/temporal_phaseNC/best.pt'

    if not ckpt_g.exists():
        raise FileNotFoundError(f"Phase G checkpoint not found: {ckpt_g}")
    if not ckpt_nc.exists():
        raise FileNotFoundError(f"Phase NC checkpoint not found: {ckpt_nc}\n"
                                "Run scripts/09a_phaseNC_train.py first.")

    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir   = PROJECT_ROOT / 'comparisons' / f'{timestamp}_phaseNC_vs_phaseG'
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = get_dataset_paths()
    speakers = {}
    for f in sorted(paths['test_clean'].rglob('*.flac')):
        spk = f.parts[-3]
        if spk not in speakers:
            speakers[spk] = f
        if len(speakers) == N_SPEAKERS:
            break
    test_files = list(speakers.values())

    print(f"\n{'='*68}")
    print("EVAL: Phase NC (non-causal) vs Phase G (causal)")
    print(f"{'='*68}")
    print(f"device: {device}  |  {N_SPEAKERS} speakers × {CLIP_SEC}s\n")

    model_g,  _ = load_model(ckpt_g,  device)
    model_nc, _ = load_nc_model(ckpt_nc, device)
    print()

    results = []
    for idx, audio_path in enumerate(test_files, 1):
        spk = audio_path.parts[-3]
        audio, file_sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            n     = int(len(audio) * SR / file_sr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio = np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32)

        recon_g,  kbps_g  = encode_decode(model_g,  audio, SR, device)
        recon_nc, kbps_nc = encode_decode(model_nc, audio, SR, device)
        pesq_g,  stoi_g  = compute_metrics(audio, recon_g,  SR)
        pesq_nc, stoi_nc = compute_metrics(audio, recon_nc, SR)

        results.append({
            'speaker': spk,
            'kbps_g': kbps_g,  'pesq_g': pesq_g,  'stoi_g': stoi_g,
            'kbps_nc': kbps_nc, 'pesq_nc': pesq_nc, 'stoi_nc': stoi_nc,
        })
        print(f"[{idx}/{N_SPEAKERS}] {spk}")
        p_g  = f"{pesq_g:.3f}"  if pesq_g  is not None else "n/a"
        p_nc = f"{pesq_nc:.3f}" if pesq_nc is not None else "n/a"
        s_g  = f"{stoi_g:.3f}"  if stoi_g  is not None else "n/a"
        s_nc = f"{stoi_nc:.3f}" if stoi_nc is not None else "n/a"
        print(f"  Phase G  : {kbps_g:.2f} kbps  pesq={p_g}  stoi={s_g}")
        print(f"  Phase NC : {kbps_nc:.2f} kbps  pesq={p_nc}  stoi={s_nc}")

    # Summary
    SEP = '=' * 68
    sep = '-' * 68
    avg_pesq_g   = avg(results, 'pesq_g')
    avg_stoi_g   = avg(results, 'stoi_g')
    avg_kbps_g   = avg(results, 'kbps_g')
    avg_pesq_nc  = avg(results, 'pesq_nc')
    avg_stoi_nc  = avg(results, 'stoi_nc')
    avg_kbps_nc  = avg(results, 'kbps_nc')

    d_pesq = avg_pesq_nc - avg_pesq_g if not (np.isnan(avg_pesq_nc) or np.isnan(avg_pesq_g)) else float('nan')
    d_stoi = avg_stoi_nc - avg_stoi_g if not (np.isnan(avg_stoi_nc) or np.isnan(avg_stoi_g)) else float('nan')

    lines = [
        '', SEP,
        'EVAL: Phase NC (non-causal) vs Phase G (causal)',
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Clips: {N_SPEAKERS} speakers x {CLIP_SEC}s  |  16 kHz mono",
        SEP, '',
        f"{'Model':<20} {'kbps':>8} {'PESQ-WB':>10} {'STOI':>8}",
        sep,
        f"  Phase G  (causal)  {avg_kbps_g:>7.2f}k {avg_pesq_g:>10.3f} {avg_stoi_g:>8.3f}",
        f"  Phase NC (non-caus){avg_kbps_nc:>7.2f}k {avg_pesq_nc:>10.3f} {avg_stoi_nc:>8.3f}",
        sep,
        f"  Delta NC - G                   pesq={d_pesq:+.3f}  stoi={d_stoi:+.3f}",
        '',
        'Interpretation: delta measures the quality cost of the real-time causality',
        'constraint. Positive delta = non-causal model benefits from future context.',
        sep, '',
        'PER-SPEAKER', sep,
    ]
    for r in results:
        dp = (r['pesq_nc'] - r['pesq_g']) if r['pesq_nc'] is not None and r['pesq_g'] is not None else float('nan')
        ds = (r['stoi_nc'] - r['stoi_g']) if r['stoi_nc'] is not None and r['stoi_g'] is not None else float('nan')
        lines += [
            f"\n  speaker {r['speaker']}",
            f"  {'Model':<20} {'kbps':>8} {'PESQ-WB':>10} {'STOI':>8}",
            f"  {'-'*48}",
            f"  Phase G  (causal)   {r['kbps_g']:>7.2f}k {r['pesq_g'] or float('nan'):>10.3f} {r['stoi_g'] or float('nan'):>8.3f}",
            f"  Phase NC (non-caus) {r['kbps_nc']:>7.2f}k {r['pesq_nc'] or float('nan'):>10.3f} {r['stoi_nc'] or float('nan'):>8.3f}",
            f"  Delta                              pesq={dp:+.3f}  stoi={ds:+.3f}",
        ]
    lines.append(SEP)
    report = '\n'.join(lines)
    print(report)

    with open(out_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    with open(out_dir / 'metrics.csv', 'w', encoding='utf-8') as f:
        f.write('speaker,model,kbps,pesq_wb,stoi\n')
        for r in results:
            f.write(f"{r['speaker']},phaseG,{r['kbps_g']:.4f},{r['pesq_g'] or ''},"
                    f"{r['stoi_g'] or ''}\n")
            f.write(f"{r['speaker']},phaseNC,{r['kbps_nc']:.4f},{r['pesq_nc'] or ''},"
                    f"{r['stoi_nc'] or ''}\n")

    print(f"\nreport:      {out_dir}/report.txt")
    print(f"metrics csv: {out_dir}/metrics.csv")


if __name__ == '__main__':
    main()
