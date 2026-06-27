#!/usr/bin/env python3
"""
Full Evaluation: Phase C vs Phase D vs Phase D-VAE
===================================================
Runs all three checkpoints on the same 5-speaker LibriSpeech test-clean clips.
Saves audio samples, metrics CSV, and text report to a timestamped folder.

Output: comparisons/YYYY-MM-DD_phaseDvae_vs_phaseC_phaseD/
"""

import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths

try:
    from pesq import pesq as pesq_fn
    PESQ_OK = True
except ImportError:
    PESQ_OK = False

try:
    from pystoi import stoi as stoi_fn
    STOI_OK = True
except ImportError:
    STOI_OK = False


# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    """Load codec from checkpoint — VAE weights ignored at inference."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    d_model = ckpt.get('d_model', 384)

    ids = set()
    for k in state:
        if 'encoder.transformer_blocks.' in k:
            p = k.split('.')
            if len(p) > 2 and p[2].isdigit():
                ids.add(int(p[2]))
    n_layers = max(ids) + 1 if ids else 6

    model = NeuralAudioCodec(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=ckpt.get('n_heads', 8),
        window_size=ckpt.get('window_size', 200),
        dropout=0.0,
        bottleneck_dim=ckpt.get('bottleneck_dim', 32),
        temporal_stride=ckpt.get('temporal_stride', 20),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def encode_decode(model, audio, sr, device, chunk_sec=5.0):
    num_levels = 8
    chunk_size = int(chunk_sec * sr)
    recon_chunks, total_bits = [], 0

    with torch.no_grad():
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            if len(chunk) < 160:
                continue
            x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            z = model.encode(x)
            z_np = z.squeeze(0).cpu().numpy()

            z_min, z_max = float(z_np.min()), float(z_np.max())
            scale = (z_max - z_min) / (num_levels - 1) + 1e-8
            q = np.clip(np.round((z_np - z_min) / scale), 0, num_levels - 1).astype(np.uint8)
            compressed = zlib.compress(q.tobytes(), level=9)
            total_bits += len(compressed) * 8

            q_dec = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(z_np.shape)
            z_rec = q_dec.astype(np.float32) * scale + z_min
            x_recon = model.decode(torch.from_numpy(z_rec).unsqueeze(0).to(device))
            recon_chunks.append(x_recon.squeeze().cpu().numpy())

    recon = np.concatenate(recon_chunks) if recon_chunks else np.zeros_like(audio)
    recon = recon[:len(audio)] if len(recon) >= len(audio) else np.pad(recon, (0, len(audio) - len(recon)))
    kbps = total_bits / (len(audio) / sr) / 1000
    return recon.astype(np.float32), kbps


def compute_metrics(ref, deg, sr):
    n = min(len(ref), len(deg))
    r = ref[:n] / (np.abs(ref[:n]).max() + 1e-8)
    d = deg[:n] / (np.abs(deg[:n]).max() + 1e-8)
    pesq = float(pesq_fn(sr, r, d, 'wb')) if PESQ_OK else None
    stoi = float(stoi_fn(r, d, sr, extended=False)) if STOI_OK else None
    return pesq, stoi


def avg(results, key):
    v = [r[key] for r in results if r[key] is not None]
    return float(np.mean(v)) if v else float('nan')


# ---------------------------------------------------------------------------

def main():
    SR = 16000
    CLIP_SEC = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoints = {
        'phaseC':    PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt',
        'phaseD':    PROJECT_ROOT / 'checkpoints_active/temporal_phaseD/best.pt',
        'phaseDvae': PROJECT_ROOT / 'checkpoints_active/temporal_phaseD_vae/best.pt',
    }

    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir = PROJECT_ROOT / 'comparisons' / f'{timestamp}_phaseDvae_vs_phaseD_vs_phaseC'
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

    print(f"\n{'='*72}")
    print("EVAL: Phase C  vs  Phase D (Noise Proxy)  vs  Phase D-VAE")
    print(f"{'='*72}")
    print(f"Device : {device}  |  Clips: 5 speakers × {CLIP_SEC}s  |  16kHz mono")
    print(f"Output : {out_dir}\n")

    models = {}
    for name, path in checkpoints.items():
        print(f"Loading {name}...", end=' ', flush=True)
        m, meta = load_model(path, device)
        models[name] = m
        print(f"OK  (phase={meta.get('phase')}, recon_loss={meta.get('train_loss', 0):.4f})")
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
            row[f'{name}_kbps']  = kbps
            row[f'{name}_pesq']  = pesq
            row[f'{name}_stoi']  = stoi
            label = {'phaseC': 'Phase C (STE)      ', 'phaseD': 'Phase D (Noise)    ', 'phaseDvae': 'Phase D-VAE        '}[name]
            print(f"  {label} → {kbps:.1f} kbps  PESQ={pesq:.3f}  STOI={stoi:.3f}")

        all_results.append(row)

    # Summary
    SEP = '=' * 72
    sep = '-' * 72

    summary = (
        f"\n{SEP}\n"
        f"EVAL: Phase C vs Phase D vs Phase D-VAE\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"{SEP}\n\n"
        f"{'':24} {'Bitrate':>10} {'PESQ (WB)':>12} {'STOI':>8}\n"
        f"{sep}\n"
    )

    rows = [
        ('Phase C  (STE)',        'phaseC'),
        ('Phase D  (Noise Proxy)', 'phaseD'),
        ('Phase D-VAE',            'phaseDvae'),
    ]
    for label, key in rows:
        summary += (
            f"{label:24} {avg(all_results, f'{key}_kbps'):>9.1f}k"
            f" {avg(all_results, f'{key}_pesq'):>12.3f}"
            f" {avg(all_results, f'{key}_stoi'):>8.3f}\n"
        )

    summary += sep + "\n"
    for label, key in [('D vs C', 'phaseD'), ('D-VAE vs C', 'phaseDvae')]:
        dp = avg(all_results, f'{key}_pesq') - avg(all_results, 'phaseC_pesq')
        ds = avg(all_results, f'{key}_stoi') - avg(all_results, 'phaseC_stoi')
        summary += f"  Δ {label:18} pesq={dp:+.3f}  stoi={ds:+.3f}\n"

    summary += f"\n{SEP}\n\nPER-SPEAKER\n{sep}\n"
    summary += (
        f"{'Speaker':<10}"
        f" {'C kbps':>7} {'C PESQ':>7} {'C STOI':>7}"
        f" {'D kbps':>7} {'D PESQ':>7} {'D STOI':>7}"
        f" {'Dvae kbps':>9} {'Dvae PESQ':>9} {'Dvae STOI':>9}\n"
    )
    summary += sep + "\n"
    for r in all_results:
        summary += (
            f"  {r['speaker']:<8}"
            f" {r['phaseC_kbps']:>7.1f} {r['phaseC_pesq']:>7.3f} {r['phaseC_stoi']:>7.3f}"
            f" {r['phaseD_kbps']:>7.1f} {r['phaseD_pesq']:>7.3f} {r['phaseD_stoi']:>7.3f}"
            f" {r['phaseDvae_kbps']:>9.1f} {r['phaseDvae_pesq']:>9.3f} {r['phaseDvae_stoi']:>9.3f}\n"
        )

    print(summary)

    with open(out_dir / 'report.txt', 'w') as f:
        f.write(summary)

    with open(out_dir / 'metrics.csv', 'w') as f:
        f.write('speaker,file,'
                'phaseC_kbps,phaseC_pesq,phaseC_stoi,'
                'phaseD_kbps,phaseD_pesq,phaseD_stoi,'
                'phaseDvae_kbps,phaseDvae_pesq,phaseDvae_stoi\n')
        for r in all_results:
            f.write(
                f"{r['speaker']},{r['file']},"
                f"{r['phaseC_kbps']:.2f},{r['phaseC_pesq']:.4f},{r['phaseC_stoi']:.4f},"
                f"{r['phaseD_kbps']:.2f},{r['phaseD_pesq']:.4f},{r['phaseD_stoi']:.4f},"
                f"{r['phaseDvae_kbps']:.2f},{r['phaseDvae_pesq']:.4f},{r['phaseDvae_stoi']:.4f}\n"
            )

    print(f"Audio  : {out_dir}/")
    print(f"Report : {out_dir}/report.txt")
    print(f"CSV    : {out_dir}/metrics.csv")


if __name__ == '__main__':
    main()
