#!/usr/bin/env python3
"""
Evaluate bottleneck neural codec: PESQ, STOI, real bitrate, latency.
Works for both bottleneck checkpoints (bottleneck_dim set) and legacy checkpoints.
Compares against the old phase3b_l2_10kbps baseline.
"""

import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths

try:
    from pesq import pesq as pesq_fn
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("Warning: pesq not available")

try:
    from pystoi import stoi as stoi_fn
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("Warning: pystoi not available")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    # Infer d_model
    d_model = ckpt.get('d_model')
    if d_model is None:
        k = 'encoder.transformer_blocks.0.attention.qkv.weight'
        if k in state:
            d_model = state[k].shape[1]
    if d_model is None:
        d_model = 256

    # Infer n_layers
    n_layers = ckpt.get('n_layers')
    if n_layers is None:
        ids = set()
        for k in state:
            if 'encoder.transformer_blocks.' in k:
                p = k.split('.')
                if len(p) > 2 and p[2].isdigit():
                    ids.add(int(p[2]))
        n_layers = max(ids) + 1 if ids else 4

    n_heads = ckpt.get('n_heads', 8)
    window_size = ckpt.get('window_size', 256)
    bottleneck_dim = ckpt.get('bottleneck_dim', None)

    model = NeuralAudioCodec(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        window_size=window_size, dropout=0.0,
        bottleneck_dim=bottleneck_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, {
        'd_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads,
        'window_size': window_size, 'bottleneck_dim': bottleneck_dim,
        'epoch': ckpt.get('epoch', '?'),
    }


def encode_decode_chunked(model, audio, sr, device, chunk_sec=1.0):
    """
    Encode audio in chunks using model.encode()/decode(), measure real bitrate.
    Returns: (reconstructed_audio, bitrate_kbps, avg_latency_ms)
    """
    chunk_size = int(chunk_sec * sr)
    recon_chunks = []
    latencies_ms = []
    total_compressed_bits = 0

    with torch.no_grad():
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            if len(chunk) < 160:
                continue
            x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            z = model.encode(x)                     # (1, T_lat, dim)
            x_recon = model.decode(z)               # (1, 1, T_audio)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            # Real bitrate: 1-bit quantize + zlib
            z_np = z.squeeze(0).cpu().numpy()
            threshold = (z_np.min() + z_np.max()) / 2.0
            z_bin = (z_np > threshold).astype(np.uint8)
            compressed = zlib.compress(z_bin.tobytes(), level=9)
            total_compressed_bits += len(compressed) * 8

            recon_chunks.append(x_recon.squeeze().cpu().numpy())

    reconstructed = np.concatenate(recon_chunks) if recon_chunks else np.zeros_like(audio)
    duration = len(audio) / sr
    bitrate_kbps = total_compressed_bits / duration / 1000.0
    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    return reconstructed, bitrate_kbps, avg_latency


def compute_metrics(ref, deg, sr):
    min_len = min(len(ref), len(deg))
    ref, deg = ref[:min_len], deg[:min_len]

    pesq_score, stoi_score = None, None

    if PESQ_AVAILABLE:
        try:
            pesq_score = pesq_fn(sr, ref, deg, 'wb')
        except Exception as e:
            pesq_score = None

    if STOI_AVAILABLE:
        try:
            stoi_score = stoi_fn(ref, deg, sr, extended=False)
        except Exception as e:
            stoi_score = None

    return pesq_score, stoi_score


def evaluate_checkpoint(label, checkpoint_path, files, device, seg_sec=8.0, chunk_sec=1.0):
    print(f"\n--- {label} ---")
    model, info = load_model(checkpoint_path, device)
    print(f"  d_model={info['d_model']}, n_layers={info['n_layers']}, "
          f"window={info['window_size']}, bottleneck={info['bottleneck_dim']}, epoch={info['epoch']}")

    all_pesq, all_stoi, all_bitrate, all_latency = [], [], [], []

    for path in tqdm(files, desc=label):
        try:
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                n = int(len(audio) * 16000 / sr)
                audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
                sr = 16000
            audio = np.clip(audio[:int(seg_sec * sr)], -1.0, 1.0).astype(np.float32)

            recon, kbps, lat_ms = encode_decode_chunked(model, audio, sr, device, chunk_sec)
            p, s = compute_metrics(audio, recon, sr)

            all_bitrate.append(kbps)
            all_latency.append(lat_ms)
            if p is not None:
                all_pesq.append(p)
            if s is not None:
                all_stoi.append(s)

            print(f"  {path.name}: {kbps:.1f} kbps  PESQ={p:.3f if p else 'N/A'}  "
                  f"STOI={s:.3f if s else 'N/A'}  lat={lat_ms:.0f}ms")
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")

    return {
        'label': label,
        'bitrate_kbps': float(np.mean(all_bitrate)) if all_bitrate else float('nan'),
        'pesq': float(np.mean(all_pesq)) if all_pesq else float('nan'),
        'stoi': float(np.mean(all_stoi)) if all_stoi else float('nan'),
        'latency_ms': float(np.mean(all_latency)) if all_latency else float('nan'),
        'n_files': len(files),
        'n_pesq': len(all_pesq),
        'n_stoi': len(all_stoi),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    paths = get_dataset_paths()
    test_files = sorted(paths['test_clean'].rglob('*.flac'))[:10]
    print(f"Evaluating on {len(test_files)} test-clean files")

    checkpoints = [
        ("Bottleneck v1 (new)",
         PROJECT_ROOT / 'checkpoints_ratedistortion/bottleneck_v1/best.pt'),
        ("Phase3b L2 10kbps (old best)",
         PROJECT_ROOT / 'checkpoints_ratedistortion/phase3b_l2_10kbps/best.pt'),
        ("Phase1 (PESQ baseline)",
         PROJECT_ROOT / 'checkpoints_emergency/phase1_multiscale_20260129_124452/best.pt'),
    ]

    results = []
    for label, ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"\nSkipping {label} — checkpoint not found: {ckpt_path}")
            continue
        r = evaluate_checkpoint(label, ckpt_path, test_files, device)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*72}")
    print("FINAL COMPARISON")
    print(f"{'='*72}")
    print(f"{'Model':<30} {'Bitrate':>10} {'PESQ':>8} {'STOI':>8} {'Latency':>10}")
    print(f"{'-'*72}")
    for r in results:
        print(f"{r['label']:<30} {r['bitrate_kbps']:>9.1f}k "
              f"{r['pesq']:>8.3f} {r['stoi']:>8.3f} {r['latency_ms']:>8.0f}ms")
    print(f"{'='*72}")
    print(f"{'AAC @ 10kbps (reference)':<30} {'~12.0':>10} {'2.060':>8} {'0.240':>8} {'N/A':>10}")
    print(f"{'='*72}")

    # Save CSV
    out = PROJECT_ROOT / 'results/eval_bottleneck_v1.csv'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        f.write('model,bitrate_kbps,pesq,stoi,latency_ms\n')
        for r in results:
            f.write(f"{r['label']},{r['bitrate_kbps']:.2f},{r['pesq']:.4f},{r['stoi']:.4f},{r['latency_ms']:.1f}\n")
        f.write(f"AAC @ 10kbps (reference),12.00,2.0600,0.2400,N/A\n")
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
