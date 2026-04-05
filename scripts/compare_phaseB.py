#!/usr/bin/env python3
"""
Fair Codec Comparison: AAC vs Phase B (temporal stride + 3-bit QAT)
====================================================================
Neural pipeline:
  source → AudioEncoder → Linear(384→32) → Conv1d stride=20
         → 3-bit uniform quantization (8 levels) → zlib
         → zlib decompress → 3-bit dequantize
         → ConvTranspose1d ×20 → Linear(32→384) → AudioDecoder → PCM

Bitrate cap: 32 × 3-bit × 100 Hz = 9.6 kbps
"""

import sys
import zlib
import time
import textwrap
import tempfile
from pathlib import Path
from datetime import datetime

import av
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths

try:
    from pesq import pesq as pesq_fn
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi as stoi_fn
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


# ---------------------------------------------------------------------------
# AAC pipeline
# ---------------------------------------------------------------------------

def aac_encode_decode(audio, sr, target_kbps=10):
    duration = len(audio) / sr
    with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as tmp:
        tmp_path = tmp.name

    container = av.open(tmp_path, 'w')
    stream = container.add_stream('aac', rate=sr)
    stream.bit_rate = target_kbps * 1000
    for i in range(0, len(audio), 1024):
        chunk = audio[i:i + 1024].astype(np.float32)
        if len(chunk) < 1024:
            chunk = np.pad(chunk, (0, 1024 - len(chunk)))
        frame = av.AudioFrame.from_ndarray(chunk[np.newaxis, :], format='fltp', layout='mono')
        frame.sample_rate = sr
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode(None):
        container.mux(pkt)
    container.close()

    actual_kbps = Path(tmp_path).stat().st_size * 8 / duration / 1000

    container = av.open(tmp_path, 'r')
    frames = []
    for frame in container.decode(audio=0):
        frames.append(frame.to_ndarray().flatten())
    container.close()
    Path(tmp_path).unlink(missing_ok=True)

    decoded = np.concatenate(frames).astype(np.float32) if frames else np.zeros_like(audio)
    if len(decoded) >= len(audio):
        decoded = decoded[:len(audio)]
    else:
        decoded = np.pad(decoded, (0, len(audio) - len(decoded)))
    return decoded, actual_kbps


# ---------------------------------------------------------------------------
# Neural pipeline (3-bit quantization)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    d_model = ckpt.get('d_model')
    if d_model is None:
        k = 'encoder.transformer_blocks.0.attention.qkv.weight'
        d_model = state[k].shape[1] if k in state else 256

    ids = set()
    for k in state:
        if 'encoder.transformer_blocks.' in k:
            p = k.split('.')
            if len(p) > 2 and p[2].isdigit():
                ids.add(int(p[2]))
    n_layers = max(ids) + 1 if ids else 4

    model = NeuralAudioCodec(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=ckpt.get('n_heads', 8),
        window_size=ckpt.get('window_size', 200),
        dropout=0.0,
        bottleneck_dim=ckpt.get('bottleneck_dim'),
        temporal_stride=ckpt.get('temporal_stride', 1),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def neural_encode_decode_3bit(model, audio, sr, device, chunk_sec=1.0):
    """3-bit uniform quantization + zlib, matching Phase B training."""
    num_levels = 8
    chunk_size = int(chunk_sec * sr)
    recon_chunks = []
    total_bits = 0
    latencies_ms = []

    with torch.no_grad():
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            if len(chunk) < 160:
                continue
            x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            z = model.encode(x)
            z_np = z.squeeze(0).cpu().numpy()

            # 3-bit quantize
            z_min, z_max = float(z_np.min()), float(z_np.max())
            scale = (z_max - z_min) / (num_levels - 1) + 1e-8
            q = np.clip(np.round((z_np - z_min) / scale), 0, num_levels - 1).astype(np.uint8)
            compressed = zlib.compress(q.tobytes(), level=9)
            total_bits += len(compressed) * 8

            # Decompress and dequantize
            q_dec = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(z_np.shape)
            z_rec = q_dec.astype(np.float32) * scale + z_min

            z_tensor = torch.from_numpy(z_rec).unsqueeze(0).to(device)
            x_recon = model.decode(z_tensor)
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000.0)
            recon_chunks.append(x_recon.squeeze().cpu().numpy())

    decoded = np.concatenate(recon_chunks) if recon_chunks else np.zeros_like(audio)
    if len(decoded) >= len(audio):
        decoded = decoded[:len(audio)]
    else:
        decoded = np.pad(decoded, (0, len(audio) - len(decoded)))

    duration = len(audio) / sr
    kbps = total_bits / duration / 1000.0
    avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    return decoded, kbps, avg_lat


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ref, deg, sr):
    n = min(len(ref), len(deg))
    ref, deg = ref[:n].copy(), deg[:n].copy()
    ref /= (np.abs(ref).max() + 1e-8)
    deg /= (np.abs(deg).max() + 1e-8)
    pesq_score, stoi_score = None, None
    if PESQ_AVAILABLE:
        try:
            pesq_score = float(pesq_fn(sr, ref, deg, 'wb'))
        except Exception:
            pass
    if STOI_AVAILABLE:
        try:
            stoi_score = float(stoi_fn(ref, deg, sr, extended=False))
        except Exception:
            pass
    return pesq_score, stoi_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    SR = 16000
    CLIP_SEC = 5
    AAC_TARGET_KBPS = 10

    neural_ckpt = PROJECT_ROOT / 'checkpoints_ratedistortion/temporal_phaseB/best.pt'
    out_dir = PROJECT_ROOT / 'comparisons' / 'temporal_phaseB'
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
    print("PHASE B: AAC vs Temporal Stride + 3-bit QAT Neural Codec")
    print(f"{'='*68}")
    print(f"Checkpoint: {neural_ckpt}")
    print(f"Output    : {out_dir}")
    print(f"{'='*68}\n")

    print("Loading Phase B model...")
    model, ckpt = load_model(neural_ckpt, device)
    print(f"  d_model={ckpt.get('d_model')}, n_layers={ckpt.get('n_layers')}, "
          f"bottleneck_dim={ckpt.get('bottleneck_dim')}, "
          f"temporal_stride={ckpt.get('temporal_stride')}, "
          f"best_loss={ckpt.get('train_loss', '?'):.4f}\n")

    all_results = []

    for idx, audio_path in enumerate(test_files, start=1):
        spk_id = audio_path.parts[-3]
        sample_dir = out_dir / f"sample_{idx:02d}_spk{spk_id}"
        sample_dir.mkdir(exist_ok=True)

        print(f"[{idx}/5] {audio_path.name}  (speaker {spk_id})")

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            n = int(len(audio) * SR / sr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio = np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32)
        sf.write(sample_dir / 'source.wav', audio, SR)

        # AAC
        aac_dec, aac_kbps = aac_encode_decode(audio, SR, AAC_TARGET_KBPS)
        aac_pesq, aac_stoi = compute_metrics(audio, aac_dec, SR)
        sf.write(sample_dir / f'aac_{aac_kbps:.0f}kbps.wav', aac_dec, SR)
        print(f"  AAC    → {aac_kbps:.1f} kbps  PESQ={aac_pesq:.3f}  STOI={aac_stoi:.3f}")

        # Neural 3-bit
        neural_dec, neural_kbps, neural_lat = neural_encode_decode_3bit(
            model, audio, SR, device)
        neural_pesq, neural_stoi = compute_metrics(audio, neural_dec, SR)
        sf.write(sample_dir / f'neural_3bit_{neural_kbps:.0f}kbps.wav', neural_dec, SR)
        print(f"  Neural → {neural_kbps:.1f} kbps  PESQ={neural_pesq:.3f}  "
              f"STOI={neural_stoi:.3f}  lat={neural_lat:.0f}ms")

        all_results.append({
            'speaker': spk_id, 'file': audio_path.name,
            'aac_kbps': aac_kbps, 'aac_pesq': aac_pesq, 'aac_stoi': aac_stoi,
            'neural_kbps': neural_kbps, 'neural_pesq': neural_pesq,
            'neural_stoi': neural_stoi, 'neural_latency_ms': neural_lat,
        })

    def mean(vals):
        v = [x for x in vals if x is not None]
        return float(np.mean(v)) if v else float('nan')

    ak = mean([r['aac_kbps'] for r in all_results])
    ap = mean([r['aac_pesq'] for r in all_results])
    as_ = mean([r['aac_stoi'] for r in all_results])
    nk = mean([r['neural_kbps'] for r in all_results])
    np_ = mean([r['neural_pesq'] for r in all_results])
    ns = mean([r['neural_stoi'] for r in all_results])
    nl = mean([r['neural_latency_ms'] for r in all_results])

    SEP = '=' * 68
    sep = '-' * 68
    report = textwrap.dedent(f"""
    CODEC COMPARISON — Phase B (Temporal Stride + 3-bit QAT)
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    {SEP}

    Architecture:
      Encoder: d_model=384, 6 transformer layers, window=200 frames (100ms)
      Bottleneck: Linear(384→32) → Conv1d stride=20 → ~100 Hz latent rate
      Quantizer: 3-bit uniform (8 levels) + zlib level-9
      Bitrate cap: 32 × 3-bit × 100 Hz = 9.6 kbps

    {SEP}
    {'Codec':<22} {'Bitrate':>10} {'PESQ':>8} {'STOI':>8} {'Latency':>10}
    {sep}
    {'AAC (actual)':<22} {ak:>9.1f}k {ap:>8.3f} {as_:>8.3f} {'N/A':>10}
    {'Neural 3-bit QAT':<22} {nk:>9.1f}k {np_:>8.3f} {ns:>8.3f} {nl:>8.0f}ms
    {SEP}

    PER-FILE RESULTS
    {sep}
    {'Speaker':<10} {'AAC kbps':>10} {'AAC PESQ':>10} {'AAC STOI':>10} {'Neur kbps':>10} {'PESQ':>8} {'STOI':>8}
    {sep}
    """)
    for r in all_results:
        report += (f"  {r['speaker']:<8} {r['aac_kbps']:>10.1f} {r['aac_pesq']:>10.3f} "
                   f"{r['aac_stoi']:>10.3f} {r['neural_kbps']:>10.1f} "
                   f"{r['neural_pesq']:>8.3f} {r['neural_stoi']:>8.3f}\n")

    print(f"\n{report}")

    with open(out_dir / 'report.txt', 'w') as f:
        f.write(report)
    with open(out_dir / 'metrics.csv', 'w') as f:
        f.write('speaker,file,aac_kbps,aac_pesq,aac_stoi,neural_kbps,neural_pesq,neural_stoi,latency_ms\n')
        for r in all_results:
            f.write(f"{r['speaker']},{r['file']},{r['aac_kbps']:.2f},{r['aac_pesq']:.4f},"
                    f"{r['aac_stoi']:.4f},{r['neural_kbps']:.2f},{r['neural_pesq']:.4f},"
                    f"{r['neural_stoi']:.4f},{r['neural_latency_ms']:.1f}\n")

    print(f"Saved report : {out_dir}/report.txt")
    print(f"Saved metrics: {out_dir}/metrics.csv")
    print(f"Audio files  : {out_dir}/")


if __name__ == '__main__':
    main()
