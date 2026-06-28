#!/usr/bin/env python3
"""
Three-way comparison: AAC vs EnCodec vs Our Neural Codec (Phase B)
===================================================================
All evaluated on the same 5 LibriSpeech test-clean speakers, 5s clips.

EnCodec operates at 24kHz internally — we resample our 16kHz audio up,
encode/decode, then resample back to 16kHz for fair metric comparison.

Our codec: 6.3 kbps (3-bit QAT, temporal stride)
EnCodec:   1.5 / 3.0 / 6.0 kbps (for context at multiple points)
AAC:       ~15.6 kbps (minimum floor at 16kHz mono)
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
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths

try:
    from encodec import EncodecModel
    from encodec.utils import convert_audio
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False
    print("Warning: encodec not installed — pip install encodec")

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


# AAC

def aac_encode_decode(audio, sr, target_kbps=10):
    duration = len(audio) / sr
    with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as tmp:
        tmp_path = tmp.name
    container = av.open(tmp_path, 'w')
    stream = container.add_stream('aac', rate=sr)
    stream.bit_rate = target_kbps * 1000
    samples_written = 0
    for i in range(0, len(audio), 1024):
        chunk = audio[i:i + 1024].astype(np.float32)
        if len(chunk) < 1024:
            chunk = np.pad(chunk, (0, 1024 - len(chunk)))
        frame = av.AudioFrame.from_ndarray(chunk[np.newaxis, :], format='fltp', layout='mono')
        frame.sample_rate = sr
        frame.pts = samples_written
        samples_written += 1024
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode(None):
        container.mux(pkt)
    container.close()
    actual_kbps = Path(tmp_path).stat().st_size * 8 / duration / 1000
    container = av.open(tmp_path, 'r')
    frames = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray()
        frames.append(arr[0])  # channel 0 — decoder may return stereo for mono input
    container.close()
    Path(tmp_path).unlink(missing_ok=True)
    decoded = np.concatenate(frames).astype(np.float32) if frames else np.zeros_like(audio)
    if len(decoded) >= len(audio):
        decoded = decoded[:len(audio)]
    else:
        decoded = np.pad(decoded, (0, len(audio) - len(decoded)))
    return decoded, actual_kbps


# EnCodec

def encodec_encode_decode(audio_np, sr, bandwidth_kbps, device):
    """
    audio_np: float32 array, shape (N,), 16kHz
    bandwidth_kbps: one of 1.5, 3.0, 6.0, 12.0, 24.0
    Returns: decoded float32 array (16kHz), actual_kbps
    """
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth_kbps)
    model = model.to(device)
    model.eval()

    # Resample 16kHz → 24kHz
    audio_t = torch.FloatTensor(audio_np).unsqueeze(0)  # (1, N)
    audio_24k = torchaudio.functional.resample(audio_t, sr, 24000)  # (1, N')
    audio_24k = audio_24k.unsqueeze(0).to(device)  # (1, 1, N')

    with torch.no_grad():
        encoded_frames = model.encode(audio_24k)
        decoded_24k = model.decode(encoded_frames)  # (1, 1, N')

    # Measure bitrate from encoded frames
    total_codes = sum(codes.numel() for codes, _ in encoded_frames)
    # Each code index uses log2(codebook_size) bits
    n_codebooks = encoded_frames[0][0].shape[1]
    bits_per_code = np.log2(model.quantizer.bins)
    duration = len(audio_np) / sr
    actual_kbps = (total_codes * bits_per_code) / duration / 1000

    # Resample back 24kHz → 16kHz
    decoded_16k = torchaudio.functional.resample(
        decoded_24k.squeeze(0).cpu(), 24000, sr
    ).squeeze(0).numpy()

    if len(decoded_16k) >= len(audio_np):
        decoded_16k = decoded_16k[:len(audio_np)]
    else:
        decoded_16k = np.pad(decoded_16k, (0, len(audio_np) - len(decoded_16k)))

    return decoded_16k.astype(np.float32), actual_kbps


# Our Neural Codec (Phase B, 3-bit QAT)

def load_our_model(checkpoint_path, device):
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
        d_model=d_model, n_layers=n_layers,
        n_heads=ckpt.get('n_heads', 8),
        window_size=ckpt.get('window_size', 200),
        dropout=0.0,
        bottleneck_dim=ckpt.get('bottleneck_dim', 32),
        temporal_stride=ckpt.get('temporal_stride', 20),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def our_encode_decode(model, audio_np, sr, device, chunk_sec=1.0):
    num_levels = 8
    chunk_size = int(chunk_sec * sr)
    recon_chunks, total_bits, latencies = [], 0, []
    with torch.no_grad():
        for start in range(0, len(audio_np), chunk_size):
            chunk = audio_np[start:start + chunk_size]
            if len(chunk) < 160:
                continue
            x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            z = model.encode(x)
            z_np = z.squeeze(0).cpu().numpy()
            z_min, z_max = float(z_np.min()), float(z_np.max())
            scale = (z_max - z_min) / (num_levels - 1) + 1e-8
            q = np.clip(np.round((z_np - z_min) / scale), 0, num_levels - 1).astype(np.uint8)
            compressed = zlib.compress(q.tobytes(), level=9)
            total_bits += len(compressed) * 8
            q_dec = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(z_np.shape)
            z_rec = q_dec.astype(np.float32) * scale + z_min
            z_t = torch.from_numpy(z_rec).unsqueeze(0).to(device)
            x_recon = model.decode(z_t)
            latencies.append((time.perf_counter() - t0) * 1000)
            recon_chunks.append(x_recon.squeeze().cpu().numpy())
    decoded = np.concatenate(recon_chunks) if recon_chunks else np.zeros_like(audio_np)
    if len(decoded) >= len(audio_np):
        decoded = decoded[:len(audio_np)]
    else:
        decoded = np.pad(decoded, (0, len(audio_np) - len(decoded)))
    kbps = total_bits / (len(audio_np) / sr) / 1000
    return decoded.astype(np.float32), kbps, float(np.mean(latencies))


# Metrics

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


# Main

def mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else float('nan')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    SR = 16000
    CLIP_SEC = 5
    ENCODEC_BANDWIDTHS = [1.5, 3.0, 6.0]

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d')
    out_dir = PROJECT_ROOT / 'comparisons' / f'{timestamp}_phaseC_aac_vs_encodec_vs_ours'
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
    print("THREE-WAY COMPARISON: AAC vs EnCodec vs Our Neural Codec (Phase B)")
    print(f"{'='*72}\n")

    our_ckpt = PROJECT_ROOT / 'checkpoints_active/temporal_phaseC/best.pt'
    print("Loading our Phase B model...")
    our_model, our_ckpt_meta = load_our_model(our_ckpt, device)
    print(f"  bottleneck_dim={our_ckpt_meta.get('bottleneck_dim')}, "
          f"temporal_stride={our_ckpt_meta.get('temporal_stride')}, "
          f"best_loss={our_ckpt_meta.get('train_loss', 0):.4f}\n")

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

        result = {'speaker': spk_id, 'file': audio_path.name}

        # AAC
        aac_dec, aac_kbps = aac_encode_decode(audio, SR, target_kbps=10)
        aac_pesq, aac_stoi = compute_metrics(audio, aac_dec, SR)
        sf.write(sample_dir / f'aac_{aac_kbps:.0f}kbps.wav', aac_dec, SR)
        result.update({'aac_kbps': aac_kbps, 'aac_pesq': aac_pesq, 'aac_stoi': aac_stoi})
        print(f"  AAC          → {aac_kbps:.1f} kbps  PESQ={aac_pesq:.3f}  STOI={aac_stoi:.3f}")

        # EnCodec at multiple bitrates
        for bw in ENCODEC_BANDWIDTHS:
            enc_dec, enc_kbps = encodec_encode_decode(audio, SR, bw, device)
            enc_pesq, enc_stoi = compute_metrics(audio, enc_dec, SR)
            sf.write(sample_dir / f'encodec_{bw}kbps.wav', enc_dec, SR)
            key = f'enc{bw}'
            result.update({f'{key}_kbps': enc_kbps, f'{key}_pesq': enc_pesq, f'{key}_stoi': enc_stoi})
            print(f"  EnCodec {bw:4.1f}  → {enc_kbps:.1f} kbps  PESQ={enc_pesq:.3f}  STOI={enc_stoi:.3f}")

        # Our codec — full-clip encode to avoid chunk boundary artifacts
        our_dec, our_kbps, our_lat = our_encode_decode(our_model, audio, SR, device,
                                                        chunk_sec=CLIP_SEC)
        our_pesq, our_stoi = compute_metrics(audio, our_dec, SR)
        sf.write(sample_dir / f'ours_3bit_{our_kbps:.0f}kbps.wav', our_dec, SR)
        result.update({'our_kbps': our_kbps, 'our_pesq': our_pesq,
                        'our_stoi': our_stoi, 'our_lat': our_lat})
        print(f"  Ours (3-bit) → {our_kbps:.1f} kbps  PESQ={our_pesq:.3f}  STOI={our_stoi:.3f}  lat={our_lat:.0f}ms")

        all_results.append(result)
        print()

    # Summary
    SEP = '=' * 72
    sep = '-' * 72
    lines = [
        f"\nCOMPARISON SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        SEP,
        f"{'Codec':<22} {'Bitrate':>10} {'PESQ':>8} {'STOI':>8}",
        sep,
        f"{'AAC (~10kbps target)':<22} {mean([r['aac_kbps'] for r in all_results]):>9.1f}k "
        f"{mean([r['aac_pesq'] for r in all_results]):>8.3f} "
        f"{mean([r['aac_stoi'] for r in all_results]):>8.3f}",
    ]
    for bw in ENCODEC_BANDWIDTHS:
        key = f'enc{bw}'
        lines.append(
            f"{'EnCodec '+str(bw)+' kbps':<22} "
            f"{mean([r[f'{key}_kbps'] for r in all_results]):>9.1f}k "
            f"{mean([r[f'{key}_pesq'] for r in all_results]):>8.3f} "
            f"{mean([r[f'{key}_stoi'] for r in all_results]):>8.3f}"
        )
    lines += [
        f"{'Ours (3-bit QAT)':<22} {mean([r['our_kbps'] for r in all_results]):>9.1f}k "
        f"{mean([r['our_pesq'] for r in all_results]):>8.3f} "
        f"{mean([r['our_stoi'] for r in all_results]):>8.3f}",
        SEP,
    ]
    report = '\n'.join(lines)
    print(report)

    with open(out_dir / 'report.txt', 'w') as f:
        f.write(report)

    # CSV
    with open(out_dir / 'metrics.csv', 'w') as f:
        header = 'speaker,file,aac_kbps,aac_pesq,aac_stoi'
        for bw in ENCODEC_BANDWIDTHS:
            k = f'enc{bw}'
            header += f',{k}_kbps,{k}_pesq,{k}_stoi'
        header += ',our_kbps,our_pesq,our_stoi,our_lat_ms'
        f.write(header + '\n')
        for r in all_results:
            row = f"{r['speaker']},{r['file']},{r['aac_kbps']:.2f},{r['aac_pesq']:.4f},{r['aac_stoi']:.4f}"
            for bw in ENCODEC_BANDWIDTHS:
                k = f'enc{bw}'
                row += f",{r[f'{k}_kbps']:.2f},{r[f'{k}_pesq']:.4f},{r[f'{k}_stoi']:.4f}"
            row += f",{r['our_kbps']:.2f},{r['our_pesq']:.4f},{r['our_stoi']:.4f},{r['our_lat']:.1f}"
            f.write(row + '\n')

    print(f"\nSaved: {out_dir}/report.txt")
    print(f"Saved: {out_dir}/metrics.csv")
    print(f"Audio: {out_dir}/")


if __name__ == '__main__':
    main()
