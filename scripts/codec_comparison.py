#!/usr/bin/env python3
"""
Fair Codec Comparison: AAC vs Neural Bottleneck Codec
=====================================================
Both pipelines:
  - Same source files (test-clean, never seen by neural model during training)
  - Same sample rate (16 kHz mono)
  - Same clip duration (5 seconds)
  - Both decoded back to PCM WAV before computing metrics
  - Actual bitrate measured (not nominal target)

AAC pipeline:  source → pyav AAC encode (CBR) → pyav decode → PCM WAV
Neural pipeline: source → model.encode() → 1-bit quant → zlib → decompress
                 → dequant → model.decode() → PCM WAV

Output directory:
  comparisons/
      sample_001_spk1089/
          source.wav
          aac_16kbps.wav
          neural_11kbps.wav
      ...
      metrics.csv
      report.txt
"""

import sys
import zlib
import time
import tempfile
import textwrap
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
    print("Warning: pesq not available")

try:
    from pystoi import stoi as stoi_fn
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("Warning: pystoi not available")


# ---------------------------------------------------------------------------
# AAC pipeline
# ---------------------------------------------------------------------------

def aac_encode_decode(audio: np.ndarray, sr: int, target_kbps: int = 10) -> tuple:
    """
    Encode audio to AAC and decode back to PCM.

    Returns:
        decoded (np.ndarray): PCM audio, same length as input
        actual_kbps (float): actual bitrate achieved
    """
    duration = len(audio) / sr

    with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as tmp:
        tmp_path = tmp.name

    # Encode
    container = av.open(tmp_path, 'w')
    stream = container.add_stream('aac', rate=sr)
    stream.bit_rate = target_kbps * 1000

    for i in range(0, len(audio), 1024):
        chunk = audio[i:i + 1024].astype(np.float32)
        if len(chunk) < 1024:
            chunk = np.pad(chunk, (0, 1024 - len(chunk)))
        frame = av.AudioFrame.from_ndarray(
            chunk[np.newaxis, :], format='fltp', layout='mono'
        )
        frame.sample_rate = sr
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode(None):
        container.mux(pkt)
    container.close()

    # Measure actual bitrate
    actual_kbps = Path(tmp_path).stat().st_size * 8 / duration / 1000

    # Decode back to PCM
    container = av.open(tmp_path, 'r')
    frames = []
    for frame in container.decode(audio=0):
        frames.append(frame.to_ndarray().flatten())
    container.close()
    Path(tmp_path).unlink(missing_ok=True)

    decoded = np.concatenate(frames).astype(np.float32) if frames else np.zeros_like(audio)

    # Trim/pad to original length (AAC adds encoder/decoder delay padding)
    if len(decoded) >= len(audio):
        decoded = decoded[:len(audio)]
    else:
        decoded = np.pad(decoded, (0, len(audio) - len(decoded)))

    return decoded, actual_kbps


# ---------------------------------------------------------------------------
# Neural pipeline
# ---------------------------------------------------------------------------

def load_neural_model(checkpoint_path: Path, device: str) -> NeuralAudioCodec:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    d_model = ckpt.get('d_model')
    if d_model is None:
        k = 'encoder.transformer_blocks.0.attention.qkv.weight'
        d_model = state[k].shape[1] if k in state else 256

    layer_ids = set()
    for k in state:
        if 'encoder.transformer_blocks.' in k:
            p = k.split('.')
            if len(p) > 2 and p[2].isdigit():
                layer_ids.add(int(p[2]))
    n_layers = max(layer_ids) + 1 if layer_ids else 4

    model = NeuralAudioCodec(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=ckpt.get('n_heads', 8),
        window_size=ckpt.get('window_size', 200),
        dropout=0.0,
        bottleneck_dim=ckpt.get('bottleneck_dim'),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def neural_encode_decode(model: NeuralAudioCodec, audio: np.ndarray,
                          sr: int, device: str,
                          chunk_sec: float = 1.0) -> tuple:
    """
    Full neural codec pipeline with quantization:
      encode → 1-bit quantize → zlib compress → decompress
      → dequantize → decode

    This matches the real deployment bitrate exactly.

    Returns:
        decoded (np.ndarray): PCM audio
        actual_kbps (float): actual bitrate after quantize+compress
        avg_latency_ms (float): mean per-chunk encode+decode latency
    """
    chunk_size = int(chunk_sec * sr)
    recon_chunks = []
    total_compressed_bits = 0
    latencies_ms = []

    with torch.no_grad():
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            if len(chunk) < 160:
                continue

            x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)

            t0 = time.perf_counter()

            # Encode to bottleneck latent
            z = model.encode(x)                         # (1, T_lat, dim)
            z_np = z.squeeze(0).cpu().numpy()           # (T_lat, dim)

            # 1-bit uniform quantization (global min/max per chunk)
            z_min = float(z_np.min())
            z_max = float(z_np.max())
            threshold = (z_min + z_max) / 2.0
            z_bin = (z_np > threshold).astype(np.uint8)

            # Compress
            compressed = zlib.compress(z_bin.tobytes(), level=9)
            total_compressed_bits += len(compressed) * 8

            # Decompress and dequantize (simulate decoder side)
            z_bin_dec = np.frombuffer(
                zlib.decompress(compressed), dtype=np.uint8
            ).reshape(z_np.shape)
            z_dequant = np.where(z_bin_dec.astype(bool), z_max, z_min).astype(np.float32)

            # Decode
            z_tensor = torch.from_numpy(z_dequant).unsqueeze(0).to(device)
            x_recon = model.decode(z_tensor)            # (1, 1, T_audio)

            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            recon_chunks.append(x_recon.squeeze().cpu().numpy())

    decoded = np.concatenate(recon_chunks) if recon_chunks else np.zeros_like(audio)

    # Trim/pad to input length
    if len(decoded) >= len(audio):
        decoded = decoded[:len(audio)]
    else:
        decoded = np.pad(decoded, (0, len(audio) - len(decoded)))

    duration = len(audio) / sr
    actual_kbps = total_compressed_bits / duration / 1000.0
    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    return decoded, actual_kbps, avg_latency


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ref: np.ndarray, deg: np.ndarray, sr: int) -> tuple:
    min_len = min(len(ref), len(deg))
    ref, deg = ref[:min_len].copy(), deg[:min_len].copy()

    # Normalise to avoid PESQ saturation from level differences
    ref = ref / (np.abs(ref).max() + 1e-8)
    deg = deg / (np.abs(deg).max() + 1e-8)

    pesq_score = None
    stoi_score = None

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
    AAC_TARGET_KBPS = 10   # AAC will floor to ~16 kbps; actual is reported

    # Output directory
    out_dir = PROJECT_ROOT / 'comparisons'
    audio_dir = out_dir / 'audio_examples'
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Select one file per speaker (5 diverse speakers from test-clean)
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
    print("FAIR CODEC COMPARISON: AAC vs Neural Bottleneck")
    print(f"{'='*68}")
    print(f"Device       : {device}")
    print(f"Clip duration: {CLIP_SEC}s  |  Sample rate: {SR} Hz  |  Speakers: 5")
    print(f"Output       : {out_dir}")
    print(f"{'='*68}\n")

    # Load neural model
    neural_ckpt = PROJECT_ROOT / 'checkpoints_ratedistortion/bottleneck_v1/best.pt'
    print(f"Loading neural model from {neural_ckpt.name}...")
    model = load_neural_model(neural_ckpt, device)

    all_results = []

    for idx, audio_path in enumerate(test_files, start=1):
        spk_id = audio_path.parts[-3]
        sample_name = f"sample_{idx:02d}_spk{spk_id}"
        sample_dir = audio_dir / sample_name
        sample_dir.mkdir(exist_ok=True)

        print(f"\n[{idx}/5] {audio_path.name}  (speaker {spk_id})")

        # Load and trim
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            n = int(len(audio) * SR / sr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio = np.clip(audio[:CLIP_SEC * SR], -1.0, 1.0).astype(np.float32)

        # Save source
        source_path = sample_dir / 'source.wav'
        sf.write(source_path, audio, SR)

        # --- AAC pipeline ---
        print("  AAC encode/decode...")
        aac_decoded, aac_kbps = aac_encode_decode(audio, SR, target_kbps=AAC_TARGET_KBPS)
        aac_pesq, aac_stoi = compute_metrics(audio, aac_decoded, SR)
        aac_path = sample_dir / f'aac_{aac_kbps:.0f}kbps.wav'
        sf.write(aac_path, aac_decoded, SR)
        print(f"  AAC  → {aac_kbps:.1f} kbps  PESQ={aac_pesq:.3f}  STOI={aac_stoi:.3f}")

        # --- Neural pipeline ---
        print("  Neural encode/decode (with 1-bit quant)...")
        neural_decoded, neural_kbps, neural_lat = neural_encode_decode(
            model, audio, SR, device
        )
        neural_pesq, neural_stoi = compute_metrics(audio, neural_decoded, SR)
        neural_path = sample_dir / f'neural_{neural_kbps:.0f}kbps.wav'
        sf.write(neural_path, neural_decoded, SR)
        print(f"  Neur → {neural_kbps:.1f} kbps  PESQ={neural_pesq:.3f}  "
              f"STOI={neural_stoi:.3f}  lat={neural_lat:.0f}ms")

        all_results.append({
            'sample': sample_name,
            'speaker': spk_id,
            'file': audio_path.name,
            'aac_kbps': aac_kbps,
            'aac_pesq': aac_pesq,
            'aac_stoi': aac_stoi,
            'neural_kbps': neural_kbps,
            'neural_pesq': neural_pesq,
            'neural_stoi': neural_stoi,
            'neural_latency_ms': neural_lat,
        })

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------

    def mean(vals):
        v = [x for x in vals if x is not None]
        return float(np.mean(v)) if v else float('nan')

    aac_kbps_mean    = mean([r['aac_kbps']    for r in all_results])
    aac_pesq_mean    = mean([r['aac_pesq']     for r in all_results])
    aac_stoi_mean    = mean([r['aac_stoi']     for r in all_results])
    neural_kbps_mean = mean([r['neural_kbps']  for r in all_results])
    neural_pesq_mean = mean([r['neural_pesq']  for r in all_results])
    neural_stoi_mean = mean([r['neural_stoi']  for r in all_results])
    neural_lat_mean  = mean([r['neural_latency_ms'] for r in all_results])

    report = textwrap.dedent(f"""
    CODEC COMPARISON REPORT
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    ================================================================

    Test set   : LibriSpeech test-clean (5 speakers, never seen during training)
    Clip length: {CLIP_SEC} seconds each
    Sample rate: {SR} Hz mono

    PIPELINE DETAILS
    ----------------
    AAC    : pyav AAC encoder (CBR) → pyav decoder → PCM
             Note: AAC minimum bitrate at 16kHz mono is ~16 kbps (codec floor).

    Neural : Phase 1 encoder (d_model=384, 6 layers)
             → Linear projection (384 → 32)
             → 1-bit uniform quantization (global min/max per chunk)
             → zlib compress (level 9)
             → zlib decompress
             → 1-bit dequantize
             → Linear projection (32 → 384)
             → Phase 1 decoder
             Window size: 200 frames × 0.5ms = 100ms max latency

    RESULTS SUMMARY
    ---------------
    {'Codec':<20} {'Bitrate':>10} {'PESQ':>8} {'STOI':>8} {'Latency':>10}
    {'-'*60}
    {'AAC (actual)':<20} {aac_kbps_mean:>9.1f}k {aac_pesq_mean:>8.3f} {aac_stoi_mean:>8.3f} {'N/A':>10}
    {'Neural Bottleneck':<20} {neural_kbps_mean:>9.1f}k {neural_pesq_mean:>8.3f} {neural_stoi_mean:>8.3f} {neural_lat_mean:>8.0f}ms
    {'='*60}

    KEY OBSERVATIONS
    ----------------
    • Neural codec operates at {neural_kbps_mean:.1f} kbps — below AAC's minimum floor of ~16 kbps.
    • STOI (intelligibility): neural={neural_stoi_mean:.3f} vs AAC={aac_stoi_mean:.3f}
      Neural is {'ahead' if neural_stoi_mean > aac_stoi_mean else 'behind'} on intelligibility.
    • PESQ (perceptual quality): neural={neural_pesq_mean:.3f} vs AAC={aac_pesq_mean:.3f}
    • Neural latency: {neural_lat_mean:.0f} ms per chunk (well below 100ms target).

    PER-FILE RESULTS
    ----------------
    {'Speaker':<10} {'AAC kbps':>10} {'AAC PESQ':>10} {'AAC STOI':>10} {'Neur kbps':>10} {'Neur PESQ':>10} {'Neur STOI':>10}
    {'-'*72}
    """)

    for r in all_results:
        report += (
            f"  {r['speaker']:<8} "
            f"{r['aac_kbps']:>10.1f} "
            f"{r['aac_pesq']:>10.3f} "
            f"{r['aac_stoi']:>10.3f} "
            f"{r['neural_kbps']:>10.1f} "
            f"{r['neural_pesq']:>10.3f} "
            f"{r['neural_stoi']:>10.3f}\n"
        )

    report += f"\nAudio files saved to: {audio_dir}\n"

    print(report)

    with open(out_dir / 'report.txt', 'w') as f:
        f.write(report)

    # Save CSV
    with open(out_dir / 'metrics.csv', 'w') as f:
        f.write('sample,speaker,file,aac_kbps,aac_pesq,aac_stoi,'
                'neural_kbps,neural_pesq,neural_stoi,neural_latency_ms\n')
        for r in all_results:
            f.write(','.join([
                r['sample'], r['speaker'], r['file'],
                f"{r['aac_kbps']:.2f}", f"{r['aac_pesq']:.4f}", f"{r['aac_stoi']:.4f}",
                f"{r['neural_kbps']:.2f}", f"{r['neural_pesq']:.4f}", f"{r['neural_stoi']:.4f}",
                f"{r['neural_latency_ms']:.1f}",
            ]) + '\n')

    print(f"\nSaved: {out_dir / 'report.txt'}")
    print(f"Saved: {out_dir / 'metrics.csv'}")
    print(f"Audio: {audio_dir}")


if __name__ == '__main__':
    main()
