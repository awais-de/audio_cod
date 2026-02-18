#!/usr/bin/env python3
"""
Fair bitrate comparison: Neural Codec vs FFmpeg encoder at same bitrate.
Both codecs run at target bitrate (e.g., 10 kbps) and quality is compared.
"""

import argparse
import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NeuralAudioCodec
from src.paths import get_dataset_paths
from src.quantization import QuantizedLatentCodec

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False

try:
    from scripts.evaluate_scipy_based import pesq_scipy, stoi_scipy
    SCIPY_METRICS_AVAILABLE = True
except Exception:
    SCIPY_METRICS_AVAILABLE = False


def resample_audio(audio, src_sr, tgt_sr):
    if src_sr == tgt_sr:
        return audio
    return np.interp(
        np.linspace(0, len(audio), int(len(audio) * tgt_sr / src_sr)),
        np.arange(len(audio)),
        audio,
    )


def compute_metrics(reference, degraded, sr):
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]

    pesq_score = None
    stoi_score = None

    if PESQ_AVAILABLE:
        try:
            pesq_score = pesq(sr, reference, degraded, 'wb')
        except Exception:
            pesq_score = None
    elif SCIPY_METRICS_AVAILABLE:
        pesq_score = pesq_scipy(reference, degraded, sr)

    if STOI_AVAILABLE:
        try:
            stoi_score = stoi(reference, degraded, sr, extended=False)
        except Exception:
            stoi_score = None
    elif SCIPY_METRICS_AVAILABLE:
        stoi_score = stoi_scipy(reference, degraded, sr)

    return pesq_score, stoi_score


class FFmpegCodec:
    def __init__(self, codec='aac', bitrate_kbps=10, sample_rate=16000):
        self.codec = codec
        self.bitrate_kbps = bitrate_kbps
        self.sample_rate = sample_rate
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffmpeg not found. Install ffmpeg and retry.")

    def encode_decode(self, audio):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_wav = os.path.join(tmpdir, 'input.wav')
            encoded_path = os.path.join(tmpdir, f'encoded.{self.codec}')
            output_wav = os.path.join(tmpdir, 'output.wav')

            sf.write(input_wav, np.clip(audio, -1.0, 1.0), self.sample_rate, subtype='PCM_16')

            encode_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', input_wav,
                '-c:a', self.codec,
                '-b:a', f'{self.bitrate_kbps}k',
                encoded_path,
            ]
            subprocess.run(encode_cmd, capture_output=True, check=True)

            decode_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', encoded_path,
                output_wav,
            ]
            subprocess.run(decode_cmd, capture_output=True, check=True)

            decoded, sr = sf.read(output_wav)
            if sr != self.sample_rate:
                decoded = resample_audio(decoded, sr, self.sample_rate)

            if len(decoded.shape) > 1:
                decoded = decoded.mean(axis=1)

            encoded_size = os.path.getsize(encoded_path)
            duration = len(audio) / self.sample_rate
            bitrate_bps = (encoded_size * 8) / max(duration, 1e-6)

            return decoded, bitrate_bps


class NeuralCodecQuantized:
    def __init__(self, checkpoint_path, device='cpu', sample_rate=16000, target_bitrate_kbps=10):
        self.device = device
        self.sample_rate = sample_rate
        self.target_bitrate_kbps = target_bitrate_kbps
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        d_model = checkpoint.get('d_model')
        if d_model is None:
            qkv_weight = state_dict.get('encoder.transformer_blocks.0.attention.qkv.weight')
            if qkv_weight is not None:
                d_model = qkv_weight.shape[1]
        if d_model is None:
            conv_weight = state_dict.get('encoder.conv_layers.3.0.conv.weight')
            if conv_weight is not None:
                d_model = conv_weight.shape[0]
        if d_model is None:
            d_model = 256

        n_layers = checkpoint.get('n_layers')
        if n_layers is None:
            layer_indices = set()
            for key in state_dict.keys():
                if key.startswith('encoder.transformer_blocks.'):
                    parts = key.split('.')
                    if len(parts) > 2 and parts[2].isdigit():
                        layer_indices.add(int(parts[2]))
            n_layers = max(layer_indices) + 1 if layer_indices else 4

        n_heads = checkpoint.get('n_heads', 8)

        self.model = NeuralAudioCodec(
            sample_rate=sample_rate,
            hop_length=checkpoint.get('hop_length', 160),
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=checkpoint.get('window_size', 256),
            dropout=0.0,
        ).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Initialize quantizer
        self.quantizer = QuantizedLatentCodec(
            target_bitrate_kbps=target_bitrate_kbps,
            latent_dim=d_model,
            frame_duration_ms=20.0
        )

    def encode_decode(self, audio, chunk_seconds=2.0):
        chunk_size = int(chunk_seconds * self.sample_rate)
        reconstructed_chunks = []
        latencies_ms = []
        total_compressed_bytes = 0

        with torch.no_grad():
            for start in range(0, len(audio), chunk_size):
                chunk = audio[start:start + chunk_size]
                chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(self.device)

                if self.device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                latent = self.model.encoder(chunk_tensor)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

                # Quantize and entropy code
                latent_np = latent.detach().cpu().numpy()
                compressed = self.quantizer.compress(latent_np)
                total_compressed_bytes += len(compressed)

                # Decompress (simulate receiving side)
                latent_reconstructed = self.quantizer.decompress(compressed)
                latent_recon_tensor = torch.from_numpy(latent_reconstructed).to(self.device)

                # Decode
                recon = self.model.decoder(latent_recon_tensor)
                reconstructed_chunks.append(recon.squeeze().cpu().numpy())

        reconstructed = np.concatenate(reconstructed_chunks) if reconstructed_chunks else np.array([], dtype=np.float32)
        duration = len(audio) / self.sample_rate
        achieved_bitrate_bps = (total_compressed_bytes * 8) / max(duration, 1e-6)
        
        return reconstructed, achieved_bitrate_bps, latencies_ms


def collect_audio_files(data_root, max_files):
    exts = ('.wav', '.flac', '.mp3', '.ogg')
    files = [p for p in Path(data_root).rglob('*') if p.suffix.lower() in exts]
    files = sorted(files)
    return files[:max_files]


def main():
    parser = argparse.ArgumentParser(description="Fair bitrate comparison: Neural vs FFmpeg at same bitrate")
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints_emergency/best.pt'))
    parser.add_argument('--data-root', type=Path, default=None, help='Folder with audio files')
    parser.add_argument('--max-files', type=int, default=10)
    parser.add_argument('--seg-sec', type=float, default=8.0)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk-sec', type=float, default=2.0)
    parser.add_argument('--target-bitrate-kbps', type=int, default=10, help='Target bitrate for both codecs')
    parser.add_argument('--ffmpeg-codec', type=str, default='aac')
    parser.add_argument('--out', type=Path, default=Path('results/compare_fair_bitrate.csv'))
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = get_dataset_paths()["test_clean"]

    files = collect_audio_files(args.data_root, args.max_files)
    if not files:
        print(f"No audio files found in {args.data_root}")
        return

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    print(f"\n{'='*80}")
    print(f"FAIR BITRATE COMPARISON")
    print(f"Target Bitrate: {args.target_bitrate_kbps} kbps")
    print(f"Codec 1: FFmpeg {args.ffmpeg_codec}")
    print(f"Codec 2: Neural (with quantization)")
    print(f"{'='*80}\n")

    neural = NeuralCodecQuantized(
        args.checkpoint, 
        device=args.device, 
        sample_rate=args.sample_rate,
        target_bitrate_kbps=args.target_bitrate_kbps
    )
    ffmpeg_codec = FFmpegCodec(
        codec=args.ffmpeg_codec, 
        bitrate_kbps=args.target_bitrate_kbps, 
        sample_rate=args.sample_rate
    )

    results = []

    for audio_path in tqdm(files, desc="Evaluating"):
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = resample_audio(audio, sr, args.sample_rate)
        audio = audio[:int(args.seg_sec * args.sample_rate)]
        duration = len(audio) / args.sample_rate

        # Neural codec with quantization
        neural_recon, neural_bitrate, neural_latencies = neural.encode_decode(audio, chunk_seconds=args.chunk_sec)
        neural_pesq, neural_stoi = compute_metrics(audio, neural_recon, args.sample_rate)

        # FFmpeg codec
        ffmpeg_recon, ffmpeg_bitrate = ffmpeg_codec.encode_decode(audio)
        ffmpeg_pesq, ffmpeg_stoi = compute_metrics(audio, ffmpeg_recon, args.sample_rate)

        results.append({
            'file': str(audio_path),
            'duration_s': duration,
            'neural_bitrate_achieved_bps': neural_bitrate,
            'neural_bitrate_target_kbps': args.target_bitrate_kbps,
            'neural_latency_ms_avg': float(np.mean(neural_latencies)) if neural_latencies else 0.0,
            'neural_latency_ms_p95': float(np.percentile(neural_latencies, 95)) if neural_latencies else 0.0,
            'neural_pesq': neural_pesq,
            'neural_stoi': neural_stoi,
            'ffmpeg_codec': args.ffmpeg_codec,
            'ffmpeg_bitrate_achieved_bps': ffmpeg_bitrate,
            'ffmpeg_bitrate_target_kbps': args.target_bitrate_kbps,
            'ffmpeg_pesq': ffmpeg_pesq,
            'ffmpeg_stoi': ffmpeg_stoi,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        'file', 'duration_s',
        'neural_bitrate_achieved_bps', 'neural_bitrate_target_kbps', 'neural_latency_ms_avg', 'neural_latency_ms_p95', 'neural_pesq', 'neural_stoi',
        'ffmpeg_codec', 'ffmpeg_bitrate_achieved_bps', 'ffmpeg_bitrate_target_kbps', 'ffmpeg_pesq', 'ffmpeg_stoi',
    ]

    with open(args.out, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in results:
            f.write(','.join(str(row.get(h, '')) for h in header) + '\n')

    print(f"\n{'='*80}")
    print(f"Saved results to {args.out}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
