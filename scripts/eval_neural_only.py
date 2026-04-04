#!/usr/bin/env python3
"""
Evaluate neural codec only (no AAC comparison for systems without ffmpeg).
Measures PESQ, STOI, bitrate, and latency.
"""

import argparse
import os
import sys
import time
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
    print("Warning: pesq not available")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("Warning: pystoi not available")

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
    parser = argparse.ArgumentParser(description="Evaluate neural codec only")
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints_emergency/best.pt'))
    parser.add_argument('--data-root', type=Path, default=None, help='Folder with audio files')
    parser.add_argument('--max-files', type=int, default=10)
    parser.add_argument('--seg-sec', type=float, default=8.0)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk-sec', type=float, default=2.0)
    parser.add_argument('--target-bitrate-kbps', type=int, default=10, help='Target bitrate')
    parser.add_argument('--out', type=Path, default=Path('results/eval_neural.csv'))
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
    print(f"NEURAL CODEC EVALUATION")
    print(f"Target Bitrate: {args.target_bitrate_kbps} kbps")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Files: {len(files)}")
    print(f"{'='*80}\n")

    neural = NeuralCodecQuantized(
        args.checkpoint, 
        device=args.device, 
        sample_rate=args.sample_rate,
        target_bitrate_kbps=args.target_bitrate_kbps
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

        results.append({
            'file': str(audio_path),
            'duration_s': duration,
            'bitrate_achieved_bps': neural_bitrate,
            'bitrate_target_kbps': args.target_bitrate_kbps,
            'latency_ms_avg': float(np.mean(neural_latencies)) if neural_latencies else 0.0,
            'latency_ms_p95': float(np.percentile(neural_latencies, 95)) if neural_latencies else 0.0,
            'pesq': neural_pesq,
            'stoi': neural_stoi,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        'file', 'duration_s',
        'bitrate_achieved_bps', 'bitrate_target_kbps', 'latency_ms_avg', 'latency_ms_p95', 'pesq', 'stoi',
    ]

    with open(args.out, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in results:
            f.write(','.join(str(row.get(h, '')) for h in header) + '\n')

    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    
    mean_bitrate = np.mean([r['bitrate_achieved_bps'] for r in results]) / 1000
    mean_latency = np.mean([r['latency_ms_avg'] for r in results])
    mean_latency_p95 = np.mean([r['latency_ms_p95'] for r in results])
    pesq_scores = [r['pesq'] for r in results if r['pesq'] is not None]
    stoi_scores = [r['stoi'] for r in results if r['stoi'] is not None]
    
    print(f"Target Bitrate:     {args.target_bitrate_kbps} kbps")
    print(f"Achieved Bitrate:   {mean_bitrate:.2f} kbps")
    print(f"Latency (avg):      {mean_latency:.2f} ms")
    print(f"Latency (p95):      {mean_latency_p95:.2f} ms")
    if pesq_scores:
        print(f"PESQ (mean):        {np.mean(pesq_scores):.3f}")
    if stoi_scores:
        print(f"STOI (mean):        {np.mean(stoi_scores):.3f}")
    
    print(f"\nSaved detailed results to {args.out}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
