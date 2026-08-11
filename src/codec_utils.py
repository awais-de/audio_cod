"""Shared inference utilities: checkpoint finding, model loading, encode-decode, metrics."""

from pathlib import Path

import numpy as np
import torch
import zlib

from src.model import NeuralAudioCodec

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


def find_checkpoint(project_root: Path) -> Path:
    """Return the best available checkpoint. Phase G is preferred — it achieves
    the highest PESQ/STOI scores across all trained phases."""
    for p in [
        project_root / 'checkpoints_active/temporal_phaseG/best.pt',
        project_root / 'checkpoints_active/temporal_phaseF/best.pt',
        project_root / 'checkpoints_active/temporal_phaseC/best.pt',
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No trained checkpoint found in checkpoints_active/. "
        "Pass --checkpoint explicitly or run training first."
    )


def load_model(ckpt_path: Path, device):
    """Load NeuralAudioCodec from checkpoint, inferring architecture from saved state."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
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
        fixed_window_mask=ckpt.get('fixed_window_mask', False),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def encode_decode(model, audio: np.ndarray, sr: int, device, chunk_sec: float = 5.0):
    """3-bit encode→compress→decompress→decode pipeline. Returns (recon, kbps)."""
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


def compute_compression_stats(model, audio: np.ndarray, sr: int, device,
                              chunk_sec: float = 5.0) -> dict:
    """
    Run the encode→quantize→compress pipeline and return compression statistics.
    Does not decode — purely measures the latent representation structure.

    Returns a dict with:
      raw_bits        — theoretical size at 3 bits × frames × 32 dims
      compressed_bits — actual zlib output size in bits
      compression_ratio — raw / compressed  (>1 means zlib saved space)
      effective_kbps  — compressed_bits / audio duration
      theoretical_kbps — raw_bits / audio duration (always ~9.6 kbps)
      dim_entropy     — (32,) array: Shannon entropy per latent dimension
                        (max = 3.0 bits for uniform distribution over 8 levels)
      dim_utilisation — (32,) array: fraction of the 8 levels actually used per dim
    """
    num_levels = 8
    chunk_size  = int(chunk_sec * sr)
    duration    = len(audio) / sr

    raw_bits        = 0
    compressed_bits = 0
    all_q           = []   # accumulate quantized values for entropy calculation

    with torch.no_grad():
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            if len(chunk) < 160:
                continue
            x   = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            z   = model.encode(x)
            z_np = z.squeeze(0).cpu().numpy()          # (frames, dims)

            z_min  = float(z_np.min())
            z_max  = float(z_np.max())
            scale  = (z_max - z_min) / (num_levels - 1) + 1e-8
            q      = np.clip(np.round((z_np - z_min) / scale),
                             0, num_levels - 1).astype(np.uint8)

            raw_bits        += q.size * 3              # 3 bits per scalar
            compressed_bits += len(zlib.compress(q.tobytes(), level=9)) * 8
            all_q.append(q)                            # (frames, 32)

    q_all = np.concatenate(all_q, axis=0)             # (total_frames, 32)

    # Per-dimension Shannon entropy and level utilisation
    dim_entropy     = np.zeros(q_all.shape[1])
    dim_utilisation = np.zeros(q_all.shape[1])
    for d in range(q_all.shape[1]):
        counts = np.bincount(q_all[:, d], minlength=num_levels).astype(float)
        probs  = counts / counts.sum()
        nonzero = probs[probs > 0]
        dim_entropy[d]     = float(-np.sum(nonzero * np.log2(nonzero)))
        dim_utilisation[d] = float(np.sum(counts > 0)) / num_levels

    return {
        'raw_bits':          raw_bits,
        'compressed_bits':   compressed_bits,
        'compression_ratio': raw_bits / compressed_bits if compressed_bits > 0 else 0.0,
        'effective_kbps':    compressed_bits / duration / 1000,
        'theoretical_kbps':  raw_bits / duration / 1000,
        'dim_entropy':       dim_entropy,        # (32,) — bits per symbol per dim
        'dim_utilisation':   dim_utilisation,    # (32,) — fraction of levels used
    }


def compute_metrics(ref: np.ndarray, deg: np.ndarray, sr: int):
    """Returns (PESQ_WB, STOI) or (None, None) if libraries unavailable."""
    n = min(len(ref), len(deg))
    r = ref[:n] / (np.abs(ref[:n]).max() + 1e-8)
    d = deg[:n] / (np.abs(deg[:n]).max() + 1e-8)
    pesq = float(pesq_fn(sr, r, d, 'wb')) if PESQ_OK else None
    stoi = float(stoi_fn(r, d, sr, extended=False)) if STOI_OK else None
    return pesq, stoi


def avg(results: list, key: str) -> float:
    """Mean of results[key], skipping None."""
    v = [r[key] for r in results if r[key] is not None]
    return float(np.mean(v)) if v else float('nan')
