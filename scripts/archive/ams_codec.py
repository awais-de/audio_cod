#!/usr/bin/env python3
"""
AMS Platform Wrapper for Neural Audio Codec

Implements the required interface:
- my_encoder_logic(audio_frame: np.ndarray) -> bytes
- my_decoder_logic(compressed_bytes: bytes) -> np.ndarray

Assumptions:
- Input audio_frame is mono float32 at 16 kHz. If stereo, the first channel is used.
- Chunk/frame size is arbitrary; encoder handles any length. Latent size is inferred from byte length.

Dependencies: torch, numpy
Optional: torchaudio (only used if available for resampling)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.model import NeuralAudioCodec

# Globals
_model: Optional[NeuralAudioCodec] = None
_device: Optional[torch.device] = None
_d_model: int = 384  # default for best_pesq_finetune checkpoint; will be updated from checkpoint if available


def _load_model(checkpoint_path: str = "checkpoints_emergency/best_pesq_finetune.pt", device: Optional[str] = None):
    """Load codec model once and cache it."""
    global _model, _device, _d_model
    if _model is not None:
        return

    # Check for GPU availability, fall back to CPU if not available
    if device:
        _device = torch.device(device)
    else:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {_device}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=False)
    _d_model = checkpoint.get("d_model", 384)
    n_layers = checkpoint.get("n_layers", 6)
    n_heads = checkpoint.get("n_heads", 8)

    _model = NeuralAudioCodec(d_model=_d_model, n_layers=n_layers, n_heads=n_heads).to(_device)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()


def _to_mono(audio_frame: np.ndarray) -> np.ndarray:
    """Ensure mono by selecting the first channel if needed."""
    if audio_frame.ndim == 1:
        return audio_frame
    if audio_frame.ndim == 2:
        return audio_frame[:, 0]
    raise ValueError(f"Unexpected audio_frame ndim: {audio_frame.ndim}")


def _resample_if_needed(audio: torch.Tensor, input_sr: int, target_sr: int = 16000) -> torch.Tensor:
    """Resample using torchaudio if available; otherwise return original."""
    if input_sr == target_sr:
        return audio
    if _HAS_TORCHAUDIO:
        return torchaudio.functional.resample(audio, input_sr, target_sr)
    # Fallback: naive numpy-based resample (linear interpolation)
    ratio = target_sr / input_sr
    orig = audio.squeeze(0).cpu().numpy()
    x_old = np.linspace(0, 1, orig.shape[-1])
    x_new = np.linspace(0, 1, int(orig.shape[-1] * ratio))
    resampled = np.interp(x_new, x_old, orig).astype(np.float32)
    return torch.from_numpy(resampled).unsqueeze(0)


def my_encoder_logic(audio_frame: np.ndarray, input_sr: int = 16000) -> bytes:
    """Encode a raw audio frame to compressed bytes.

    Args:
        audio_frame: np.ndarray of shape (samples,) or (samples, channels), float32/float64
        input_sr: sample rate of input frame (default 16 kHz)
    Returns:
        bytes representing latent tensor in float16
    """
    _load_model()

    # Ensure float32 and mono
    audio_np = _to_mono(np.asarray(audio_frame, dtype=np.float32))
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # shape (1, samples)

    # Resample if needed
    audio_tensor = _resample_if_needed(audio_tensor, input_sr, 16000)

    # Normalize to prevent clipping
    peak = audio_tensor.abs().max().item() + 1e-8
    audio_tensor = audio_tensor / peak

    # Add channel dim: (1, 1, samples)
    audio_tensor = audio_tensor.unsqueeze(0).to(_device)

    with torch.no_grad():
        latent = _model.encoder(audio_tensor)           # (1, seq, d_model)
        latent = latent.transpose(1, 2).contiguous()    # -> (1, d_model, seq)
        latent_np = latent.cpu().half().numpy()         # float16 for bandwidth
        return latent_np.tobytes()


def my_decoder_logic(compressed_bytes: bytes) -> np.ndarray:
    """Decode compressed bytes back to audio samples.

    Args:
        compressed_bytes: bytes produced by my_encoder_logic
    Returns:
        np.ndarray of shape (samples,) in float32 at 16 kHz
    """
    _load_model()

    # Reconstruct latent tensor shape: (1, d_model, seq_len)
    bytes_per_elem = 2  # float16
    total_elems = len(compressed_bytes) // bytes_per_elem
    seq_len = total_elems // _d_model
    latent_np = np.frombuffer(compressed_bytes, dtype=np.float16).reshape(1, _d_model, seq_len).copy()
    latent = torch.from_numpy(latent_np).float().to(_device)
    latent = latent.transpose(1, 2).contiguous()  # (1, seq, d_model) for decoder

    with torch.no_grad():
        audio = _model.decoder(latent)  # (1, 1, samples)
        audio_np = audio.squeeze().cpu().numpy().astype(np.float32)
        return audio_np


if __name__ == "__main__":
    # Simple self-test using random noise (does not require audio hardware)
    import time
    dummy = np.random.randn(16000).astype(np.float32)  # 1s of audio at 16 kHz
    t0 = time.time()
    data = my_encoder_logic(dummy)
    t1 = time.time()
    rec = my_decoder_logic(data)
    t2 = time.time()
    print(f"Encoded {len(dummy)} samples to {len(data)} bytes in {(t1 - t0)*1000:.2f} ms")
    print(f"Decoded back to {rec.shape[0]} samples in {(t2 - t1)*1000:.2f} ms")
    print(f"Round-trip peak: {np.abs(rec).max():.4f}")
