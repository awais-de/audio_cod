"""
Quantization and entropy coding for neural audio codec.
Implements rate-distortion constrained compression.
"""

import numpy as np
import torch
import torch.nn as nn
import zlib
from typing import Tuple


class UniformQuantizer:
    """Uniform scalar quantizer for latent tensors"""

    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.scale = None

    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantize tensor to num_bits precision, returning indices and scale."""
        x_flat = x.flatten()
        x_min = np.min(x_flat)
        x_max = np.max(x_flat)

        if np.abs(x_max - x_min) < 1e-6:
            x_max = x_min + 1.0

        scale = (x_max - x_min) / (self.num_levels - 1)
        q = np.round((x_flat - x_min) / scale).astype(np.uint8 if self.num_bits <= 8 else np.uint16)
        q = np.clip(q, 0, self.num_levels - 1)

        return q.reshape(x.shape), scale, x_min

    def dequantize(self, q: np.ndarray, scale: float, x_min: float) -> np.ndarray:
        """Reconstruct float tensor from quantized indices."""
        x_recon = q.astype(np.float32) * scale + x_min
        return x_recon


class VectorQuantizer:
    """Simple vector quantizer (codebook-based)"""

    def __init__(self, num_bits: int = 8, vector_dim: int = 16):
        self.num_bits = num_bits
        self.codebook_size = 2 ** num_bits
        self.vector_dim = vector_dim
        self.codebook = None

    def fit(self, X: np.ndarray):
        """Learn codebook via k-means on training data (N, D)."""
        from sklearn.cluster import KMeans

        X_flat = X.reshape(-1, self.vector_dim)
        kmeans = KMeans(n_clusters=self.codebook_size, n_init=10, max_iter=100)
        kmeans.fit(X_flat)
        self.codebook = kmeans.cluster_centers_

    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize using nearest codebook entry."""
        shape = x.shape
        x_flat = x.reshape(-1, self.vector_dim)

        distances = np.linalg.norm(x_flat[:, None, :] - self.codebook[None, :, :], axis=2)
        indices = np.argmin(distances, axis=1)
        x_quant = self.codebook[indices]

        return indices.reshape(shape[:-1]), x_quant.reshape(shape)


class BitrateController:
    """Control bitrate via quantization level"""

    @staticmethod
    def bits_per_frame(target_bitrate_kbps: int, frame_duration_ms: float = 20.0) -> int:
        """Total bits available per frame at the given bitrate."""
        return int((target_bitrate_kbps * 1000) * (frame_duration_ms / 1000.0))

    @staticmethod
    def required_bits_per_coefficient(
        total_bits: int,
        latent_dim: int,
        overhead_bits: int = 16
    ) -> float:
        """Bits per latent coefficient after metadata overhead."""
        available_bits = total_bits - overhead_bits
        return available_bits / latent_dim

    @staticmethod
    def required_num_bits(bits_per_coeff: float) -> int:
        """Round bits-per-coefficient to the nearest integer in [1, 8]."""
        return max(1, min(8, int(np.round(bits_per_coeff))))


class QuantizedLatentCodec:
    """Quantize + entropy code latent vectors"""

    def __init__(self, target_bitrate_kbps: int = 10, latent_dim: int = 128,
                 frame_duration_ms: float = 20.0, use_vq: bool = False):
        self.target_bitrate_kbps = target_bitrate_kbps
        self.latent_dim = latent_dim
        self.frame_duration_ms = frame_duration_ms
        self.use_vq = use_vq

        total_bits = BitrateController.bits_per_frame(target_bitrate_kbps, frame_duration_ms)
        bits_per_coeff = BitrateController.required_bits_per_coefficient(total_bits, latent_dim)
        self.num_bits = BitrateController.required_num_bits(bits_per_coeff)

        self.quantizer = UniformQuantizer(num_bits=self.num_bits)
        print(f"[QuantizedLatentCodec] Target: {target_bitrate_kbps} kbps")
        print(f"  Bits per frame: {total_bits}")
        print(f"  Bits per coefficient: {bits_per_coeff:.2f}")
        print(f"  Using {self.num_bits}-bit quantization")

    def compress(self, latent_np: np.ndarray) -> bytes:
        """Quantize and entropy-code latent, returning a compressed bytestring."""
        q, scale, x_min = self.quantizer.quantize(latent_np)

        metadata = {
            'shape': latent_np.shape,
            'scale': float(scale),
            'x_min': float(x_min),
            'num_bits': self.num_bits,
        }

        q_bytes = q.tobytes()
        q_compressed = zlib.compress(q_bytes, level=9)

        import json
        header = json.dumps(metadata).encode('utf-8')
        header_len = len(header).to_bytes(4, byteorder='big')

        return header_len + header + q_compressed

    def decompress(self, compressed_bytes: bytes) -> np.ndarray:
        """Entropy-decode and dequantize, returning the reconstructed latent."""
        import json

        header_len = int.from_bytes(compressed_bytes[:4], byteorder='big')
        header = json.loads(compressed_bytes[4:4+header_len].decode('utf-8'))
        q_compressed = compressed_bytes[4+header_len:]

        q_bytes = zlib.decompress(q_compressed)
        dtype = np.uint8 if self.num_bits <= 8 else np.uint16
        q = np.frombuffer(q_bytes, dtype=dtype).reshape(header['shape'])

        latent_np = self.quantizer.dequantize(q, header['scale'], header['x_min'])

        return latent_np

    def bitrate_achieved(self, original_size: int, compressed_size: int, duration_s: float) -> float:
        """Achieved bitrate in bits/second."""
        bits = compressed_size * 8
        return bits / duration_s
