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
        """
        Args:
            num_bits: Bits per coefficient (1-8)
        """
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.scale = None
    
    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Quantize tensor to num_bits precision
        
        Args:
            x: latent tensor (any shape)
            
        Returns:
            q: quantized indices (0 to 2^num_bits - 1)
            scale: scaling factor for dequantization
        """
        x_flat = x.flatten()
        x_min = np.min(x_flat)
        x_max = np.max(x_flat)
        
        # Avoid division by zero
        if np.abs(x_max - x_min) < 1e-6:
            x_max = x_min + 1.0
        
        scale = (x_max - x_min) / (self.num_levels - 1)
        q = np.round((x_flat - x_min) / scale).astype(np.uint8 if self.num_bits <= 8 else np.uint16)
        q = np.clip(q, 0, self.num_levels - 1)
        
        return q.reshape(x.shape), scale, x_min
    
    def dequantize(self, q: np.ndarray, scale: float, x_min: float) -> np.ndarray:
        """
        Dequantize back to float
        
        Args:
            q: quantized indices
            scale: scaling factor
            x_min: minimum value used in quantization
            
        Returns:
            x_recon: reconstructed float tensor
        """
        x_recon = q.astype(np.float32) * scale + x_min
        return x_recon


class VectorQuantizer:
    """Simple vector quantizer (codebook-based)"""
    
    def __init__(self, num_bits: int = 8, vector_dim: int = 16):
        """
        Args:
            num_bits: Bits per vector (codebook size = 2^num_bits)
            vector_dim: Dimension of each vector
        """
        self.num_bits = num_bits
        self.codebook_size = 2 ** num_bits
        self.vector_dim = vector_dim
        self.codebook = None
    
    def fit(self, X: np.ndarray):
        """
        Learn codebook via k-means on training data
        
        Args:
            X: training latents (N, D)
        """
        from sklearn.cluster import KMeans
        
        X_flat = X.reshape(-1, self.vector_dim)
        kmeans = KMeans(n_clusters=self.codebook_size, n_init=10, max_iter=100)
        kmeans.fit(X_flat)
        self.codebook = kmeans.cluster_centers_
    
    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize using nearest codebook entry
        
        Args:
            x: latent tensor (any shape)
            
        Returns:
            indices: codebook indices
            x_quant: quantized vectors
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.vector_dim)
        
        # Find nearest codebook entry
        distances = np.linalg.norm(x_flat[:, None, :] - self.codebook[None, :, :], axis=2)
        indices = np.argmin(distances, axis=1)
        x_quant = self.codebook[indices]
        
        return indices.reshape(shape[:-1]), x_quant.reshape(shape)


class BitrateController:
    """Control bitrate via quantization level"""
    
    @staticmethod
    def bits_per_frame(target_bitrate_kbps: int, frame_duration_ms: float = 20.0) -> int:
        """
        Calculate total bits available per frame
        
        Args:
            target_bitrate_kbps: Target bitrate in kbps
            frame_duration_ms: Frame duration in milliseconds
            
        Returns:
            bits: Total bits available per frame
        """
        return int((target_bitrate_kbps * 1000) * (frame_duration_ms / 1000.0))
    
    @staticmethod
    def required_bits_per_coefficient(
        total_bits: int,
        latent_dim: int,
        overhead_bits: int = 16
    ) -> float:
        """
        Calculate bits per coefficient after overhead
        
        Args:
            total_bits: Total bits per frame
            latent_dim: Latent dimensionality
            overhead_bits: Metadata overhead (e.g., scale, min)
            
        Returns:
            bits_per_coeff: Bits per coefficient
        """
        available_bits = total_bits - overhead_bits
        return available_bits / latent_dim
    
    @staticmethod
    def required_num_bits(bits_per_coeff: float) -> int:
        """
        Round bits per coefficient to nearest integer
        
        Args:
            bits_per_coeff: Bits per coefficient (float)
            
        Returns:
            num_bits: Integer bits to use
        """
        return max(1, min(8, int(np.round(bits_per_coeff))))


class QuantizedLatentCodec:
    """Quantize + entropy code latent vectors"""
    
    def __init__(self, target_bitrate_kbps: int = 10, latent_dim: int = 128, 
                 frame_duration_ms: float = 20.0, use_vq: bool = False):
        """
        Args:
            target_bitrate_kbps: Target bitrate
            latent_dim: Latent dimensionality
            frame_duration_ms: Frame duration
            use_vq: Use vector quantization (requires training)
        """
        self.target_bitrate_kbps = target_bitrate_kbps
        self.latent_dim = latent_dim
        self.frame_duration_ms = frame_duration_ms
        self.use_vq = use_vq
        
        # Calculate bitrate constraint
        total_bits = BitrateController.bits_per_frame(target_bitrate_kbps, frame_duration_ms)
        bits_per_coeff = BitrateController.required_bits_per_coefficient(total_bits, latent_dim)
        self.num_bits = BitrateController.required_num_bits(bits_per_coeff)
        
        self.quantizer = UniformQuantizer(num_bits=self.num_bits)
        print(f"[QuantizedLatentCodec] Target: {target_bitrate_kbps} kbps")
        print(f"  Bits per frame: {total_bits}")
        print(f"  Bits per coefficient: {bits_per_coeff:.2f}")
        print(f"  Using {self.num_bits}-bit quantization")
    
    def compress(self, latent_np: np.ndarray) -> bytes:
        """
        Quantize and entropy code latent
        
        Args:
            latent_np: Latent array (any shape)
            
        Returns:
            compressed_bytes: Compressed bitstream
        """
        # Quantize
        q, scale, x_min = self.quantizer.quantize(latent_np)
        
        # Metadata: shape, scale, min
        metadata = {
            'shape': latent_np.shape,
            'scale': float(scale),
            'x_min': float(x_min),
            'num_bits': self.num_bits,
        }
        
        # Entropy code (zlib)
        q_bytes = q.tobytes()
        q_compressed = zlib.compress(q_bytes, level=9)
        
        # Package: metadata (text) + compressed data
        import json
        header = json.dumps(metadata).encode('utf-8')
        header_len = len(header).to_bytes(4, byteorder='big')
        
        return header_len + header + q_compressed
    
    def decompress(self, compressed_bytes: bytes) -> np.ndarray:
        """
        Entropy decode and dequantize latent
        
        Args:
            compressed_bytes: Compressed bitstream
            
        Returns:
            latent_np: Reconstructed latent
        """
        import json
        
        # Unpack header
        header_len = int.from_bytes(compressed_bytes[:4], byteorder='big')
        header = json.loads(compressed_bytes[4:4+header_len].decode('utf-8'))
        q_compressed = compressed_bytes[4+header_len:]
        
        # Entropy decode
        q_bytes = zlib.decompress(q_compressed)
        dtype = np.uint8 if self.num_bits <= 8 else np.uint16
        q = np.frombuffer(q_bytes, dtype=dtype).reshape(header['shape'])
        
        # Dequantize
        latent_np = self.quantizer.dequantize(q, header['scale'], header['x_min'])
        
        return latent_np
    
    def bitrate_achieved(self, original_size: int, compressed_size: int, duration_s: float) -> float:
        """
        Calculate actual achieved bitrate
        
        Args:
            original_size: Size of original latents (bytes)
            compressed_size: Size of compressed data (bytes)
            duration_s: Duration of audio (seconds)
            
        Returns:
            bitrate_bps: Achieved bitrate in bits/second
        """
        bits = compressed_size * 8
        return bits / duration_s
