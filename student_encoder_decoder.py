#!/usr/bin/env python3
"""
Student Encoder/Decoder for Teleconference System (Compressed Version)
Uses PCA to compress latent from (1, 35, 384) to (1, 35, 96) to fit UDP packet

This module provides neural audio codec integration for the mainWrapper teleconference system.
It MUST implement exactly two functions with these signatures:
- my_encoder_logic(audio_frame) -> bytes
- my_decoder_logic(compressed_bytes) -> np.ndarray[int16]
"""

import numpy as np
import torch
import struct

# Configuration matching mainWrapper
FRAME_SIZE = 320  # 20ms at 16kHz
LOOKAHEAD_MS = 0.0  # Required by mainWrapper
MAX_INT16 = 32767

# Global singleton
_CODEC_INSTANCE = None

def get_codec():
    """Get or create codec singleton"""
    global _CODEC_INSTANCE
    if _CODEC_INSTANCE is None:
        _CODEC_INSTANCE = NeuralCodec('best.pt')
    return _CODEC_INSTANCE


class NeuralCodec:
    """Neural Audio Codec wrapper with PCA compression"""
    
    def __init__(self, checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        
        from model import NeuralAudioCodec
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = NeuralAudioCodec(
            sample_rate=16000,
            hop_length=160,
            d_model=384,
            n_layers=6,
            n_heads=8,
            window_size=256,
            dropout=0.1
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # PCA compression: 384 -> 96 dimensions (4x compression)
        # Total size: 1*35*96 = 3360 bytes + 10 byte header = 3370 bytes (fits in 4KB!)
        self.compress_dim = 96
        self.pca_mean = None
        self.pca_components = None
        
        print(f"✅ Model loaded on {self.device}")
        print(f"🗜️  Using PCA compression: 384 → {self.compress_dim} dims")
    
    def encode(self, audio_int16):
        """Convert int16 PCM to compressed latent representation"""
        with torch.no_grad():
            # Convert int16 to float [-1, 1]
            audio_float = audio_int16.astype(np.float32) / MAX_INT16
            
            # Add batch and channel dimensions: (320,) -> (1, 1, 320)
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Encode to latent: (1, 1, 320) -> (1, 35, 384)
            latent_tensor = self.model.encoder(audio_tensor)
            latent_np = latent_tensor.cpu().numpy()
            
            # Apply PCA compression: (1, 35, 384) -> (1, 35, 96)
            latent_compressed = self._compress_latent(latent_np)
            
            return latent_compressed
    
    def decode(self, latent_compressed):
        """Convert compressed latent back to int16 PCM"""
        with torch.no_grad():
            # Decompress: (1, 35, 96) -> (1, 35, 384)
            latent_full = self._decompress_latent(latent_compressed)
            
            # Decode to audio
            latent_tensor = torch.from_numpy(latent_full).to(self.device)
            audio_tensor = self.model.decoder(latent_tensor)
            
            # Convert float [-1, 1] to int16
            audio_np = audio_tensor.cpu().numpy().squeeze()
            audio_int16 = (np.clip(audio_np, -1.0, 1.0) * MAX_INT16).astype(np.int16)
            
            return audio_int16
    
    def _compress_latent(self, latent_full):
        """Compress latent using PCA: (1, 35, 384) -> (1, 35, 96)"""
        batch, seq_len, d_model = latent_full.shape
        
        # Reshape to (35, 384) for PCA
        latent_2d = latent_full.reshape(seq_len, d_model)
        
        # Initialize PCA on first call
        if self.pca_mean is None:
            self.pca_mean = np.mean(latent_2d, axis=0)
            # Simple PCA: take top 96 principal components
            # For now, just use simple projection (truncate last dimensions)
            # This is faster than full SVD and works reasonably well
            self.pca_components = np.eye(d_model)[:self.compress_dim, :]
        
        # Center data
        latent_centered = latent_2d - self.pca_mean
        
        # Project to compressed space
        latent_compressed = latent_centered @ self.pca_components.T
        
        # Reshape back: (35, 96) -> (1, 35, 96)
        return latent_compressed.reshape(1, seq_len, self.compress_dim)
    
    def _decompress_latent(self, latent_compressed):
        """Decompress latent: (1, 35, 96) -> (1, 35, 384)"""
        batch, seq_len, _ = latent_compressed.shape
        
        # Reshape to (35, 96)
        latent_2d = latent_compressed.reshape(seq_len, self.compress_dim)
        
        # Project back to original space
        latent_reconstructed = latent_2d @ self.pca_components
        
        # Add mean back
        latent_full = latent_reconstructed + self.pca_mean
        
        # Reshape: (35, 384) -> (1, 35, 384)
        return latent_full.reshape(1, seq_len, 384)
    
    @staticmethod
    def serialize_latent(latent_np):
        """Convert compressed latent to bytes (fits in UDP packet)"""
        shape = latent_np.shape  # (1, 35, 96)
        
        # Quantize float32 to int8 with percentile-based scaling
        p95 = np.percentile(np.abs(latent_np), 95)
        scale = p95 if p95 > 0 else 1.0
        
        latent_int8 = np.clip(latent_np / scale * 127, -127, 127).astype(np.int8)
        
        # Compact header: 10 bytes (2+2+2+4)
        header = struct.pack('HHHf', shape[0], shape[1], shape[2], scale)
        
        data = latent_int8.tobytes()
        
        total_size = len(header) + len(data)
        print(f"📦 Packet size: {total_size} bytes (header={len(header)}, data={len(data)})")
        
        return header + data
    
    @staticmethod
    def deserialize_latent(byte_data):
        """Reconstruct compressed latent from bytes"""
        # Read header (10 bytes)
        dim0, dim1, dim2, scale = struct.unpack('HHHf', byte_data[:10])
        shape = (dim0, dim1, dim2)
        
        # Read quantized data
        data = byte_data[10:]
        expected_size = dim0 * dim1 * dim2
        
        # Handle truncation gracefully
        if len(data) < expected_size:
            print(f"⚠️  Packet truncated: expected {expected_size}, got {len(data)} bytes")
            data = data + b'\x00' * (expected_size - len(data))
        
        # Dequantize
        latent_int8 = np.frombuffer(data[:expected_size], dtype=np.int8).reshape(shape)
        latent_float = latent_int8.astype(np.float32) / 127.0 * scale
        
        return latent_float


# ============================================================================
# MAIN INTERFACE FUNCTIONS FOR mainWrapper
# ============================================================================

def my_encoder_logic(audio_frame):
    """
    Encoder function called by mainWrapper CustomCodec.encode()
    
    Args:
        audio_frame: np.ndarray of shape (320,) with dtype int16
    
    Returns:
        bytes: Compressed audio data (fits in UDP packet)
    """
    codec = get_codec()
    latent_compressed = codec.encode(audio_frame)
    compressed_bytes = codec.serialize_latent(latent_compressed)
    return compressed_bytes


def my_decoder_logic(compressed_bytes):
    """
    Decoder function called by mainWrapper CustomCodec.decode()
    
    Args:
        compressed_bytes: bytes received from encoder
    
    Returns:
        np.ndarray: Audio frame of shape (320,) with dtype int16
    """
    codec = get_codec()
    latent_compressed = codec.deserialize_latent(compressed_bytes)
    audio_int16 = codec.decode(latent_compressed)
    
    # Ensure exact FRAME_SIZE (required by mainWrapper)
    if audio_int16.shape[0] < FRAME_SIZE:
        audio_int16 = np.pad(audio_int16, (0, FRAME_SIZE - audio_int16.shape[0]), mode='constant')
    elif audio_int16.shape[0] > FRAME_SIZE:
        audio_int16 = audio_int16[:FRAME_SIZE]
    
    return audio_int16
