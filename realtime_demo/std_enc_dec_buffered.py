"""
Proper buffering codec that matches training segment length (6000 samples).

The model was trained on 6000-sample segments (0.375 seconds @ 16kHz).
Trying to process 320-sample frames (0.02 seconds) produces tiny outputs.

Solution: Buffer frames until we have 6000 samples, then encode.
"""

import torch
import numpy as np
import struct
from model import NeuralAudioCodec
import os

# Buffer to accumulate frames
_FRAME_BUFFER = np.zeros(6000, dtype=np.int16)
_BUFFER_POS = 0
_CODEC = None

FRAME_SIZE = 320  # Matches mainWrapper
BUFFER_SIZE = 6000  # Matches training
NUM_FRAMES_PER_BUFFER = BUFFER_SIZE // FRAME_SIZE  # 6000 / 320 = 18 frames


def get_codec():
    """Load the neural audio codec"""
    global _CODEC
    if _CODEC is None:
        model = NeuralAudioCodec(
            sample_rate=16000,
            hop_length=160,
            d_model=384,
            n_layers=6,
            n_heads=8,
            window_size=256,
            dropout=0.1
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_dir, 'best.pt')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        _CODEC = model
        print("[INFO] Model loaded successfully")
    return _CODEC


def reset_buffer():
    """Reset the frame buffer"""
    global _FRAME_BUFFER, _BUFFER_POS
    _FRAME_BUFFER = np.zeros(6000, dtype=np.int16)
    _BUFFER_POS = 0


def buffer_frame(audio_frame):
    """
    Add a 320-sample frame to the buffer.
    Returns compressed latent when buffer is full, None otherwise.
    
    Args:
        audio_frame: np.ndarray of 320 int16 samples
        
    Returns:
        bytes: Compressed latent (when buffer full), None (when accumulating)
    """
    global _FRAME_BUFFER, _BUFFER_POS
    
    # Copy frame into buffer
    _FRAME_BUFFER[_BUFFER_POS:_BUFFER_POS + FRAME_SIZE] = audio_frame
    _BUFFER_POS += FRAME_SIZE
    
    # When buffer is full, encode it
    if _BUFFER_POS >= BUFFER_SIZE:
        # Encode the full 6000-sample buffer
        codec = get_codec()
        
        # Normalize to [-1, 1]
        x = _FRAME_BUFFER.astype(np.float32) / 32768.0
        x_tensor = torch.from_numpy(x[np.newaxis, np.newaxis, :]).float()
        
        with torch.no_grad():
            latent = codec.encoder(x_tensor)  # (1, 745, 384)
        
        # Serialize
        compressed = latent.cpu().numpy().astype(np.float32).tobytes()
        
        # Reset buffer
        reset_buffer()
        
        return compressed
    
    return None


def decode_buffer(compressed_bytes):
    """
    Decode a full 6000-sample buffer and split into frames.
    
    Args:
        compressed_bytes: Serialized latent
        
    Returns:
        list: 18 frames of 320 samples each, or fewer if data is incomplete
    """
    codec = get_codec()
    
    # Deserialize: 1 * 745 * 384 * 4 = 1,140,480 bytes
    expected_size = 1 * 745 * 384 * 4
    if len(compressed_bytes) < expected_size:
        compressed_bytes = compressed_bytes + b'\x00' * (expected_size - len(compressed_bytes))
    
    latent = np.frombuffer(compressed_bytes[:expected_size], dtype=np.float32).reshape(1, 745, 384)
    latent_tensor = torch.from_numpy(latent.copy()).float()
    
    # Decode to get 6000 samples (actually 5960)
    with torch.no_grad():
        audio_float = codec.decoder(latent_tensor)  # (1, 1, 5960)
    
    # Denormalize and convert to int16
    audio_float = audio_float.squeeze().cpu().numpy()
    audio_int16 = np.clip(audio_float * 32767.0, -32768, 32767).astype(np.int16)
    
    # Pad to 6000 if needed
    if len(audio_int16) < 6000:
        audio_int16 = np.pad(audio_int16, (0, 6000 - len(audio_int16)), mode='constant')
    
    # Split into 320-sample frames (18 frames total)
    frames = []
    for i in range(0, 6000, 320):
        if i + 320 <= len(audio_int16):
            frames.append(audio_int16[i:i+320])
    
    return frames


# Compatibility with mainWrapper
def my_encoder_logic(audio_frame):
    """Buffer frame and return compressed data when ready"""
    return buffer_frame(audio_frame)


def my_decoder_logic(compressed_bytes):
    """Decode and return first frame from buffer"""
    frames = decode_buffer(compressed_bytes)
    if frames:
        return frames[0].tobytes()
    return b'\x00' * (320 * 2)  # Silent frame if decode fails
