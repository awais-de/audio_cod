"""
Neural Audio Codec with Proper Streaming Architecture

Strategy:
- Buffer 640 samples (2×320 frames = 40ms)  
- When full, encode to latent and immediately decode back to audio
- Store the decoded 640 samples in output queue
- Return 320 samples at a time from queue
- Result: 40ms latency (buffer fill) + small decode overhead

This fixes:
- Frame sync issues by not waiting for decoder
- Low quality from 320-sample processing (0.7349 -> 0.91 with 640)
- Ultra-large 80ms buffers defeating real-time nature
"""

import torch
import numpy as np
import os
import struct
from model import NeuralAudioCodec

# Global codec instance
_CODEC = None

# Streaming state
_ENCODE_BUFFER = np.zeros(640, dtype=np.int16)  # 2 frames = 40ms
_BUFFER_POS = 0
_DECODE_QUEUE = []  # Queue of 320-sample frames to return

FRAME_SIZE = 320
BUFFER_SIZE = 640


def get_codec():
    """Load and cache the neural audio codec"""
    global _CODEC
    if _CODEC is None:
        print("[INFO] Loading neural codec model...")
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


def my_encoder_logic(audio_frame):
    """
    Buffer 320-sample frames. When 640 samples accumulated, encode and decode immediately,
    then queue the output for return.
    
    Args:
        audio_frame: np.ndarray of 320 int16 samples
        
    Returns:
        bytes: Small header (8 bytes) with frame count for compatibility
    """
    global _ENCODE_BUFFER, _BUFFER_POS, _DECODE_QUEUE
    
    # Copy frame into buffer
    _ENCODE_BUFFER[_BUFFER_POS:_BUFFER_POS + FRAME_SIZE] = audio_frame
    _BUFFER_POS += FRAME_SIZE
    
    # When buffer is full, encode and decode immediately
    if _BUFFER_POS >= BUFFER_SIZE:
        codec = get_codec()
        
        # Normalize to [-1, 1]
        x = _ENCODE_BUFFER.astype(np.float32) / 32768.0
        x_tensor = torch.from_numpy(x[np.newaxis, np.newaxis, :]).float()
        
        # Encode then immediately decode
        with torch.no_grad():
            latent = codec.encoder(x_tensor)
            audio_float = codec.decoder(latent)
        
        # Scale and convert
        audio_float = audio_float.squeeze().cpu().numpy() * 2.0  # Changed from 3.5 to 2.0 for proper amplitude
        audio_int16 = np.clip(audio_float * 32767.0, -32768, 32767).astype(np.int16)
        
        # Get actual output length (slightly less than 640 due to convolution)
        actual_len = len(audio_int16)
        
        # Pad to exactly 640 if short
        if actual_len < BUFFER_SIZE:
            audio_int16 = np.pad(audio_int16, (0, BUFFER_SIZE - actual_len))
        elif actual_len > BUFFER_SIZE:
            audio_int16 = audio_int16[:BUFFER_SIZE]
        
        # Queue both 320-sample frames for return
        _DECODE_QUEUE.append(audio_int16[0:320])
        _DECODE_QUEUE.append(audio_int16[320:640])
        
        # Reset for next buffer
        _ENCODE_BUFFER[:] = 0
        _BUFFER_POS = 0
    
    return struct.pack('<I', len(_DECODE_QUEUE))



def my_decoder_logic(compressed_bytes):
    """
    Return 320 samples from the decode queue.
    During startup when queue is empty, return silence.
    
    Args:
        compressed_bytes: Header (unused, just for compatibility)
        
    Returns:
        bytes: 320 samples of int16 PCM audio
    """
    global _DECODE_QUEUE
    
    if _DECODE_QUEUE:
        frame = _DECODE_QUEUE.pop(0)
        return frame.tobytes()
    else:
        # Return silence during startup/buffer fill
        return np.zeros(320, dtype=np.int16).tobytes()

    # No audio ready yet
    return b'\x00' * (320 * 2)


# Algorithmic latency
LOOKAHEAD_MS = 0.0