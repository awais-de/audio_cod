"""
encoder_decoder.py - Neural Audio Codec Integration
Provides my_encoder_logic() and my_decoder_logic() for mainWrapper
"""

import torch
import numpy as np
import struct

# Configuration - MUST match mainWrapper config
MAX_INT16 = 32767
FRAME_SIZE = 320  # Must match mainV2.py config
LOOKAHEAD_MS = 0.0  # Neural codec has minimal lookahead

# Singleton codec instance
_CODEC_INSTANCE = None

def get_codec():
    """Initialize codec once and reuse"""
    global _CODEC_INSTANCE
    if _CODEC_INSTANCE is None:
        _CODEC_INSTANCE = NeuralCodec('best.pt')
    return _CODEC_INSTANCE


class NeuralCodec:
    def __init__(self, checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        
        from src.model import NeuralAudioCodec
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = NeuralAudioCodec(
            sample_rate=16000,
            hop_length=160,
            d_model=256,
            n_layers=4,
            n_heads=8,
            window_size=256,
            dropout=0.1
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()
        self.device = DEVICE
        print(f"✅ Model loaded on {DEVICE}")
    
    def encode(self, audio_int16):
        with torch.no_grad():
            audio_float = audio_int16.astype(np.float32) / MAX_INT16
            audio_tensor = torch.from_numpy(audio_float).to(self.device).unsqueeze(0).unsqueeze(0)
            latent = self.model.encode(audio_tensor)
            return latent.cpu().numpy()
    
    def decode(self, latent):
        with torch.no_grad():
            latent_tensor = torch.from_numpy(latent).to(self.device)
            audio_float = self.model.decode(latent_tensor)
            audio_np = audio_float.cpu().numpy().squeeze()
            audio_np = np.clip(audio_np, -1.0, 1.0)
            return (audio_np * MAX_INT16).astype(np.int16)
    
    @staticmethod
    def serialize_latent(latent_np):
        shape = latent_np.shape
        header = struct.pack('I', len(shape))
        for dim in shape:
            header += struct.pack('I', dim)
        data = latent_np.astype(np.float32).tobytes()
        return header + data
    
    @staticmethod
    def deserialize_latent(byte_data):
        offset = 0
        num_dims = struct.unpack('I', byte_data[offset:offset+4])[0]
        offset += 4
        
        shape = []
        for _ in range(num_dims):
            dim = struct.unpack('I', byte_data[offset:offset+4])[0]
            shape.append(dim)
            offset += 4
        
        data = byte_data[offset:]
        return np.frombuffer(data, dtype=np.float32).reshape(shape)


# ============================================================================
# FUNCTIONS FOR mainWrapper CustomCodec
# ============================================================================

def my_encoder_logic(audio_frame):
    """
    Called by mainWrapper.CustomCodec.encode()
    Input: audio_frame (NumPy array, dtype=np.int16, shape=(FRAME_SIZE,))
    Output: bytes object (compressed payload)
    """
    codec = get_codec()
    latent = codec.encode(audio_frame)
    return codec.serialize_latent(latent)


def my_decoder_logic(compressed_bytes):
    """
    Called by mainWrapper.CustomCodec.decode()
    Input: compressed_bytes (bytes object)
    Output: NumPy array (dtype=np.int16, shape=(FRAME_SIZE,))
    """
    codec = get_codec()
    latent = codec.deserialize_latent(compressed_bytes)
    audio_int16 = codec.decode(latent)
    
    # Ensure correct size
    if audio_int16.shape[0] < FRAME_SIZE:
        audio_int16 = np.pad(audio_int16, (0, FRAME_SIZE - audio_int16.shape[0]))
    elif audio_int16.shape[0] > FRAME_SIZE:
        audio_int16 = audio_int16[:FRAME_SIZE]
    
    return audio_int16



# Quick test
if __name__ == "__main__":
    print("Testing encoder/decoder functions...")
    codec = get_codec()
    
    test_audio = np.zeros(FRAME_SIZE, dtype=np.int16)
    compressed = my_encoder_logic(test_audio)
    decoded = my_decoder_logic(compressed)
    
    print(f"✅ Encoder: {test_audio.shape} → {len(compressed)} bytes ({FRAME_SIZE*2/len(compressed):.1f}x compression)")
    print(f"✅ Decoder: {len(compressed)} bytes → {decoded.shape} {decoded.dtype}")
    print("Ready for mainWrapper!")
