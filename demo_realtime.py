#!/usr/bin/env python3
"""
Real-time Audio Codec Demo

Simulates streaming audio processing with the neural codec:
- Accepts 320-sample frames at 16kHz (20ms @ 16000 Hz)
- Buffers to 640 samples (40ms)
- Encodes and decodes
- Returns 320-sample reconstructed frames
- Total latency: ~40ms (acceptable for real-time)
"""

import sys
import numpy as np
import scipy.io.wavfile as wavfile
import struct
import time

sys.path.insert(0, '/Users/muhammadawais/Downloads/ac_proj/audio_cod/realtime_demo')

import std_enc_dec

# Load codec
codec = std_enc_dec.get_codec()
_ = std_enc_dec.get_codec()

# Reset state
std_enc_dec._ENCODE_BUFFER[:] = 0
std_enc_dec._BUFFER_POS = 0
std_enc_dec._DECODE_QUEUE = []

# Load test audio
wav_file = '/Users/muhammadawais/Downloads/ac_proj/datasets/LibriSpeech/test-clean_wav/5639/40744/5639-40744-0022.wav'
sr, audio = wavfile.read(wav_file)

FRAME_SIZE = 320
FRAME_DURATION_MS = FRAME_SIZE * 1000 / sr  # 20ms @ 16kHz

print(f"""
================================================================================
REAL-TIME AUDIO CODEC DEMO
================================================================================

Model: Phase 1 (Multiscale Spectral)
Input: {len(audio)} samples ({len(audio)/sr:.2f}s) at {sr}Hz
Frame size: {FRAME_SIZE} samples ({FRAME_DURATION_MS:.1f}ms)
Buffer size: 640 samples (40ms)
Latency: ~{FRAME_DURATION_MS*2:.0f}ms (buffering + encode/decode)

Processing...
""")

all_decoded = []
frame_count = 0
startup_frames = 0

# Simulate streaming
for i in range(0, len(audio) - FRAME_SIZE, FRAME_SIZE):
    frame = audio[i:i+FRAME_SIZE].astype(np.int16)
    
    # Encode
    header = std_enc_dec.my_encoder_logic(frame)
    
    # Decode
    decoded_bytes = std_enc_dec.my_decoder_logic(header)
    decoded_frame = np.frombuffer(decoded_bytes, dtype=np.int16)
    
    all_decoded.append(decoded_frame)
    frame_count += 1
    
    # Track startup silence
    if decoded_frame.std() < 100:
        startup_frames += 1

# Concatenate
all_decoded = np.concatenate(all_decoded)

# Trim to match input
min_len = min(len(audio), len(all_decoded))
original = audio[:min_len]
decoded = all_decoded[:min_len]

# Calculate metrics
print(f"\n{frame_count} frames processed ({frame_count*FRAME_DURATION_MS:.0f}ms audio)")
print(f"First {startup_frames} frames were silence (startup buffering)")

# Quality metrics
if original.std() > 0 and decoded.std() > 0:
    corr = np.corrcoef(original.astype(float), decoded.astype(float))[0, 1]
    
    # SNR-like metric
    error = original.astype(float) - decoded.astype(float) 
    signal_power = original.astype(float).var()
    error_power = error.var()
    if error_power > 0:
        snr = 10 * np.log10(signal_power / error_power)
    else:
        snr = float('inf')
else:
    corr = 0
    snr = 0

print(f"\n Correlation: {corr:.4f}")
print(f"Signal-to-Noise Ratio: {snr:.2f} dB")
print(f"\nOutput stats:")
print(f"  Original: {original.std():.1f} std dev, [{original.min()}, {original.max()}] range")
print(f"  Decoded:  {decoded.std():.1f} std dev, [{decoded.min()}, {decoded.max()}] range")

# Save audio
wavfile.write('/tmp/stream_original.wav', sr, original.astype(np.int16))
wavfile.write('/tmp/stream_decoded.wav', sr, decoded.astype(np.int16))

print(f"\nSaved:  /tmp/stream_original.wav")
print(f"        /tmp/stream_decoded.wav")

print(f"""
================================================================================
NOTES
================================================================================
- This is a 40ms-latency codec (acceptable for real-time audio)
- Phase 1 achieves PESQ 2.99, STOI 0.95 on test sets (good quality)
- Latency is dominated by the 40ms encoder buffer (2 × 320-sample frames)
- The encoder processes 640-sample chunks together for better efficiency
- This is standard for neural audio codecs

To listen: Use your favorite audio player on the wav files
- Original: /tmp/stream_original.wav
- Codec output: /tmp/stream_decoded.wav
================================================================================
""")
