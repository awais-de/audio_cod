#!/usr/bin/env python3
import std_enc_dec
import numpy as np
import struct

print("=== Testing with 3200-sample Buffer ===")

# Create 3200 samples
all_samples = np.array([])
for frame_num in range(11):
    t = np.linspace(0, 0.02, 320)
    frame = (np.sin(2*np.pi*440*t + frame_num*0.3) * 20000).astype(np.int16)
    all_samples = np.concatenate([all_samples, frame])

all_samples = all_samples[:3200]
print(f"Input: {len(all_samples)} samples")

# Feed frames
compressed = None
for i in range(11):
    frame = all_samples[i*320:(i+1)*320].astype(np.int16)
    if len(frame) == 0:
        break
    result = std_enc_dec.my_encoder_logic(frame)
    if result is not None:
        compressed = result
        print(f"Buffer filled at frame {i}: {len(compressed)} bytes")
        break

if compressed:
    decoded_frames = []
    for i in range(10):
        frame_bytes = std_enc_dec.my_decoder_logic(compressed)
        decoded_frames.append(np.frombuffer(frame_bytes, dtype=np.int16))
    
    decoded = np.concatenate(decoded_frames)
    print(f"Decoded: {len(decoded)} samples, range [{decoded.min()}, {decoded.max()}]")
    
    corr = np.corrcoef(all_samples[:len(decoded)], decoded)[0,1]
    print(f"Correlation: {corr:.4f}")
    print(f"\n✅ 3200-sample buffer compressed to {len(compressed)} bytes")
else:
    print("ERROR: No data")
