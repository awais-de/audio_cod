#!/usr/bin/env python3
"""Test the buffered codec"""
import numpy as np
import std_enc_dec

# Create exactly 6000 samples (need to add 240 more samples)
# 18 * 320 = 5760, need 6000, so add 2 more frames partially
all_samples = np.array([])
for frame_num in range(19):  # 19 frames = 6080 samples, then truncate to 6000
    t = np.linspace(0, 0.02, 320)
    frame = (np.sin(2*np.pi*440*t + frame_num*0.3) * 20000).astype(np.int16)
    all_samples = np.concatenate([all_samples, frame])

all_samples = all_samples[:6000]  # Truncate to 6000

print(f"Input: {len(all_samples)} samples, range [{int(all_samples.min())}, {int(all_samples.max())}]")

# Feed frames through encoder (will return compressed when buffer reaches 6000)
compressed = None
num_frames_sent = 0
for i in range(19):  # Send 19 frames to fill buffer (19*320 = 6080)
    frame = all_samples[i*320:(i+1)*320].astype(np.int16)
    if len(frame) == 0:
        break
    result = std_enc_dec.my_encoder_logic(frame)
    num_frames_sent += 1
    if result is not None:
        compressed = result
        print(f"Frame {i}: Encoder returned {len(result)} bytes of compressed latent (buffer full!)")
        break

print(f"Sent {num_frames_sent} frames, buffer size = {num_frames_sent * 320}")

if not compressed:
    print("ERROR: Buffer never filled! Check buffer logic.")

if compressed:
    # Decode and collect frames
    all_decoded = []
    for i in range(18):
        frame_bytes = std_enc_dec.my_decoder_logic(compressed)
        frame_audio = np.frombuffer(frame_bytes, dtype=np.int16)
        all_decoded.append(frame_audio)
    
    all_decoded = np.concatenate(all_decoded)
    print(f"\nDecoded: {len(all_decoded)} samples, range [{int(all_decoded.min())}, {int(all_decoded.max())}]")
    
    # Compare
    corr = np.corrcoef(all_samples[:len(all_decoded)], all_decoded)[0, 1]
    mae = np.mean(np.abs(all_samples.astype(float)[:len(all_decoded)] - all_decoded.astype(float)))
    print(f"Correlation: {corr:.4f}")
    print(f"Mean absolute error: {mae:.2f}")
    
    if corr > 0.9:
        print("\n✅ CODEC IS NOW WORKING! Audio should sound good.")
    else:
        print("\n❌ Still issues with codec")
else:
    print("ERROR: No compressed data")
