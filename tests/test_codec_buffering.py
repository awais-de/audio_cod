#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import std_enc_dec
import numpy as np

print("Test 1: Encoder with 3 frames (should buffer)")
for i in range(3):
    t = np.linspace(0, 0.02, 320)
    frame = (np.sin(2*np.pi*440*t) * 15000).astype(np.int16)
    result = std_enc_dec.my_encoder_logic(frame)
    print(f"  Frame {i}: returned {len(result) if result else 0} bytes")

print("\nTest 2: Decoder with that data")
if result:
    out = std_enc_dec.my_decoder_logic(result)
    print(f"  Decoded: {len(out)} bytes")
    audio = np.frombuffer(out, dtype=np.int16)
    print(f"  Audio range: [{audio.min()}, {audio.max()}]")
else:
    print("  No data to decode")

print("\nTest 3: Fill 18 frames total")
for i in range(15):  # Already did 3
    t = np.linspace(0, 0.02, 320)
    frame = (np.sin(2*np.pi*440*t + i*0.1) * 15000).astype(np.int16)
    result = std_enc_dec.my_encoder_logic(frame)
    if i % 5 == 4:
        print(f"  Frame {i+3}: returned {len(result) if result else 0} bytes")

print("\n✅ All tests completed")
