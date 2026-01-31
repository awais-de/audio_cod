"""
Simple codec test without audio hardware
"""
import numpy as np
import std_enc_dec
import wave

print("="*60)
print("Testing Neural Audio Codec")
print("="*60)

# Generate a test signal: 440Hz sine wave for 20ms (320 samples at 16kHz)
sample_rate = 16000
duration_ms = 20
num_samples = 320

t = np.linspace(0, duration_ms/1000, num_samples, endpoint=False)
frequency = 440  # A4 note
test_audio = (np.sin(2 * np.pi * frequency * t) * 32000).astype(np.int16)

print(f"\n1. Original Signal:")
print(f"   - Samples: {len(test_audio)}")
print(f"   - Range: [{test_audio.min()}, {test_audio.max()}]")
print(f"   - RMS: {np.sqrt(np.mean(test_audio**2)):.1f}")

# Encode
print(f"\n2. Encoding...")
compressed = std_enc_dec.my_encoder_logic(test_audio)
print(f"   - Compressed size: {len(compressed)} bytes")
print(f"   - Original size: {len(test_audio) * 2} bytes")
print(f"   - Compression ratio: {(len(test_audio) * 2) / len(compressed):.2f}:1")

# Decode
print(f"\n3. Decoding...")
reconstructed = std_enc_dec.my_decoder_logic(compressed)
print(f"   - Samples: {len(reconstructed)}")
print(f"   - Range: [{reconstructed.min()}, {reconstructed.max()}]")
print(f"   - RMS: {np.sqrt(np.mean(reconstructed**2)):.1f}")

# Quality metrics
correlation = np.corrcoef(test_audio, reconstructed)[0, 1]
mse = np.mean((test_audio - reconstructed) ** 2)
snr = 10 * np.log10(np.mean(test_audio**2) / (mse + 1e-10))

print(f"\n4. Quality Metrics:")
print(f"   - Correlation: {correlation:.4f} ({correlation*100:.1f}%)")
print(f"   - SNR: {snr:.2f} dB")
print(f"   - MSE: {mse:.1f}")

# Save to WAV files for manual listening (optional)
print(f"\n5. Saving WAV files...")
with wave.open('test_original.wav', 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(test_audio.tobytes())
print(f"   - test_original.wav")

with wave.open('test_reconstructed.wav', 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(reconstructed.tobytes())
print(f"   - test_reconstructed.wav")

print(f"\n{'='*60}")
if correlation > 0.9:
    print("✅ SUCCESS! Codec is working well (correlation > 90%)")
elif correlation > 0.7:
    print("⚠️  PARTIAL: Codec works but quality could be better")
else:
    print("❌ FAILED: Codec quality is poor")
print(f"{'='*60}")
print("\nYou can play the WAV files with: open test_original.wav test_reconstructed.wav")
