import numpy as np
import std_enc_dec_buffered as codec

# Simulate 18 frames (6000 samples total) of audio
all_audio = np.array([])
for frame_num in range(18):
    t = np.linspace(0, 0.02, 320)
    frame = (np.sin(2*np.pi*440*t + frame_num*6.28) * 20000).astype(np.int16)
    all_audio = np.concatenate([all_audio, frame])

print(f'Input: {len(all_audio)} samples, range [{int(all_audio.min())}, {int(all_audio.max())}]')

# Feed frames to buffer
compressed = None
for i in range(18):
    frame = all_audio[i*320:(i+1)*320].astype(np.int16)
    result = codec.my_encoder_logic(frame)
    if result is not None:
        compressed = result
        print(f'Frame {i}: Encoder returned {len(result)} bytes')

if compressed:
    frames_out = codec.decode_buffer(compressed)
    print(f'Decoder: Got {len(frames_out)} frames')
    if frames_out:
        recon = np.concatenate(frames_out[:18])
        print(f'Reconstructed: {len(recon)} samples, range [{int(recon.min())}, {int(recon.max())}]')
        
        # Compare
        corr = np.corrcoef(all_audio[:len(recon)], recon)[0,1]
        print(f'Correlation: {corr:.4f}')
else:
    print('ERROR: No compressed data returned!')
