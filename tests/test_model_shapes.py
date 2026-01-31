import torch
from model import NeuralAudioCodec

model = NeuralAudioCodec(d_model=384, n_layers=6, n_heads=8, window_size=256, dropout=0.1)

# Test with training segment (6000 samples)
test_audio = torch.randn(1, 1, 6000)
with torch.no_grad():
    latent = model.encoder(test_audio)
    print(f'6000 samples -> Encoder: {latent.shape}')
    reconstructed = model.decoder(latent)
    print(f'6000 samples -> Decoder: {reconstructed.shape}')
    print(f'Decoder output range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]')

# Test with 320 samples
test_audio_small = torch.randn(1, 1, 320)
with torch.no_grad():
    latent_small = model.encoder(test_audio_small)
    print(f'\n320 samples -> Encoder: {latent_small.shape}')
    reconstructed_small = model.decoder(latent_small)
    print(f'320 samples -> Decoder: {reconstructed_small.shape}')
    print(f'Decoder output range: [{reconstructed_small.min():.3f}, {reconstructed_small.max():.3f}]')
