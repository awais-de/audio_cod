"""Shared loss functions, noise utilities, and datasets for codec training scripts."""

import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import IterableDataset


def pink_noise(n: int) -> np.ndarray:
    """Generate pink noise (1/f spectrum) of length n."""
    f = np.fft.rfftfreq(n)
    f[0] = 1.0
    spectrum = (np.random.randn(len(f)) + 1j * np.random.randn(len(f))) / np.sqrt(f)
    spectrum[0] = 0
    noise = np.fft.irfft(spectrum, n=n).astype(np.float32)
    return noise / (np.abs(noise).max() + 1e-8)


def add_noise(clean: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
    """Mix clean speech with noise at a given SNR (dB)."""
    signal_power = np.mean(clean ** 2) + 1e-8
    noise = pink_noise(len(clean)) if noise_type == 'pink' else np.random.randn(len(clean)).astype(np.float32)
    noise_power = np.mean(noise ** 2) + 1e-8
    noise = noise * np.sqrt(signal_power / (10 ** (snr_db / 10)) / noise_power)
    return np.clip(clean + noise, -1.0, 1.0).astype(np.float32)


def multi_scale_stft_loss(x_recon, x_target, fft_sizes=(256, 512, 1024), hop=160):
    """Linear magnitude multi-scale STFT loss (Phases A, B, C, D, D-VAE)."""
    if x_recon.ndim == 3:
        x_recon = x_recon.squeeze(1)
    if x_target.ndim == 3:
        x_target = x_target.squeeze(1)
    n = min(x_recon.shape[-1], x_target.shape[-1])
    x_recon, x_target = x_recon[..., :n], x_target[..., :n]
    total = torch.tensor(0.0, device=x_recon.device)
    for n_fft in fft_sizes:
        win = torch.hann_window(n_fft, device=x_recon.device)
        Sr = torch.stft(x_recon, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        St = torch.stft(x_target, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        Mr, Mt = torch.abs(Sr), torch.abs(St)
        total = total + torch.mean((Mr - Mt) ** 2) + torch.mean(torch.abs(Mr - Mt))
    return total / len(fft_sizes)


def log_scale_stft_loss(x_recon, x_target, fft_sizes=(256, 512, 1024), hop=160):
    """Log-magnitude multi-scale STFT loss (Phase E).

    Uses log1p magnitude instead of linear: compresses dynamic range so quiet
    frequency components (consonants, sibilants) receive proportional gradients.
    """
    if x_recon.ndim == 3:
        x_recon = x_recon.squeeze(1)
    if x_target.ndim == 3:
        x_target = x_target.squeeze(1)
    n = min(x_recon.shape[-1], x_target.shape[-1])
    x_recon, x_target = x_recon[..., :n], x_target[..., :n]
    total = torch.tensor(0.0, device=x_recon.device)
    for n_fft in fft_sizes:
        win = torch.hann_window(n_fft, device=x_recon.device)
        Sr = torch.stft(x_recon, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        St = torch.stft(x_target, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        # log-magnitude: compresses dynamic range, equalises gradient across frequencies
        Mr = torch.log1p(torch.abs(Sr))
        Mt = torch.log1p(torch.abs(St))
        total = total + torch.mean((Mr - Mt) ** 2) + torch.mean(torch.abs(Mr - Mt))
    return total / len(fft_sizes)


class CombinedSpectralLoss(nn.Module):
    """
    Triple loss: linear STFT + log STFT + log mel spectrogram (Phases F, G).

    Linear STFT: penalises large spectral errors, stable gradients for dominant
                 speech energy (same as Phase C).
    Log STFT:    compresses dynamic range so quiet components (fricatives,
                 consonants, high-frequency detail) get proportional gradients.
    Log Mel:     80-band mel filterbank matches human auditory frequency
                 resolution — formant frequencies (500-3000 Hz) weighted more
                 than inaudible bands.

    All three averaged to keep total loss comparable in scale to Phase C.
    """
    def __init__(self, sample_rate=16000, fft_sizes=(256, 512, 1024), hop=160, n_mels=80):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop = hop
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=max(fft_sizes),
            hop_length=hop,
            n_mels=n_mels,
            f_min=0.0,
            f_max=float(sample_rate) / 2,
        )

    def forward(self, x_recon, x_target):
        if x_recon.ndim == 3:
            x_recon = x_recon.squeeze(1)
        if x_target.ndim == 3:
            x_target = x_target.squeeze(1)
        n = min(x_recon.shape[-1], x_target.shape[-1])
        x_recon, x_target = x_recon[..., :n], x_target[..., :n]

        lin_total = torch.tensor(0.0, device=x_recon.device)
        log_total = torch.tensor(0.0, device=x_recon.device)

        for n_fft in self.fft_sizes:
            win = torch.hann_window(n_fft, device=x_recon.device)
            Sr = torch.stft(x_recon, n_fft=n_fft, hop_length=self.hop, window=win, return_complex=True)
            St = torch.stft(x_target, n_fft=n_fft, hop_length=self.hop, window=win, return_complex=True)
            Mr, Mt = torch.abs(Sr), torch.abs(St)
            lin_total = lin_total + torch.mean((Mr - Mt) ** 2) + torch.mean(torch.abs(Mr - Mt))
            Lr, Lt = torch.log1p(Mr), torch.log1p(Mt)
            log_total = log_total + torch.mean((Lr - Lt) ** 2) + torch.mean(torch.abs(Lr - Lt))

        lin_loss = lin_total / len(self.fft_sizes)
        log_loss = log_total / len(self.fft_sizes)

        mel_r = torch.log1p(self.mel.to(x_recon.device)(x_recon))
        mel_t = torch.log1p(self.mel.to(x_target.device)(x_target))
        mel_loss = torch.mean((mel_r - mel_t) ** 2) + torch.mean(torch.abs(mel_r - mel_t))

        return (lin_loss + log_loss + mel_loss) / 3.0


def ste_quantize_3bit(z: torch.Tensor) -> torch.Tensor:
    """3-bit STE quantization (Straight-Through Estimator)."""
    num_levels = 8
    z_min, z_max = z.min(), z.max()
    scale = (z_max - z_min) / (num_levels - 1) + 1e-8
    z_norm = (z - z_min) / scale
    z_int = torch.clamp(torch.round(z_norm), 0, num_levels - 1)
    z_quant = z_int * scale + z_min
    return z + (z_quant - z).detach()


def uniform_noise_quantize(z: torch.Tensor) -> torch.Tensor:
    """Differentiable proxy for 3-bit uniform quantization (Phase D).

    Injects U(-0.5, 0.5) * scale noise — same distribution as quantization error.
    """
    num_levels = 8
    z_min, z_max = z.min(), z.max()
    scale = (z_max - z_min) / (num_levels - 1) + 1e-8
    noise = torch.empty_like(z).uniform_(-0.5, 0.5) * scale
    return torch.clamp(z + noise, z_min, z_max)


class AudioChunkDataset(IterableDataset):
    """Plain audio chunks — no noise augmentation (Phases A, B)."""
    def __init__(self, data_root, chunk_seconds=1.0, sample_rate=16000, epoch_size=1000):
        self.chunk_size = int(chunk_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.epoch_size = epoch_size
        exts = ('.wav', '.flac', '.mp3', '.ogg')
        self.files = sorted(p for p in Path(data_root).rglob('*') if p.suffix.lower() in exts)
        if not self.files:
            raise ValueError(f"No audio files in {data_root}")
        print(f"Dataset: {len(self.files)} files")

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for _ in range(self.epoch_size):
            path = self.files[np.random.randint(0, len(self.files))]
            try:
                audio, sr = sf.read(path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != self.sample_rate:
                    n = int(len(audio) * self.sample_rate / sr)
                    audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
                if len(audio) > self.chunk_size:
                    start = np.random.randint(0, len(audio) - self.chunk_size)
                    chunk = audio[start:start + self.chunk_size]
                else:
                    chunk = np.pad(audio, (0, self.chunk_size - len(audio)))
                yield torch.FloatTensor(np.clip(chunk, -1.0, 1.0).astype(np.float32)).unsqueeze(0)
            except Exception:
                continue


class NoisyAudioDataset(IterableDataset):
    """Audio chunks with random noise augmentation (Phases C, D, D-VAE, E, F, G)."""
    def __init__(self, data_root, chunk_seconds=1.0, sample_rate=16000,
                 epoch_size=1000, noise_prob=0.6, snr_range=(5, 20)):
        self.chunk_size = int(chunk_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.epoch_size = epoch_size
        self.noise_prob = noise_prob
        self.snr_min, self.snr_max = snr_range
        exts = ('.wav', '.flac', '.mp3', '.ogg')
        self.files = sorted(p for p in Path(data_root).rglob('*') if p.suffix.lower() in exts)
        if not self.files:
            raise ValueError(f"No audio files in {data_root}")
        print(f"Dataset: {len(self.files)} files  |  noise_prob={noise_prob}  "
              f"SNR={snr_range[0]}-{snr_range[1]}dB")

    def _load_chunk(self, path):
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            n = int(len(audio) * self.sample_rate / sr)
            audio = np.interp(np.linspace(0, len(audio), n), np.arange(len(audio)), audio)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        if len(audio) > self.chunk_size:
            start = np.random.randint(0, len(audio) - self.chunk_size)
            return audio[start:start + self.chunk_size]
        return np.pad(audio, (0, self.chunk_size - len(audio)))

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for _ in range(self.epoch_size):
            path = self.files[np.random.randint(0, len(self.files))]
            try:
                chunk = self._load_chunk(path)
                if np.random.random() < self.noise_prob:
                    snr = np.random.uniform(self.snr_min, self.snr_max)
                    noise_type = np.random.choice(['white', 'pink', 'babble'])
                    if noise_type == 'babble':
                        try:
                            babble = self._load_chunk(self.files[np.random.randint(0, len(self.files))])
                            babble_power = np.mean(babble ** 2) + 1e-8
                            sig_power = np.mean(chunk ** 2) + 1e-8
                            babble_scaled = babble * np.sqrt(sig_power / (10 ** (snr / 10)) / babble_power)
                            chunk = np.clip(chunk + babble_scaled, -1.0, 1.0).astype(np.float32)
                        except Exception:
                            chunk = add_noise(chunk, 'white', snr)
                    else:
                        chunk = add_noise(chunk, noise_type, snr)
                yield torch.FloatTensor(chunk).unsqueeze(0)
            except Exception:
                continue


def measure_real_bitrate(model, audio_files, device, n_files=5, chunk_samples=16000) -> float:
    """Measure actual compressed bitrate on real audio files."""
    model.eval()
    total_bits, total_dur = 0, 0.0
    num_levels = 8
    with torch.no_grad():
        for path in audio_files[:n_files]:
            try:
                audio, sr = sf.read(path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
                for start in range(0, len(audio), chunk_samples):
                    chunk = audio[start:start + chunk_samples]
                    if len(chunk) < 160:
                        continue
                    x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
                    z = model.encode(x).squeeze(0).cpu().numpy()
                    z_min, z_max = z.min(), z.max()
                    scale = (z_max - z_min) / (num_levels - 1) + 1e-8
                    q = np.clip(np.round((z - z_min) / scale), 0, num_levels - 1).astype(np.uint8)
                    total_bits += len(zlib.compress(q.tobytes(), level=9)) * 8
                    total_dur += len(chunk) / sr
            except Exception:
                continue
    return total_bits / total_dur / 1000.0 if total_dur > 0 else float('nan')
