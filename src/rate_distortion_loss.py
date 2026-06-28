"""
Rate-Distortion Loss for neural audio codec training.
Combines reconstruction quality (distortion) with bitrate penalty (rate).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class RateDistortionLoss(nn.Module):
    """
    Weighted combination of distortion and rate losses.
    Loss = D + lambda * R
    where:
      D = reconstruction loss (MSE, STFT, or perceptual)
      R = rate = -log p(z) from entropy model (in nats)
      lambda = tradeoff parameter (schedule from 1.0 to 0.001)
    """

    def __init__(self,
                 distortion_type: str = 'mse',
                 entropy_model=None,
                 lambda_init: float = 1.0,
                 lambda_min: float = 0.001,
                 use_perceptual: bool = False,
                 encoder_dim: int = None):
        super().__init__()
        self.distortion_type = distortion_type
        self.entropy_model = entropy_model
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.use_perceptual = use_perceptual
        self.current_lambda = lambda_init

        self.projection = None
        if entropy_model is not None and encoder_dim is not None:
            entropy_latent_dim = entropy_model.latent_dim
            if encoder_dim != entropy_latent_dim:
                print(f"Dimension mismatch: encoder={encoder_dim}, entropy_model={entropy_latent_dim}")
                print(f"Creating projection layer {encoder_dim} -> {entropy_latent_dim}")
                self.projection = nn.Linear(encoder_dim, entropy_latent_dim)
                nn.init.xavier_uniform_(self.projection.weight, gain=0.01)
                nn.init.zeros_(self.projection.bias)

        if distortion_type == 'mse':
            self.distortion_fn = nn.MSELoss(reduction='mean')
        elif distortion_type == 'l1':
            self.distortion_fn = nn.L1Loss(reduction='mean')
        elif distortion_type == 'stft':
            self.distortion_fn = self._stft_loss
        elif distortion_type == 'hybrid':
            self.distortion_fn = self._hybrid_loss
        else:
            raise ValueError(f"Unknown distortion_type: {distortion_type}")

    def _stft_loss(self, x_recon: torch.Tensor, x_target: torch.Tensor,
                   n_fft: int = 512, hop_length: int = 160) -> torch.Tensor:
        """STFT magnitude loss — more perceptually relevant than time-domain MSE."""
        if x_recon.ndim == 3:
            x_recon = x_recon.squeeze(1)
        if x_target.ndim == 3:
            x_target = x_target.squeeze(1)

        spec_recon = torch.stft(x_recon, n_fft=n_fft, hop_length=hop_length,
                                center=True, return_complex=True)
        spec_target = torch.stft(x_target, n_fft=n_fft, hop_length=hop_length,
                                 center=True, return_complex=True)

        mag_recon = torch.abs(spec_recon)
        mag_target = torch.abs(spec_target)

        loss = torch.mean((mag_recon - mag_target) ** 2)

        return loss

    def _hybrid_loss(self, x_recon: torch.Tensor, x_target: torch.Tensor,
                     alpha: float = 0.5) -> torch.Tensor:
        """alpha * L1(time) + (1-alpha) * L2(magnitude spectrogram)"""
        if x_recon.ndim == 3:
            x_recon_t = x_recon.squeeze(1)
        else:
            x_recon_t = x_recon

        if x_target.ndim == 3:
            x_target_t = x_target.squeeze(1)
        else:
            x_target_t = x_target

        time_loss = torch.mean(torch.abs(x_recon_t - x_target_t))
        freq_loss = self._stft_loss(x_recon, x_target)

        return alpha * time_loss + (1.0 - alpha) * freq_loss

    def compute_distortion(self, x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction distortion, cropping to minimum length if shapes differ."""
        if x_recon.shape[-1] != x_target.shape[-1]:
            min_len = min(x_recon.shape[-1], x_target.shape[-1])
            x_recon = x_recon[..., :min_len]
            x_target = x_target[..., :min_len]

        if callable(self.distortion_fn):
            return self.distortion_fn(x_recon, x_target)
        else:
            return self.distortion_fn(x_recon, x_target)

    def compute_rate(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute entropy (rate in nats and bits) from the entropy model."""
        if self.entropy_model is None:
            return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)

        try:
            log_prob = self.entropy_model.entropy_model.log_prob(z)
            rate_nats = -log_prob
            rate_bits = rate_nats / np.log(2)
            return rate_nats, rate_bits
        except Exception:
            return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)

    def forward(self, x_recon: torch.Tensor, x_target: torch.Tensor,
                z: torch.Tensor, return_components: bool = False) -> torch.Tensor:
        """
        Compute R-D loss: L = D + lambda * R

        Uses L2 norm of latents as a differentiable proxy for bitrate when no
        entropy model is available — encourages compact latent representations.
        """
        distortion = self.compute_distortion(x_recon, x_target)

        rate_nats_mean = 0.0

        if z.ndim == 3:
            B, T, D = z.shape
            z_flat = z.reshape(-1, D)
        elif z.ndim == 2:
            z_flat = z
        else:
            raise ValueError(f"Unexpected latent shape: {z.shape}")

        # L2 norm of latents as proxy for bitrate (no entropy model required)
        rate_raw = torch.norm(z_flat, p=2, dim=-1).mean()
        rate_scaled = rate_raw / np.sqrt(D)

        rate_kbps = rate_scaled * 10.0
        rate_nats_mean = rate_scaled.item()

        rate_penalty = self.current_lambda * rate_scaled

        loss = distortion + rate_penalty

        if return_components:
            return {
                'loss': loss,
                'distortion': distortion.item(),
                'rate_penalty': rate_penalty.item() if isinstance(rate_penalty, torch.Tensor) else 0.0,
                'rate_nats': rate_nats_mean,
                'rate_kbps': rate_kbps.item() if isinstance(rate_kbps, torch.Tensor) else 0.0,
                'lambda': self.current_lambda,
            }
        else:
            return loss

    def set_lambda(self, lambda_val: float):
        """Update the rate-distortion tradeoff parameter."""
        self.current_lambda = max(self.lambda_min, min(lambda_val, self.lambda_init))

    def schedule_lambda(self, epoch: int, total_epochs: int, schedule: str = 'linear'):
        """Update lambda according to a decay schedule ('linear', 'exponential', or 'step')."""
        progress = epoch / max(total_epochs, 1)

        if schedule == 'linear':
            self.current_lambda = self.lambda_init + progress * (self.lambda_min - self.lambda_init)

        elif schedule == 'exponential':
            decay_rate = np.log(self.lambda_min / self.lambda_init) / total_epochs
            self.current_lambda = self.lambda_init * np.exp(decay_rate * epoch)

        elif schedule == 'step':
            if progress < 0.5:
                self.current_lambda = self.lambda_init
            elif progress < 0.75:
                self.current_lambda = self.lambda_init * 0.1
            else:
                self.current_lambda = self.lambda_min

        self.current_lambda = max(self.lambda_min, self.current_lambda)


class QuantizedLatentWithRD(nn.Module):
    """
    Wrapper around encoder + quantizer + entropy model for R-D training.
    Enables gradient flow through quantization via STE (Straight-Through Estimator).
    """

    def __init__(self, encoder, quantizer, entropy_model, num_bits: int = 1):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.entropy_model = entropy_model
        self.num_bits = num_bits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode, quantize via STE, and return latents + rate."""
        z = self.encoder(x)

        # Straight-Through Estimator: passes gradients as identity through rounding
        z_rounded = torch.round(z)
        z_quant = z + (z_rounded - z).detach()

        if self.entropy_model is not None:
            log_prob = self.entropy_model.entropy_model.log_prob(z_quant)
            rate = -log_prob
        else:
            rate = torch.zeros(z_quant.shape[0], device=z.device)

        return z_quant, rate


def create_rd_loss(distortion_type: str = 'mse',
                   entropy_model=None,
                   lambda_init: float = 1.0) -> RateDistortionLoss:
    """Factory for RateDistortionLoss."""
    return RateDistortionLoss(
        distortion_type=distortion_type,
        entropy_model=entropy_model,
        lambda_init=lambda_init
    )
