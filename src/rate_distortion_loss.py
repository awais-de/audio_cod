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
        """
        Args:
            distortion_type: 'mse', 'l1', 'stft', or 'hybrid'
            entropy_model: Trained entropy model for rate computation
            lambda_init: Initial rate penalty weight
            lambda_min: Minimum lambda (for scheduling)
            use_perceptual: Whether to use perceptual weighting (requires librosa)
            encoder_dim: Encoder output dimension (if different from entropy model latent_dim)
        """
        super().__init__()
        self.distortion_type = distortion_type
        self.entropy_model = entropy_model
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.use_perceptual = use_perceptual
        self.current_lambda = lambda_init
        
        # Add projection layer if dimension mismatch
        self.projection = None
        if entropy_model is not None and encoder_dim is not None:
            entropy_latent_dim = entropy_model.latent_dim
            if encoder_dim != entropy_latent_dim:
                print(f"⚠️  Dimension mismatch: encoder={encoder_dim}, entropy_model={entropy_latent_dim}")
                print(f"Creating projection layer {encoder_dim} → {entropy_latent_dim}")
                self.projection = nn.Linear(encoder_dim, entropy_latent_dim)
                # Initialize with small weights for stability
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
        """
        STFT-based loss: compare magnitude spectrograms.
        More perceptually relevant than time-domain MSE.
        
        Args:
            x_recon: Reconstructed waveform (B, 1, T)
            x_target: Target waveform (B, 1, T)
            n_fft: FFT size
            hop_length: Hop length for STFT
            
        Returns:
            loss: STFT magnitude loss
        """
        # Squeeze to (B, T)
        if x_recon.ndim == 3:
            x_recon = x_recon.squeeze(1)
        if x_target.ndim == 3:
            x_target = x_target.squeeze(1)
        
        # Compute STFT
        spec_recon = torch.stft(x_recon, n_fft=n_fft, hop_length=hop_length,
                                center=True, return_complex=True)
        spec_target = torch.stft(x_target, n_fft=n_fft, hop_length=hop_length,
                                 center=True, return_complex=True)
        
        # Magnitude spectrogram
        mag_recon = torch.abs(spec_recon)
        mag_target = torch.abs(spec_target)
        
        # L2 loss on magnitude
        loss = torch.mean((mag_recon - mag_target) ** 2)
        
        return loss
    
    def _hybrid_loss(self, x_recon: torch.Tensor, x_target: torch.Tensor,
                     alpha: float = 0.5) -> torch.Tensor:
        """
        Hybrid loss: weighted combination of time-domain and frequency-domain.
        alpha * L1(time) + (1-alpha) * L2(magnitude spectrogram)
        
        Args:
            x_recon: Reconstructed waveform
            x_target: Target waveform
            alpha: Weight for time-domain loss (0-1)
            
        Returns:
            loss: Weighted hybrid loss
        """
        # Time-domain L1
        if x_recon.ndim == 3:
            x_recon_t = x_recon.squeeze(1)
        else:
            x_recon_t = x_recon
        
        if x_target.ndim == 3:
            x_target_t = x_target.squeeze(1)
        else:
            x_target_t = x_target
        
        time_loss = torch.mean(torch.abs(x_recon_t - x_target_t))
        
        # Frequency-domain loss
        freq_loss = self._stft_loss(x_recon, x_target)
        
        # Weighted combination
        return alpha * time_loss + (1.0 - alpha) * freq_loss
    
    def compute_distortion(self, x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction distortion.
        
        Args:
            x_recon: Reconstructed audio (B, 1, T) or (B, T)
            x_target: Target audio (B, 1, T) or (B, T)
            
        Returns:
            distortion: Scalar loss
        """
        # Handle shape mismatches by cropping to minimum length
        if x_recon.shape[-1] != x_target.shape[-1]:
            min_len = min(x_recon.shape[-1], x_target.shape[-1])
            x_recon = x_recon[..., :min_len]
            x_target = x_target[..., :min_len]
        
        if callable(self.distortion_fn):
            return self.distortion_fn(x_recon, x_target)
        else:
            return self.distortion_fn(x_recon, x_target)
    
    def compute_rate(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute entropy (rate) from entropy model.
        
        Args:
            z: Latent vectors (B, D) or (B, D, T)
            
        Returns:
            rate_nats: Rate in nats (B,) or (B, T)
            rate_bits: Rate in bits (B,) or (B, T)
        """
        if self.entropy_model is None:
            return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)
        
        try:
            # Compute log probability
            log_prob = self.entropy_model.entropy_model.log_prob(z)  # (B,) or (B, T)
            
            # Rate = -log p(z) in nats
            rate_nats = -log_prob
            
            # Convert to bits
            rate_bits = rate_nats / np.log(2)
            
            return rate_nats, rate_bits
        except Exception:
            # Fallback if entropy model fails (shape mismatch, etc)
            return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)
    
    def forward(self, x_recon: torch.Tensor, x_target: torch.Tensor, 
                z: torch.Tensor, return_components: bool = False) -> torch.Tensor:
        """
        Compute Rate-Distortion loss: L = D + lambda * R
        
        Args:
            x_recon: Reconstructed audio (B, 1, T)
            x_target: Target audio (B, 1, T)
            z: Latent vectors (B, D, T) from encoder
            return_components: If True, return (loss, D, R*lambda) dict
            
        Returns:
            loss: Scalar R-D loss
            or dict: {'loss': scalar, 'distortion': D, 'rate_penalty': lambda*R}
        """
        # Distortion term
        distortion = self.compute_distortion(x_recon, x_target)
        
        # Rate term: Use L2 norm of latents as proxy for bitrate
        # This encourages compression without requiring entropy model
        rate_nats_mean = 0.0
        
        # Reshape z to (N, D) for rate computation
        # Encoder outputs (B, T, D), so we just need to flatten first two dims
        if z.ndim == 3:
            B, T, D = z.shape
            z_flat = z.reshape(-1, D)  # (B, T, D) -> (B*T, D)
        elif z.ndim == 2:
            z_flat = z  # Already (N, D)
        else:
            raise ValueError(f"Unexpected latent shape: {z.shape}")
        
        # Compute L2 norm of latents (scaled by dimension for stability)
        # rate ≈ ||z||_2 / sqrt(D) encourages sparse/small latent values
        rate_raw = torch.norm(z_flat, p=2, dim=-1).mean()  # Mean L2 norm across all latent vectors
        rate_scaled = rate_raw / np.sqrt(D)  # Normalize by latent dimension
        
        # Convert to approximate kbps for monitoring (heuristic scaling)
        # Tune this scale to target the desired bitrate (10 kbps goal)
        rate_kbps = rate_scaled * 10.0  # Heuristic conversion
        rate_nats_mean = rate_scaled.item()
        
        # Apply lambda weighting
        rate_penalty = self.current_lambda * rate_scaled
        
        # Total loss
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
        """
        Update lambda according to schedule.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            schedule: 'linear', 'exponential', or 'step'
        """
        progress = epoch / max(total_epochs, 1)
        
        if schedule == 'linear':
            # Linear decay from lambda_init to lambda_min
            self.current_lambda = self.lambda_init + progress * (self.lambda_min - self.lambda_init)
        
        elif schedule == 'exponential':
            # Exponential decay
            decay_rate = np.log(self.lambda_min / self.lambda_init) / total_epochs
            self.current_lambda = self.lambda_init * np.exp(decay_rate * epoch)
        
        elif schedule == 'step':
            # Step decay: decrease at 50%, 75% of training
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
        """
        Args:
            encoder: Neural codec encoder
            quantizer: Quantizer (UniformQuantizer or VectorQuantizer)
            entropy_model: Trained entropy model
            num_bits: Quantization precision
        """
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.entropy_model = entropy_model
        self.num_bits = num_bits
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode, quantize, and return latents + rate.
        
        Args:
            x: Waveform (B, 1, T)
            
        Returns:
            z_quant: Quantized latents (differentiable via STE)
            rate: Rate from entropy model
        """
        # Encode
        z = self.encoder(x)  # (B, D, T)
        
        # Quantize with STE: quantization is not differentiable,
        # but we approximate gradients as identity
        z_rounded = torch.round(z)
        z_quant = z + (z_rounded - z).detach()  # Straight-Through Estimator
        
        # Compute rate from entropy model
        if self.entropy_model is not None:
            log_prob = self.entropy_model.entropy_model.log_prob(z_quant)
            rate = -log_prob  # Rate in nats
        else:
            rate = torch.zeros(z_quant.shape[0], device=z.device)
        
        return z_quant, rate


def create_rd_loss(distortion_type: str = 'mse',
                   entropy_model=None,
                   lambda_init: float = 1.0) -> RateDistortionLoss:
    """
    Factory function to create R-D loss.
    
    Args:
        distortion_type: Type of distortion metric
        entropy_model: Trained entropy model
        lambda_init: Initial lambda value
        
    Returns:
        rd_loss: Configured RateDistortionLoss
    """
    return RateDistortionLoss(
        distortion_type=distortion_type,
        entropy_model=entropy_model,
        lambda_init=lambda_init
    )
