"""
Learned entropy models for neural audio codec latents.
Implements Gaussian Mixture Models (GMM) for encoding latent distributions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import warnings


class GaussianMixtureModel(nn.Module):
    """
    Learnable Gaussian Mixture Model for modeling latent distribution.
    Used for entropy coding via learned probability distribution.
    """
    
    def __init__(self, latent_dim: int, num_components: int = 8):
        """
        Args:
            latent_dim: Dimension of latent space
            num_components: Number of Gaussian components in mixture
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        
        # Log mixture weights (before softmax)
        self.register_parameter(
            'log_mixture_weights',
            nn.Parameter(torch.zeros(num_components))
        )
        
        # Per-component means
        self.register_parameter(
            'component_means',
            nn.Parameter(torch.randn(num_components, latent_dim) * 0.1)
        )
        
        # Per-component log-variances (for numerical stability)
        self.register_parameter(
            'component_log_vars',
            nn.Parameter(torch.zeros(num_components, latent_dim))
        )
    
    def mixture_weights(self) -> torch.Tensor:
        """Get normalized mixture weights (K,)"""
        return F.softmax(self.log_mixture_weights, dim=0)
    
    def log_prob_per_component(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under each Gaussian component.
        
        Args:
            z: Latent tensor (*, D) where D = latent_dim
            
        Returns:
            log_probs_per_comp: (*, K) log probability under each component
        """
        original_shape = z.shape
        z_flat = z.reshape(-1, self.latent_dim)  # (B, D)
        
        means = self.component_means  # (K, D)
        log_vars = self.component_log_vars  # (K, D)
        vars = torch.exp(log_vars)
        
        # Compute (z - mean)^2 / var for each component
        # Shape: (B, K, D)
        z_expanded = z_flat.unsqueeze(1)  # (B, 1, D)
        diff = z_expanded - means.unsqueeze(0)  # (B, K, D)
        
        # (z - mean)^2 / var
        mahal = (diff ** 2) / vars.unsqueeze(0)  # (B, K, D)
        
        # Log determinant: sum of log(var) per dimension
        const = 0.5 * self.latent_dim * np.log(2 * np.pi)
        log_det = 0.5 * torch.sum(log_vars, dim=1)  # (K,)
        
        # log p(z|k) = -const - log_det - 0.5 * mahal
        log_prob = -const - log_det.unsqueeze(0) - 0.5 * mahal.sum(dim=2)  # (B, K)
        
        return log_prob.reshape(*original_shape[:-1], self.num_components)
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under the mixture model: log p(z) = log sum_k w_k * p(z|k)
        
        Args:
            z: Latent tensor (*, D)
            
        Returns:
            log_prob: (*,) log probability
        """
        log_probs_per_comp = self.log_prob_per_component(z)  # (*, K)
        weights = self.mixture_weights()  # (K,)
        
        # log sum exp trick: log(sum w_k * p_k) = log(sum exp(log w_k + log p_k))
        log_weights_expanded = torch.log(weights).reshape(
            *([1] * (log_probs_per_comp.ndim - 1)), self.num_components
        )
        
        log_weighted_probs = log_weights_expanded + log_probs_per_comp
        
        # Numerically stable log_sum_exp
        max_val = log_weighted_probs.max(dim=-1, keepdim=True)[0]
        log_sum = max_val + torch.log(
            torch.sum(torch.exp(log_weighted_probs - max_val), dim=-1, keepdim=True)
        )
        
        return log_sum.squeeze(-1)
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample from the mixture model.
        
        Args:
            num_samples: Number of samples to draw
            
        Returns:
            samples: (num_samples, latent_dim) samples
        """
        with torch.no_grad():
            weights = self.mixture_weights()  # (K,)
            component_ids = torch.multinomial(weights, num_samples, replacement=True)  # (N,)
            
            means = self.component_means  # (K, D)
            log_vars = self.component_log_vars  # (K, D)
            stds = torch.exp(0.5 * log_vars)  # (K, D)
            
            selected_means = means[component_ids]  # (N, D)
            selected_stds = stds[component_ids]  # (N, D)
            
            noise = torch.randn(num_samples, self.latent_dim, device=means.device)
            samples = selected_means + selected_stds * noise
            
            return samples


class ConditionalGMM(nn.Module):
    """
    Factorized Gaussian Mixture Model where each dimension has independent mixture.
    More expressive than a single global GMM, more efficient than full covariance.
    """
    
    def __init__(self, latent_dim: int, num_components: int = 8):
        """
        Args:
            latent_dim: Dimension of latent space
            num_components: Number of Gaussian components per dimension
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        
        # Per-dimension mixture model
        # Shape: (latent_dim, num_components)
        self.register_parameter(
            'log_mixture_weights',
            nn.Parameter(torch.zeros(latent_dim, num_components))
        )
        
        self.register_parameter(
            'component_means',
            nn.Parameter(torch.randn(latent_dim, num_components) * 0.1)
        )
        
        self.register_parameter(
            'component_log_vars',
            nn.Parameter(torch.zeros(latent_dim, num_components))
        )
    
    def mixture_weights(self) -> torch.Tensor:
        """Get normalized mixture weights (D, K)"""
        return F.softmax(self.log_mixture_weights, dim=1)
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability assuming factorized form: log p(z) = sum_d log p(z_d)
        
        Args:
            z: Latent tensor (*, D)
            
        Returns:
            log_prob: (*,) log probability
        """
        original_shape = z.shape
        z_flat = z.reshape(-1, self.latent_dim)  # (B, D)
        
        log_prob_total = torch.zeros(z_flat.shape[0], device=z.device)
        
        for d in range(self.latent_dim):
            z_d = z_flat[:, d:d+1]  # (B, 1)
            means_d = self.component_means[d]  # (K,)
            log_vars_d = self.component_log_vars[d]  # (K,)
            weights_d = F.softmax(self.log_mixture_weights[d], dim=0)  # (K,)
            
            # p(z_d|k) for each component
            vars_d = torch.exp(log_vars_d)
            mahal = (z_d - means_d.unsqueeze(0)) ** 2 / vars_d.unsqueeze(0)  # (B, K)
            
            const = 0.5 * np.log(2 * np.pi)
            log_probs_per_comp = -const - 0.5 * log_vars_d.unsqueeze(0) - 0.5 * mahal  # (B, K)
            
            # log p(z_d) = log sum_k w_k p(z_d|k)
            log_weighted = log_probs_per_comp + torch.log(weights_d).unsqueeze(0)
            max_val = log_weighted.max(dim=1, keepdim=True)[0]
            log_prob_d = max_val.squeeze(1) + torch.log(
                torch.sum(torch.exp(log_weighted - max_val), dim=1)
            )
            
            log_prob_total += log_prob_d
        
        return log_prob_total.reshape(original_shape[:-1])


class EntropyBottleneck(nn.Module):
    """
    Entropy bottleneck using learned probability model.
    Enables rate-distortion computation: R = -log2(p(z))
    """
    
    def __init__(self, latent_dim: int, num_components: int = 8, 
                 model_type: str = 'global_gmm'):
        """
        Args:
            latent_dim: Dimension of latent space
            num_components: Number of mixture components
            model_type: 'global_gmm' or 'factorized_gmm'
        """
        super().__init__()
        
        if model_type == 'global_gmm':
            self.entropy_model = GaussianMixtureModel(latent_dim, num_components)
        elif model_type == 'factorized_gmm':
            self.entropy_model = ConditionalGMM(latent_dim, num_components)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.latent_dim = latent_dim
        self.model_type = model_type
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute entropy (bits) for latent vector.
        
        Args:
            z: Latent tensor (B, D) or (B, D, T)
            
        Returns:
            rate: (B,) or (B, T) rate in nats (use / ln(2) to convert to bits)
            z: Same latent (pass-through)
        """
        log_prob = self.entropy_model.log_prob(z)  # (*,) log probability in nats
        rate = -log_prob  # Rate = -log p(z) in nats
        
        return rate, z
    
    def compute_rate_in_bits(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute rate in bits.
        
        Args:
            z: Latent tensor
            
        Returns:
            rate_bits: Rate in bits
        """
        rate_nats, _ = self.forward(z)
        rate_bits = rate_nats / np.log(2)
        return rate_bits
    
    def entropy_loss(self, z: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Entropy regularization loss for training.
        Encourages learning a distribution that compresses well.
        
        Args:
            z: Latent tensor
            beta: Weight for entropy term (default 1.0)
            
        Returns:
            loss: Scalar entropy loss
        """
        log_prob = self.entropy_model.log_prob(z)
        # Negative log likelihood = entropy regularization
        return beta * (-log_prob.mean())


def create_entropy_model(latent_dim: int, num_components: int = 8,
                         model_type: str = 'global_gmm',
                         device: str = 'cpu') -> EntropyBottleneck:
    """
    Factory function to create entropy model.
    
    Args:
        latent_dim: Latent dimension
        num_components: Number of mixture components
        model_type: Type of entropy model
        device: Device to place model on
        
    Returns:
        entropy_model: Initialized EntropyBottleneck
    """
    model = EntropyBottleneck(latent_dim, num_components, model_type)
    return model.to(device)
