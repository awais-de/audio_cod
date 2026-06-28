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
    """Learnable GMM for modelling latent distribution used in entropy coding."""

    def __init__(self, latent_dim: int, num_components: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components

        self.register_parameter(
            'log_mixture_weights',
            nn.Parameter(torch.zeros(num_components))
        )

        self.register_parameter(
            'component_means',
            nn.Parameter(torch.randn(num_components, latent_dim) * 0.1)
        )

        self.register_parameter(
            'component_log_vars',
            nn.Parameter(torch.zeros(num_components, latent_dim))
        )

    def mixture_weights(self) -> torch.Tensor:
        """Normalised mixture weights (K,)"""
        return F.softmax(self.log_mixture_weights, dim=0)

    def log_prob_per_component(self, z: torch.Tensor) -> torch.Tensor:
        """Log probability under each Gaussian component: returns (*, K)"""
        original_shape = z.shape
        z_flat = z.reshape(-1, self.latent_dim)

        means = self.component_means
        log_vars = self.component_log_vars
        vars = torch.exp(log_vars)

        z_expanded = z_flat.unsqueeze(1)
        diff = z_expanded - means.unsqueeze(0)

        mahal = (diff ** 2) / vars.unsqueeze(0)

        const = 0.5 * self.latent_dim * np.log(2 * np.pi)
        log_det = 0.5 * torch.sum(log_vars, dim=1)

        log_prob = -const - log_det.unsqueeze(0) - 0.5 * mahal.sum(dim=2)

        return log_prob.reshape(*original_shape[:-1], self.num_components)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """log p(z) = log sum_k w_k * p(z|k) via the log-sum-exp trick."""
        log_probs_per_comp = self.log_prob_per_component(z)
        weights = self.mixture_weights()

        log_weights_expanded = torch.log(weights).reshape(
            *([1] * (log_probs_per_comp.ndim - 1)), self.num_components
        )

        log_weighted_probs = log_weights_expanded + log_probs_per_comp

        max_val = log_weighted_probs.max(dim=-1, keepdim=True)[0]
        log_sum = max_val + torch.log(
            torch.sum(torch.exp(log_weighted_probs - max_val), dim=-1, keepdim=True)
        )

        return log_sum.squeeze(-1)

    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample (num_samples, latent_dim) from the mixture."""
        with torch.no_grad():
            weights = self.mixture_weights()
            component_ids = torch.multinomial(weights, num_samples, replacement=True)

            means = self.component_means
            log_vars = self.component_log_vars
            stds = torch.exp(0.5 * log_vars)

            selected_means = means[component_ids]
            selected_stds = stds[component_ids]

            noise = torch.randn(num_samples, self.latent_dim, device=means.device)
            samples = selected_means + selected_stds * noise

            return samples


class ConditionalGMM(nn.Module):
    """
    Factorized GMM: each latent dimension has an independent mixture.
    More expressive than a global GMM, more efficient than full covariance.
    """

    def __init__(self, latent_dim: int, num_components: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components

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
        """Normalised per-dimension mixture weights (D, K)"""
        return F.softmax(self.log_mixture_weights, dim=1)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """log p(z) = sum_d log p(z_d) assuming factorized form."""
        original_shape = z.shape
        z_flat = z.reshape(-1, self.latent_dim)

        log_prob_total = torch.zeros(z_flat.shape[0], device=z.device)

        for d in range(self.latent_dim):
            z_d = z_flat[:, d:d+1]
            means_d = self.component_means[d]
            log_vars_d = self.component_log_vars[d]
            weights_d = F.softmax(self.log_mixture_weights[d], dim=0)

            vars_d = torch.exp(log_vars_d)
            mahal = (z_d - means_d.unsqueeze(0)) ** 2 / vars_d.unsqueeze(0)

            const = 0.5 * np.log(2 * np.pi)
            log_probs_per_comp = -const - 0.5 * log_vars_d.unsqueeze(0) - 0.5 * mahal

            log_weighted = log_probs_per_comp + torch.log(weights_d).unsqueeze(0)
            max_val = log_weighted.max(dim=1, keepdim=True)[0]
            log_prob_d = max_val.squeeze(1) + torch.log(
                torch.sum(torch.exp(log_weighted - max_val), dim=1)
            )

            log_prob_total += log_prob_d

        return log_prob_total.reshape(original_shape[:-1])


class EntropyBottleneck(nn.Module):
    """
    Entropy bottleneck using a learned probability model.
    Enables rate computation: R = -log2(p(z))
    """

    def __init__(self, latent_dim: int, num_components: int = 8,
                 model_type: str = 'global_gmm'):
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
        """Returns rate in nats and the pass-through latent."""
        log_prob = self.entropy_model.log_prob(z)
        rate = -log_prob  # rate in nats; divide by ln(2) for bits

        return rate, z

    def compute_rate_in_bits(self, z: torch.Tensor) -> torch.Tensor:
        """Rate in bits."""
        rate_nats, _ = self.forward(z)
        rate_bits = rate_nats / np.log(2)
        return rate_bits

    def entropy_loss(self, z: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Entropy regularisation loss: encourages a compressible latent distribution."""
        log_prob = self.entropy_model.log_prob(z)
        return beta * (-log_prob.mean())


def create_entropy_model(latent_dim: int, num_components: int = 8,
                         model_type: str = 'global_gmm',
                         device: str = 'cpu') -> EntropyBottleneck:
    """Factory for EntropyBottleneck."""
    model = EntropyBottleneck(latent_dim, num_components, model_type)
    return model.to(device)
