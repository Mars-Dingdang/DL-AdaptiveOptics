"""Lightweight conditional diffusion backbone (skeleton).

This module provides a minimal, extensible conditional diffusion implementation
for image restoration tasks. It is designed as a project scaffold:
- Gaussian diffusion schedule.
- Conditional UNet-like denoiser.
- Training loss (noise prediction objective).
- Deterministic DDIM-style sampling interface.

Note:
    This skeleton is intentionally compact and can be upgraded in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DiffusionConfig:
    """Configuration for conditional diffusion model."""

    image_channels: int = 3
    cond_channels: int = 3
    base_channels: int = 64
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return time embeddings with shape [B, dim]."""
        half = self.dim // 2
        device = t.device
        emb_scale = math.log(10000) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb_scale)
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    """Residual block with time embedding modulation."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply residual block conditioned on timestep embedding."""
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.skip(x)


class ConditionUNet(nn.Module):
    """Compact conditional U-Net-like denoiser for diffusion."""

    def __init__(self, image_channels: int, cond_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        time_dim = base_channels * 4

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        in_ch = image_channels + cond_channels
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2

        self.in_conv = nn.Conv2d(in_ch, c1, kernel_size=3, padding=1)
        self.down1 = ResBlock(c1, c1, time_dim)
        self.downsample1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)

        self.down2 = ResBlock(c2, c2, time_dim)
        self.downsample2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)

        self.mid1 = ResBlock(c3, c3, time_dim)
        self.mid2 = ResBlock(c3, c3, time_dim)

        self.upsample1 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock(c2 + c2, c2, time_dim)

        self.upsample2 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.up2 = ResBlock(c1 + c1, c1, time_dim)

        self.out_conv = nn.Conv2d(c1, image_channels, kernel_size=3, padding=1)

    def forward(self, x_noisy: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict additive noise epsilon given noisy sample, condition, and timestep."""
        t_emb = self.time_embed(t)

        x = torch.cat([x_noisy, cond], dim=1)
        x0 = self.in_conv(x)

        d1 = self.down1(x0, t_emb)
        d2_in = self.downsample1(d1)

        d2 = self.down2(d2_in, t_emb)
        d3_in = self.downsample2(d2)

        m = self.mid1(d3_in, t_emb)
        m = self.mid2(m, t_emb)

        u1 = self.upsample1(m)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up1(u1, t_emb)

        u2 = self.upsample2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up2(u2, t_emb)

        return self.out_conv(u2)


class ConditionalDiffusionModel(nn.Module):
    """Conditional diffusion wrapper with training and sampling utilities."""

    def __init__(self, config: DiffusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiffusionConfig()

        self.denoiser = ConditionUNet(
            image_channels=self.config.image_channels,
            cond_channels=self.config.cond_channels,
            base_channels=self.config.base_channels,
        )

        betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.config.timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion: add noise to clean image at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)

        a = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return a * x_start + b * noise

    def p_losses(self, x_start: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Noise prediction objective (MSE)."""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        noise_pred = self.denoiser(x_noisy=x_noisy, cond=cond, t=t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample_ddim(self, cond: torch.Tensor, steps: int = 20, eta: float = 0.0) -> torch.Tensor:
        """DDIM-like deterministic sampling interface.

        Args:
            cond: Condition image tensor [B, C, H, W], typically degraded input.
            steps: Number of denoising steps.
            eta: DDIM stochasticity parameter. eta=0 gives deterministic path.

        Returns:
            Restored image tensor in approx [-1, 1].
        """
        device = cond.device
        b, _c, h, w = cond.shape
        total_t = self.config.timesteps
        steps = max(2, min(steps, total_t))

        x = torch.randn((b, self.config.image_channels, h, w), device=device, dtype=cond.dtype)
        time_indices = torch.linspace(total_t - 1, 0, steps, device=device).long()

        for i in range(len(time_indices) - 1):
            t = time_indices[i]
            t_next = time_indices[i + 1]

            t_batch = torch.full((b,), int(t.item()), device=device, dtype=torch.long)
            eps = self.denoiser(x_noisy=x, cond=cond, t=t_batch)

            alpha_t = self.alpha_cumprod[t]
            alpha_next = self.alpha_cumprod[t_next]

            x0 = (x - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t)
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
            noise = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)

            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma * sigma, min=0.0)) * eps
            x = torch.sqrt(alpha_next) * x0 + dir_xt + sigma * noise

        return x.clamp(-1.0, 1.0)


def build_conditional_diffusion(config: DiffusionConfig | None = None) -> ConditionalDiffusionModel:
    """Factory function for conditional diffusion model."""
    return ConditionalDiffusionModel(config=config)
