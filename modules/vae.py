"""Conditional VAE for turbulence restoration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class VAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    cond_channels: int = 3
    base_channels: int = 64
    latent_channels: int = 128


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        return self.down(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConditionalVAE(nn.Module):
    """Conditional VAE where posterior is inferred from (degraded, clear)."""

    def __init__(self, config: VAEConfig | None = None) -> None:
        super().__init__()
        cfg = config or VAEConfig()
        self.config = cfg

        c1 = cfg.base_channels
        c2 = c1 * 2
        c3 = c2 * 2

        enc_in = cfg.cond_channels + cfg.out_channels
        self.enc_in = ConvBlock(enc_in, c1)
        self.enc_down1 = DownBlock(c1, c2)
        self.enc_down2 = DownBlock(c2, c3)

        self.mu_head = nn.Conv2d(c3, cfg.latent_channels, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv2d(c3, cfg.latent_channels, kernel_size=3, padding=1)

        self.prior_proj = nn.Sequential(
            nn.Conv2d(cfg.cond_channels, cfg.latent_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        self.dec_in = nn.Sequential(
            nn.Conv2d(cfg.latent_channels + cfg.cond_channels, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
        )
        self.dec_up1 = UpBlock(c3, c3, c2)
        self.dec_up2 = UpBlock(c2, c2, c1)
        self.dec_out = nn.Conv2d(c1, cfg.out_channels, kernel_size=3, padding=1)

    def encode(self, degraded: torch.Tensor, clear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([degraded, clear], dim=1)
        h0 = self.enc_in(x)
        h1, skip1 = self.enc_down1(h0)
        h2, skip2 = self.enc_down2(h1)
        mu = self.mu_head(h2)
        logvar = self.logvar_head(h2)
        return mu, logvar, skip1, skip2

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, degraded: torch.Tensor, z: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor) -> torch.Tensor:
        cond_down = nn.functional.interpolate(degraded, size=z.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([z, self.prior_proj(cond_down)], dim=1)
        x = self.dec_in(x)
        x = self.dec_up1(x, skip2)
        x = self.dec_up2(x, skip1)
        return self.dec_out(x)

    def forward(self, degraded: torch.Tensor, clear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, skip1, skip2 = self.encode(degraded=degraded, clear=clear)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(degraded=degraded, z=z, skip1=skip1, skip2=skip2)
        return recon, mu, logvar

    @torch.no_grad()
    def reconstruct(self, degraded: torch.Tensor) -> torch.Tensor:
        """Reconstruct clear image from degraded-only input at inference time."""
        # Duplicate degraded as a proxy pair to reuse posterior encoder path when GT clear is unavailable.
        x = torch.cat([degraded, degraded], dim=1)
        h0 = self.enc_in(x)
        h1, skip1 = self.enc_down1(h0)
        h2, skip2 = self.enc_down2(h1)
        # Use zero latent as deterministic fallback inference path.
        # This keeps reconstruction stable and avoids stochastic output flicker in demo/eval.
        z = torch.zeros(
            (h2.shape[0], self.config.latent_channels, h2.shape[2], h2.shape[3]),
            device=degraded.device,
            dtype=degraded.dtype,
        )
        recon = self.decode(degraded=degraded, z=z, skip1=skip1, skip2=skip2)
        return recon


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute batch-mean KL(q(z|x)||N(0,I)) for latent tensors [B, C, H, W]."""
    return 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)


def build_conditional_vae(config: VAEConfig | None = None) -> ConditionalVAE:
    return ConditionalVAE(config=config)
