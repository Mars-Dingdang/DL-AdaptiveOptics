"""Conditional GAN models and physics-aware losses for image restoration.

This module implements a Pix2Pix-style conditional GAN backbone:
- Generator: U-Net like encoder-decoder with skip connections.
- Discriminator: PatchGAN classifier.
- Physics-aware consistency: optional degradation consistency loss.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GANLossWeights:
    """Weights for generator objective terms."""

    lambda_l1: float = 100.0
    lambda_physics: float = 10.0


class DownsampleBlock(nn.Module):
    """Downsampling Conv block for generator/discriminator."""

    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Upsampling ConvTranspose block for generator."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.block(x)


class Pix2PixGenerator(nn.Module):
    """U-Net style generator used in conditional GAN restoration."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        self.down1 = DownsampleBlock(in_channels, c1, use_bn=False)
        self.down2 = DownsampleBlock(c1, c2, use_bn=True)
        self.down3 = DownsampleBlock(c2, c3, use_bn=True)
        self.down4 = DownsampleBlock(c3, c4, use_bn=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        self.up1 = UpsampleBlock(c4, c3, dropout=0.1)
        self.up2 = UpsampleBlock(c3 + c3, c2, dropout=0.0)
        self.up3 = UpsampleBlock(c2 + c2, c1, dropout=0.0)

        self.out = nn.ConvTranspose2d(c1 + c1, out_channels, kernel_size=4, stride=2, padding=1)
        self.out_act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Degraded image tensor in range [-1, 1].

        Returns:
            Restored image tensor in range [-1, 1].
        """
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        b = self.bottleneck(d4)

        u1 = self.up1(b)
        u1 = torch.cat([u1, d3], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)

        out = self.out(u3)
        return self.out_act(out)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for conditional image translation."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        # Conditional discriminator receives concatenated (input, target).
        self.block = nn.Sequential(
            nn.Conv2d(in_channels * 2, c1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c3, c4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for conditional discrimination.

        Args:
            condition: Degraded image tensor [B, C, H, W].
            target: Real or generated clear image tensor [B, C, H, W].

        Returns:
            Patch authenticity logits [B, 1, H', W'].
        """
        x = torch.cat([condition, target], dim=1)
        return self.block(x)


def _build_gaussian_kernel(
    channels: int,
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a depth-wise 2D Gaussian kernel for torch conv2d."""
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    sigma = max(1e-4, float(sigma))
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    kernel_2d = torch.exp(-(grid_x * grid_x + grid_y * grid_y) / (2.0 * sigma * sigma))
    kernel_2d = kernel_2d / kernel_2d.sum().clamp_min(1e-8)

    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


def apply_degradation_approx(
    clear_pred: torch.Tensor,
    sigma: float = 1.2,
    kernel_size: int = 11,
) -> torch.Tensor:
    """Approximate forward degradation model on predicted clear image.

    This is a lightweight physics proxy:
    1) apply Gaussian low-pass blur to mimic optical PSF smoothing,
    2) keep value range unchanged (no stochastic noise here for stability).
    """
    if clear_pred.ndim != 4:
        raise ValueError("clear_pred must be a 4D tensor [B, C, H, W]")

    b, c, _h, _w = clear_pred.shape
    del b

    kernel = _build_gaussian_kernel(
        channels=c,
        kernel_size=kernel_size,
        sigma=sigma,
        device=clear_pred.device,
        dtype=clear_pred.dtype,
    )
    pad = kernel_size // 2
    degraded = F.conv2d(clear_pred, kernel, stride=1, padding=pad, groups=c)
    return degraded


def degradation_consistency_loss(
    clear_pred: torch.Tensor,
    degraded_input: torch.Tensor,
    sigma: float = 1.2,
    kernel_size: int = 11,
) -> torch.Tensor:
    """Compute physics-inspired degradation consistency loss.

    Idea:
    The generated clear image should, after a forward degradation operator,
    reconstruct the observed degraded input.
    """
    recon_degraded = apply_degradation_approx(clear_pred=clear_pred, sigma=sigma, kernel_size=kernel_size)
    return F.l1_loss(recon_degraded, degraded_input)


def generator_adversarial_loss(pred_fake_logits: torch.Tensor) -> torch.Tensor:
    """Standard GAN generator objective with BCE logits."""
    target_real = torch.ones_like(pred_fake_logits)
    return F.binary_cross_entropy_with_logits(pred_fake_logits, target_real)


def discriminator_loss(pred_real_logits: torch.Tensor, pred_fake_logits: torch.Tensor) -> torch.Tensor:
    """Standard discriminator objective with BCE logits."""
    target_real = torch.ones_like(pred_real_logits)
    target_fake = torch.zeros_like(pred_fake_logits)

    loss_real = F.binary_cross_entropy_with_logits(pred_real_logits, target_real)
    loss_fake = F.binary_cross_entropy_with_logits(pred_fake_logits, target_fake)
    return 0.5 * (loss_real + loss_fake)


def generator_total_loss(
    fake_clear: torch.Tensor,
    real_clear: torch.Tensor,
    degraded_input: torch.Tensor,
    pred_fake_logits: torch.Tensor,
    weights: GANLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total generator loss with L1 + adversarial + physics terms."""
    w = weights or GANLossWeights()

    loss_g_adv = generator_adversarial_loss(pred_fake_logits)
    loss_l1 = F.l1_loss(fake_clear, real_clear)
    loss_phy = degradation_consistency_loss(clear_pred=fake_clear, degraded_input=degraded_input)

    total = loss_g_adv + w.lambda_l1 * loss_l1 + w.lambda_physics * loss_phy
    stats = {
        "g_total": float(total.detach().item()),
        "g_adv": float(loss_g_adv.detach().item()),
        "g_l1": float(loss_l1.detach().item()),
        "g_phy": float(loss_phy.detach().item()),
    }
    return total, stats


def init_weights_normal(module: nn.Module, std: float = 0.02) -> None:
    """Initialize Conv/BN layers with common Pix2Pix settings."""
    classname = module.__class__.__name__.lower()
    if "conv" in classname and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "batchnorm" in classname and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 1.0, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


def build_pix2pix_models(
    in_channels: int = 3,
    out_channels: int = 3,
    base_channels: int = 64,
) -> tuple[Pix2PixGenerator, PatchDiscriminator]:
    """Factory for Pix2Pix generator/discriminator pair."""
    generator = Pix2PixGenerator(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    discriminator = PatchDiscriminator(in_channels=in_channels, base_channels=base_channels)

    generator.apply(init_weights_normal)
    discriminator.apply(init_weights_normal)
    return generator, discriminator
