"""TSR-WGAN model blocks and losses for turbulence sequence restoration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import VGG19_Weights, vgg19
except ImportError:  # pragma: no cover - older torchvision fallback
    VGG19_Weights = None
    from torchvision.models import vgg19


@dataclass(frozen=True)
class GANLossWeights:
    """Weights for TSR-WGAN generator and critic losses."""

    lambda_perceptual: float = 10.0
    lambda_pixel: float = 10000.0
    lambda_gp: float = 10.0


def _init_conv_weights(module: nn.Module, std: float = 0.02) -> None:
    classname = module.__class__.__name__.lower()
    if ("conv" in classname or "linear" in classname) and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


class ResBlock(nn.Module):
    """Residual block used by TSR-WGAN."""

    def __init__(self, channels: int, norm: type[nn.Module] = nn.InstanceNorm2d) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            norm(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            norm(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AlignModule(nn.Module):
    """Per-frame spatial feature extractor from the reference TSR-WGAN."""

    def __init__(self, feature_channels: int = 8) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(3, feature_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.level1_block1 = ResBlock(feature_channels)
        self.level1_block2 = ResBlock(feature_channels)
        self.level1_block3 = ResBlock(feature_channels)
        self.level2_block1 = ResBlock(feature_channels)
        self.level2_block2 = ResBlock(feature_channels)
        self.level3_block1 = ResBlock(feature_channels)
        self.downsample1 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.downsample2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        level1 = self.activation(self.conv_in(x))
        level2 = self.activation(self.downsample1(level1))
        level3 = self.activation(self.downsample2(level2))

        level3 = self.upsample(self.level3_block1(level3))
        level2 = self.level2_block1(level2) + level3
        level2 = self.upsample(self.level2_block2(level2))
        level1 = self.level1_block2(self.level1_block1(level1)) + level2
        return self.level1_block3(level1)


class TemporalSpatialIntegration(nn.Module):
    """Temporal-spatial residual integration block from TSR-WGAN."""

    def __init__(self, input_channels: int, frames_num: int, base_channels: int) -> None:
        super().__init__()
        if frames_num < 3 or frames_num % 2 == 0:
            raise ValueError("frames_num must be an odd integer >= 3 for TSR-WGAN.")

        self.integration_window = (frames_num + 1) // 2
        self.temporal_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReplicationPad3d((2, 2, 2, 2, 1, 1)),
                    nn.Conv3d(input_channels, base_channels, kernel_size=(self.integration_window, 5, 5), padding=0),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.integration_window)
            ]
        )
        self.fusion = nn.Sequential(
            nn.ReplicationPad3d((1, 1, 1, 1, 0, 0)),
            nn.Conv3d(base_channels, base_channels * 4, kernel_size=(3 * self.integration_window, 3, 3)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for index, conv3d in enumerate(self.temporal_convs):
            window = x[:, :, index : index + self.integration_window, :, :]
            outputs.append(conv3d(window))

        stacked = torch.cat(outputs, dim=2)
        fused = self.fusion(stacked)
        return fused[:, :, 0, :, :]


class TSRWGANGenerator(nn.Module):
    """TSR-WGAN generator adapted to sequence tensors [B, T, C, H, W]."""

    def __init__(
        self,
        frames_num: int = 7,
        align_channels: int = 8,
        base_channels: int = 16,
        downsample_stages: int = 2,
        num_resblocks: int = 9,
        learn_residual: bool = True,
    ) -> None:
        super().__init__()
        if frames_num < 3 or frames_num % 2 == 0:
            raise ValueError("TSR-WGAN generator expects an odd number of frames >= 3.")

        self.frames_num = frames_num
        self.learn_residual = learn_residual
        self.align = AlignModule(feature_channels=align_channels)
        self.temporal_attention = nn.Conv2d(align_channels, align_channels, kernel_size=7, padding=3)

        layers: list[nn.Module] = [TemporalSpatialIntegration(align_channels, frames_num, base_channels)]
        layers.extend(
            [
                nn.ReplicationPad2d(3),
                nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=7),
                nn.InstanceNorm2d(base_channels * 8),
                nn.ReLU(inplace=True),
            ]
        )

        channels = base_channels * 8
        for _ in range(downsample_stages):
            next_channels = channels * 2
            layers.extend(
                [
                    nn.Conv2d(channels, next_channels, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = next_channels

        for _ in range(num_resblocks):
            layers.append(ResBlock(channels))

        for _ in range(downsample_stages):
            next_channels = channels // 2
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels,
                        next_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.InstanceNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = next_channels

        layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(channels, 3, kernel_size=7)])
        self.reconstruction = nn.Sequential(*layers)
        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("TSR-WGAN generator expects input with shape [B, T, C, H, W].")
        if x.shape[1] != self.frames_num:
            raise ValueError(f"Expected {self.frames_num} frames, but got {x.shape[1]}.")
        if x.shape[2] != 3:
            raise ValueError(f"Expected 3 input channels per frame, but got {x.shape[2]}.")

        aligned_features: list[torch.Tensor] = []
        for frame_index in range(self.frames_num):
            aligned_features.append(self.align(x[:, frame_index, :, :, :]).unsqueeze(2))
        feature_set = torch.cat(aligned_features, dim=2)

        center_index = self.frames_num // 2
        center_feature = feature_set[:, :, center_index, :, :]
        reweight_maps: list[torch.Tensor] = []
        for frame_index in range(self.frames_num):
            neighbour = feature_set[:, :, frame_index, :, :]
            correlation = self.temporal_attention(center_feature * neighbour)
            correlation = F.softmax(correlation, dim=1).unsqueeze(2)
            reweight_maps.append(correlation)
        correlation_prob = torch.cat(reweight_maps, dim=2)
        reweighted_features = feature_set + correlation_prob * feature_set

        output = self.reconstruction(reweighted_features)
        if self.learn_residual:
            output = output + x[:, center_index, :, :, :]
        return self.output_activation(output)


class TSRWGANCritic(nn.Module):
    """Critic from the reference TSR-WGAN operating on a fused frame clip."""

    def __init__(self, input_channels: int = 15, base_channels: int = 64) -> None:
        super().__init__()
        global_layers: list[nn.Module] = [
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        multiplier = 1
        for index in range(4):
            prev_multiplier = multiplier
            multiplier = min(2**index, 8)
            global_layers.extend(
                [
                    nn.Conv2d(
                        base_channels * prev_multiplier,
                        base_channels * multiplier,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.InstanceNorm2d(base_channels * multiplier),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        global_layers.append(nn.Conv2d(base_channels * multiplier, 8, kernel_size=3, padding=1, bias=True))
        self.global_model = nn.Sequential(*global_layers)

        local_layers: list[nn.Module] = [
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(2 * base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * base_channels, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.local_model = nn.Sequential(*local_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("TSR-WGAN critic expects input with shape [B, C, H, W].")
        if x.shape[2] < 25 or x.shape[3] < 25:
            raise ValueError("TSR-WGAN critic requires spatial size >= 25 for the local branch.")

        max_h = x.shape[2] - 25 + 1
        max_w = x.shape[3] - 25 + 1
        loc_h = int(torch.randint(0, max_h, (1,), device=x.device).item())
        loc_w = int(torch.randint(0, max_w, (1,), device=x.device).item())
        local_input = x[:, :, loc_h : loc_h + 25, loc_w : loc_w + 25]
        global_output = self.global_model(x)
        local_output = self.local_model(local_input)
        if local_output.shape[2:] != global_output.shape[2:]:
            local_output = F.adaptive_avg_pool2d(local_output, output_size=global_output.shape[2:])
        return torch.cat((global_output, local_output), dim=1)


def select_discriminator_frame_indices(frames_num: int, clip_frames: int = 5) -> list[int]:
    """Select a symmetric frame subset centered around the middle frame."""
    if clip_frames < 3 or clip_frames % 2 == 0:
        raise ValueError("clip_frames must be an odd integer >= 3.")
    if frames_num < clip_frames:
        raise ValueError("frames_num must be >= clip_frames.")

    center = frames_num // 2
    radius = clip_frames // 2
    return list(range(center - radius, center + radius + 1))


def flatten_frame_clip(sequence: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
    """Flatten selected frames from [B, T, C, H, W] to [B, T*C, H, W]."""
    if sequence.ndim != 5:
        raise ValueError("Expected sequence tensor with shape [B, T, C, H, W].")
    clip = sequence[:, frame_indices, :, :, :]
    batch, frames, channels, height, width = clip.shape
    return clip.reshape(batch, frames * channels, height, width)


def build_fake_discriminator_input(
    degraded_sequence: torch.Tensor,
    fake_clear: torch.Tensor,
    frame_indices: list[int],
) -> torch.Tensor:
    """Replace the center frame in a selected degraded clip with the generated result."""
    if degraded_sequence.ndim != 5:
        raise ValueError("degraded_sequence must have shape [B, T, C, H, W].")
    if fake_clear.ndim != 4:
        raise ValueError("fake_clear must have shape [B, C, H, W].")

    clip = degraded_sequence[:, frame_indices, :, :, :].clone()
    center = len(frame_indices) // 2
    clip[:, center, :, :, :] = fake_clear
    batch, frames, channels, height, width = clip.shape
    return clip.reshape(batch, frames * channels, height, width)


class PerceptualLoss(nn.Module):
    """VGG19 feature loss matching the reference TSR-WGAN training recipe."""

    def __init__(self, feature_layer: int = 14) -> None:
        super().__init__()
        self.criterion = nn.L1Loss()
        try:
            if VGG19_Weights is not None:
                features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            else:  # pragma: no cover - older torchvision fallback
                features = vgg19(pretrained=True).features
        except Exception:
            if VGG19_Weights is not None:
                features = vgg19(weights=None).features
            else:  # pragma: no cover - older torchvision fallback
                features = vgg19(pretrained=False).features

        selected_layers: list[nn.Module] = []
        for index, layer in enumerate(features):
            selected_layers.append(layer)
            if index == feature_layer:
                break
        self.model = nn.Sequential(*selected_layers)
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_features = self.model(fake)
        real_features = self.model(real).detach()
        return self.criterion(fake_features, real_features)


def compute_gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_gp: float,
) -> torch.Tensor:
    """Compute WGAN-GP penalty on critic inputs."""
    batch_size = real.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device, dtype=real.dtype)
    composed = alpha * real + (1.0 - alpha) * fake
    composed.requires_grad_(True)
    critic_out = critic(composed)
    gradients = autograd.grad(
        outputs=critic_out,
        inputs=composed,
        grad_outputs=torch.ones_like(critic_out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * float(lambda_gp)


def generator_total_loss(
    critic: nn.Module,
    fake_discriminator_input: torch.Tensor,
    fake_clear: torch.Tensor,
    real_clear: torch.Tensor,
    perceptual_loss_fn: nn.Module,
    weights: GANLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute TSR-WGAN generator loss."""
    wgan_loss = -critic(fake_discriminator_input).mean()
    perceptual = perceptual_loss_fn(fake_clear, real_clear)
    pixel = F.mse_loss(fake_clear, real_clear)
    total = weights.lambda_perceptual * perceptual + weights.lambda_pixel * pixel + wgan_loss
    stats = {
        "g_total": float(total.detach().item()),
        "g_percep": float(perceptual.detach().item()),
        "g_pixel": float(pixel.detach().item()),
        "g_wgan": float(wgan_loss.detach().item()),
    }
    return total, stats


def critic_total_loss(
    critic: nn.Module,
    fake_discriminator_input: torch.Tensor,
    real_discriminator_input: torch.Tensor,
    lambda_gp: float,
    include_gp: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute TSR-WGAN critic loss with gradient penalty."""
    d_fake = critic(fake_discriminator_input.detach())
    d_real = critic(real_discriminator_input)
    wasserstein = d_fake.mean() - d_real.mean()
    if include_gp:
        gp = compute_gradient_penalty(
            critic=critic,
            real=real_discriminator_input,
            fake=fake_discriminator_input.detach(),
            lambda_gp=lambda_gp,
        )
    else:
        gp = torch.zeros((), device=real_discriminator_input.device, dtype=real_discriminator_input.dtype)
    total = wasserstein + gp
    stats = {
        "d_total": float(total.detach().item()),
        "d_wasserstein": float(wasserstein.detach().item()),
        "d_gp": float(gp.detach().item()),
        "d_real": float(d_real.mean().detach().item()),
        "d_fake": float(d_fake.mean().detach().item()),
    }
    return total, stats


def build_tsr_wgan_models(
    frames_num: int = 7,
    critic_clip_frames: int = 5,
    align_channels: int = 8,
    base_channels: int = 16,
    critic_base_channels: int = 64,
    downsample_stages: int = 2,
    num_resblocks: int = 9,
    learn_residual: bool = True,
) -> tuple[TSRWGANGenerator, TSRWGANCritic]:
    """Factory for TSR-WGAN generator/critic."""
    frame_indices = select_discriminator_frame_indices(frames_num=frames_num, clip_frames=critic_clip_frames)
    generator = TSRWGANGenerator(
        frames_num=frames_num,
        align_channels=align_channels,
        base_channels=base_channels,
        downsample_stages=downsample_stages,
        num_resblocks=num_resblocks,
        learn_residual=learn_residual,
    )
    critic = TSRWGANCritic(input_channels=len(frame_indices) * 3, base_channels=critic_base_channels)
    generator.apply(_init_conv_weights)
    critic.apply(_init_conv_weights)
    return generator, critic


Pix2PixGenerator = TSRWGANGenerator
PatchDiscriminator = TSRWGANCritic


def build_pix2pix_models(
    in_channels: int = 3,
    out_channels: int = 3,
    base_channels: int = 64,
    frames_num: int | None = None,
    critic_clip_frames: int = 5,
    align_channels: int = 8,
    generator_base_channels: int | None = None,
    critic_base_channels: int | None = None,
    downsample_stages: int = 2,
    num_resblocks: int = 9,
    learn_residual: bool = True,
) -> tuple[TSRWGANGenerator, TSRWGANCritic]:
    del out_channels
    resolved_frames_num = frames_num
    if resolved_frames_num is None:
        resolved_frames_num = max(3, in_channels // 3) if in_channels % 3 == 0 else 7

    generator, critic = build_tsr_wgan_models(
        frames_num=resolved_frames_num,
        critic_clip_frames=critic_clip_frames,
        align_channels=align_channels,
        base_channels=max(8, base_channels // 4) if generator_base_channels is None else generator_base_channels,
        critic_base_channels=base_channels if critic_base_channels is None else critic_base_channels,
        downsample_stages=downsample_stages,
        num_resblocks=num_resblocks,
        learn_residual=learn_residual,
    )
    return generator, critic