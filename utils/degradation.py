"""Physical degradation simulator for remote sensing image restoration.

This module provides a configurable atmospheric turbulence simulation pipeline.
It combines:
1. Zernike-based phase-screen generation.
2. PSF (point spread function) blur derived from wavefront phase distortion.
3. Optional Gaussian and motion blur.
4. Sensor noise and compression artifacts.

Main entrypoint:
    add_atmospheric_turbulence(image, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np


@dataclass
class TurbulenceParams:
    """Config for atmospheric turbulence and sensor degradation."""

    zernike_order: int = 6
    phase_strength: float = 1.5
    psf_kernel_size: int = 31
    gaussian_sigma_range: tuple[float, float] = (0.3, 1.2)
    motion_blur_prob: float = 0.25
    motion_blur_kernel_range: tuple[int, int] = (7, 17)
    poisson_scale_range: tuple[float, float] = (40.0, 120.0)
    gaussian_noise_std_range: tuple[float, float] = (0.003, 0.02)
    jpeg_quality_range: tuple[int, int] = (70, 98)


def _ensure_odd(value: int) -> int:
    """Ensure a positive odd integer."""
    if value < 1:
        value = 1
    if value % 2 == 0:
        value += 1
    return value


def _to_float_image(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0, 1], preserving channels."""
    if image.ndim == 2:
        image = image[..., None]

    if image.dtype == np.uint8:
        out = image.astype(np.float32) / 255.0
    else:
        out = image.astype(np.float32)
        if out.max() > 1.0:
            out = out / 255.0

    out = np.clip(out, 0.0, 1.0)
    return out


def _radial_polynomial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """Compute Zernike radial polynomial R_n^m(r)."""
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(r, dtype=np.float32)

    radial = np.zeros_like(r, dtype=np.float32)
    upper = (n - m) // 2
    for k in range(upper + 1):
        numerator = ((-1) ** k) * math.factorial(n - k)
        denominator = (
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial((n - m) // 2 - k)
        )
        radial += (numerator / denominator) * (r ** (n - 2 * k))

    return radial


def _zernike(n: int, m: int, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute Zernike basis Z_n^m(r, theta)."""
    radial = _radial_polynomial(n, m, r)
    if m >= 0:
        return radial * np.cos(m * theta)
    return radial * np.sin(abs(m) * theta)


def _zernike_mode_sequence(max_order: int) -> list[tuple[int, int]]:
    """Return (n, m) sequence excluding piston term (0, 0)."""
    modes: list[tuple[int, int]] = []
    for n in range(1, max_order + 1):
        for m in range(-n, n + 1, 2):
            modes.append((n, m))
    return modes


def generate_phase_screen(
    size: int,
    max_order: int,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random wavefront phase screen using Zernike modes.

    The phase screen approximates atmospheric turbulence-induced wavefront errors.
    """
    if size < 8:
        raise ValueError("size must be >= 8 for stable PSF simulation")

    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    rr = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)

    aperture = rr <= 1.0
    phase = np.zeros((size, size), dtype=np.float32)

    modes = _zernike_mode_sequence(max_order=max_order)
    coeffs = rng.normal(loc=0.0, scale=1.0, size=len(modes)).astype(np.float32)

    for coeff, (n, m) in zip(coeffs, modes):
        phase += coeff * _zernike(n=n, m=m, r=rr, theta=theta)

    # Physically, phase only exists inside the circular pupil aperture.
    phase *= aperture.astype(np.float32)

    phase_std = float(np.std(phase[aperture])) + 1e-8
    phase = (strength / phase_std) * phase
    return phase


def phase_screen_to_psf(phase_screen: np.ndarray, kernel_size: int) -> np.ndarray:
    """Convert phase screen to a PSF kernel using Fourier optics."""
    if phase_screen.ndim != 2:
        raise ValueError("phase_screen must be 2D")

    h, w = phase_screen.shape
    if h != w:
        raise ValueError("phase_screen must be square")

    size = h
    kernel_size = _ensure_odd(kernel_size)
    if kernel_size > size:
        kernel_size = _ensure_odd(size)

    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    rr = np.sqrt(xx * xx + yy * yy)
    aperture = (rr <= 1.0).astype(np.float32)

    # Pupil function: A(x,y) * exp(j * phi(x,y)), phi is phase distortion.
    pupil = aperture * np.exp(1j * phase_screen)

    # PSF is squared magnitude of Fourier transform of pupil field.
    optical_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    psf = np.abs(optical_field) ** 2
    psf = psf.astype(np.float32)

    center = size // 2
    half = kernel_size // 2
    cropped = psf[center - half : center + half + 1, center - half : center + half + 1]
    cropped_sum = float(cropped.sum()) + 1e-8
    cropped /= cropped_sum
    return cropped


def apply_psf_blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Apply PSF convolution channel-wise with reflective border handling."""
    if image.ndim != 3:
        raise ValueError("image must be HxWxC")

    blurred_channels: list[np.ndarray] = []
    for c in range(image.shape[2]):
        channel = cv2.filter2D(image[..., c], ddepth=-1, kernel=psf, borderType=cv2.BORDER_REFLECT101)
        blurred_channels.append(channel)

    blurred = np.stack(blurred_channels, axis=-1)
    return np.clip(blurred, 0.0, 1.0)


def _sample_motion_kernel(length: int, angle: float) -> np.ndarray:
    """Generate a normalized linear motion blur kernel."""
    length = _ensure_odd(length)
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    kernel[center, :] = 1.0

    rot_mat = cv2.getRotationMatrix2D(center=(center, center), angle=angle, scale=1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (length, length), flags=cv2.INTER_LINEAR)
    kernel_sum = float(kernel.sum()) + 1e-8
    kernel /= kernel_sum
    return kernel


def add_sensor_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    poisson_scale_range: tuple[float, float],
    gaussian_noise_std_range: tuple[float, float],
) -> tuple[np.ndarray, float, float]:
    """Add Poisson and Gaussian noise to simulate sensor perturbations."""
    poisson_scale = float(rng.uniform(*poisson_scale_range))
    poisson_scale = max(poisson_scale, 1e-6)

    noisy = np.clip(image, 0.0, 1.0)
    poisson_counts = rng.poisson(lam=noisy * poisson_scale).astype(np.float32)
    noisy = poisson_counts / poisson_scale

    gaussian_std = float(rng.uniform(*gaussian_noise_std_range))
    gaussian_noise = rng.normal(0.0, gaussian_std, size=noisy.shape).astype(np.float32)
    noisy = noisy + gaussian_noise

    return np.clip(noisy, 0.0, 1.0), poisson_scale, gaussian_std


def add_jpeg_artifact(
    image: np.ndarray,
    rng: np.random.Generator,
    quality_range: tuple[int, int],
) -> tuple[np.ndarray, int]:
    """Add JPEG compression artifacts in-memory."""
    low, high = quality_range
    quality = int(rng.integers(low=max(1, low), high=min(100, high) + 1))

    encoded_success, encoded = cv2.imencode(
        ".jpg",
        (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8),
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not encoded_success:
        return np.clip(image, 0.0, 1.0), quality

    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return np.clip(image, 0.0, 1.0), quality

    if decoded.ndim == 2:
        decoded = decoded[..., None]
    decoded_f = decoded.astype(np.float32) / 255.0

    if decoded_f.shape != image.shape:
        decoded_f = cv2.resize(decoded_f, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        if decoded_f.ndim == 2:
            decoded_f = decoded_f[..., None]

    return np.clip(decoded_f, 0.0, 1.0), quality


def add_atmospheric_turbulence(
    image: np.ndarray,
    params: TurbulenceParams | None = None,
    rng: np.random.Generator | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Apply atmospheric turbulence and sensor degradation pipeline.

    Args:
        image: Input image, shape HxW or HxWxC. Supports uint8 or float arrays.
        params: Degradation hyper-parameters.
        rng: Optional random generator for reproducibility.
        return_meta: If True, returns both image and sampled parameters.

    Returns:
        Degraded image in float32 [0, 1], shape HxWxC.
    """
    params = params or TurbulenceParams()
    rng = rng or np.random.default_rng()

    image_f = _to_float_image(image)

    sigma = float(rng.uniform(*params.gaussian_sigma_range))
    sigma = max(0.0, sigma)
    if sigma > 1e-6:
        ksize = _ensure_odd(int(round(6.0 * sigma)) + 1)
        image_f = cv2.GaussianBlur(
            src=image_f,
            ksize=(ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT101,
        )

    phase_screen = generate_phase_screen(
        size=max(image_f.shape[0], image_f.shape[1]),
        max_order=params.zernike_order,
        strength=params.phase_strength,
        rng=rng,
    )
    psf = phase_screen_to_psf(phase_screen=phase_screen, kernel_size=params.psf_kernel_size)
    image_f = apply_psf_blur(image=image_f, psf=psf)

    use_motion = bool(rng.uniform(0.0, 1.0) < params.motion_blur_prob)
    motion_len = 0
    motion_angle = 0.0
    if use_motion:
        low, high = params.motion_blur_kernel_range
        motion_len = _ensure_odd(int(rng.integers(low=max(3, low), high=max(low + 1, high + 1))))
        motion_angle = float(rng.uniform(0.0, 180.0))
        motion_kernel = _sample_motion_kernel(length=motion_len, angle=motion_angle)
        image_f = apply_psf_blur(image=image_f, psf=motion_kernel)

    image_f, poisson_scale, gaussian_std = add_sensor_noise(
        image=image_f,
        rng=rng,
        poisson_scale_range=params.poisson_scale_range,
        gaussian_noise_std_range=params.gaussian_noise_std_range,
    )

    image_f, jpeg_quality = add_jpeg_artifact(
        image=image_f,
        rng=rng,
        quality_range=params.jpeg_quality_range,
    )

    image_f = np.clip(image_f, 0.0, 1.0).astype(np.float32)

    if not return_meta:
        return image_f

    meta: dict[str, Any] = {
        "gaussian_sigma": sigma,
        "zernike_order": params.zernike_order,
        "phase_strength": params.phase_strength,
        "psf_kernel_size": params.psf_kernel_size,
        "motion_used": use_motion,
        "motion_kernel_size": motion_len,
        "motion_angle": motion_angle,
        "poisson_scale": poisson_scale,
        "gaussian_noise_std": gaussian_std,
        "jpeg_quality": jpeg_quality,
    }
    return image_f, meta
