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
import importlib
import math
from typing import Any

import cv2
import numpy as np


_TURBSIM_ADAPTERS: dict[str, Any] = {}


class SimpleParametricAdapter:
    """Fast fallback simulator using lightweight geometric blur/noise approximations."""

    name = "simple_parametric"

    def simulate_sequence(
        self,
        clean_image: np.ndarray,
        num_frames: int,
        context: dict[str, float],
        params: Any,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        clean = np.clip(clean_image.astype(np.float32), 0.0, 1.0)
        if clean.ndim != 3:
            raise ValueError("simple_parametric expects HxWxC input")

        h, w = clean.shape[:2]
        n_frames = max(1, int(num_frames))

        wind = float(context.get("wind_speed", 1.0))
        cn2 = float(context.get("cn2", 1e-15))
        strength = float(getattr(params, "turbulence_strength", 0.5))

        shift_sigma = 0.8 + 1.8 * np.clip(wind, 0.0, 3.0)
        blur_sigma_base = 0.6 + 4.0 * np.clip((cn2 / 5e-15), 0.0, 1.0) * np.clip(strength, 0.0, 1.5)

        frames: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []

        for frame_idx in range(n_frames):
            jitter_x = float(rng.normal(0.0, shift_sigma))
            jitter_y = float(rng.normal(0.0, shift_sigma))
            affine = np.array([[1.0, 0.0, jitter_x], [0.0, 1.0, jitter_y]], dtype=np.float32)
            warped = cv2.warpAffine(
                clean,
                affine,
                dsize=(w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )

            blur_sigma = float(blur_sigma_base * (0.85 + 0.3 * rng.random()))
            ksize = int(max(3, 2 * int(np.ceil(blur_sigma * 2.0)) + 1))
            blurred = cv2.GaussianBlur(warped, (ksize, ksize), sigmaX=blur_sigma, sigmaY=blur_sigma)

            noisy, poisson_scale, gaussian_std = add_sensor_noise(
                image=blurred,
                rng=rng,
                poisson_scale_range=getattr(params, "poisson_scale_range", (40.0, 120.0)),
                gaussian_noise_std_range=getattr(params, "gaussian_noise_std_range", (0.003, 0.02)),
            )
            compressed, jpeg_quality = add_jpeg_artifact(
                image=noisy,
                rng=rng,
                quality_range=getattr(params, "jpeg_quality_range", (70, 98)),
            )

            frame = np.clip(compressed, 0.0, 1.0).astype(np.float32)
            frames.append(frame)
            metas.append(
                {
                    "frame_idx": int(frame_idx),
                    "sim_backend": self.name,
                    "cn2": cn2,
                    "focal_length": float(context.get("focal_length", 0.0)),
                    "wind_speed": wind,
                    "time_step": float(context.get("time_step", 0.03)),
                    "jitter_x": jitter_x,
                    "jitter_y": jitter_y,
                    "blur_sigma": blur_sigma,
                    "poisson_scale": float(poisson_scale),
                    "gaussian_std": float(gaussian_std),
                    "jpeg_quality": int(jpeg_quality),
                    "device": "cpu-fast",
                }
            )

        return np.stack(frames, axis=0).astype(np.float32), metas


def _get_turbsim_adapter(backend: str) -> Any:
    """Lazily create and cache the requested TurbulenceSim adapter."""
    backend_norm = str(backend).lower().strip()
    adapter = _TURBSIM_ADAPTERS.get(backend_norm)
    if adapter is not None:
        return adapter

    if backend_norm == "simple_parametric":
        adapter = SimpleParametricAdapter()
    elif backend_norm == "turbsim_gpu_v1":
        gpu_pkg = importlib.import_module("utils.TurbulenceSimGPU")
        adapter_cls = getattr(gpu_pkg, "TurbulenceSimGPUAdapter", None)
        if adapter_cls is None:
            adapter_cls = getattr(gpu_pkg, "TurbulenceSimV1Adapter")
        adapter = adapter_cls()
    else:
        from utils.TurbulenceSim.simulator import TurbulenceSimV1Adapter

        adapter = TurbulenceSimV1Adapter()

    _TURBSIM_ADAPTERS[backend_norm] = adapter
    return adapter


@dataclass
class TurbulenceParams:
    """Config for atmospheric turbulence and sensor degradation."""

    backend: str = "turbsim_gpu_v1"
    zernike_order: int = 6
    phase_strength: float = 0.12
    psf_kernel_size: int = 5
    gaussian_sigma_range: tuple[float, float] = (0.3, 1.2)
    motion_blur_prob: float = 0.25
    motion_blur_kernel_range: tuple[int, int] = (7, 17)
    poisson_scale_range: tuple[float, float] = (40.0, 120.0)
    gaussian_noise_std_range: tuple[float, float] = (0.003, 0.02)
    jpeg_quality_range: tuple[int, int] = (70, 98)
    cn2_range: tuple[float, float] = (1e-16, 5e-15)
    focal_length_range: tuple[float, float] = (6000.0, 18000.0)
    wind_speed_range: tuple[float, float] = (0.5, 2.0)
    sequence_time_step: float = 0.03
    aperture_diameter: float = 0.2
    wavelength: float = 0.525e-6
    object_size: float = 2.06
    turbulence_strength: float = 0.45
    turbsim_luma_only: bool = False
    turbsim_patch_grid_downsample: int = 1
    turbsim_psf_resolution: int = 32
    turbsim_reuse_psf_per_frame: bool = False


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
    """Apply single-frame atmospheric turbulence using TurbulenceSim-v1.

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

    seq, metas = add_atmospheric_turbulence_sequence(
        image=image,
        num_frames=1,
        params=params,
        rng=rng,
        context=None,
        return_meta=True,
    )
    frame = seq[0]
    if not return_meta:
        return frame
    return frame, metas[0]


def sample_turbulence_context(
    params: TurbulenceParams,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample sample-level physical parameters for sequence simulation."""
    return {
        "cn2": float(rng.uniform(*params.cn2_range)),
        "focal_length": float(rng.uniform(*params.focal_length_range)),
        "wind_speed": float(rng.uniform(*params.wind_speed_range)),
        "time_step": float(params.sequence_time_step),
    }


def add_atmospheric_turbulence_sequence(
    image: np.ndarray,
    num_frames: int,
    params: TurbulenceParams | None = None,
    rng: np.random.Generator | None = None,
    context: dict[str, float] | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate a turbulence sequence with shape [T, H, W, C] via TurbulenceSim-v1."""
    params = params or TurbulenceParams()
    rng = rng or np.random.default_rng()

    n_frames = max(1, int(num_frames))
    image_f = _to_float_image(image)
    sample_context = context or sample_turbulence_context(params=params, rng=rng)

    adapter = _get_turbsim_adapter(params.backend)
    seq, metas = adapter.simulate_sequence(
        clean_image=image_f,
        num_frames=n_frames,
        context=sample_context,
        params=params,
        rng=rng,
    )
    if return_meta:
        return seq, metas
    return seq
