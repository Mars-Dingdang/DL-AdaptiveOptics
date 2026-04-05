"""Adapter wrapper for Purdue TurbulenceSim-v1 sequence generation."""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm.auto import tqdm

from . import TurbSim_v1_main as turbsim


class TurbulenceSimV1Adapter:
    """Sequence simulator using real TurbulenceSim-v1 operators."""

    def __init__(self) -> None:
        self.name = "turbsim_v1"
        self.default_aperture_diameter = 0.2
        self.default_wavelength = 0.525e-6
        self.default_object_size = 2.06

    def _context_to_r0(self, cn2: float, length_m: float, wavelength_m: float) -> float:
        """Map Cn2 to Fried parameter r0 using Kolmogorov model."""
        cn2 = max(float(cn2), 1e-20)
        length_m = max(float(length_m), 1.0)
        wavelength_m = max(float(wavelength_m), 1e-9)
        k = (2.0 * np.pi) / wavelength_m
        r0 = (0.423 * (k**2) * cn2 * length_m) ** (-3.0 / 5.0)
        return float(np.clip(r0, 0.08, 6.0))

    def _resolve_parameters(
        self,
        clean_image: np.ndarray,
        context: dict[str, float],
        params: Any,
    ) -> dict[str, float]:
        n = int(clean_image.shape[0])
        length_m = float(context.get("focal_length", 7000.0))
        cn2 = float(context.get("cn2", 1e-15))
        wavelength = float(getattr(params, "wavelength", self.default_wavelength))
        aperture_d = float(getattr(params, "aperture_diameter", self.default_aperture_diameter))
        obj_size = float(getattr(params, "object_size", self.default_object_size))
        turbulence_strength = float(getattr(params, "turbulence_strength", 1.0))
        effective_cn2 = cn2 * max(turbulence_strength, 1e-3)
        r0 = self._context_to_r0(cn2=effective_cn2, length_m=length_m, wavelength_m=wavelength)
        blend_alpha = float(np.clip(turbulence_strength, 0.0, 1.0))

        return {
            "N": n,
            "D": aperture_d,
            "L": length_m,
            "wvl": wavelength,
            "r0": r0,
            "obj_size": obj_size,
            "wind_speed": float(context.get("wind_speed", 5.0)),
            "time_step": float(context.get("time_step", 0.03)),
            "turbulence_strength": turbulence_strength,
            "blend_alpha": blend_alpha,
        }

    def _simulate_single_channel(
        self,
        channel: np.ndarray,
        sim_param: dict[str, float],
        shared_psd: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        # Turbulence is achromatic at this stage; keep channel draws aligned to avoid false color artifacts.
        np.random.seed(int(seed))
        p = turbsim.p_obj(
            N=int(sim_param["N"]),
            D=float(sim_param["D"]),
            L=float(sim_param["L"]),
            r0=float(sim_param["r0"]),
            wvl=float(sim_param["wvl"]),
            obj_size=float(sim_param["obj_size"]),
        )
        p["S"] = shared_psd
        tilted, _ = turbsim.genTiltImg(channel, p)
        blurred = turbsim.genBlurImage(p, tilted)
        blurred = np.nan_to_num(blurred, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(blurred, 0.0, 1.0).astype(np.float32)

    def simulate_sequence(
        self,
        clean_image: np.ndarray,
        num_frames: int,
        context: dict[str, float],
        params: Any,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Generate turbulence sequence with output shape [T, H, W, C]."""
        n_frames = max(1, int(num_frames))
        clean = np.clip(clean_image.astype(np.float32), 0.0, 1.0)
        if clean.ndim != 3 or clean.shape[0] != clean.shape[1]:
            raise ValueError("TurbulenceSim-v1 adapter expects square HxWxC input.")

        sim_param = self._resolve_parameters(clean_image=clean, context=context, params=params)
        p_base = turbsim.p_obj(
            N=int(sim_param["N"]),
            D=float(sim_param["D"]),
            L=float(sim_param["L"]),
            r0=float(sim_param["r0"]),
            wvl=float(sim_param["wvl"]),
            obj_size=float(sim_param["obj_size"]),
        )
        shared_psd = turbsim.gen_PSD(p_base)
        phase0 = float(rng.uniform(0.0, 2.0 * np.pi))

        frames: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []

        for frame_idx in tqdm(range(n_frames), desc="CPU turbulence frames", unit="frame", leave=False):
            # Add smooth frame-to-frame turbulence intensity variation.
            frame_scale = 1.0 + 0.08 * np.sin((2.0 * np.pi * frame_idx) / max(2.0, float(n_frames)) + phase0)
            sim_param_frame = dict(sim_param)
            sim_param_frame["r0"] = float(np.clip(sim_param["r0"] / max(frame_scale, 0.6), 0.08, 6.0))
            frame_seed = int(rng.integers(0, 2**31 - 1))
            channels: list[np.ndarray] = []
            for channel_idx in range(clean.shape[2]):
                channel = clean[..., channel_idx]
                out = self._simulate_single_channel(
                    channel=channel,
                    sim_param=sim_param_frame,
                    shared_psd=shared_psd,
                    seed=frame_seed,
                )
                channels.append(out)

            frame_img = np.stack(channels, axis=-1).astype(np.float32)
            # Strength gate: keep structural fidelity for low-strength settings.
            frame_img = (1.0 - float(sim_param["blend_alpha"])) * clean + float(sim_param["blend_alpha"]) * frame_img
            frame_img = np.clip(frame_img, 0.0, 1.0).astype(np.float32)
            frame_meta = {
                "frame_idx": frame_idx,
                "sim_backend": self.name,
                "cn2": float(context.get("cn2", 0.0)),
                "focal_length": float(context.get("focal_length", 0.0)),
                "wind_speed": float(context.get("wind_speed", 0.0)),
                "time_step": float(context.get("time_step", 0.03)),
                "r0": float(sim_param["r0"]),
                "r0_frame": float(sim_param_frame["r0"]),
                "aperture_diameter": float(sim_param["D"]),
                "wavelength": float(sim_param["wvl"]),
                "turbulence_strength": float(sim_param["turbulence_strength"]),
                "blend_alpha": float(sim_param["blend_alpha"]),
            }
            frames.append(frame_img)
            metas.append(frame_meta)

        return np.stack(frames, axis=0).astype(np.float32), metas
