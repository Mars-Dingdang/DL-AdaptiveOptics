#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


DEFAULT_STRENGTHS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep turbulence_strength, optionally build eval sets, run restoration model evaluation, "
            "and plot PSNR/SSIM curves."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_unet.pt"))
    parser.add_argument("--input-root", type=Path, default=Path("data/clean_patches/nwpu_parquet/images"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for per-strength outputs, metrics, and summary plots.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="",
        choices=["", "unet", "gan", "diffusion", "vae"],
        help="Optional model type override passed through to eval.py.",
    )
    parser.add_argument("--start-index", type=int, default=2000)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-save", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument(
        "--strengths",
        type=int,
        nargs="+",
        default=DEFAULT_STRENGTHS,
        help="Use integer percentages, e.g. 50 55 60 ... 100",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a strength if both metrics files already exist.",
    )
    parser.add_argument(
        "--reuse-existing-datasets",
        action="store_true",
        help="Reuse prebuilt data/eval_{seven,single}_frame_mildXX datasets instead of rebuilding them.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter used to run child scripts.",
    )
    return parser.parse_args()


def infer_output_root(checkpoint: Path, model_type: str) -> Path:
    suffix = model_type.strip().lower()
    if not suffix:
        checkpoint_name = checkpoint.stem.lower()
        if "wgan" in checkpoint_name or "gan" in checkpoint_name:
            suffix = "gan"
        elif "diffusion" in checkpoint_name:
            suffix = "diffusion"
        elif "vae" in checkpoint_name:
            suffix = "vae"
        else:
            suffix = "unet"
    return Path(f"outputs/eval_strength_sweep_{suffix}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def parse_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("")
    print("[RUN]", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env)


def build_row(strength_percent: int, single_data: dict[str, float], seven_data: dict[str, float]) -> dict[str, float | int | None]:
    return {
        "strength_percent": strength_percent,
        "turbulence_strength": strength_percent / 100.0,
        "psnr_single": single_data.get("psnr"),
        "ssim_single": single_data.get("ssim"),
        "psnr_seven": seven_data.get("psnr"),
        "ssim_seven": seven_data.get("ssim"),
    }


def write_summary_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strength_percent",
        "turbulence_strength",
        "psnr_single",
        "ssim_single",
        "psnr_seven",
        "ssim_seven",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(
    x_values: list[float],
    y_single: list[float],
    y_seven: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_single, marker="o", linewidth=2, label="Single frame")
    plt.plot(x_values, y_seven, marker="s", linewidth=2, label="Seven frame")
    plt.xlabel("Turbulence strength")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    repo_root = Path.cwd()
    output_root = args.output_root if args.output_root is not None else infer_output_root(args.checkpoint, args.model_type)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    rows: list[dict] = []

    for strength_percent in args.strengths:
        strength_value = strength_percent / 100.0
        suffix = f"mild{strength_percent}"

        seven_lmdb = repo_root / f"data/eval_seven_frame_{suffix}"
        single_lmdb = repo_root / f"data/eval_single_frame_{suffix}"
        seven_build_viz_dir = output_root / "build_viz" / "seven_frame" / suffix
        single_build_viz_dir = output_root / "build_viz" / "single_frame" / suffix
        seven_eval_dir = output_root / "seven_frame" / suffix
        single_eval_dir = output_root / "single_frame" / suffix
        seven_sample_dir = seven_eval_dir / "samples"
        single_sample_dir = single_eval_dir / "samples"

        seven_metrics = metrics_dir / f"metric_seven_frame_{suffix}.txt"
        single_metrics = metrics_dir / f"metric_single_frame_{suffix}.txt"

        if args.skip_existing and seven_metrics.exists() and single_metrics.exists():
            seven_data = parse_metrics(seven_metrics)
            single_data = parse_metrics(single_metrics)
            rows.append(build_row(strength_percent, single_data, seven_data))
            print(f"[SKIP] strength={strength_value:.2f} uses existing metrics")
            continue

        cfg = load_yaml(args.config)
        tmp_path: Path | None = None
        if not args.reuse_existing_datasets:
            cfg.setdefault("degradation", {})
            cfg["degradation"]["turbulence_strength"] = float(strength_value)

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                prefix=f"eval_strength_{strength_percent}_",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp_path = Path(tmp.name)
        try:
            eval_config_path = args.config
            if tmp_path is not None:
                dump_yaml(cfg, tmp_path)
                eval_config_path = tmp_path

                run_command(
                    [
                        args.python,
                        "data/build_seven_frame_eval.py",
                        "--config", str(eval_config_path),
                        "--input-root", str(args.input_root),
                        "--output-lmdb-root", str(seven_lmdb),
                        "--viz-dir", str(seven_build_viz_dir),
                        "--start-index", str(args.start_index),
                        "--count", str(args.count),
                        "--num-frames", str(args.num_frames),
                        "--seed", str(args.seed),
                        "--force",
                    ],
                    env=env,
                )

            run_command(
                [
                    args.python,
                    "eval.py",
                    "--config", str(eval_config_path),
                    "--checkpoint", str(args.checkpoint),
                    "--model-type", str(args.model_type),
                    "--split", "test",
                    "--test-root", str(seven_lmdb),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--out-dir", str(seven_eval_dir),
                    "--sample-dir", str(seven_sample_dir),
                    "--metrics-path", str(seven_metrics),
                    "--save-images",
                    "--max-save", str(args.max_save),
                ],
                env=env,
            )

            if tmp_path is not None:
                run_command(
                    [
                        args.python,
                        "data/build_single_frame_eval.py",
                        "--config", str(eval_config_path),
                        "--input-root", str(args.input_root),
                        "--output-lmdb-root", str(single_lmdb),
                        "--viz-dir", str(single_build_viz_dir),
                        "--start-index", str(args.start_index),
                        "--count", str(args.count),
                        "--num-frames", str(args.num_frames),
                        "--seed", str(args.seed),
                        "--force",
                    ],
                    env=env,
                )

            run_command(
                [
                    args.python,
                    "eval.py",
                    "--config", str(eval_config_path),
                    "--checkpoint", str(args.checkpoint),
                    "--model-type", str(args.model_type),
                    "--split", "test",
                    "--test-root", str(single_lmdb),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--out-dir", str(single_eval_dir),
                    "--sample-dir", str(single_sample_dir),
                    "--metrics-path", str(single_metrics),
                    "--save-images",
                    "--max-save", str(args.max_save),
                ],
                env=env,
            )

        finally:
            try:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        seven_data = parse_metrics(seven_metrics)
        single_data = parse_metrics(single_metrics)

        row = build_row(strength_percent, single_data, seven_data)
        rows.append(row)

        print(
            "[DONE] strength={:.2f} | single: PSNR={:.4f}, SSIM={:.4f} | seven: PSNR={:.4f}, SSIM={:.4f}".format(
                strength_value,
                row["psnr_single"],
                row["ssim_single"],
                row["psnr_seven"],
                row["ssim_seven"],
            )
        )

    rows.sort(key=lambda item: item["turbulence_strength"])

    csv_path = summary_dir / "strength_sweep_metrics.csv"
    write_summary_csv(rows, csv_path)

    x_values = [row["turbulence_strength"] for row in rows]
    psnr_single = [row["psnr_single"] for row in rows]
    psnr_seven = [row["psnr_seven"] for row in rows]
    ssim_single = [row["ssim_single"] for row in rows]
    ssim_seven = [row["ssim_seven"] for row in rows]

    plot_curve(
        x_values=x_values,
        y_single=psnr_single,
        y_seven=psnr_seven,
        ylabel="PSNR",
        title="PSNR vs Turbulence Strength",
        out_path=summary_dir / "psnr_vs_turbulence_strength.png",
    )

    plot_curve(
        x_values=x_values,
        y_single=ssim_single,
        y_seven=ssim_seven,
        ylabel="SSIM",
        title="SSIM vs Turbulence Strength",
        out_path=summary_dir / "ssim_vs_turbulence_strength.png",
    )

    print("")
    print(f"[SUMMARY] CSV: {csv_path}")
    print(f"[SUMMARY] PSNR plot: {summary_dir / 'psnr_vs_turbulence_strength.png'}")
    print(f"[SUMMARY] SSIM plot: {summary_dir / 'ssim_vs_turbulence_strength.png'}")


if __name__ == "__main__":
    main()