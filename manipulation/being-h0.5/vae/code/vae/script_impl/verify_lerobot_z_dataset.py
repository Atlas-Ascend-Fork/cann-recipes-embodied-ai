#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot_z_common import (
    EXPECTED_DIM, PRESERVED_DIM, SHADOW_JOINT_DIM, STATE_KEYS, WRIST_DIM, Z_DIM,
    encode_in_batches, sha256_file,
)
from native_vae import NativeVAE

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
                        'Verify that a z_gesture LeRobot conversion preserves wrist and Shadow joints exa'
                        'ctly while replacing finite 24D latent values.'
                    )
    )
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--converted-dataset", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--atol",
        type=float,
        default=0.002,
        help="Absolute CPU-vs-NPU encoding tolerance (calibrated on all 17,700 rows).",
    )
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise RuntimeError("Metric input is empty or non-finite")
    return {
        "mae": float(np.mean(np.abs(flat))),
        "rmse": float(np.sqrt(np.mean(np.square(flat)))),
        "max_abs": float(np.max(np.abs(flat))),
    }


def main() -> None:
    _verify_dataset(parse_args())


@dataclass
class _ParquetContext:
    source: Path
    converted: Path
    vae: NativeVAE
    batch_size: int
    atol: float
    rtol: float


@dataclass
class _ParquetFileContext:
    source: Path
    converted: Path
    vae: NativeVAE
    batch_size: int
    atol: float
    rtol: float
    differences: dict[str, list[np.ndarray]]
    reencoding_errors: dict[str, list[np.ndarray]]


@dataclass
class _MetadataContext:
    source: Path
    converted: Path
    checkpoint: Path
    checkpoint_sha256: str
    parquet_count: int
    total_rows: int


@dataclass
class _ParquetVerification:
    relatives: list[Path]
    total_rows: int
    differences: dict[str, list[np.ndarray]]
    reencoding_errors: dict[str, list[np.ndarray]]


def _verify_parquet_file(context: _ParquetFileContext, relative: Path):
    source, converted, vae = context.source, context.converted, context.vae
    batch_size, atol, rtol = context.batch_size, context.atol, context.rtol
    differences, reencoding_errors = context.differences, context.reencoding_errors
    source_table = pq.read_table(source / relative)
    converted_table = pq.read_table(converted / relative)
    if source_table.num_rows != converted_table.num_rows or source_table.schema.names != converted_table.schema.names:
        raise RuntimeError(f"Parquet structure changed: {relative}")
    for name in source_table.schema.names:
        if name not in STATE_KEYS and not source_table[name].equals(converted_table[name]):
            raise RuntimeError(f"Non-target parquet column changed at {relative}:{name}")
    for key in STATE_KEYS:
        source_values = np.asarray(source_table[key].combine_chunks().to_pylist(), dtype=np.float32)
        converted_values = np.asarray(converted_table[key].combine_chunks().to_pylist(), dtype=np.float32)
        if (
            source_values.shape != converted_values.shape
            or source_values.ndim != 2
            or source_values.shape[1] != EXPECTED_DIM
        ):
            raise RuntimeError(f"Invalid shape at {relative}:{key}: {source_values.shape}")
        if not np.array_equal(source_values[:, :PRESERVED_DIM], converted_values[:, :PRESERVED_DIM]):
            raise RuntimeError(f"Preserved prefix changed at {relative}:{key}")
        latent = converted_values[:, PRESERVED_DIM:]
        if latent.shape[1] != EXPECTED_DIM - PRESERVED_DIM or not np.isfinite(latent).all():
            raise RuntimeError(f"Invalid latent at {relative}:{key}")
        expected = encode_in_batches(vae, source_values[:, WRIST_DIM:WRIST_DIM + SHADOW_JOINT_DIM], batch_size)
        error = latent - expected
        if not np.allclose(latent, expected, atol=atol, rtol=rtol):
            raise RuntimeError(f"Latent mismatch at {relative}:{key}; max_abs={float(np.max(np.abs(error)))}")
        differences[key].append(latent - source_values[:, PRESERVED_DIM:])
        reencoding_errors[key].append(error)
    return source_table.num_rows


def _verify_parquets(context: _ParquetContext):
    source, converted, vae = context.source, context.converted, context.vae
    batch_size, atol, rtol = context.batch_size, context.atol, context.rtol
    source_paths = sorted(source.glob("data/**/*.parquet"))
    converted_paths = sorted(converted.glob("data/**/*.parquet"))
    relatives = [path.relative_to(source) for path in source_paths]
    if relatives != [path.relative_to(converted) for path in converted_paths] or not relatives:
        raise RuntimeError("Source/converted parquet manifests differ")
    differences = {key: [] for key in STATE_KEYS}
    reencoding_errors = {key: [] for key in STATE_KEYS}
    total_rows = 0
    for index, relative in enumerate(relatives, start=1):
        rows = _verify_parquet_file(
            _ParquetFileContext(
                source, converted, vae, batch_size, atol, rtol,
                differences, reencoding_errors,
            ),
            relative,
        )
        total_rows += rows
        LOGGER.info("[verify] %s/%s %s rows=%s", index, len(relatives), relative, rows)
    return _ParquetVerification(relatives, total_rows, differences, reencoding_errors)


def _verify_metadata(context: _MetadataContext) -> int:
    source, converted = context.source, context.converted
    checkpoint, checkpoint_sha256 = context.checkpoint, context.checkpoint_sha256
    parquet_count, total_rows = context.parquet_count, context.total_rows
    source_videos = sorted(path.relative_to(source) for path in source.glob("videos/**/*.mp4"))
    converted_videos = sorted(path.relative_to(converted) for path in converted.glob("videos/**/*.mp4"))
    if source_videos != converted_videos:
        raise RuntimeError("Source/converted video manifests differ")
    changed = [
        str(path) for path in source_videos
        if (source / path).stat().st_size != (converted / path).stat().st_size
    ]
    if changed:
        raise RuntimeError(f"Copied video sizes differ: {changed[:5]}")
    required = (
        "meta/info.json", "meta/episodes.jsonl", "meta/episodes_stats.jsonl",
        "meta/tasks.jsonl", "meta/stats.json", "collection_metadata.json",
    )
    missing = [relative for relative in required if not (converted / relative).is_file()]
    if missing:
        raise RuntimeError(f"Missing converted metadata: {missing}")
    marker_path = converted / ".z_gesture_conversion_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else None
    if not isinstance(marker, dict):
        raise FileNotFoundError(f"Conversion marker is missing: {marker_path}")
    expected = {
        "source_dataset": str(source), "vae_checkpoint": str(checkpoint),
        "vae_checkpoint_sha256": checkpoint_sha256, "parquet_files": parquet_count,
        "frames": int(total_rows), "latent_dimension": Z_DIM,
        "encoding": "posterior_mean_z_mu", "roundtrip_verified": True,
        "preserved_prefix_dimensions": PRESERVED_DIM,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(f"Conversion marker mismatch for {key}")
    collection = json.loads((converted / "collection_metadata.json").read_text(encoding="utf-8"))
    metadata = collection.get("z_gesture")
    valid_metadata = (
        isinstance(metadata, dict)
        and metadata.get("checkpoint") == str(checkpoint)
        and metadata.get("checkpoint_sha256") == checkpoint_sha256
        and int(metadata.get("dimension", -1)) == Z_DIM
        and metadata.get("value") == "posterior mean z_mu (deterministic)"
    )
    if not valid_metadata:
        raise RuntimeError("collection_metadata z_gesture metadata is invalid")
    return len(source_videos)


def _verify_dataset(args: argparse.Namespace) -> None:
    source = args.source_dataset.expanduser().resolve()
    converted = args.converted_dataset.expanduser().resolve()
    checkpoint = args.vae_checkpoint.expanduser().resolve()
    if args.batch_size <= 0 or args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("batch size must be positive and tolerances must be non-negative")
    if not source.is_dir() or not converted.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError(f"source={source} converted={converted} checkpoint={checkpoint}")
    vae = NativeVAE.from_pretrained(checkpoint=checkpoint, device=args.device)
    checkpoint_sha256 = sha256_file(checkpoint)
    result = _verify_parquets(_ParquetContext(source, converted, vae, args.batch_size, args.atol, args.rtol))
    relatives = result.relatives
    total_rows, differences = result.total_rows, result.differences
    errors = result.reencoding_errors
    video_count = _verify_metadata(_MetadataContext(
        source, converted, checkpoint, checkpoint_sha256, len(relatives), total_rows
    ))
    report = {
        "source_dataset": str(source), "converted_dataset": str(converted), "parquet_files": len(relatives),
        "video_files": video_count, "rows": int(total_rows), "preserved_prefix_dimensions": PRESERVED_DIM,
        "latent_dimensions": EXPECTED_DIM - PRESERVED_DIM,
        "latent_change_vs_source": {key: stats(np.concatenate(chunks, axis=0)) for key, chunks in differences.items()},
        "vae_checkpoint": str(checkpoint), "vae_checkpoint_sha256": checkpoint_sha256,
        "verification_device": args.device, "content_tolerance": {"atol": args.atol, "rtol": args.rtol},
        "latent_reencoding_error": {key: stats(np.concatenate(chunks, axis=0)) for key, chunks in errors.items()},
        "non_target_parquet_columns_exact": True, "marker_verified": True, "collection_metadata_verified": True,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("[done] %s: rows=%s", output, total_rows)

if __name__ == "__main__":
    main()
