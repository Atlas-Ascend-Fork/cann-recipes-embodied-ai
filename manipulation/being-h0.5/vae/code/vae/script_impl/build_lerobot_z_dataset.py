#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from BeingH.dataset.parquet_utils import calculate_dataset_statistics
from lerobot_z_common import (
    EXPECTED_DIM,
    PRESERVED_DIM,
    SHADOW_JOINT_DIM,
    STATE_KEYS,
    WRIST_DIM,
    Z_DIM,
    encode_in_batches,
    sha256_file,
)
from native_vae import NativeVAE

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EncodeContext:
    source: Path
    output: Path
    parquet_paths: list[Path]
    vae: NativeVAE
    batch_size: int
    compression: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode Shadow joint state/action with a NativeVAE checkpoint "
            "and build a self-contained 52D LeRobot v2.1 dataset."
        )
    )
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def numeric_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def replace_column(table: pa.Table, key: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(key)
    if index < 0:
        raise KeyError(f"Missing parquet column {key!r}")
    field = table.schema.field(index)
    encoded = pa.array(values.tolist(), type=field.type)
    return table.set_column(index, field, encoded)


def update_collection_metadata(
    output_dataset: Path, source_dataset: Path, checkpoint: Path, checkpoint_sha256: str
) -> None:
    path = output_dataset / "collection_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metadata["source_dataset"] = str(source_dataset.resolve())
    metadata["repo_id"] = f"local/{output_dataset.name}"
    metadata["state"] = "52D: measured wrist 6D + measured Shadow finger 22D + state z_gesture 24D"
    metadata["action"] = "52D: target wrist 6D + target Shadow finger 22D + action z_gesture 24D"
    metadata["z_gesture"] = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "dimension": Z_DIM,
        "value": "posterior mean z_mu (deterministic)",
        "state_source": "observation.state[6:28]",
        "action_source": "action[6:28]",
        "builder": str(Path(__file__).resolve()),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    _build_dataset(parse_args())


def _encode_parquet_files(context: _EncodeContext) -> dict[int, dict[str, Any]]:
    source, output, parquet_paths = context.source, context.output, context.parquet_paths
    vae, batch_size, compression = context.vae, context.batch_size, context.compression
    episode_stats: dict[int, dict[str, Any]] = {}
    for item_index, source_parquet in enumerate(parquet_paths, start=1):
        relative = source_parquet.relative_to(source)
        output_parquet = output / relative
        table = pq.read_table(source_parquet)
        updated_values: dict[str, np.ndarray] = {}
        for key in STATE_KEYS:
            raw = np.asarray(
                table[key].combine_chunks().to_pylist(), dtype=np.float32
            )
            if raw.ndim != 2 or raw.shape[1] != EXPECTED_DIM:
                raise ValueError(f"{source_parquet}:{key} expected [N,{EXPECTED_DIM}], got {raw.shape}")
            z = encode_in_batches(
                vae, raw[:, WRIST_DIM:WRIST_DIM + SHADOW_JOINT_DIM], batch_size
            )
            values = raw.copy()
            values[:, WRIST_DIM + SHADOW_JOINT_DIM:] = z
            updated_values[key] = values
            table = replace_column(table, key, values)
        pq.write_table(table, output_parquet, compression=compression)
        roundtrip = pq.read_table(output_parquet)
        for name in table.schema.names:
            if name not in STATE_KEYS and not table[name].equals(roundtrip[name]):
                raise RuntimeError(f"Non-target column changed on write at {relative}:{name}")
        for key, expected in updated_values.items():
            actual = np.asarray(roundtrip[key].combine_chunks().to_pylist(), dtype=np.float32)
            if not np.array_equal(actual, expected):
                maximum = float(np.max(np.abs(actual - expected)))
                raise RuntimeError(
                    f"Encoded values changed on parquet round-trip at {relative}:{key}; "
                    f"max_abs={maximum}"
                )
        episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        episode_index = int(episode_values[0])
        if not np.all(episode_values == episode_index):
            raise ValueError(f"Mixed episode indices in {source_parquet}")
        episode_stats[episode_index] = {
            key: numeric_stats(value) for key, value in updated_values.items()
        }
        LOGGER.info(
            "[encode] %s/%s episode=%s frames=%s",
            item_index, len(parquet_paths), episode_index, table.num_rows,
        )
    return episode_stats


def write_episode_stats(output: Path, episode_stats: dict[int, dict[str, Any]]) -> None:
    """Merge episode statistics and preserve one JSON record per line."""
    stats_path = output / "meta/episodes_stats.jsonl"
    existing = (
        [
            json.loads(line)
            for line in stats_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if stats_path.exists()
        else []
    )
    records = {int(record["episode_index"]): record for record in existing}
    for episode_index, values in episode_stats.items():
        record = records.setdefault(episode_index, {"episode_index": episode_index, "stats": {}})
        record.setdefault("stats", {}).update(values)
    stats_path.write_text(
        "".join(json.dumps(records[index], ensure_ascii=False) + "\n" for index in sorted(records)),
        encoding="utf-8",
    )


def _prepare_dataset(args: argparse.Namespace) -> tuple[Path, Path, Path, list[Path], str]:
    source = args.source_dataset.expanduser().resolve()
    output = args.output_dataset.expanduser().resolve()
    checkpoint = args.vae_checkpoint.expanduser().resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint}")
    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}. Choose a new directory.")
    parquet_paths = sorted(source.glob("data/**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files below {source / 'data'}")
    shutil.copytree(source, output, ignore=shutil.ignore_patterns(".cache"))
    return source, output, checkpoint, parquet_paths, sha256_file(checkpoint)


def _finalize_dataset(output: Path, source: Path, checkpoint: Path, checkpoint_sha256: str) -> None:
    output_parquets = sorted(output.glob("data/**/*.parquet"))
    statistics = calculate_dataset_statistics(output_parquets)
    (output / "meta/stats.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    update_collection_metadata(output, source, checkpoint, checkpoint_sha256)
    frame_count = sum(pq.read_metadata(path).num_rows for path in output_parquets)
    marker = output / ".z_gesture_conversion_complete.json"
    marker.write_text(json.dumps({
        "source_dataset": str(source), "vae_checkpoint": str(checkpoint),
        "vae_checkpoint_sha256": checkpoint_sha256, "parquet_files": len(output_parquets),
        "frames": frame_count, "latent_dimension": Z_DIM, "encoding": "posterior_mean_z_mu",
        "roundtrip_verified": True, "preserved_prefix_dimensions": PRESERVED_DIM,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("[done] %s: episodes=%s frames=%s marker=%s", output, len(output_parquets), frame_count, marker)


def _build_dataset(args: argparse.Namespace) -> None:
    source, output, checkpoint, parquet_paths, checkpoint_sha256 = _prepare_dataset(args)
    LOGGER.info("[copy] %s -> %s", source, output)
    vae = NativeVAE.from_pretrained(checkpoint=checkpoint, device=args.device)
    episode_stats = _encode_parquet_files(_EncodeContext(
        source, output, parquet_paths, vae, args.batch_size, args.compression
    ))
    write_episode_stats(output, episode_stats)
    _finalize_dataset(output, source, checkpoint, checkpoint_sha256)


def main() -> None:
    _build_dataset(parse_args())


if __name__ == "__main__":
    main()
