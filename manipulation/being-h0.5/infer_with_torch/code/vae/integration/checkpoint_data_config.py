# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the data transform recorded by a Being-H checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection


def _values(value: Any, keys: set[str]) -> list[str]:
    if isinstance(value, dict):
        found: list[str] = []
        for key, child in value.items():
            if key in keys:
                found += (
                    [child]
                    if isinstance(child, str)
                    else [x for x in child if isinstance(x, str)]
                    if isinstance(child, list)
                    else []
                )
            found += _values(child, keys)
        return found
    if isinstance(value, list):
        return [item for child in value for item in _values(child, keys)]
    return []


def _select(names: list[str], supported: set[str], source: str) -> tuple[str, str] | None:
    names = list(dict.fromkeys(names))
    if not names:
        return None
    unsupported = sorted(set(names) - supported)
    if unsupported:
        raise ValueError(f"{source} records unsupported data config(s) {unsupported}; supported={sorted(supported)}")
    if len(names) != 1:
        raise ValueError(f"{source} records multiple data configs {names}; pass --data-config-name explicitly")
    return (names[0], source)


_collect_named_values = _values


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _legacy_config_paths(checkpoint: Path) -> list[Path]:
    paths: list[Path] = []
    for root in (checkpoint, checkpoint.parent):
        run_config = root / "run_config"
        for suffix in ("*.yaml", "*.yml"):
            paths.extend(sorted(run_config.glob(suffix)))
    return list(dict.fromkeys(paths))


def resolve_checkpoint_data_config(
    checkpoint: Path, requested: str | None, supported: Collection[str]
) -> tuple[str, str]:
    checkpoint, allowed = (checkpoint.expanduser().resolve(), set(supported))
    if requested is not None:
        if requested not in allowed:
            raise ValueError(f"Unsupported data config {requested!r}; supported={sorted(allowed)}")
        return (requested, "command line")
    config = checkpoint / "config.json"
    if config.is_file():
        selected = _select(
            _values(json.loads(config.read_text(encoding="utf-8")), {"data_config_name", "data_config_names"}),
            allowed,
            str(config),
        )
        if selected:
            return selected
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to inspect legacy run_config") from error
    paths = _legacy_config_paths(checkpoint)
    names = [
        name
        for path in paths
        for name in _values(yaml.safe_load(path.read_text(encoding="utf-8")), {"data_config_names"})
    ]
    selected = _select(names, allowed, ", ".join(map(str, paths)))
    if selected:
        return selected
    raise ValueError(
        f"Cannot determine the data config used by checkpoint {checkpoint}; "
        "pass --data-config-name explicitly for this legacy checkpoint"
    )
