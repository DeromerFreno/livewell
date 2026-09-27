from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from livewell.gear.schema import ExperimentSpec


def _deep_merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in top.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _assign(tree: dict[str, Any], dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    cursor = tree
    for key in keys[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    cursor[keys[-1]] = _coerce(raw)


def load_yaml_tree(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    base = load_yaml_tree((path.parent / str(parent)).resolve())
    return _deep_merge(base, raw)


def build_experiment(path: Path, overrides: list[str] | None = None) -> ExperimentSpec:
    tree = load_yaml_tree(path)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        dotted, raw = item.split("=", 1)
        _assign(tree, dotted.strip(), raw.strip())
    return ExperimentSpec.model_validate(tree)
