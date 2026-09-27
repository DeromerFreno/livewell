from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
        root = logging.getLogger("livewell")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"livewell.{name}")


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def write_json(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    with os.fdopen(fd, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)
