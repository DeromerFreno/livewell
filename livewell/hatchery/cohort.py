from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


@dataclass(frozen=True)
class Cohort:
    label: IntArray
    dynamics_score: Array
    endpoint_score: Array
    dynamics_call: IntArray
    endpoint_call: IntArray
    center: IntArray
    region: IntArray
    survival_time: Array
    survival_event: IntArray
    survival_group: IntArray
    lead_times: Array


class CohortRecordError(ValueError):
    """Raised when a cohort records file does not match the documented layout."""


_PAIR_KEYS = (
    "label",
    "dynamics_score",
    "endpoint_score",
    "dynamics_call",
    "endpoint_call",
    "center",
    "region",
)
_SURVIVAL_KEYS = ("survival_time", "survival_event", "survival_group")
_BINARY_KEYS = ("label", "dynamics_call", "endpoint_call", "survival_event", "survival_group")
_REQUIRED_KEYS = (*_PAIR_KEYS, *_SURVIVAL_KEYS, "lead_times")


def load_cohort(path: Path) -> Cohort:
    """Read the de-identified cohort records from ``path``.

    The records are access-controlled and are not distributed with this
    repository, so the caller supplies the path. The file is a single JSON
    object whose keys are the eleven :class:`Cohort` fields, each mapped to a
    flat list:

    ``label``, ``dynamics_score``, ``endpoint_score``, ``dynamics_call``,
    ``endpoint_call``, ``center`` and ``region`` carry one entry per
    organoid--outcome pair; ``survival_time``, ``survival_event`` and
    ``survival_group`` carry one entry per patient; ``lead_times`` carries one
    entry per recorded early-warning interval. The three groups need not be the
    same length.

    Raises :class:`CohortRecordError` if the file cannot be read, is not a JSON
    object, is missing a key, or holds a field that is not finite, not binary
    where it must be, or too short to summarise.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CohortRecordError(f"{path}: cannot read cohort records: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CohortRecordError(f"{path}: not a JSON document: {exc}") from exc
    if not isinstance(raw, dict):
        raise CohortRecordError(f"{path}: expected a JSON object, got {type(raw).__name__}")

    columns: dict[str, Array] = {}
    for key in _REQUIRED_KEYS:
        if key not in raw:
            raise CohortRecordError(f"{path}: missing required key {key!r}")
        try:
            column = np.asarray(raw[key], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CohortRecordError(f"{path}: key {key!r} is not a list of numbers: {exc}") from exc
        if column.ndim != 1:
            raise CohortRecordError(f"{path}: key {key!r} must be a flat list, got {column.ndim}D")
        if column.size == 0:
            raise CohortRecordError(f"{path}: key {key!r} is empty")
        if not bool(np.all(np.isfinite(column))):
            raise CohortRecordError(f"{path}: key {key!r} contains a non-finite value")
        columns[key] = column

    lengths = {key: int(column.size) for key, column in columns.items()}
    if len({lengths[key] for key in _PAIR_KEYS}) != 1:
        raise CohortRecordError(f"{path}: pair-level keys disagree on length: {lengths}")
    if len({lengths[key] for key in _SURVIVAL_KEYS}) != 1:
        raise CohortRecordError(f"{path}: survival keys disagree on length: {lengths}")
    if lengths["lead_times"] < 2:
        raise CohortRecordError(f"{path}: lead_times needs at least two entries")

    for key in _BINARY_KEYS:
        values = columns[key]
        if not bool(np.all(np.isin(values, (0.0, 1.0)))):
            raise CohortRecordError(f"{path}: key {key!r} must be binary, got {values.tolist()[:8]}")

    return Cohort(
        label=_as_int(columns["label"]),
        dynamics_score=_as_float(columns["dynamics_score"]),
        endpoint_score=_as_float(columns["endpoint_score"]),
        dynamics_call=_as_int(columns["dynamics_call"]),
        endpoint_call=_as_int(columns["endpoint_call"]),
        center=_as_int(columns["center"]),
        region=_as_int(columns["region"]),
        survival_time=_as_float(columns["survival_time"]),
        survival_event=_as_int(columns["survival_event"]),
        survival_group=_as_int(columns["survival_group"]),
        lead_times=_as_float(columns["lead_times"]),
    )


def _as_int(column: Array) -> IntArray:
    return np.rint(column).astype(np.int_)


def _as_float(column: Array) -> Array:
    return np.asarray(column, dtype=np.float64)
