from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from livewell.hatchery.cohort import CohortRecordError, load_cohort

_REQUIRED = (
    "label",
    "dynamics_score",
    "endpoint_score",
    "dynamics_call",
    "endpoint_call",
    "center",
    "region",
    "survival_time",
    "survival_event",
    "survival_group",
    "lead_times",
)


def _degenerate_records() -> dict[str, Any]:
    """A deliberately invalid structure, used only to trigger a rejection.

    Every field is the smallest degenerate value of the wrong kind for a study
    cohort. It is never valid input, is written straight into a temporary
    directory, and is not part of the package.
    """
    records: dict[str, Any] = {key: [0] for key in _REQUIRED}
    records["lead_times"] = [0, 0]
    return records


def _write(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "records.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CohortRecordError):
        load_cohort(tmp_path / "absent.json")


def test_non_json_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text("not a json document", encoding="utf-8")
    with pytest.raises(CohortRecordError):
        load_cohort(path)


def test_non_object_document_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CohortRecordError):
        load_cohort(_write(tmp_path, [1, 2, 3]))


def test_empty_object_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CohortRecordError):
        load_cohort(_write(tmp_path, {}))


def test_missing_required_key_is_rejected(tmp_path: Path) -> None:
    records = _degenerate_records()
    del records["survival_group"]
    with pytest.raises(CohortRecordError):
        load_cohort(_write(tmp_path, records))


def test_non_binary_label_is_rejected(tmp_path: Path) -> None:
    records = _degenerate_records()
    records["label"] = [2]
    with pytest.raises(CohortRecordError):
        load_cohort(_write(tmp_path, records))


def test_non_finite_value_is_rejected(tmp_path: Path) -> None:
    records = _degenerate_records()
    records["dynamics_score"] = [float("nan")]
    with pytest.raises(CohortRecordError):
        load_cohort(_write(tmp_path, records))


def test_empty_column_is_rejected(tmp_path: Path) -> None:
    records = _degenerate_records()
    records["survival_time"] = []
    with pytest.raises(CohortRecordError):
        load_cohort(_write(tmp_path, records))


def test_pair_length_mismatch_is_rejected(tmp_path: Path) -> None:
    records = _degenerate_records()
    records["dynamics_call"] = [0, 1]
    with pytest.raises(CohortRecordError):
        load_cohort(_write(tmp_path, records))


def test_short_lead_time_series_is_rejected(tmp_path: Path) -> None:
    records = _degenerate_records()
    records["lead_times"] = [0]
    with pytest.raises(CohortRecordError):
        load_cohort(_write(tmp_path, records))
