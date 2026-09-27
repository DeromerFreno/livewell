#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment/main.yaml}"
python -m livewell.panel steer --config "${CONFIG}" --seed 1
python -m livewell.panel correlate --config "${CONFIG}" --records "${COHORT_RECORDS:?set COHORT_RECORDS to the de-identified cohort records path}"
python -m livewell.panel ablate --config "${CONFIG}" --seed 1
