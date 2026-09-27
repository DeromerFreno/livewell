#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment/main.yaml}"
WORKDIR="${2:-runs}"
torchrun --standalone --nproc_per_node=8 -m livewell.panel cycle \
    --config "${CONFIG}" --workdir "${WORKDIR}"
