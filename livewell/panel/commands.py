from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from cleo.commands.command import Command
from cleo.helpers import option

from livewell.cycling.ablation import ablation_sweep
from livewell.cycling.trainer import run_cycle
from livewell.gear.schema import ExperimentSpec
from livewell.gear.settings import build_experiment
from livewell.hatchery.cohort import load_cohort
from livewell.lens.affine import fit_affine_koopman
from livewell.lens.observables import KoopmanModel
from livewell.probes.array import SensorArray
from livewell.pumps.loop import ClosedLoop
from livewell.reef.plant import Plant
from livewell.survey.report import assemble

_CONFIG = option(
    "config", None, "Experiment configuration file", flag=False, default="configs/experiment/main.yaml"
)
_SET = option("set", None, "Override as dotted.key=value", flag=False, multiple=True)


def _resolve(command: Command) -> ExperimentSpec:
    path = Path(str(command.option("config")))
    overrides = list(command.option("set") or [])
    return build_experiment(path, overrides)


class CycleCommand(Command):
    name = "cycle"
    description = "Identify the Koopman lift offline from chip state transitions"
    options = [
        _CONFIG,
        _SET,
        option("workdir", None, "Checkpoint directory", flag=False, default="runs"),
        option("steps", None, "Optimizer-step budget", flag=False, default=None),
    ]

    def handle(self) -> int:
        spec = _resolve(self)
        budget = self.option("steps")
        record = run_cycle(
            spec,
            Path(str(self.option("workdir"))),
            int(budget) if budget is not None else None,
        )
        self.line(f"steps={record.steps} final_loss={record.final:.6f} checkpoint={record.checkpoint}")
        return 0


class SteerCommand(Command):
    name = "steer"
    description = "Run the closed loop against open-loop dosing and report tracking error"
    options = [
        _CONFIG,
        _SET,
        option("seed", None, "Random seed", flag=False, default="1"),
        option("steps", None, "Loop steps", flag=False, default="144"),
        option("resistance", None, "Phenotype resistance", flag=False, default="0.6"),
    ]

    def handle(self) -> int:
        spec = _resolve(self)
        plant = Plant(spec.plant)
        sensors = SensorArray(spec.probe, spec.plant.n_output, spec.plant.dt_hours)
        model = fit_affine_koopman(
            plant, spec.safety, float(str(self.option("resistance"))), np.random.default_rng(0)
        )
        loop = ClosedLoop(spec, plant, sensors, model, ablations=frozenset(spec.ablate))
        report = loop.evaluate(
            resistance=float(str(self.option("resistance"))),
            seed=int(str(self.option("seed"))),
            steps=int(str(self.option("steps"))),
        )
        self.line(
            f"open_track={report.open_track:.2f} closed_track={report.closed_track:.2f} "
            f"reduction={report.reduction:.3f} feasible={report.feasible}"
        )
        return 0


class CorrelateCommand(Command):
    name = "correlate"
    description = "Assemble the organoid clinical-correlation readout"
    options = [
        _CONFIG,
        _SET,
        option(
            "records",
            None,
            "Path to the de-identified cohort records file",
            flag=False,
            default=None,
        ),
    ]

    def handle(self) -> int:
        spec = _resolve(self)
        configured = self.option("records") or spec.cohort.records
        if configured is None:
            self.line_error(
                "no cohort records configured: pass --records PATH. The cohort is "
                "access-controlled and is not distributed with this repository."
            )
            return 1
        path = Path(str(configured))
        if not path.is_file():
            self.line_error(
                f"cohort records not found: {path}. The cohort is access-controlled "
                "and is not distributed with this repository."
            )
            return 1
        report = assemble(load_cohort(path))
        self.line(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return 0


class AblateCommand(Command):
    name = "ablate"
    description = "Sweep component removals and report tracking error"
    options = [
        _CONFIG,
        _SET,
        option("seed", None, "Random seed", flag=False, default="1"),
        option("steps", None, "Loop steps", flag=False, default="144"),
    ]

    def handle(self) -> int:
        spec = _resolve(self)
        rows = ablation_sweep(spec, int(str(self.option("seed"))), int(str(self.option("steps"))))
        for row in rows:
            self.line(f"{row.name:18s} track={row.track_error:6.2f} reduction={row.reduction:.3f}")
        return 0


class ExportCommand(Command):
    name = "export"
    description = "Export the observable dictionary to ONNX"
    options = [
        _CONFIG,
        _SET,
        option("checkpoint", None, "Checkpoint to load", flag=False, default=None),
        option("out", None, "ONNX output path", flag=False, default="lift.onnx"),
    ]

    def handle(self) -> int:
        spec = _resolve(self)
        plant = Plant(spec.plant)
        model = KoopmanModel(spec.lift, spec.plant, plant.output_matrix)
        ckpt = self.option("checkpoint")
        if ckpt is not None:
            state = torch.load(str(ckpt), map_location="cpu")
            model.load_state_dict(state["state"])
        model.eval()
        dummy = torch.zeros(1, spec.plant.n_state)
        torch.onnx.export(
            model.dictionary,
            (dummy,),
            str(self.option("out")),
            input_names=["state"],
            output_names=["observable"],
            dynamic_axes={"state": {0: "batch"}, "observable": {0: "batch"}},
            dynamo=False,
        )
        self.line(f"wrote {self.option('out')}")
        return 0
