# livewell

A perfused hepatocellular-carcinoma chip behaves like a small aquarium: a living
load sits inside a water loop whose chemistry drifts, and the only way to keep
the load on a chosen trajectory is to read the water continuously and adjust the
pumps. `livewell` is the control software for that loop. It reconstructs the
hidden remodeling state of the tumour microenvironment from a drifting
multimodal probe stream and steers perfusion, dissolved oxygen, and drug dosing
inside hard biological-safety limits, then correlates the resulting dynamics-aware
readout against documented organoid outcomes.

The package is laid out the way an aquarist keeps a system running:

| module | role |
| --- | --- |
| `reef` | the living plant - two-phase remodeling dynamics and the clean output map |
| `probes` | the in-situ sensor array - detection limits, multi-day drift, per-channel noise |
| `lens` | the Koopman observable dictionary and the lifted transition/identification |
| `sounding` | the moving-horizon estimator that reconstructs the latent state |
| `keeper` | the safety-constrained predictive controller and its hand-rolled QP |
| `pumps` | actuation, loop closure, and the hard safety set |
| `survey` | tracking, observability, and the clinical-correlation statistics |
| `hatchery` | the cohort container and the records loader for the access-controlled cohort |
| `cycling` | offline identification of the lift and the component-ablation sweep |
| `gear` | seeds, atomic checkpoints, logging, distributed wiring, configuration |
| `panel` | the command line |

## Filling the sump (installation)

```
python -m pip install -r requirements.txt
python -m pip install -e .
```

Conda:

```
conda env create -f environment.yml
conda activate livewell
python -m pip install -e .
```

Container:

```
docker build -t livewell .
docker run --gpus all livewell steer --config configs/experiment/main.yaml --seed 1
```

The pinned stack targets Python 3.8 with CUDA 11.7 and PyTorch 1.13.1; the
versions are fixed to keep the offline identification numerically repeatable.

## Water source (data)

There is no public training set. The matched clinical records that support the
clinical-correlation analysis are de-identified and are held under the access
terms of the governing ethics approval reported in the manuscript; they are not
redistributed here. Only the clinical-correlation step reads them, from your
access-controlled copy. `datasets.txt` lists the verified reference pages for the
three commercial cell lines named in the device characterization and records the
access terms.

This repository ships no cohort data. `panel correlate` reads the de-identified,
access-controlled cohort records from a file whose path you supply - either as
`--records PATH` or as `cohort.records` in the experiment configuration - and
exits non-zero with an explanatory message when none is given. The loader layout
is documented in `livewell/hatchery/cohort.py`. The study estimates computed from
that cohort are reported in the manuscript.

## Running the loop

Identify the lift offline (the heavy run; reduce it only through the
unit-test smoke profile):

```
python -m livewell.panel cycle --config configs/experiment/main.yaml --workdir runs
bash scripts/launch_cycle.sh configs/experiment/main.yaml runs
```

Steer a chip and compare against open-loop dosing:

```
python -m livewell.panel steer --config configs/experiment/main.yaml --seed 1
```

Expected: the closed loop cuts the trajectory-tracking error by at least one
third relative to open-loop dosing (reduction ~0.37 in the model-plant
evaluation), with every applied actuation inside the safety set.

Assemble the organoid clinical-correlation readout:

```
python -m livewell.panel correlate --config configs/experiment/main.yaml \
    --records /path/to/cohort_records.json
```

This step needs the de-identified, access-controlled cohort records, which are
not distributed with this repository, so the path is yours to supply. The command
prints the two AUROCs, the DeLong test, kappa, the Cox fit, and the lead-time
summary; the study estimates are reported in the manuscript.

Sweep the component ablations:

```
python -m livewell.panel ablate --config configs/experiment/main.yaml --seed 1
```

Expected: the full system keeps the one-third reduction; removing the
safety-constrained controller drops it to roughly the open-loop level, and
removing the hypoxia chamber lowers raw tracking error while erasing the
two-phase signal the device is built to read.

Export the observable dictionary:

```
python -m livewell.panel export --config configs/experiment/main.yaml --out lift.onnx
```

## Power and plumbing budget (compute)

The offline identification in `configs/experiment/main.yaml` targets eight
A100-80GB devices in fp32 with a per-device batch of 4096 and gradient
accumulation of 4 - an effective batch of 131072. It runs for 4000 epochs at a
learning rate of 3e-6 with an 8000-step warmup. This is a multi-day run on the
full hardware; it is not meant to be shortened for reporting. The online
estimator and controller run on a single CPU at a per-step latency well under
the hours-scale biology they manage.

## Bench checks (tests)

```
pytest -q
ruff check .
mypy --strict livewell
```

The suite covers single-trajectory overfitting of the lift, the observability
property behind the estimator, optimality of the moving-horizon solve, hard
feasibility of every controller action, the closed-loop tracking margin, the
survival and concordance statistics, the lead-time summary, the rejection paths
of the cohort-records loader, and a two-step smoke of the offline
identification.
