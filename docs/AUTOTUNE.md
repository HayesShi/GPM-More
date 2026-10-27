# Automatic per-dataset tuning

The queue runs a finite validation search on the complete raw_sum model. Its
full-feedback arm is a matched raw_sum control. Both arms use the same forward
computation, training horizon, data, walk caches, batch size, and clipping rule.

Copy `configs/autotune.example.json`, supply prepared data and genuine structural
pretraining checkpoints, then run:

```powershell
python -m gpm_more.cli autotune --plan configs/my-autotune.json --output runs/autotune
```

Paths in the plan are resolved relative to the plan file. Checkpoints and datasets
are user-supplied assets and are not distributed with this repository. The current
input validator supports historical `natural-pattern-checkpoint-v1` structural
pretraining checkpoints with `phase=pretrain`, matching dataset/seed,
and `test_labels_read=false`. An adjacent `complete.json` must record 1,000
completed pretraining updates and identify the saved structural-risk-selected
step, which can precede update 1,000. Every active encoder tensor must exist and have the
correct shape. Inactive historical tensors are listed in the inventory.

The validator also supports `paper-aligned-natural-gpm-v1` structural pretraining
for Photo, Pubmed, and Computers. These sources must be the `latest.pt` resume
state at fixed final epoch 100, with the matching completed 100-epoch receipt,
incidence_rotation configuration, constant pretraining LR, and independent source
seed. The transferred checkpoint is not selected by a downstream validation probe.
The historical receipt may name its validation-selected `best.pt`; the inventory
distinguishes that recorded file from the actual final `latest.pt` transferred.
Historical probes reset RNG streams and affect subsequent encoder dropout; this
is recorded as source provenance. Probe heads receive frozen, detached encoder
embeddings. The new optimizer never imports source optimizer or probe-head state.

The example searches three base learning rates, two late learning-rate factors,
and two feedback families: `full` and `late_stop` after update 400. This gives six
candidates per family and twelve per dataset. Late LR scaling starts after the
same boundary in both families. Each candidate trains for all 1,000 updates;
validation uses 12 walks every 50 updates, including the initial state. All other
model and optimizer settings remain fixed.

Fixed random seeds alone did not preserve the CUDA training prefix in a local
matched-feedback check. The queue therefore requires deterministic execution. It sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, enables
`torch.use_deterministic_algorithms(True)` and cuDNN deterministic kernels, and
disables cuDNN benchmarking and TF32. Separate full and late-stop runs
with the same seed and LR recipe must have exactly matching training prefixes
through the feedback-switch boundary. Execution settings are saved in complete
checkpoints and checked on resume. Unsupported deterministic operations raise an
error instead of silently continuing with a different prefix.

Only seed 0 selects hyperparameters. After every search candidate for a dataset
completes successfully, the queue freezes one winner per feedback family in
`selections/<dataset>.json`. The primary score is best validation R2 for
regression, AP for binary classification, or accuracy for multiclass
classification. Exact score ties use the earlier candidate in the declared grid;
checkpoint ties retain the earlier update. The two selected recipes then run with
seeds 1 and 2. Their results never change the selected configurations. A failed
candidate prevents selection for its dataset and remains visible with its log.

Each fine-tuning seed uses the corresponding independently pretrained source
seed, then resets only the existing Linear task head with CPU seed
`70000000 + seed`. No optimizer state is imported from structural pretraining.
All candidates sharing a seed receive identical initial active weights and task
head. `train --init-weights` retains its original behavior by default; add
`--reset-head` to request this task-head reset explicitly. Only reset head tensors
may differ in output dimension; every active encoder tensor is loaded strictly.

Report seed-0 search results separately from confirmation seeds 1 and 2. A mean
over seeds 0, 1, and 2 includes the selection seed and should say so. The selection
record also identifies the seed-0 full-feedback run with the same base LR and
late LR factor as the winning late-stop run. This matched counterpart helps
separate the feedback effect from selecting a different optimization recipe.
The trainer never reads test labels.

The output includes:

- `plan.json` and `inventory.json`: frozen resolved plan and input identities.
- `code_snapshot/`: the package copied before training; every child imports this
  snapshot so later repository edits cannot change an ongoing comparison.
- `state.json`, `summary.json`, `summary.csv`, and `SUMMARY.md`: live queue status,
  primary validation scores, selected/best updates, timings, and seed mapping.
- `configs/`, `logs/`, `failures/`, and `events.jsonl`: every candidate, child log,
  failure record, and queue transition.
- `runs/<job>/`: best/latest checkpoints, validation predictions, training steps,
  curves, and progress. Completed jobs are never launched twice.

Rerun the same command to resume. The queue rejects changes to its frozen plan.
A partially trained trial resumes its complete checkpoint with optimizer,
scheduler, and random-generator state. A trial interrupted before its first
checkpoint is preserved under `failures/` before initialization is retried.

Create `STOP_REQUESTED` inside the output directory to stop at the next validation
checkpoint. Remove that file before resuming. The queue also stops at a checkpoint
when its cumulative wall budget expires, and pauses between jobs if free output
disk space falls below `min_free_gb`. The wall budget can overrun by the time needed
to reach the next validation checkpoint. The maximum is three datasets, twelve
search candidates each, and three unique seeds. With the example grid, each
dataset requires twelve search and four confirmation runs.

Only one training child is launched at a time. OS locks prevent duplicate queue
workers; optional `gpu_lock` sets a shared lock path for cooperating campaigns.
Persisted child PIDs and a shared GPU lease prevent a restarted queue from
overlapping a still-running child after its former worker exits. Checkpoints from
an unrelated trainer cannot be resumed as this trainer's optimizer trajectory.

On Windows, a hidden background worker can be launched with redirected logs:

```powershell
Start-Process -FilePath python -ArgumentList '-m gpm_more.cli autotune --plan configs/my-autotune.json --output runs/autotune' -WindowStyle Hidden -RedirectStandardOutput queue.stdout.log -RedirectStandardError queue.stderr.log
```

The queue is a local validation experiment runner. It does not edit paper tables
or schedule a test-set evaluation.

## More datasets and sequential campaigns

`configs/autotune.classification.example.json` defines the same twelve-candidate
search for Photo, Pubmed, and Computers: 36 seed-0 search runs and twelve frozen
confirmation runs. It uses user-supplied task caches and final structural
pretraining sources. A two-dataset campaign followed by this three-dataset
campaign covers five datasets without changing an existing frozen plan.

For a graph NPZ containing train/validation mask matrices, select a column
explicitly during preparation:

```powershell
python -m gpm_more.cli prepare --graph sources/photo/task.npz --output data/photo --name photo --task multiclass_classification --split-index 0 --seed 0
```

This decodes only the selected column's train/validation label rows. Keep the
same original split column for all fine-tuning seeds so seed effects are not
confounded with split changes. The historical Photo/Pubmed/Computers source
caches contain three seed-42 low-label 10/10/80 splits. Column 0 is preserved for
this protocol; raw official features are retained. Their earlier 128-SRW bank
belongs to pretraining. This preparation generates a declared new 16-SRW/8-NBRW
cache shared by every candidate and fine-tuning seed. The earlier official loader
cached complete labels; that historical access is distinct from selected-row
extraction and the current validation search.

To queue additional campaigns behind an existing campaign:

```powershell
python -m gpm_more.cli dispatch --plan configs/dispatch.example.json --output runs/dispatcher
```

The dispatcher saves its resolved plan, copied campaign recipes, package snapshot,
`state.json`, `summary.json`, `SUMMARY.md`, and transition log. Status reports
include planned search/confirmation counts, the dispatcher PID, prerequisite
queue PID, and active campaign. It checks prerequisites every two seconds in the
example, waits for shared GPU locks, and runs one campaign at a time. Editing a
referenced recipe while it waits cannot change its frozen copy. A restart rejects
recipe changes and never restarts a campaign recorded as complete.

A completed predecessor permits the next campaign. A terminal failed or
wall-budget predecessor can also permit it when no stop file or training child
remains; the predecessor's status remains visible. A predecessor with a
`STOP_REQUESTED` file, low disk status, or an unfinished queue whose worker has
exited holds dispatch for recovery. A busy GPU lock is a waiting state.

Create `STOP_REQUESTED` inside the dispatcher output to stop waiting. During an
active campaign it forwards the stop to that campaign's `STOP_REQUESTED` file,
which stops at the next validation checkpoint. Remove both files before an
authorized resume. The dispatcher never clears or bypasses a predecessor's stop
file. Its own wall budget likewise requests a checkpoint stop in an active
campaign; each campaign also keeps its separate frozen budget.
