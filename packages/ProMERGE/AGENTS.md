# ProMERGE package guidance

## Scope and working rules

This file applies to `packages/ProMERGE/` and its descendants. Follow
`../../AGENTS.md` and `/Users/jhgao/.codex/AGENTS.md` as well. This
package guide supplements the repository guidance at the user's request.
Read the current source before editing; README.md describes the public
interface but is not a substitute for checking the implementation.

- Keep changes within the requested scope. Preserve model behavior,
  public signatures, script interfaces, checkpoint keys, and output
  formats unless the user requests a change to them.
- This is the current standalone implementation. Do not substitute
  `cellmaps_coembedding.promerge` from the bundled baseline fork.
- Local source inspection and static checks are appropriate. Do not
  run training or full analyses locally, or interact with the remote
  server through SSH, SCP, rsync, or Slurm. Prepare server commands for
  the user; training belongs in the existing `cellmap` environment.
- Preserve inputs, generated outputs, and checkpoints. Do not use
  existing result directories for validation that could overwrite them.
- Do not install dependencies for documentation or static checks.
  Follow the parent instructions for environments and installations.
- Reload target files immediately before edits. This partial manual
  mirror is not necessarily a Git checkout; check before using Git.

## Package structure

| File | Responsibility |
| --- | --- |
| `runner.py` | Public generator, constructor defaults and validation, path parsing, stage orchestration, alignment, and row yielding. |
| `promerge/__init__.py` | Compatibility exports used by `runner.py` and other callers. |
| `promerge/model.py` | Compatibility exports for model classes, estimators, and layers. |
| `promerge/data.py` | Context protein unions, row maps, masks, negative sampling, anchors, and Procrustes alignment. |
| `promerge/base_model.py` | Per-modality base encoders and decoders. |
| `promerge/query_model.py` | Anchored query pathways, subtract fusion, copied decoders, adapters, and estimator attachment. |
| `promerge/layers.py` | MLP construction and identity-initialized recalibration. |
| `promerge/estimators.py` | CLUB Gaussian approximation and MINE critic/EMA. |
| `promerge/base_training.py` | Base construction, checkpoint loading, losses, and optimization. |
| `promerge/query_training.py` | Query losses, separate estimator updates, warmup/ramp, replay, and gate controller. |
| `promerge/losses.py` | Cosine, triplet, variance, mean-offset, and gate penalties. |
| `promerge/training_utils.py` | Seeds, shuffled batches, ramp, and snapshot scheduling. |
| `promerge/diagnostics.py` | Epoch statistics, latent geometry, progress printing, and non-stopping collapse warnings. |
| `promerge/inference.py` | Chunked evaluation-mode encoding and reconstruction under `no_grad`. |
| `promerge/outputs.py` | Aggregation, scores, support, TSVs, and model/snapshot writing. |

The only third-party imports are PyTorch, NumPy, and pandas. There is
no package installer metadata or package CLI. Existing analysis scripts
insert this directory on `sys.path` and import
`ProteinCoEmbeddingGenerator` from `runner`. Preserve imports through
both `promerge` and `promerge.model` when reorganizing modules.

## Execution and data contracts

1. `get_next_embedding()` is a lazy generator. Consuming it seeds RNGs,
   selects CUDA or CPU, resolves inputs, and reads every requested TSV.
2. All requested context-modality pairs must have existing files,
   including base inputs when `load_base` is set. Filenames split at
   the last hyphen and must match the requested names exactly.
3. `contexts[0]` is the base. It is trained or rebuilt from the saved
   state, then encoded to frozen per-modality anchors. A base-only run
   writes outputs and returns without yielding rows.
4. Each query is aligned and trained against the same base. Queries
   are not chained. Matching-width base decoders are deep-copied, then
   trained with query adapters and base replay; the base model itself
   is not updated by query training.
5. Query output is written before its combined latent rows are yielded.
   Exhaust the generator to finish all queries. Yielded rows contain
   no context label, and base rows are never yielded.

`ContextData` uses a sorted protein union and per-modality row maps with
`-1` for missing proteins. Dense batch placeholders must remain masked
out of applicable losses. Preserve input row order for per-modality
outputs and negative-sampling pools. There is no automatic imputation
or imputation provenance flag.

Alignment uses uncentered orthogonal Procrustes on shared proteins.
Different widths or insufficient overlap skip alignment, not training.
Width mismatch also prevents decoder reuse and replay for that modality;
it does not remove its latent anchors. Query reconstructions remain in
the coordinates used for training, with no inverse rotation.

## Model and training contracts

- Base training combines all ordered modality reconstruction pairs,
  cross-modality triplets, variance regularization, and mean-offset
  regularization. Reconstruction is cosine distance on jointly present
  proteins; triplet negatives come from the entire target modality pool.
- CLUB/MINE queries predict a normalized shared state, replacing it
  with the measured same-modality anchor where available. The contrast
  block receives trunk features, detached shared state, and anchor
  presence. It produces a normalized direction and scalar gate logit.
- `angle = gate_angle_max * sigmoid(logit)`,
  `specific = sin(angle) * direction`, and the combined latent is
  `cos(angle) * shared + specific`, normalized if `l2_norm=True`.
  There is no explicit orthogonality constraint. Preserve the six-item
  `encode()` return tuple used by training and inference.
- `subtract` uses normalized query encoding minus the shared state,
  then fuses shared and residual through an MLP. Missing same-modality
  anchors yield zero residual. It has no MI estimator or gate penalty;
  replay goes through fusion with a zero residual. Some unused gated
  modules are still constructed; removing them changes checkpoint keys.
- Query reconstruction uses decoder plus adapter. Replay bypasses the
  adapter and reconstructs sampled base inputs. Shared-anchor loss uses
  only proteins with both query measurements and same-modality anchors.
- Query triplets compare directions, weighted by detached gate products.
  Triplet and MI contributions ramp with epoch; estimator warmup only
  delays MI contribution to the main loss, not estimator fitting.
- Estimators have a separate optimizer and fit detached directions.
  Main parameters exclude `clubs.` and `mines.`. CLUB takes one fit
  step per batch and divides its main bound by latent dimension. MINE
  takes configurable steps, updates EMA only on critic fitting, and
  uses a zero-floored estimate without dimension scaling in the loss.
  `lambda_mi=0` still constructs and trains the selected estimator.
- The gate penalty is mean `softplus(logit)`. Adaptive control starts at
  `gate_lambda_min`, changes after each epoch using mean sigmoid
  openness, and clips to the configured limits. `lambda_gate` and
  `gate_ramp_floor` apply only with `gate_target=None`. Do not confuse
  openness with the sine-scaled gate magnitude or claim a guaranteed
  target. The controller is unused by `subtract`.
- Collapse guards only print; they never stop training. Their grace,
  patience, gate bounds, and top-three variance threshold are internal
  to `diagnostics.py`, not public arguments. Check the source before
  documenting warnings; there is no gate-target-stagnation warning.

## Checkpoints, outputs, and interpretation

- Checkpoints contain `state_dict` only: no configuration, optimizer,
  scheduler, input provenance, or RNG state. Preserve module names and
  tensor shapes during behavior-preserving work.
- `load_base` reads `{root}/{base}/{base}_model.pth`, constructs the
  model with current arguments, and re-encodes the supplied base inputs.
  It does not read saved latent TSVs or write base outputs. Construction
  consumes RNG state; loading and training are not equivalent RNG paths.
- Combined latents average available modality vectors and optionally
  normalize. Combined specific vectors average without normalization.
  Scores instead average the per-modality specific-vector norms.
- `support.tsv` reports supplied-input presence, including externally
  imputed data. Its query `base` flag means presence in any base
  modality, not necessarily every modality used for a protein's score.
  Preserve this distinction and the `n_modality` score column.
- Output names, protein ordering, headers, and missing-score NaNs are
  downstream contracts. Inspect `outputs.py` and affected callers under
  `../../scripts/` before changing them. PPI and CCA are downstream
  utilities, not training steps in this package.
- Snapshots occur after updates at zero-based epoch multiples, including
  epoch 0. They save latents, query specific vectors, and model state,
  without scores, support, reconstructions, or log snapshots. Inference
  switches to eval mode; training restores train mode each epoch.
- Seed setup does not enable deterministic algorithms. Do not promise
  reproducibility across devices, versions, or different training/load
  paths, or change random draws incidentally during a refactor.

## Editing and validation

Follow the inherited Python style: 72-character lines, four spaces,
double-quoted strings, aligned closing delimiters, standalone opening
and closing function-docstring quotes, and comment blocks of at most
three lines. Keep qualifying file-path strings on one line. Avoid
unrequested global parameters, dependency additions, or broad cleanup.

For documentation edits, check constructor names/defaults, module paths,
output names, and example syntax against source. Do not run training.
For code edits, choose focused static checks and compare public imports,
signatures, checkpoint structure, and relevant output behavior. Keep
README parameter and usage documentation synchronized with API changes.

There is no package-local test suite or build configuration. Tests in
`../cellmaps_coembedding_1102243_AE/` exercise the older fork and do not
validate this implementation. Never report them as standalone ProMERGE
tests. Distinguish performed static checks from runtime checks proposed
for the user.

A minimal import check for the user to run on the server is:

```bash
cd /home/jig049/ProMERGE/packages/ProMERGE
conda activate cellmap
python -c "import runner, promerge, promerge.model"
```

For model or training changes, also prepare a targeted server smoke test
covering the affected branch, checkpoint reuse, or output contract. Use
a separate output directory and the required Slurm settings from the
parent instructions when preparing jobs. Do not submit them yourself.
