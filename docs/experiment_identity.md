# Dataset and experiment identity

Human-readable labels and immutable scientific identity serve different purposes.
Labels organize queues, local directories, and W&B views. Exact Dataset IDs,
resolved configurations, hashes, revisions, checkpoints, and opaque W&B run IDs
remain authoritative for reproducibility and resume.

## Dataset references

An authored experiment may select a Dataset either by exact package ID or by an
explicit task-local reference:

```yaml
data:
  train_dataset:
    name: lentil+chickpea_id
    revision: 0
  ood_datasets:
  - name: kidney_bean_near_family_ood
    revision: 0
```

A reference is immutable metadata at
`02_datasets/refs/<task>/<name>/r<revision>.json`.
It binds the logical name and revision to an exact Dataset ID, manifest hash,
payload hash, Dataset digest, view, regime, materials, and source-package
provenance. There is no mutable `latest` alias.

Resolution validates the record against the admitted package before training.
The resolved config persists both the exact IDs used by data loaders and the
complete reference evidence. Resume uses that saved evidence and fails if the
current reference or package evidence drifts. Existing exact-ID configs remain
valid; missing legacy revision fields resolve through the legacy compatibility
schema only.

Inspect references without loading training data:

```bash
python -m src.datasets.dataset_packages refs --task transient_drying
python -m src.datasets.dataset_packages resolve   --task transient_drying   --name lentil+chickpea_id   --revision 0
python -m src.datasets.dataset_packages inspect-ref   --task transient_drying   --name kidney_bean_near_family_ood   --revision 0
```

Package generation requires `dataset_revision`. A successfully admitted
package publishes its task- and view-local reference only after the complete
manifest and payload hash validate. Reusing the same binding is idempotent;
binding the same task, name, and revision to different evidence is a conflict
that requires a new explicit revision.

## Experiment labels and revisions

Current resolved configs require an explicit non-negative `run.revision`.
Changing scientific inputs while retaining the same presentation label requires
incrementing that revision. The complete resolved-config digest and authored
config content hash are recorded independently of the label.

A transient two-stage plan has one concise parent label and two flat child
bundles:

```text
<parent>_a0
<parent>_b
<parent>/experiment.json
```

The parent record binds the exact child paths, resolved-config digests, Dataset
evidence, run revision, seed, authored-config hash, and the A0-to-B checkpoint
handoff. It is immutable. A fresh command rejects an existing parent or child;
it never silently reuses a completed Stage A0. Continue only with an explicit
`--resume <exact-child-directory>`. Current-schema resume also requires the
matching parent record. Legacy saved runs keep their legacy names and resume
schema.

## Queue and W&B presentation

Training queue labels are derived from the resolved child run label, for example
`train-<parent>_a0-<short-log-id>`. The full config path and invocation remain
in the queue descriptor and log metadata, so the short process label is not
identity evidence.

Current transient W&B runs use the parent label as `group` and `stage_a0`,
`stage_a_plus`, or `stage_b` as `job_type`. Opaque persisted W&B IDs remain the
only resume identity. Current history projects approximately 25 authoritative
completed-epoch series into the `Overview`, `Loss`, `Accuracy`,
`Optimization`, `Curriculum`, and `Performance` namespaces. Local history
and summaries remain authoritative. The only unit projection is CUDA allocated
bytes to GiB at the W&B boundary.

Historical W&B runs and existing personal-workspace panels are not rewritten.
The maintained observer registers only the curated projection above. Remove
obsolete `Transient/*` panels or reset the personal workspace once if the UI
retains old panel definitions from historical runs.
