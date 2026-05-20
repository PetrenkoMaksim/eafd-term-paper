# EAFD Term Paper

Repository for experiments, configuration, notes, and artifacts used in the EAFD term paper.

## Structure

- `scripts/` - ordered setup, preparation, training, export, run, and sync scripts.
- `external/` - external project checkouts used by the experiments.
- `configs/` - experiment configuration notes.
- `notes/` - run logs and artifact manifests.
- `artifacts/` - generated outputs and synced results.

## Workflow

Run scripts in numeric order as needed:

```sh
./scripts/00_bootstrap_server.sh
./scripts/01_setup_ptls_env.sh
./scripts/02_prepare_rosbank_ptls.sh
./scripts/03_train_rosbank_embeddings.sh
./scripts/04_export_rosbank_to_eafd.py
./scripts/05_setup_eafd_env.sh
./scripts/06_run_eafd_smoke.sh
./scripts/07_run_eafd_full.sh
./scripts/08_sync_artifacts.sh
```
