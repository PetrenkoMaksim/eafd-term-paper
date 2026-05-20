# EAFD Term Paper Pipeline

This repository makes Maksim Petrenko's EAFD term paper pipeline reproducible for the Rosbank event-sequence task from arXiv paper [2603.15713](https://arxiv.org/abs/2603.15713), Embedding-Aware Feature Discovery for event sequences.

The pipeline has two separate stages:

- Stage A: raw Rosbank data -> PTLS preprocessing -> CoLES/MeLES embedding training -> PTLS inference -> export to the original EAFD/turbo-stats format -> reusable artifact.
- Stage B: original EAFD/turbo-stats feature discovery using `gpt-oss-20b` through vLLM on a single A100.

Important: `gpt-oss-20b` is not the embedding model. It is only the LLM feature-generation agent for EAFD. Embeddings are trained or precomputed first with PTLS / CoLES / MeLES.

## Repository Layout

```text
scripts/    ordered setup, preprocessing, training, export, run, and sync scripts
configs/    reproducibility notes for the EAFD Rosbank run
notes/      run log and artifact manifest
artifacts/  generated archives; only README.md is tracked
external/   external checkouts: ptls-experiments, pytorch-lifestream, turbo-stats
```

Large data, model files, caches, logs, and archives are intentionally ignored by git.

## Server Access

Example SSH command:

```sh
ssh -p 40228 root@50.217.254.161 -L 8080:localhost:8080
```

The port and IP are examples and change with the rented machine.

Recommended tmux session:

```sh
tmux new -s eafd
tmux attach -t eafd
```

## Stage A

From a clean A100 server:

```sh
cd /workspace
git clone https://github.com/PetrenkoMaksim/eafd-term-paper.git
cd eafd-term-paper
bash scripts/00_bootstrap_server.sh
bash scripts/01_setup_ptls_env.sh
bash scripts/02_prepare_rosbank_ptls.sh
bash scripts/03_train_rosbank_embeddings.sh smoke
bash scripts/03_train_rosbank_embeddings.sh full
bash scripts/03_train_rosbank_embeddings.sh inference
python scripts/04_export_rosbank_to_eafd.py
bash scripts/99_validate_stageA_ready.sh
bash scripts/08_sync_artifacts.sh
```

If Stage A embeddings already exist:

```sh
cd /workspace/eafd-term-paper
git pull
python scripts/04_export_rosbank_to_eafd.py
bash scripts/99_validate_stageA_ready.sh
bash scripts/08_sync_artifacts.sh
```

Expected final Stage A artifact:

```text
artifacts/rosbank_eafd_ready_v1.tar.gz
```

Copy the artifact to Mac Google Drive:

```sh
rsync -avz --progress -e "ssh -p PORT" \
  root@IP:/workspace/eafd-term-paper/artifacts/rosbank_eafd_ready_v1.tar.gz \
  "/Users/maksimpetrenko/Library/CloudStorage/GoogleDrive-madsm1006@gmail.com/My Drive/eafd-term-paper/artifacts/"
```

## Stage B

After the EAFD-ready artifact exists:

```sh
bash scripts/05_setup_eafd_env.sh
bash scripts/06_run_eafd_smoke.sh
bash scripts/07_run_eafd_full.sh
```

Stage B expects these files under `external/turbo-stats`:

```text
data/rosbank/transactions.csv
data/rosbank/target.csv
data/rosbank/test_ids.csv
embeddings/rosbank/coles.pickle
```

The EAFD scripts override `dataset.col_target=target_flag` because the original Rosbank config may expect `churn`.

## Common Bugs And Fixes

- Compressed CSV mislabeled as `.csv`: `02_prepare_rosbank_ptls.sh` detects gzip with `file`, renames to `.gz`, and decompresses.
- Spark `Yarn Local dirs can't be empty`: Spark local directories and `LOCAL_DIRS` are set under `/workspace/spark-tmp`.
- `ptls` not installed: `01_setup_ptls_env.sh` clones `pytorch-lifestream` and installs it editable.
- PyTorch CUDA mismatch: PTLS env reinstalls PyTorch from the CUDA 12.8 wheel index.
- Lightning rejects old `gpus` / `auto_select_gpus`: training script removes deprecated trainer keys and uses `+trainer.accelerator=gpu +trainer.devices=1`.
- `onnxruntime` missing: installed in the PTLS env.
- `torch.load` `weights_only=True`: inference passes `inference.seq_encoder.weights_only=false`, with a Hydra `+` fallback.
- Rosbank target column mismatch: EAFD runs use `dataset.col_target=target_flag`.
- vLLM `max_model_len=100000` too large: EAFD run scripts patch it to `16384` for A100 40GB stability.
