# Artifact Manifest

Generated artifacts are intentionally not tracked by git. Store checksums after each run:

```sh
shasum -a 256 artifacts/*.tar.gz
```

| Artifact | Source Script | Location | Description |
| --- | --- | --- | --- |
| `rosbank_eafd_ready_v1.tar.gz` | `scripts/08_sync_artifacts.sh` | `artifacts/rosbank_eafd_ready_v1.tar.gz` | EAFD-ready Rosbank `data/rosbank` files and `embeddings/rosbank/coles.pickle`. |
| `rosbank_ptls_stageA_v1.tar.gz` | `scripts/08_sync_artifacts.sh` | `artifacts/rosbank_ptls_stageA_v1.tar.gz` | Trained PTLS model, `mles_embeddings.pickle`, and PTLS logs. |

## Fill After Run

| Artifact | SHA-256 | Size | Created At | Notes |
| --- | --- | --- | --- | --- |
| `rosbank_eafd_ready_v1.tar.gz` | TODO | TODO | TODO | TODO |
| `rosbank_ptls_stageA_v1.tar.gz` | TODO | TODO | TODO | TODO |
