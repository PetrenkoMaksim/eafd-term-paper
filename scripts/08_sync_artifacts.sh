#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts

test -d external/turbo-stats/data/rosbank
test -d external/turbo-stats/embeddings/rosbank

tar -czf artifacts/rosbank_eafd_ready_v1.tar.gz \
    external/turbo-stats/data/rosbank \
    external/turbo-stats/embeddings/rosbank

if [ -f external/ptls-experiments/scenario_rosbank/data/mles_embeddings.pickle ] && \
   [ -f external/ptls-experiments/scenario_rosbank/models/mles_model.p ]; then
    tar -czf artifacts/rosbank_ptls_stageA_v1.tar.gz \
        external/ptls-experiments/scenario_rosbank/data/mles_embeddings.pickle \
        external/ptls-experiments/scenario_rosbank/models/mles_model.p \
        external/ptls-experiments/scenario_rosbank/logs
else
    echo "Skipping rosbank_ptls_stageA_v1.tar.gz because PTLS model or embeddings are missing"
fi

ls -lh artifacts/*.tar.gz

cat <<'EOF'

Manual copy example for Mac / Google Drive:
rsync -avz --progress \
  -e "ssh -p 40228" \
  root@50.217.254.161:/workspace/eafd-term-paper/artifacts/rosbank_eafd_ready_v1.tar.gz \
  "/Users/maksimpetrenko/Library/CloudStorage/GoogleDrive-madsm1006@gmail.com/My Drive/eafd-term-paper/artifacts/"
EOF

if [ -n "${SYNC_SSH_PORT:-}" ] && [ -n "${SYNC_SSH_HOST:-}" ] && [ -n "${SYNC_DEST:-}" ]; then
    rsync -avz --progress \
        -e "ssh -p ${SYNC_SSH_PORT}" \
        "root@${SYNC_SSH_HOST}:/workspace/eafd-term-paper/artifacts/rosbank_eafd_ready_v1.tar.gz" \
        "$SYNC_DEST"
fi
