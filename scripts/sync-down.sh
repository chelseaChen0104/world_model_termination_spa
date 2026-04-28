#!/bin/bash
# Pull AutoDL → local: data/, logs/, outputs/ (excluding large model weights by default).
# Cloud is source-of-truth for data and outputs.
#
# Usage:
#   bash scripts/sync-down.sh             # data/ + logs/ + outputs/ metadata (no weights)
#   bash scripts/sync-down.sh --with-models   # also pull *.safetensors / *.bin / *.pt
set -e
REPO_LOCAL="/Users/yunboliu/Documents/Documents/Lbb/world_model_termination_spa"
REPO_REMOTE="autodl:/root/autodl-tmp/world_model_termination_spa"

WEIGHT_EXCLUDES=(--exclude=*.safetensors --exclude=*.bin --exclude=*.pt --exclude=*.pth)
if [ "$1" = "--with-models" ]; then
    WEIGHT_EXCLUDES=()
    echo "[sync-down] pulling model weights too — this may be large"
fi

mkdir -p "$REPO_LOCAL/data" "$REPO_LOCAL/logs" "$REPO_LOCAL/outputs"

# data/ — SFT parquets, etc.
rsync -avz --human-readable "$REPO_REMOTE/data/" "$REPO_LOCAL/data/" 2>/dev/null || true

# logs/ — training and data-gen logs
rsync -avz --human-readable "$REPO_REMOTE/logs/" "$REPO_LOCAL/logs/" 2>/dev/null || true

# outputs/ — checkpoints, with optional weight exclusion
rsync -avz --human-readable "${WEIGHT_EXCLUDES[@]}" \
    "$REPO_REMOTE/outputs/" "$REPO_LOCAL/outputs/" 2>/dev/null || true

echo "[sync-down] done"
du -sh "$REPO_LOCAL/data" "$REPO_LOCAL/logs" "$REPO_LOCAL/outputs" 2>/dev/null
