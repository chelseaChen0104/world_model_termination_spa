#!/bin/bash
# Pull AutoDL → local: data/, logs/, outputs/ (excluding large model weights by default).
# Cloud is source-of-truth for data and outputs.
#
# Usage:
#   bash scripts/sync-down.sh                       # pull from both, no model weights
#   bash scripts/sync-down.sh --target autodl       # pull from autodl1 only
#   bash scripts/sync-down.sh --target autodl2      # pull from autodl2 only
#   bash scripts/sync-down.sh --with-models         # also pull *.safetensors / *.bin / *.pt
#
# Outputs from each remote land in subdirs to avoid collision when both targets active:
#   data/   ← merged (separate sub-dirs per experiment, e.g., sudoku_llm_policy_easy vs sudoku_4x4_llm_policy)
#   logs/   ← merged (file names tagged per machine via prefix)
#   outputs/ ← merged (separate sub-dirs per checkpoint)
set -e
REPO_LOCAL="/Users/yunboliu/Documents/Documents/Lbb/world_model_termination_spa"

TARGET="${TARGET:-all}"
WITH_MODELS=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --with-models) WITH_MODELS=true; shift ;;
        *) shift ;;
    esac
done

case "$TARGET" in
    all)     TARGETS=("autodl" "autodl2") ;;
    autodl)  TARGETS=("autodl") ;;
    autodl2) TARGETS=("autodl2") ;;
    *) echo "[sync-down] unknown target: $TARGET" >&2; exit 1 ;;
esac

WEIGHT_EXCLUDES=(--exclude=*.safetensors --exclude=*.bin --exclude=*.pt --exclude=*.pth)
if $WITH_MODELS; then
    WEIGHT_EXCLUDES=()
    echo "[sync-down] pulling model weights too — this may be large"
fi

mkdir -p "$REPO_LOCAL/data" "$REPO_LOCAL/logs" "$REPO_LOCAL/outputs"

for T in "${TARGETS[@]}"; do
    REMOTE="$T:/root/autodl-tmp/world_model_termination_spa"
    echo "[sync-down] ← $T"
    rsync -avz --human-readable "$REMOTE/data/" "$REPO_LOCAL/data/" 2>/dev/null || true
    rsync -avz --human-readable "$REMOTE/logs/" "$REPO_LOCAL/logs/" 2>/dev/null || true
    rsync -avz --human-readable "${WEIGHT_EXCLUDES[@]}" \
        "$REMOTE/outputs/" "$REPO_LOCAL/outputs/" 2>/dev/null || true
done

echo "[sync-down] done"
du -sh "$REPO_LOCAL/data" "$REPO_LOCAL/logs" "$REPO_LOCAL/outputs" 2>/dev/null
