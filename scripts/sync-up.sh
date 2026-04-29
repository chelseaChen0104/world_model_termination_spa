#!/bin/bash
# Push local code → AutoDL instance(s). Code is source-of-truth on local.
# Excludes: data/, outputs/, logs/, caches, IDE state.
#
# Usage:
#   bash scripts/sync-up.sh                # push to both (default)
#   bash scripts/sync-up.sh --target autodl    # push to autodl1 only
#   bash scripts/sync-up.sh --target autodl2   # push to autodl2 only
#   bash scripts/sync-up.sh --target all        # push to both (explicit)
#
# Target SSH aliases (from ~/.ssh/config): autodl, autodl2
set -e
REPO_LOCAL="/Users/yunboliu/Documents/Documents/Lbb/world_model_termination_spa"

TARGET="${TARGET:-all}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Resolve target list
case "$TARGET" in
    all)     TARGETS=("autodl" "autodl2") ;;
    autodl)  TARGETS=("autodl") ;;
    autodl2) TARGETS=("autodl2") ;;
    *) echo "[sync-up] unknown target: $TARGET" >&2; exit 1 ;;
esac

# NOTE: do NOT use --delete-excluded — would wipe data/, logs/, outputs/ on remote
# (they're "excluded" because we don't want to push them, not because we want to delete them).
for T in "${TARGETS[@]}"; do
    echo "[sync-up] → $T"
    rsync -avz --human-readable \
        --exclude=.DS_Store --exclude=.claude --exclude=__pycache__ --exclude=*.pyc \
        --exclude=.venv --exclude=/data --exclude=/outputs --exclude=/logs --exclude=.git \
        "$REPO_LOCAL/" "$T:/root/autodl-tmp/world_model_termination_spa/"
done
echo "[sync-up] done"
