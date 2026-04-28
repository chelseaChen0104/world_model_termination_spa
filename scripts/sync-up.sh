#!/bin/bash
# Push local code → AutoDL. Code is source-of-truth on local.
# Excludes: data/, outputs/, logs/, caches, IDE state.
set -e
REPO_LOCAL="/Users/yunboliu/Documents/Documents/Lbb/world_model_termination_spa"
REPO_REMOTE="autodl:/root/autodl-tmp/world_model_termination_spa"

# NOTE: do NOT use --delete-excluded — it would wipe data/, logs/, outputs/ on the remote
# (they're "excluded" because we don't want to push them, not because we want to delete them).
rsync -avz --human-readable \
    --exclude=.DS_Store --exclude=.claude --exclude=__pycache__ --exclude=*.pyc \
    --exclude=.venv --exclude=/data --exclude=/outputs --exclude=/logs --exclude=.git \
    "$REPO_LOCAL/" "$REPO_REMOTE/"
echo "[sync-up] done"
