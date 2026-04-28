---
description: Pull AutoDL → local (data/, logs/, outputs/ — excludes model weights unless $ARGUMENTS contains --with-models)
---

Run `bash scripts/sync-down.sh $ARGUMENTS` from the repo root. Show what was pulled (sizes via the printed `du -sh`), then briefly note any new files relative to last sync (check by file count or by listing the most recent files in `data/` and `logs/`).
