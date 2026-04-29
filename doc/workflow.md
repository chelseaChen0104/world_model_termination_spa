# Sync Workflow — Mac ↔ Cloud ↔ GitHub

How code, data, and results flow between local Mac, AutoDL instances (`autodl`, `autodl2`), and GitHub. Read this before doing anything that crosses machine boundaries.

## The three places

| Where | Source of truth for | Why |
|---|---|---|
| **Local Mac** | All code, docs, sync scripts. Working copy of generated data after `sync-down`. | Editing happens here. Claude Code session runs here. |
| **AutoDL clouds** (`autodl`, `autodl2`) | GPU compute. Generated data, training checkpoints, run logs. | We don't have local GPU. Heavy lifting happens in the cloud. |
| **GitHub** (`chelseaChen0104/world_model_termination_spa`) | Versioned code backup + collaboration share. | Insurance against losing local repo + clean history for reproducibility. |

## What lives where

| Artifact | Local Mac | autodl1 / autodl2 | GitHub |
|---|---|---|---|
| `src/`, `scripts/`, `doc/`, `CLAUDE.md`, `evaluate_rl.py`, configs | ✅ source-of-truth | ✅ identical via `sync-up` | ✅ via `git push` |
| `.vscode/settings.json`, `.claude/commands/` | ✅ | ✅ (commands only; `.claude/settings.json` is per-user, gitignored) | ✅ (commands), ❌ (settings) |
| `data/` (generated parquets) | ✅ merged copy via `sync-down` | ✅ each cloud has its own subset, *disjoint paths* | ❌ **never** — gitignored, too big |
| `outputs/` (model checkpoints) | optional via `sync-down --with-models` | ✅ per cloud | ❌ never — too big |
| `logs/` (run logs) | ✅ merged via `sync-down` | ✅ per cloud | ❌ never |

## The two cloud instances

We run two separate AutoDL boxes, both H800, both at the same repo path `/root/autodl-tmp/world_model_termination_spa/`. They run different experiments in parallel, with **disjoint data/output paths** to avoid collision:

| | autodl1 | autodl2 |
|---|---|---|
| SSH alias | `autodl` | `autodl2` |
| Typical use | Big runs (9×9 multi-difficulty data gen, large SFTs) | Quick experiments (4×4 sanity check, eval probes) |
| Data path examples | `data/sudoku_llm_policy_easy/`, `..._medium/`, `..._hard/` | `data/sudoku_4x4_llm_policy/` |
| Output dirs | `outputs/sft_sudoku_minimal*` | `outputs/sft_sudoku_4x4_*` |

**Code is identical on both** (same `sync-up`). **Generated data is per-cloud** and never collides because we use distinct directory names per experiment.

## The sync commands

### Push code: local → cloud(s)

```bash
bash scripts/sync-up.sh                    # default: both clouds
bash scripts/sync-up.sh --target autodl    # autodl1 only
bash scripts/sync-up.sh --target autodl2   # autodl2 only
```

What it does: rsync excluding `data/`, `outputs/`, `logs/`, `.git`, `.venv`, caches, IDE state. Code stays in sync. Run after **every** code edit.

### Pull results: cloud(s) → local

```bash
bash scripts/sync-down.sh                  # default: both clouds, no model weights
bash scripts/sync-down.sh --target autodl  # one cloud
bash scripts/sync-down.sh --with-models    # also pull *.safetensors / *.bin / *.pt
```

What it does: rsync `data/`, `logs/`, `outputs/` (excluding heavyweight checkpoint files unless `--with-models`). Subdirs are disjoint so the merge is conflict-free. Run after each long cloud run.

### GitHub checkpoint

Manual git operations from local Mac:

```bash
git status                         # what's pending
git add <specific-paths>           # NEVER `git add -A` — too easy to commit secrets
git commit -m "checkpoint: ..."    # write meaningful message
git push                           # backup to GitHub
```

Run this **periodically** — at least after each meaningful experimental milestone, ideally daily during active work.

## Slash commands inside Claude Code

Same effect as the bash scripts:

```
/sync-up
/sync-down
```

Both call the underlying scripts. They were defined in `.claude/commands/`.

## Operational rules of thumb

1. **Edit locally only.** Don't edit code on the cloud — the next `sync-up` would overwrite your changes.
2. **`sync-up` after every edit.** Cheap (~1 sec for typical edits). Keeps both clouds in sync.
3. **`sync-down` after every long run.** Especially important: if a cloud instance dies, anything not synced down is GONE. GitHub doesn't back up data.
4. **Treat clouds as ephemeral.** Disk full / container reset / instance stop → all generated data on that instance can disappear. Source of truth for *results* is the local Mac after `sync-down`.
5. **Commit + push to GitHub at meaningful checkpoints.** Code is small, GitHub is fast — the bottleneck is just remembering to do it.
6. **Slash commands trigger the scripts only on the current session's working dir.** They won't auto-sync if you're operating from a different directory.

## What to do when things go wrong

| Problem | Fix |
|---|---|
| Cloud disk full mid-training | `ssh <cloud> 'rm -rf <old-checkpoint-dir>'`. Train can usually continue if you free space before the next save step. |
| Cloud SSH suddenly hangs | Likely an Anthropic / network blip. Wait 30s, retry. If persistent, check AutoDL dashboard. |
| Edited a script directly on cloud | Bad. Either pull the change down to Mac first (`scp`) or accept it'll be overwritten on next `sync-up`. Going forward, only edit local. |
| `sync-down` shows 0 bytes transferred but data exists on cloud | Probably permissions. Check `ls -la` on cloud — files should be readable. Or the rsync exclude is too aggressive (the `data/` vs `/data/` problem we hit earlier). |
| GitHub push rejected (email privacy) | Expected — GitHub blocks pushes that would expose your private email. Use the noreply email format: `<id>+<username>@users.noreply.github.com`. Already configured for this repo. |
| Forgot to `sync-up` before launching a cloud run, used old code | Stop the run, sync-up, restart. Wasted GPU time, but at least the run uses the right code. |

## Important `.gitignore` lesson (learned the hard way)

The initial commit was missing the entire `src/data/` source directory. Cause: `.gitignore` had:

```
data/        # WRONG — also matches src/data/
```

Fix: anchor with a leading slash so it only matches the top-level directory:

```
/data/       # correct — only top-level data, not src/data/
/outputs/
/logs/
```

Same lesson applies to `rsync --exclude`. **Any path-like pattern should be anchored** (`/data/` not `data/`) when you want only the root directory.

## Cheat sheet

| What I just did | Run |
|---|---|
| Edited a `.py` file | `bash scripts/sync-up.sh` |
| Big cloud run finished | `bash scripts/sync-down.sh` |
| Want to check 4×4 results only | `bash scripts/sync-down.sh --target autodl2` |
| Need to pull weights for local inspection | `bash scripts/sync-down.sh --with-models` |
| Reached a milestone | `git status; git add ...; git commit -m '...'; git push` |
| Started new cloud experiment | `sync-up` first, then ssh to cloud and launch in tmux |
| Cloud dying / instance ending | `sync-down` IMMEDIATELY before it goes |
