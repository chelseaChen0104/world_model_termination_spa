# SAVE SFT + RL Hand-off — Sudoku 4×4

> **Scope**: train the SAVE viability scorer **f_φ** for Sudoku 4×4 using the toy-stage data generated 2026-05-04. Downstream of that, optionally RL-tune f_φ. This doc is **Sudoku-only**; Pentomino + Hidato have their own pending hand-offs.

**Last updated**: 2026-05-04
**Status**: data ready, training not yet started.

---

## 1. Goal

Produce a trained **f_φ checkpoint** that:

1. Given input `(state s, proposed action a)`, emits three tagged predictions:
   - `<next_state>ŝ'</next_state>` — predicted next state (transition modeling)
   - `<viability>true|false</viability>` — predicted v(T(s,a))  ← **the key SAVE output**
   - `<state_viable>true|false</state_viable>` — auxiliary; predicts v(s) for diagnostics
2. Plugs into the CVCP inference loop alongside our existing base policy `π_θ` = `rl_b5_phase3_v8_anchor/final` (Algorithm 2 in the SAVE paper).

f_φ is **a different model from π_θ** — same Qwen2.5-1.5B-Instruct backbone, different parameters, different role (scorer vs action generator).

---

## 2. Inputs

### 2.1 SAVE data (already generated, toy stage)

On **autodl2**: `/root/autodl-tmp/world_model_termination_spa/data/sudoku4/`
- `train_balanced.jsonl` — 1500 records, sources `lt:ht:sol:prt = 3:3:3:3` per record
- `val_natural_calibration.jsonl` — 500 records, sources `lt:ht = 4:4` (no oracle injection)
- `test_natural_policy.jsonl` — 500 records, sources `lt:ht = 4:4`

Local mirror: `/Volumes/yy_drive/SPA_termination/save_artifacts/toy_full/sudoku/`

Schema: `save_sibling_set_v1.2`. See [doc/data_generation_sudoku.md](data_generation_sudoku.md) §2 for field reference.

**Per-record structure** (relevant for f_φ training):
- `state` block: `state_text` (rendered prompt input), `state_struct` (4×4 int grid)
- `candidates[i]`: `action_text`, `action_struct`, `next_state.next_state_text`, **`next_state.next_viable`** ← supervision target, `progress`, `candidate_class`
- `set_stats.mixed`: gates `L_rank` activation
- `deceptive_pairs`: extracted (a+, a-) tuples for paper §3.4 evaluation

**Key training data stats** (from validation):
- Train: 1500 records × ~4.16 candidates/record = ~6240 (state, action) pairs
- Mixed sibling sets: 651/1500 (43%) — these are the only records where ranking loss `L_rank` activates
- Deceptive pairs: 2,813 in train (well above paper's ≥100 requirement for §3.4)

### 2.2 Base model checkpoint

**Backbone**: `Qwen/Qwen2.5-1.5B-Instruct` (HuggingFace).
**Why this size**: matches `π_θ` (rl_b5) backbone — paper §3.2 capacity-dissociation experiment uses multiple sizes; for toy/pilot, 1.5B is the default.

Paths:
- HuggingFace ID for fresh download: `Qwen/Qwen2.5-1.5B-Instruct`
- Local mirror on autodl machines: typically `~/.cache/huggingface/hub/...`

### 2.3 Existing infrastructure to reuse

- `src/training/simple_sft_trainer.py` — HuggingFace Trainer-based SFT (single GPU). Used for B-5/B-7/B-8 SFT runs in this project. Reusable.
- `transformers.AutoModelForCausalLM` + `AutoTokenizer` — standard.
- Chat template: same as π_θ (Qwen2.5-Instruct chat format).

**Do NOT reuse**:
- `src/data/sft_formatter.py` — that's for the original (state → action) SFT data. f_φ training data has a different format (see §3.1).
- The legacy `<solvable>` tag from old project work — f_φ uses `<viability>` per the SAVE paper.

---

## 3. f_φ training format & losses

### 3.1 Per-record → per-sample conversion

The SAVE schema stores 1 record per state with K candidates. For SFT training, each record expands to **K samples**, one per candidate:

```
Per record (1500 train records):
  for each candidate c in record.candidates (where c.local_valid is true):
      yield SFTSample(
          prompt = render_prompt(state, c.action_text),
          response = render_response(c.next_state.next_state_text,
                                       c.next_state.next_viable,
                                       record.state.state_viable),
          label_for_ranking = (c.candidate_class, record.set_stats.mixed,
                                record.candidates_with_pairs)
      )
```

Skip `parse_invalid` and `local_invalid` candidates — they have no `next_state` and contribute no transition signal. They contribute only to the policy's invalid-action distribution, which f_φ doesn't model.

### 3.2 Prompt / response template (paper §2.2 + Appendix A)

**Prompt** (input to f_φ):

```
[system message: "You are a viability scorer for Sudoku 4×4. Given a current
state and a proposed action, predict the next state, whether the next state
remains recoverable (viable), and whether the current state itself is viable."]

[user message:]
Current state:
{state_text in B-5 format with | and --- separators}

Proposed action: {action_text, e.g., "place 4 at row 2 col 2"}
```

**Response** (target output for f_φ):

```
<next_state>
{next_state_text in B-5 format}
</next_state>
<viability>{true|false}</viability>
<state_viable>{true|false}</state_viable>
```

Note the response has THREE tagged fields — the paper requires all three for the joint loss.

### 3.3 Loss components (paper §2.3 eq. 4-8)

For a minibatch of B sibling sets:

```
L_trans = -sum_{b,i in valid candidates} log P(s_{t+1}^(b,i) tokens | prompt_{b,i})
            (cross-entropy on next_state tokens)

L_viab  = -sum_{b,i} [y^(b,i) * log v̂_φ(b,i) + (1-y^(b,i)) * log (1 - v̂_φ(b,i))]
            (per-candidate BCE on viability label;
             v̂_φ extracted from <viability> token logits)

L_rank  = -sum_{b in mixed sibling sets, (i,j) in viable-doomed pairs}
              log σ(ℓ^(b,i) - ℓ^(b,j))
            (pairwise ranking; ℓ = pre-sigmoid logit on viability;
             ONLY active when set_stats.mixed == True)

L_state = -sum_b [u_t^(b) * log û_φ(b) + (1-u_t^(b)) * log (1 - û_φ(b))]
            (BCE on state-level viability; u_t = state_viable from oracle)

L_total = L_trans + λ·L_viab + η·L_rank + μ·L_state
```

**Loss weights** (paper Appendix D, defaults to use unless we tune):
- λ = 1.0 (viability)
- η = 1.0 (ranking; the paper's key contribution)
- μ = 0.5 (state-level; auxiliary, smaller weight)

### 3.4 Implementation note

`L_rank` requires extracting the **scalar logit** `ℓ` for the viability prediction (pre-sigmoid). The simplest approach: take the model's logit on the `true` vs `false` token at the `<viability>` position; compute `ℓ = logit_true - logit_false` per candidate.

For implementation:
1. Forward pass each (s, a) → get logits at the `<viability>{...}</viability>` position
2. ℓ = logits[true_token_id] - logits[false_token_id]
3. For mixed sibling sets, compute pairwise ranking loss across the K logits at that position

Within a sibling set, all K candidates share the prompt prefix up through `Proposed action: ...`. To get K logits in one forward pass, batch the K samples together (each sample is its own (prompt, response)).

---

## 4. Recommended hyperparameters (toy stage)

```
Backbone:               Qwen2.5-1.5B-Instruct
Optimizer:              AdamW
Learning rate:          1e-5 (matches our project's SFT runs; tune to 5e-5 if loss plateaus)
Batch size:             4 per device × 8 grad-accum steps = effective 32
Epochs:                 3
Gradient clipping:      1.0
Max seq length:         512 (state + action + response tags fit comfortably)
LR schedule:            linear warmup 100 steps + linear decay
Mixed precision:        bfloat16
Eval frequency:         every 100 steps
Save frequency:         every 500 steps + end of training
```

Target wall: ~3-5 hr on H800/A800 single GPU for ~6000 train samples × 3 epochs.

If SFT loss plateaus too early or AUC stays at chance: increase epochs to 5, raise LR to 5e-5, or check that ranking loss is actually firing (mixed sibling sets present).

---

## 5. Evaluation metrics & toy → pilot promotion gate

Compute on `val_natural_calibration.jsonl` (500 records) periodically during training, and on `test_natural_policy.jsonl` once at end:

| Metric | Computation | Toy-stage acceptable | Paper-target |
|---|---|---|---|
| **Viability ROC-AUC** | per-candidate v̂_φ predictions vs oracle next_viable; sklearn `roc_auc_score` | ≥ 0.65 | ≥ 0.85 |
| **Same-state pairwise ranking acc** | for each (a+, a-) pair within mixed sets: count where v̂_φ(s, a+) > v̂_φ(s, a-) | ≥ 0.55 | ≥ 0.80 |
| **Deceptive benchmark accuracy** | for each pair in `deceptive_pairs`: 1 if v̂_φ ranks a+ above a-, else 0 | ≥ 0.50 | ≥ 0.75 |
| **Brier score (viability)** | mean squared error of v̂_φ vs binary label | ≤ 0.20 | ≤ 0.10 |
| **ECE @ 10 bins (viability)** | expected calibration error | ≤ 0.10 | ≤ 0.05 |
| **Transition accuracy** | exact-match on `<next_state>` tokens | ≥ 0.80 | ≥ 0.95 |

**Toy → Pilot promotion gate** (per [plan_2026-05-03_save_data_generation.md](plan_2026-05-03_save_data_generation.md) §3-stage roadmap):

> "SFT loss decreasing + sanity checks all pass"

Concretely:
- ✅ training loss decreases monotonically over 3 epochs
- ✅ validation viability AUC > 0.55 (above chance)
- ✅ same-state pairwise > 0.50 (better than coin flip on hardest cases)
- ✅ 0 schema violations on outputs

If any fail, **don't scale to pilot**. Investigate causes (low mixed rate? insufficient state coverage? model size mismatch?) and either re-tune training or regenerate toy data with adjustments.

---

## 6. Optional RL phase (after SFT)

If SFT alone isn't enough — typical signal: viability AUC on test stays below ~0.75 — apply RL on top of the SFT'd f_φ. **Skip this for toy stage** unless SFT clearly underperforms.

### 6.1 RL setup (only if SFT plateau)

Use the existing `rl_trainer_v6.py` infrastructure as a starting point, adapted for f_φ:

- **Policy**: f_φ_SFT checkpoint (from §3-§5)
- **Reward**: 
  - Per-step: BCE-style reward on viability prediction (compare v̂_φ to oracle next_viable)
  - Plus optional: action-quality reward if RL is also tuning the policy (NOT applicable to f_φ since it doesn't generate actions)
- **Reference model for KL**: f_φ_SFT (anchor)

### 6.2 Why RL might help (or not)

For f_φ, RL is **less natural than for π_θ** because f_φ is a classifier, not a policy. The standard RL paradigm (sampling actions, getting rewards) doesn't directly map. What might help instead:

- **GRPO-style group-relative ranking** over the K candidates per record, treating viability as a reward signal
- **Online hard-example mining** — focus training on cases where SFT'd f_φ is wrong
- **Calibration tuning** via temperature scaling or isotonic regression (no actual RL — just post-hoc calibration)

For toy stage: skip RL entirely. If SFT works (gates pass), proceed to pilot. If SFT doesn't work, **first try more SFT data + tuning**, not RL.

---

## 7. Output artifacts

After f_φ training succeeds:

```
On autodl2 (recommended; co-located with SAVE data):
/root/autodl-tmp/world_model_termination_spa/outputs/save_sudoku4_f_phi/
├── final/                   # canonical checkpoint (model.safetensors + tokenizer + config)
├── checkpoint-N/            # periodic checkpoints (delete intermediates after final)
├── train_loss.jsonl         # per-step training loss
├── eval_metrics.jsonl       # per-eval-step viability AUC, pairwise acc, etc.
└── runs/                    # TensorBoard event files
```

Local archive (for safekeeping):
```
/Volumes/yy_drive/SPA_termination/save_models/sudoku4_f_phi/
└── final/                   # rsync'd from autodl2
```

---

## 8. Sanity checks (run before declaring done)

After training, run on val + test:

1. **Schema-valid outputs**: every f_φ response on val/test parses to the three tagged fields, with `<viability>` ∈ {true, false}. ≥ 99% parse success.
2. **No collapse to single class**: f_φ predictions span both `true` and `false` (not stuck on one). Ratio ≥ 0.10 for the minority class.
3. **Calibration sanity**: predicted high-confidence (`viability prob > 0.9`) → empirical viable rate ≥ 0.85.
4. **Deceptive benchmark check**: f_φ ranks ≥ 50% of (a+, a-) pairs correctly (a+ scores higher). Random = 50%; oracle = 100%.
5. **Transition accuracy**: when model outputs `<next_state>...</next_state>`, the predicted state matches oracle T(s, a) at ≥ 80% token-level accuracy. Low transition accuracy means f_φ doesn't even understand env mechanics.

If any fails: investigate before promoting toy → pilot.

---

## 9. Hand-off checklist

- [ ] Train data verified (1500 train, 500 val, 500 test) on autodl2 + local mirror
- [ ] Backbone model accessible (`Qwen2.5-1.5B-Instruct` loaded successfully)
- [ ] Per-record → per-sample converter implemented (skips `local_invalid` / `parse_invalid`; emits prompt+response per §3.2)
- [ ] Loss function with all 4 components implemented (`L_trans`, `L_viab`, `L_rank`, `L_state`); `L_rank` correctly gated on `set_stats.mixed`
- [ ] Training script produces decreasing loss curve over 3 epochs
- [ ] Eval at end-of-training reports viability AUC, pairwise acc, deceptive bench acc
- [ ] All 5 sanity checks in §8 pass
- [ ] Final checkpoint saved to `outputs/save_sudoku4_f_phi/final/`
- [ ] Checkpoint rsync'd to local archive
- [ ] Toy → pilot gate decision recorded in `doc/eval_<date>_save_sudoku_toy_sft.md`

---

## 10. Working with autodl2 from your local Mac

Everything in this hand-off runs on **autodl2** (a remote GPU machine). You drive it from your laptop via SSH. Convention used in this project:

### 10.1 SSH alias

`~/.ssh/config` has an alias `autodl2` pointing at the cloud host:

```
Host autodl2
    HostName connect.bjb1.seetacloud.com
    Port 12158
    User root
    IdentityFile ~/.ssh/id_ed25519_autodl
```

Any local command works: `ssh autodl2 "..."`, `rsync ... autodl2:...`, `scp ... autodl2:...`.

### 10.2 Standard paths on autodl2

```
~/autodl-tmp/world_model_termination_spa/    ← project root
├── scripts/                                  ← all SAVE-related code lives here
├── data/sudoku4/                             ← SAVE Sudoku data (toy)
├── outputs/                                  ← model checkpoints (rl_b5, save_sudoku4_f_phi/, …)
├── logs/                                     ← training + data-gen logs
└── src/environments/                         ← legacy env code (READ ONLY for SAVE work)

/root/miniconda3/bin/python                   ← canonical Python (has pydantic, transformers, torch installed)
```

**Critical**: SAVE work never edits `src/environments/*` (additivity contract per CLAUDE.md). Always operate under `scripts/`, `data/sudoku4/`, `outputs/save_*/`.

### 10.3 Six common workflows

#### A. Push a script edit to autodl2

```bash
# After editing scripts/foo.py locally
rsync -av /Users/yunboliu/Documents/Documents/Lbb/world_model_termination_spa/scripts/foo.py \
    autodl2:/root/autodl-tmp/world_model_termination_spa/scripts/
```

Push only the file you changed. Don't `rsync` the whole tree — too slow and risks overwriting other work.

#### B. Inspect a file or run a quick check

```bash
ssh autodl2 "wc -l /root/autodl-tmp/world_model_termination_spa/data/sudoku4/*.jsonl"
ssh autodl2 "ls -lh /root/autodl-tmp/world_model_termination_spa/outputs/"
ssh autodl2 "tail -50 /root/autodl-tmp/world_model_termination_spa/logs/save_sft.log"
```

These are read-only and finish in <1s. Cheap to run.

#### C. Launch a long-running training job (use tmux for durability)

```bash
ssh autodl2 "cd /root/autodl-tmp/world_model_termination_spa && \
    tmux new-session -d -s save_sft 'bash scripts/run_save_sudoku_sft.sh > logs/save_sft.log 2>&1'"
```

Why tmux: `nohup ... &` in a transient SSH session sometimes dies on disconnect; tmux is bulletproof. The session persists across SSH disconnects; either local client can re-attach with `tmux attach -t save_sft`.

#### D. Monitor a running job

```bash
# Tail the log
ssh autodl2 "tail -f /root/autodl-tmp/world_model_termination_spa/logs/save_sft.log"

# Or attach to the tmux session for full screen
ssh -t autodl2 "tmux attach -t save_sft"   # detach with Ctrl-b d

# GPU usage check
ssh autodl2 "nvidia-smi"

# Process status
ssh autodl2 "pgrep -af save_sft"
```

#### E. Pull final artifacts to local archive

```bash
# Single checkpoint
mkdir -p /Volumes/yy_drive/SPA_termination/save_models/sudoku4_f_phi
rsync -avh autodl2:/root/autodl-tmp/world_model_termination_spa/outputs/save_sudoku4_f_phi/final/ \
    /Volumes/yy_drive/SPA_termination/save_models/sudoku4_f_phi/final/

# Logs + eval metrics
rsync -av autodl2:/root/autodl-tmp/world_model_termination_spa/logs/save_sft.log \
    /Volumes/yy_drive/SPA_termination/save_models/sudoku4_f_phi/
```

Use `--info=progress2` flag for visible progress on multi-GB transfers.

#### F. Kill a stuck job

```bash
ssh autodl2 "pgrep -f 'simple_sft_trainer' | xargs -r kill"
ssh autodl2 "tmux kill-session -t save_sft"
```

### 10.4 Coordinating two local clients (laptop + desktop)

If you work from two machines targeting the same autodl2:

| Resource | Concurrent OK? |
|---|---|
| SSH connections, log tailing, status checks | ✅ Yes — one's a "monitor", other's the "executor" |
| GPU training jobs (one at a time) | ⚠️ Run only ONE training/eval job on the GPU at a time. tmux makes this safe — both clients see the same session list |
| Editing same script + rsync push | ❌ Avoid — last write wins; coordinate via git or designate one client as the editor |
| `pip install` | ❌ Don't run from both at once |

In practice: one client launches the SFT training in tmux; either client tails logs / attaches to the session for monitoring. Don't both kick off training jobs simultaneously.

### 10.5 Recovery patterns

**SSH session dies mid-job**: tmux/nohup-launched jobs survive. Reconnect with `ssh autodl2`, then `tmux attach -t <session>` or `pgrep -af <job_name>` to verify it's still running.

**Job hangs / OOMs**: `nvidia-smi` to check GPU state; `pkill -9 -f <job_name>` to force-kill; check `dmesg` for OOM signs (rare on H800/A800 80GB but possible on smaller GPUs).

**Disk full**: `df -h /root/autodl-tmp` to check; clean up intermediate checkpoints (`outputs/<run>/checkpoint-N/` not strictly needed once `final/` is saved).

**autodl2 restarts / disconnects**: jobs in tmux survive a brief network blip but die if the machine itself reboots. The `final/` checkpoint is your durable artifact — pull it to local archive ASAP.

### 10.6 Conventions for this hand-off specifically

- **Launch SFT in tmux** named `save_sft_sudoku` (or similar); not bare `nohup`
- **Log to** `logs/save_sft_sudoku_<date>.log` — date-stamped so multiple runs don't overwrite
- **Save final to** `outputs/save_sudoku4_f_phi/final/`; **delete intermediate checkpoints** once final is saved + rsync'd to local archive
- **Pull to local** `/Volumes/yy_drive/SPA_termination/save_models/sudoku4_f_phi/` immediately after training succeeds — durable backup before any cloud-side cleanup

---

## 11. Coordinating with the inference pipeline

After f_φ exists, plug into CVCP per Algorithm 2 (paper §2.4-§2.5):

```python
# Pseudocode for inference (NOT part of this hand-off — for the next person)
def cvcp_inference(state, π_θ, f_φ, τ_keep, τ_fb, K=8):
    candidates = π_θ.sample(state, K=K, temperature=1.0)
    viab_scores = [f_φ.score(state, a) for a in candidates]
    kept = [a for a, v in zip(candidates, viab_scores) if v >= τ_keep]
    if kept:
        return max(kept, key=lambda a: π_θ.logprob(a, state))   # π_θ tie-break
    else:
        if max(viab_scores) >= τ_fb:
            return candidates[viab_scores.index(max(viab_scores))]
        else:
            return TERMINATE
```

τ_keep and τ_fb are calibrated on `val_natural_calibration.jsonl` after f_φ is trained. Calibration target: `Pr[v(T(s,a))=1 | v̂_φ(s,a) ≥ τ_keep] ≥ 1 − ε_keep`, with `ε_keep` = 0.05 typically.

This calibration step is straightforward (one threshold sweep on val) but happens **after** f_φ is trained, not during. Plan a small follow-up doc for calibration when f_φ training is done.

---

## End of hand-off
