# SFT Track B — Diagnosis and Proposed Pivot to Single-Step Training (2026-04-28)

Combined eval result + research decision. Continuation of [eval_2026-04-28_sft_track_b_tier_a.md](eval_2026-04-28_sft_track_b_tier_a.md). After the multi-turn-prompt re-run, we have enough signal to commit to a structural change in our training format.

## TL;DR

- The first SFT model (`outputs/sft_sudoku_llm_policy/final`) trained cleanly (loss dropped 3×, no overfitting, 100% format compliance) but **fails at the actual classification task**:
  - **Single-turn eval:** constant `<solvable>true</solvable>` for all 200 samples, BP recall 0%
  - **Multi-turn eval:** 66% solvable accuracy (better than chance), but **BP recall is only 5%** — model misses 95/100 BPs
- Diagnosis: the model became a **temporal smoother** that echoes prior assistant turns rather than grounding its prediction in the grid.
- Root cause: 84.7% of training samples have a trivial echo shortcut (pre-BP and post-BP filler), and the 15.3% BP-transition samples don't get enough gradient weight to overcome it. Multi-turn structure makes the shortcut available; cross-entropy treats the easy bulk and the hard minority equally.
- Proposed change: **Switch to single-step training samples (one (s, a, s') triple per sample), keep the full SPA-style XML response (observation + prediction + termination tags + answer).** Removes the echo shortcut, recovers all dropped data, matches SPA's actual SFT shape exactly.

---

## 1. Eval Tier A — both runs

### 1.1 Single-turn (initial Tier A)

Eval set: 200 samples generated fresh from live `SudokuEnv` via random play, balanced 100 solvable / 100 unsolvable / 5 BPs. Prompt: `[system, user_state]` only.

| | Result |
|---|---|
| Format compliance (any tag) | 100% |
| `<answer>` tag present | 69% |
| Solvable accuracy | 50.0% (= class prior) |
| Solvable confusion | predicted **True for all 200** |
| BP recall | 0.0% (caught 0/5) |
| BP F1 | 0.0% |

The model output `<solvable>true</solvable>` for every sample, regardless of input. Constant predictor.

### 1.2 Multi-turn (re-run on val_filtered.parquet)

Eval set: 300 samples loaded directly from training-format parquet, balanced 100 solvable / 100 BP / 100 post-BP filler. Prompt: full multi-turn context as in training.

| | Result |
|---|---|
| Format compliance | 100% |
| `<answer>` tag present | 73.7% |
| Solvable accuracy | 66.3% |
| Solvable confusion | TP=94, FN=6, FP=95, TN=105 |
| Solvable F1 | 65.1% (vacuous — recall 94% from over-prediction) |
| **BP recall** | **5.0%** (caught 5/100) |
| BP precision | 45.5% |
| BP F1 | 9.0% |

The model is no longer constant — it predicts `<solvable>false</solvable>` for ~37% of samples — but only 5/100 of those flips happen at the actual BP transitions.

### 1.3 What changed between the two runs

The multi-turn prompts unlocked a non-trivial classifier. But the unlock is **temporal echo**, not grid grounding:

| Class | Predicted False rate | Why |
|---|---|---|
| Pre-BP solvable (n=100) | 6% | Model echoes prior `solvable=true` from priors → 94% correct |
| Post-BP filler (n=100) | ~99% (estimate) | Model echoes prior `solvable=false` from priors → ~99% correct |
| **BP transitions (n=100)** | **5%** | **Priors say true; current state says false; model fails to flip** |

The model handles the cases where priors give the answer. It fails the cases where priors actively mislead and the grid is the only reliable signal — i.e., the cases that require the actual capability we want.

---

## 2. Diagnosis — Why the SFT model is a temporal smoother

### 2.1 Training-data structure invites the shortcut

Filtered training data (6,221 samples):

| Class | Count | Share | What "echo" gets right |
|---|---|---|---|
| (Solvable=True, BP=False) — pre-BP | 1,531 | 24.6% | Echo "true" → 100% correct |
| (Solvable=False, BP=False) — post-BP filler | 3,739 | 60.1% | Echo "false" → 100% correct |
| **(Solvable=False, BP=True) — BP transitions** | **951** | **15.3%** | Echo "true" → 0% correct |

**84.7% of samples are trivially solved by echo.** Cross-entropy loss applied uniformly across tokens means the gradient signal is dominated by the easy 84.7%. The 15.3% BP samples — the only ones requiring real grid-conditioning — get drowned out.

### 2.2 Multi-turn prompt structure exposes the shortcut

In the multi-turn samples, prior assistant turns explicitly contain previous `<solvable>X</solvable>` tags. The model can attend to those and copy. With most of training rewarding exactly that copying, the model learns it.

Single-step samples wouldn't have any priors in the prompt — there's nothing to copy from. The model would be forced to use the grid.

### 2.3 The "who provides the action" wrinkle

A second issue surfaced during the discussion: in our current format, `<solvable>` predicts the solvability of `s_{t+1}` (after the model's own action), not of `s_t` (the current state). This is action-conditional, not state-classification.

| Tag | What it actually represents |
|---|---|
| `<observation>` | s_t — current state |
| `<prediction>` | s_{t+1} — state after my chosen action |
| `<solvable>` | Is s_{t+1} solvable? (depends on a_t) |
| `<breaking_point>` | Did this action cause the transition? |
| `<answer>` | The action itself (chosen by the model) |

This is fine for "agent reasoning about its own next action," but it's **not the same as state-classification termination prediction**. It's a design choice we hadn't made explicitly. (See §4.3 below.)

---

## 3. What we conclude

The current multi-turn world-model SFT, as built, does **not** demonstrate that an SPA-style world model lifts termination prediction. The model learned a shortcut (echo) instead of grounding in the env's dynamics. This isn't a training failure (loss curves were healthy) — it's a **data-format failure** baked into the multi-turn structure.

Going through SPEC.md §3's success-criteria checklist on this checkpoint:

| Success criterion | Status |
|---|---|
| Pass@1 / Pass@8 reported | ❌ not run (model would emit constant rollouts; we'd waste GPU hours) |
| BP recall, F1 reported | ✅ — and they're poor (5% / 9%) |
| In-distribution eval | ✅ (multi-turn matches training) |
| Balanced eval | ✅ (300 samples, 100 per class) |
| Improvement over baselines | ❌ not yet attempted (Vanilla RL, SE-RL, VAGEN still TODO) |
| Single-GPU reproducible | ✅ |

The result is informative but **negative**: standard multi-turn SFT on Sudoku with our class distribution doesn't teach termination prediction.

---

## 4. Proposed pivot — single-step samples with full SPA-style XML

### 4.1 The change

Re-format existing trajectory data so each sample is **one (s_t, a_t, s_{t+1}) triple**, not a multi-turn rollout:

```
PROMPT  (input):
  [system]      sudoku_full system prompt
  [user]        Current state: {s_t}

RESPONSE (target — same XML as before):
  <think>
    <observation>{s_t}</observation>          ← current state, ground-truth
    <prediction>{s_{t+1}}</prediction>        ← next state given a_t, ground-truth
    <terminate_prob>...</terminate_prob>
    <steps_left>...</steps_left>
    <solvable>{is_solvable}</solvable>        ← oracle label (see §4.3 for semantics)
    <breaking_point>{is_bp}</breaking_point>  ← oracle label
  </think>
  <answer>{a_t}</answer>                      ← action chosen during data gen
```

Each step in each trajectory becomes its own training row. No history in the prompt.

### 4.2 What this fixes

| Issue with current setup | How single-step fixes it |
|---|---|
| Echo shortcut (84.7% of samples solvable by copying priors) | No priors in prompt → no echo → all samples force grid-conditioning |
| 60% of samples dropped by length filter | Each sample is short (~600 tokens) → no filtering needed → recover ~9,600 lost training samples |
| BP samples drowned in gradient | Every sample contributes equally to grid-feature learning; BP samples carry their proper 15% weight |
| Drift from SPA's actual recipe | **Closer** to SPA's actual SFT shape (their samples are also single-step (s, a, s') triples) |
| Long training time (~1.5 h) | ~6× shorter sequences → ~10–20 min training |

### 4.3 Open question: state-classification vs action-conditional semantics

Before retraining, we should pin down what `<solvable>` actually means in the response. Two options:

**Option 1 (current, action-conditional):**
- `<solvable>` = is_solvable(s_{t+1}) given the model's chosen action
- `<breaking_point>` = did this action cause the transition
- Useful for "agent self-checking its own next move"
- Matches the data we already have

**Option 2 (state-only classification):**
- `<solvable>` = is_solvable(s_t) — current state, decoupled from action
- `<breaking_point>` = was the previous action a BP (i.e., prev step's action put us here)
- Useful for "external monitor of any agent's state"
- Aligns more cleanly with SPEC.md §1's predictive-gap criterion (the gap exists *between* the BP and formal failure, regardless of next action)

Recommendation: **Option 2**, because our research question is specifically state classification. The agent's next action shouldn't change whether the current state is doomed. We still keep `<answer>` so the model can act, but the termination tags are properties of the state.

This requires a one-line label change in the formatter: instead of `step.is_solvable` (which is post-action), use `prev_solvable` (or equivalently shift labels by one step).

### 4.4 Migration plan

1. **Decide Option 1 vs Option 2** (~1 conversation turn).
2. **Modify `SFTFormatter.format_step()`** to build single-step samples with the chosen semantics. Preserve the existing chat-template structure so the trainer needs no changes.
3. **Reformat the existing 1,280 trajectories** into single-step parquets:
   - `data/sudoku_llm_policy_singlestep/wm_train.parquet`
   - `data/sudoku_llm_policy_singlestep/wm_val.parquet`
   - Estimated samples: ~16,000 train (vs current 6,221) — recovering all length-filtered samples.
4. **Re-train SFT** on single-step parquet:
   - Output: `outputs/sft_sudoku_singlestep/`
   - Estimated wall time: ~15–25 min on H800
5. **Re-run Tier A eval** (single-turn or multi-turn — both equivalent now):
   - If BP recall lifts to 30%+, the diagnosis holds and we proceed to Tier B/C.
   - If BP recall is still <20%, the issue is class weighting / data scale, not multi-turn structure — next move is class-weighted loss + more BP-rich data generation.

### 4.5 What this preserves and what it loses

**Preserves:**
- Full SPA-style XML response (observation + prediction + termination tags + answer)
- Ground-truth replacement for `<observation>` and `<prediction>` (matches SPA §2.2)
- All termination tags as research extension on top of SPA
- LLM-policy data lineage (actions chosen by Qwen during data gen, not random)
- Single-GPU constraint, GRPO RL plan unchanged for downstream

**Loses:**
- Multi-turn rollout exposure during SFT — but RL is multi-turn, so the model still learns rollout context at the RL stage
- The Q5 research question ("does multi-turn help BP detection") gets answered at SFT stage and recorded; multi-turn is no longer load-bearing for our pipeline

---

## 5. Implications for SPEC.md and pipeline_design.md

To be reflected in the next spec rev:

- **§7 locked decisions:** "Multi-turn SFT format with sliding window" should be **un-locked** and replaced with "Single-step SFT samples with full SPA-style response." Multi-turn at SFT was not a load-bearing design decision — it was inherited from a misreading of SPA's recipe.
- **§7 open decisions:** add "State-classification vs action-conditional semantics for `<solvable>` and `<breaking_point>`" if we leave that open beyond this report.
- **§2 Q5** (multi-turn vs single-turn) is answered: single-turn (specifically single-step) is better for our task because the label is fully determined by the grid and history enables a destructive shortcut.
- **pipeline_design.md** Stage 2 SFT section: update the data flow to show single-step samples, drop multi-turn references.

---

## 6. Reference

- [eval_2026-04-28_data_strategy.md](eval_2026-04-28_data_strategy.md) — pre-training data inspection (predicted the BP-step skew that contributed to this failure)
- [eval_2026-04-28_sft_track_b_tier_a.md](eval_2026-04-28_sft_track_b_tier_a.md) — single-turn-eval write-up
- [pipeline_design.md](pipeline_design.md) — current pipeline (needs update per §5)
- [SPEC.md](SPEC.md) — research framing (needs update per §5)
- SPA paper §2.2 — the actual training recipe we should be matching
