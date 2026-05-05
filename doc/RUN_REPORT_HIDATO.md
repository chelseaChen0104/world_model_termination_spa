# Hidato 5×4 SAVE Run Report

> Living document. Mirrors `doc/RUN_REPORT.md` (Sudoku) for the Hidato env.
> Auto-updated as experiments complete on autodl.

## Setup

| Component | Value |
|---|---|
| Env | Hidato 5×4 (Hamiltonian-path number-placement, N=20) |
| Puzzle bank | v3 600 puzzles (algorithmic; toy used v2 200 puzzles) |
| Policy (lt/ht sampler) | `rl_b_h1_v8_anchor` (B-H1 RL'd 1.5B Qwen) |
| Toy data | 1500 / 346 / 345 (train/val/test) sibling sets |
| Pilot data | 3000 / 1000 / 1000 sibling sets |
| Pilot SFT samples (after expansion) | 5219 / 1126 / 1115 |
| Trainer | `scripts/sudoku_scripts/save_sft_train.py` (paper-aligned, L_trans + λL_viab + ηL_rank + μL_state) |
| Hyperparams | epochs=3, lr=1e-5, batch_sets=8, warmup=100, λ=η=1.0, μ=0.5 |
| Calibration | `scripts/sudoku_scripts/save_sft_calibrate.py` (temperature + τ_keep + τ_fb) |

---

## Phase A — Q1 + Q2 cross-size sweep (toy data)

**Status:** in progress on autodl (`tmux hidato_q1_sweep`)

Models in sweep (cheapest first):
1. Qwen/Qwen2.5-0.5B-Instruct
2. Qwen/Qwen2.5-1.5B-Instruct
3. meta-llama/Llama-3.2-1B-Instruct
4. Qwen/Qwen2.5-3B-Instruct (gradient_checkpointing)
5. meta-llama/Llama-3.2-3B-Instruct (gradient_checkpointing)
6. Qwen/Qwen2.5-7B-Instruct (paged_adamw_8bit + gradient_checkpointing)

### Q1 Table 1 — Cross-size on toy (TBD as runs complete)

| Model | Transition cell-acc ↑ | Viability AUC ↑ | Brier ↓ | ECE@10 ↓ | Pairwise ↑ | Status |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | TBD | TBD | TBD | TBD | TBD | running |
| Qwen2.5-1.5B (toy) | TBD | TBD | TBD | TBD | TBD | queued |
| Llama-3.2-1B | TBD | TBD | TBD | TBD | TBD | queued |
| Qwen2.5-3B | TBD | TBD | TBD | TBD | TBD | queued |
| Llama-3.2-3B | TBD | TBD | TBD | TBD | TBD | queued |
| Qwen2.5-7B | TBD | TBD | TBD | TBD | TBD | queued |

### Q1 Table 1 (pilot row, already done)

| Model | Viability AUC ↑ | Brier ↓ | ECE@10 ↓ | Pairwise ↑ | Deceptive ↑ | n_test |
|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B (pilot, paper-aligned)** | **0.862** | **0.154** | **0.087** | **0.826** | **0.826** | 1115 |

Calibration (pilot 1.5B):
- T = 1.854
- τ_keep = 0.911 (precision_on_True 0.954, kept_fraction 0.155)
- **τ_fb = 0.110** ✅ (precision_on_False 1.000, below_fraction 0.0018) — paper-strict ε_fb=0.05 is FEASIBLE on Hidato pilot

For comparison, autodl2 Sudoku pilot:
- AUC 0.931, Brier 0.089, ECE 0.054, Pairwise 0.833 (slightly stronger; denser SFT data — 4.15 vs 2.14 candidates per record)

---

## Phase B — Q3 deceptive bench (TBD)

**Status:** pending q4_methods.py port to env-agnostic

7 methods to evaluate on test deceptive pairs:
- Policy top-1
- Best-of-K (K=8)
- Local progress heuristic (Hidato: path coverage with adjacency penalty)
- Prompted score-only
- Learned progress-score
- **SAVE**
- Oracle viability

Metric: non-viable selection rate (lower is better).

---

## Phase C — Q4 online rollout (TBD)

**Status:** pending q4 port

Same 7 methods × N=200 puzzles. Track:
- Pass@1
- Dead-end entry rate
- First-DE-step
- Non-viable selected
- Tokens per episode
- NetCompute = (PolicyTokens + K · EvalTokens) / PolicyTokens_top1

---

## Phase D — Q5 termination (TBD)

**Status:** pending q4 port + Q5 harness

6 termination variants:
- No termination
- Greedy termination
- **SAVE termination** at ε_fb={0.05, 0.10, 0.20, 0.30}
- SAVE + retry
- Random matched-rate
- Oracle termination

Metrics: false term rate, true doomed term rate, tokens saved, Pass@1 drop.

---

## Phase E — Ablations (TBD)

| Ablation | Status |
|---|---|
| K-sweep (K=1, 2, 4, 8) | pending Q4 port |
| no-calibration (T=1.0) | pending Q4 port |
| single-threshold (τ_fb = τ_keep) | pending Q4 port |
| no-rank (η=0.0 retrain) | needs separate training run |
| state-conditioned only (only L_state) | needs separate training run |

---

## Disk + GPU notes

- autodl /root/autodl-tmp: 20 GB free at start of Q1 sweep
- Q1 sweep estimated total disk usage: ~26 GB downloads + ~50 GB outputs (across 6 sizes)
- Mitigation: cleanup intermediate `checkpoint-N` dirs after each size; output `final/` only
- GPU: H800 80GB; OOM risk at 7B requires `paged_adamw_8bit`

---

## Anomalies / failures

(none yet — will fill as runs progress)

---

## Artifact paths (autodl)

```
outputs/save_hidato5x4_f_phi_paper/{eval_test,calibration}.json   # 1.5B pilot, paper-aligned
outputs/save_hidato5x4_f_phi_Qwen2.5-0.5B/{eval_test,calibration}.json   # Phase A
outputs/save_hidato5x4_f_phi_Qwen2.5-1.5B/...
outputs/save_hidato5x4_f_phi_Llama-3.2-1B/...
outputs/save_hidato5x4_f_phi_Qwen2.5-3B/...
outputs/save_hidato5x4_f_phi_Llama-3.2-3B/...
outputs/save_hidato5x4_f_phi_Qwen2.5-7B/...

logs/hidato_q1_sweep.log
logs/hidato_size_<short>.log
```

Local archive: `/Volumes/yy_drive/SPA_termination/save_models/hidato5x4_f_phi/` (v1) and `/Volumes/yy_drive/SPA_termination/save_models/hidato5x4_f_phi_v2/` (v2, partial).

---

*Last updated: report initialized 2026-05-05; Q1 sweep launched.*
