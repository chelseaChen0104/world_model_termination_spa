"""Parse training logs for eval_loss values and plot side-by-side.

Reads logs/sft_*.log and logs/sft_4x4.log from the local repo (synced from clouds).
Extracts {'eval_loss': X, 'epoch': Y} dicts and plots them.

Output: doc/plots/loss_curves.png
"""
import os
import re
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "doc", "plots", "loss_curves.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


# (run-name, log-file, color, line-style, label, total_steps_per_epoch)
# Last field is used as fallback when the eval_loss line is mangled by interleaved tqdm
# output (in which case 'epoch': X is overwritten and we infer epoch from the
# nearest "<step>/<total>" tqdm marker). Set to None to skip the fallback.
RUNS = [
    ("9x9 multi-turn (B-0)",      "sft_b.log",          "tab:red",    "--", "9x9 multi-turn (failed: temporal echo)",       None),
    ("9x9 single-step B-1/B-2",   "sft_b_minimal.log",  "tab:orange", "-",  "9x9 single-step, post-BP kept (B-2)",          None),
    ("9x9 single-step B-3",       "sft_b2.log",         "tab:blue",   "-",  "9x9 single-step, no post-BP (B-3)",            None),
    ("4x4 single-step",           "sft_4x4.log",        "tab:green",  "-",  "4x4 single-step, no post-BP",                  None),
    # B-4: 5 epochs × 155 updates/epoch = 775 total. Crashed at step 600 (epoch 3.87).
    ("9x9 SPA hparams (B-4)",     "sft_b4.log",         "tab:purple", "-",  "9x9 + SPA hparams (B-4, lr=1e-4 ep=5*)",       155),
    # B-5: 6571 train / bs 16 ≈ 410 updates/epoch × 5 epochs = 2050 steps. eval_steps=10 → 205 points.
    ("4x4 SPA replication (B-5)", "sft_b5.log",         "tab:olive",  "-",  "4x4 + SPA hparams + SPA-scale data (B-5)",     410),
]


def extract(log_path, steps_per_epoch=None):
    """Return list of (epoch, eval_loss) tuples found in the log.

    Primary path: parse `{'eval_loss': X, ..., 'epoch': Y}` dicts.
    Fallback (used when tqdm overwrites the dict): for each `'eval_loss': X`
    occurrence, find the next `<step>/<total>` tqdm marker and compute
    epoch = step / steps_per_epoch.
    """
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r", errors="ignore") as f:
        txt = f.read()
    pat = re.compile(r"\{'eval_loss': ([0-9.eE+-]+),.*?'epoch': ([0-9.]+)\}")
    out = [(float(m[1]), float(m[0])) for m in pat.findall(txt)]
    if not out and steps_per_epoch:
        seen_vals = []
        for m in re.finditer(r"'eval_loss': ([0-9.eE+-]+)", txt):
            v = float(m.group(1))
            sm = re.search(r"(\d+)/\d+", txt[m.end():m.end() + 400])
            if sm and (not seen_vals or seen_vals[-1] != v):
                step = int(sm.group(1))
                out.append((step / steps_per_epoch, v))
                seen_vals.append(v)
    out.sort()
    return out


fig, ax = plt.subplots(figsize=(10, 6))
for run_name, log_file, color, ls, label, spe in RUNS:
    log_path = os.path.join(LOG_DIR, log_file)
    points = extract(log_path, steps_per_epoch=spe)
    if not points:
        print(f"  no eval_loss found in {log_file} — skipping")
        continue
    epochs = [p[0] for p in points]
    losses = [p[1] for p in points]
    ax.plot(epochs, losses, color=color, linestyle=ls, marker="o", markersize=5, label=f"{label} (n={len(points)})")
    print(f"  {label}: {len(points)} points, epoch {epochs[0]:.2f}–{epochs[-1]:.2f}, loss {min(losses):.4f}–{max(losses):.4f}")

ax.set_xlabel("Epoch")
ax.set_ylabel("eval_loss")
ax.set_title("SFT eval_loss curves — 9x9 vs 4x4 vs SPA hparams")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

# Y-axis: log scale to see relative changes more clearly
ax.set_yscale("log")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=140)
print(f"\nWrote: {OUT_PATH}")
