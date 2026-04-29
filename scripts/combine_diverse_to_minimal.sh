#!/bin/bash
# After generate_diverse_data.sh completes:
#   1. Reformat each difficulty's parquet to single-step minimal format
#   2. Concatenate train+val splits across all 3 difficulties
#   3. Output combined diverse-minimal parquets at data/sudoku_llm_policy_diverse_minimal/
#
# Usage: bash scripts/combine_diverse_to_minimal.sh

set -e
PYTHON=${PYTHON:-/root/miniconda3/bin/python}

echo "=== reformatting each difficulty to minimal single-step ==="
for diff in easy medium hard; do
    src=data/sudoku_llm_policy_$diff
    dst=data/sudoku_llm_policy_${diff}_minimal
    if [ ! -d "$src" ]; then
        echo "  skip $diff: $src not found"
        continue
    fi
    echo "  --> $diff"
    $PYTHON scripts/reformat_to_minimal.py \
        --input-dir "$src" \
        --output-dir "$dst"
done

echo
echo "=== concatenating across difficulties ==="
mkdir -p data/sudoku_llm_policy_diverse_minimal
$PYTHON - <<'PY'
import pandas as pd
import os

OUT = "data/sudoku_llm_policy_diverse_minimal"
os.makedirs(OUT, exist_ok=True)

for split in ["wm_train.parquet", "wm_val.parquet"]:
    parts = []
    for d in ["easy", "medium", "hard"]:
        p = f"data/sudoku_llm_policy_{d}_minimal/{split}"
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df["difficulty"] = d  # tag rows with difficulty for later analysis
            parts.append(df)
            print(f"  {p}: {len(df)} rows")
    if not parts:
        print(f"  no parts for {split}")
        continue
    out = pd.concat(parts, ignore_index=True)
    # Shuffle so difficulties are mixed during training
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_path = os.path.join(OUT, split)
    out.to_parquet(out_path)
    print(f"  -> {out_path}: {len(out)} rows")
PY

echo
echo "=== filter post-BP for B-3-style training (optional, if doing class-balanced run) ==="
$PYTHON scripts/filter_post_bp.py --input-dir data/sudoku_llm_policy_diverse_minimal

echo
echo "DONE. Available variants:"
echo "  data/sudoku_llm_policy_diverse_minimal/wm_train.parquet (full)"
echo "  data/sudoku_llm_policy_diverse_minimal/wm_train_no_post_bp.parquet (class-balanced)"
