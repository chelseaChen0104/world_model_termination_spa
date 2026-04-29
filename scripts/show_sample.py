"""Show a representative training sample side-by-side: full multi-turn vs single-turn."""
import pandas as pd
import json

df = pd.read_parquet("data/sudoku_llm_policy/wm_val_filtered.parquet")

target = None
for i in range(len(df)):
    info = df.iloc[i]["extra_info"]
    if isinstance(info, str):
        info = json.loads(info)
    if info.get("step") == 4 and info.get("is_breaking_point"):
        target = i
        break
if target is None:
    for i in range(len(df)):
        info = df.iloc[i]["extra_info"]
        if isinstance(info, str):
            info = json.loads(info)
        if info.get("step") == 4:
            target = i
            break

row = df.iloc[target]
info = row["extra_info"]
if isinstance(info, str):
    info = json.loads(info)
msgs = list(row["prompt"]) if hasattr(row["prompt"], "tolist") else row["prompt"]

print(f"Sample index: {target}")
print(f"extra_info: {info}")
print(f"Number of messages in prompt: {len(msgs)}")
print()
print("=" * 70)
print("FULL MULTI-TURN PROMPT (training distribution)")
print("=" * 70)
for i, m in enumerate(msgs):
    content = m["content"]
    role = m["role"]
    full_len = len(content)
    if full_len > 350:
        content = content[:350] + f" ... [{full_len - 350} more chars]"
    print()
    print(f"--- msg[{i}] role={role} ({full_len} chars) ---")
    print(content)
print()
print("=" * 70)
print("TARGET RESPONSE (what model trains to predict)")
print("=" * 70)
resp = row["response"]
if len(resp) > 800:
    print(resp[:800] + " ...")
else:
    print(resp)
print(f"\n(total response length: {len(resp)} chars)")
print()
print("=" * 70)
print("SINGLE-TURN VERSION (what evaluate_model gave at Tier A)")
print("=" * 70)
sys_c = msgs[0]["content"]
sys_role = msgs[0]["role"]
last_msg = msgs[-1]
print()
print(f"--- msg[0] role={sys_role} ({len(sys_c)} chars) ---")
if len(sys_c) > 350:
    print(sys_c[:350] + " ...")
else:
    print(sys_c)
print()
print(f"--- msg[1] role={last_msg['role']} ({len(last_msg['content'])} chars) ---")
print(last_msg["content"])
print()
print("[no other messages — model has to predict target above without history]")
