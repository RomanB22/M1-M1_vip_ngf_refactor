from pathlib import Path
import ast
import pandas as pd

path = Path("camesJan26.txt")

rows = []
for line in path.read_text(errors="ignore").splitlines():
    if line.startswith("inserting data into storage: "):
        payload = line[len("inserting data into storage: "):].strip()
        try:
            rows.append(ast.literal_eval(payload))
        except Exception:
            # Skip malformed lines if any
            pass

df = pd.DataFrame(rows)

# Sort by loss (smallest to largest)
df_sorted = df.sort_values("loss", ascending=True)

print(df_sorted[["trial_label", "loss"]].head(40))
