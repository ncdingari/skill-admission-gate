"""Table for E21 (conditional-null stress tests): false-admission rate (%) by departure and gate."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results/E21/e21_null_stress_grid.csv")
LAB = {"control": "None (design of Section~\\ref{sec:setup})",
       "coin_hard": "Exam-level coin, $D_i\\equiv\\pm1$",
       "coin_shift_0.05": "Exam-level coin, lift $\\pm0.05$", "coin_shift_0.10": "Exam-level coin, lift $\\pm0.10$",
       "snapshot_both": "Model change mid-exam, both arms", "snapshot_skill": "Model change mid-exam, skill arm only",
       "block_5_0.5": "Shared effect, blocks of 5, sd 0.5", "block_10_0.5": "Shared effect, blocks of 10, sd 0.5",
       "block_10_1.0": "Shared effect, blocks of 10, sd 1.0",
       "repeat_0.01": "Repeated seeds, 1\\%", "repeat_0.05": "Repeated seeds, 5\\%", "repeat_0.20": "Repeated seeds, 20\\%",
       "exclude_0.01": "Excluding skill-arm losses, 1\\%", "exclude_0.05": "Excluding skill-arm losses, 5\\%",
       "exclude_0.20": "Excluding skill-arm losses, 20\\%"}
G = [("grid", "Grid"), ("prpl_c050", "PrPl"), ("msprt", "mSPRT"), ("sprt", "SPRT"), ("fixed_t", "Fixed-$n$")]
p = df.pivot(index="cell", columns="gate", values="false_admit") * 100
lines = ["\\begin{tabular}{l" + "r" * len(G) + "}", "\\toprule",
         "Departure from the design & " + " & ".join(g[1] for g in G) + " \\\\", "\\midrule"]
for c in LAB:
    lines.append(LAB[c] + " & " + " & ".join(f"{p.loc[c, g]:.1f}" for g, _ in G) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "paper/tables/t_null_stress.tex").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
