"""Table for E18: expected variants at matched power (default model, r = 1), under the alternative and a useless skill."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "results/E18/e18_matched_power.csv")
d = d[(d.model == "default") & (d.r == 1)]
G = [("fixed_t", "Fixed-$n$"), ("sprt", "SPRT"), ("msprt", "mSPRT"), ("prpl", "PrPl"), ("grid", "Grid")]
L = ["\\begin{tabular}{ll" + "r" * len(G) + "}", "\\toprule",
     "Lift & Power & " + " & ".join(g for _, g in G) + " \\\\", "\\midrule"]
for (lift, tgt), g in d.groupby(["lift", "target"]):
    a = g.set_index("gate")
    L.append(f"{lift:.3g} & {tgt:.0%} & " .replace("%", "\\%") + " & ".join(
        f"{a.variants_alt[k]:.0f} ({a['variants_null_0.0'][k]:.0f})" if k in a.index else "--" for k, _ in G) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "paper/tables/t_matched_power.tex").write_text("\n".join(L) + "\n")
print("\n".join(L))
