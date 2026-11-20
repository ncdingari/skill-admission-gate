"""Figures F2 to F6 from results/ CSVs only (F1 is TikZ). Usage: python figures/make_figs.py [--quick] [F2 ...]

Each figure is written to figures/out/<stem>.{html,png,pdf,options.json} by export.save (Okabe-Ito
palette, 300 dpi PNG, vector PDF).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import export  # noqa: E402
import style as S  # noqa: E402

C = S.PALETTE
GATE_NAME = {"G1_fixed_t": "Fixed-n t-test", "G2_peek_t": "t-test, re-checked each variant",
             "G3_peek_ci": "95% CI, re-checked every 10", "G4_sprt": "SPRT, discordant pairs",
             "G5_msprt": "mSPRT", "G7_grid": "Betting, grid bet (proposed)", "G7_grid_futm": "Betting + futility margin"}
GATE_COLOR = {"G1_fixed_t": C["black"], "G2_peek_t": C["vermillion"], "G3_peek_ci": C["orange"],
              "G4_sprt": C["reddish_purple"], "G5_msprt": C["sky_blue"], "G7_grid": C["blue"],
              "G7_grid_futm": C["bluish_green"]}


def line(name, xs, ys, color, dash="solid", sym="circle", grid=0, width=3):
    return {"type": "line", "name": name, "xAxisIndex": grid, "yAxisIndex": grid,
            "data": [[float(x), float(y)] for x, y in zip(xs, ys) if np.isfinite(y)],
            "symbol": sym, "symbolSize": 10, "lineStyle": {"color": color, "type": dash, "width": width},
            "itemStyle": {"color": color}}


def f2(res, out):
    e1 = pd.read_csv(res / "E1" / "e1_false_admission.csv")
    e8 = pd.read_csv(res / "E8" / "e8_empirical_null_gates.csv")
    gates = ["G2_peek_t", "G3_peek_ci", "G1_fixed_t", "G4_sprt", "G5_msprt", "G7_grid", "G7_grid_futm"]
    scen = [("boundary", "Null boundary (model)", C["blue"]), ("two_point_worst", "Worst case (all pairs discordant)", C["orange"]),
            ("placebo_webdev", "Placebo, lift -1.3 pp", C["bluish_green"])]
    o = S.base_opts("", "False-admission rate (%)", "", "value", "category")
    o["grid"][0].update(left=330, top=120, bottom=110)
    o["yAxis"][0].update(data=[GATE_NAME[g] for g in gates], inverse=True)
    o["yAxis"][0]["axisLabel"]["fontSize"] = 22
    for key, name, col in scen:
        d = e1[e1.cell == key].set_index("gate")
        o["series"].append({"type": "bar", "name": name, "barGap": "10%", "itemStyle": {"color": col},
                            "data": [round(100 * d.admit_rate.get(g, np.nan), 2) if g in d.index else None for g in gates]})
    d = e8.set_index("gate")
    o["series"].append({"type": "bar", "name": "SWE-Skills-Bench resampled null", "itemStyle": {"color": C["reddish_purple"]},
                        "data": [round(100 * d.admit_rate[g], 2) if g in d.index else None for g in gates]})
    o["series"][0]["markLine"] = {"symbol": "none", "silent": True, "label": {"show": True, "position": "start", "formatter": "α = 5%", **S.text(22)},
                                  "lineStyle": {"color": C["black"], "type": "dashed", "width": 2}, "data": [{"xAxis": 5}]}
    o["legend"] = [S.legend(left=20, top=10, itemWidth=30, itemHeight=18, width=1000)]
    export.save(S.make_chart(o, 820), "F2_false_admission", out)


def f3(res, out):
    e2 = pd.read_csv(res / "E2" / "e2_power_cost.csv")
    e2 = e2[e2.config == "main"]
    gates = ["G1_fixed_t", "G3_peek_ci", "G4_sprt", "G5_msprt", "G7_grid", "G7_grid_futm"]
    o = S.skeleton()
    o["grid"] = [{"left": 120, "right": 30, "top": 130, "height": 330}, {"left": 120, "right": 30, "top": 600, "height": 330}]
    o["xAxis"] = [S.axis("", "value", 0, min=-0.05, max=0.3), S.axis("True lift Δ (fraction of tasks)", "value", 1, min=-0.05, max=0.3)]
    o["yAxis"] = [S.axis("P(admit)", "value", 0, gap=70, min=0, max=1), S.axis("Mean variants E[N]", "value", 1, gap=80)]
    o["title"] = [S.title("(a)", 10, 120), S.title("(b)", 10, 590)]
    for g in gates:
        d = e2[e2.gate == g].sort_values("true_lift")
        dash = "dashed" if g == "G3_peek_ci" else "solid"
        o["series"].append(line(GATE_NAME[g], d.true_lift, d.admit_rate, GATE_COLOR[g], dash, grid=0))
        o["series"].append(line(GATE_NAME[g], d.true_lift, d.mean_n, GATE_COLOR[g], dash, grid=1))
    o["legend"] = [S.legend([GATE_NAME[g] for g in gates], left=20, top=5, width=1000, itemWidth=30, itemHeight=16)]
    export.save(S.make_chart(o, 1010), "F3_power_cost", out)


def f4(res, out):
    d = pd.read_csv(res / "E3" / "e3_resubmission_grid.csv")
    o = S.base_opts("", "Resubmissions of a useless skill", "P(admitted at least once)")
    o["yAxis"][0].update(min=0, max=1, nameGap=70)
    o["grid"][0].update(left=120, top=90, bottom=110)
    for rule, name, col, dash in (("peek_ci_per_attempt", "95% CI re-checked, each attempt", C["orange"], "dashed"),
                                  ("per_exam_alpha", "Betting gate at α each attempt", C["blue"], "solid"),
                                  ("lineage_spending", "Betting gate, lineage α-spending", C["bluish_green"], "solid")):
        x = d[d.rule == rule].sort_values("attempt")
        o["series"].append(line(name, x.attempt, x.p_admitted_by, col, dash))
    o["legend"] = [S.legend(left=20, top=10, width=1000, itemWidth=30, itemHeight=16)]
    export.save(S.make_chart(o, 600), "F4_resubmission", out)


def f5(res, out):
    d = pd.read_csv(res / "E9" / "e9_variants_needed.csv")
    d = d[d.lift > 0]
    o = S.base_opts("", "Observed lift of the curated skill (SkillsBench)", "Median variants to certify", "value", "log")
    o["grid"][0].update(left=130, top=90, bottom=110)
    o["yAxis"][0].update(min=10, max=10000, nameGap=80)
    for col_, name, c in (("median_n_alpha_grid", "Per-exam α = 0.05", C["blue"]), ("median_n_bonf_grid", "Library-wide α/87", C["vermillion"])):
        y = d[col_].replace(np.inf, 5000).clip(upper=5000)
        o["series"].append({"type": "scatter", "name": name, "symbolSize": 12, "itemStyle": {"color": c},
                            "data": [[float(a), float(b)] for a, b in zip(d.lift, y)]})
    o["series"][0]["markLine"] = {"symbol": "none", "silent": True,
                                  "lineStyle": {"color": C["black"], "type": "dashed", "width": 2},
                                  "label": {**S.text(20), "formatter": "{b}", "position": "insideEndTop"},
                                  "data": [{"yAxis": 18, "name": "18 configurations released"},
                                           {"yAxis": 600, "name": "n_max = 600"}]}
    o["legend"] = [S.legend(left=20, top=10, itemWidth=30, itemHeight=16)]
    export.save(S.make_chart(o, 640), "F5_variants_needed", out)


def f6(res, out):
    d = pd.read_csv(res / "E4" / "e4_estimation.csv").sort_values("true_lift")
    o = S.base_opts("", "True lift Δ", "Lift printed on the credential")
    o["grid"][0].update(left=120, top=90, bottom=110)
    o["xAxis"][0].update(min=0, max=0.26)
    o["yAxis"][0].update(min=0, max=0.3, nameGap=70)
    o["series"] = [line("Δ̂, betting gate (admitted)", d.true_lift, d.mean_dhat_admitted_bet, C["blue"]),
                   line("Δ̂, fixed-n t-test (admitted)", d.true_lift, d.mean_dhat_admitted_fixed, C["black"]),
                   line("Anytime-valid lower bound at stop", d.true_lift, d.mean_lcb_admitted, C["bluish_green"], "dashed"),
                   line("Truth (y = x)", [0, 0.26], [0, 0.26], C["reddish_purple"], "dotted", "none", width=2)]
    o["legend"] = [S.legend(left=20, top=10, width=1000, itemWidth=30, itemHeight=16)]
    export.save(S.make_chart(o, 620), "F6_winners_curse", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=["F2", "F3", "F4", "F5", "F6"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    res = ROOT / ("results_quick" if a.quick else "results")
    out = HERE / ("out_quick" if a.quick else "out")
    for f in a.figs:
        globals()[f.lower()](res, out)
        print(f, "ok")
