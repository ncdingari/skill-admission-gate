#!/usr/bin/env python3
"""Write paper/numbers.tex (one macro per number quoted in the text) and paper/tables/*.tex,
from results/ CSVs and params.yaml only. Do not edit the outputs by hand.

Usage: python tools/make_numbers.py [--quick]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nixigate.params import load_params, v  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
RES = ROOT / ("results_quick" if a.quick else "results")
P = load_params()
M: dict[str, str] = {}


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def num(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise ValueError("NaN")
    return f"{x:.{d}f}"


def put(name, val):
    assert name.isalpha(), name
    M[name] = val


r0 = lambda e, f: pd.read_csv(RES / e / f"{f}.csv")  # noqa: E731
# Revision B: the grid bet is primary. E3, E3b and E14 read their NIXI_RULE=grid reruns; E1, E2, E5, E8 carry
# both bets in one file and the headline macros below use the grid gates (G7_*, grid_*). r0 reads PrPl files.
GRID_EXPS = ("E3", "E3b", "E14")
r = lambda e, f: r0(e, f + "_grid") if e in GRID_EXPS and (RES / e / f"{f}_grid.csv").exists() else r0(e, f)  # noqa: E731
BET, BETF = "G7_grid", "G7_grid_futm"

# ------------------------------------------------ parameters
put("PAlpha", num(v(P, "alpha"), 2)); put("PAlphaF", num(v(P, "alpha_futility"), 2))
put("PNmax", str(int(v(P, "n_max")))); put("PCap", num(v(P, "bet_cap_c"), 2)); put("PCapAlt", num(v(P, "bet_cap_c_alt"), 2))
put("PLiftOne", pct(v(P, "target_lift"), 0)); put("PFutM", pct(v(P, "futility_margin"), 0))
put("PFutMFrac", f"{v(P, 'futility_margin'):.2f}")
put("PBase", pct(v(P, "base_pass_rate"), 1)); put("PCurated", pct(v(P, "curated_lift"), 1))
put("PSdTheta", num(v(P, "sd_theta"), 1)); put("PSdLift", num(v(P, "sd_lift"), 1)); put("PRho", num(v(P, "rho_crn"), 1))
put("PWindow", str(int(v(P, "window_size")))); put("PWindows", str(int(v(P, "n_windows"))))
put("PNullShare", pct(v(P, "null_share"), 0)); put("PResub", str(int(v(P, "resubmissions"))))
put("PReps", f"{int(v(P, 'mc_reps')):,}".replace(",", "{,}")); put("PRepsPower", f"{int(v(P, 'mc_reps_power')):,}".replace(",", "{,}"))
put("PDrift", num(v(P, "drift_logit"), 1)); put("PGenSpace", str(int(v(P, "gen_space_small"))))
put("PCoverMax", pct(v(P, "leak_cover_max"), 0)); put("PTranscriptKB", str(int(v(P, "transcript_kb"))))

# ------------------------------------------------ E1
e1 = r("E1", "e1_false_admission")
b = e1[e1.cell == "boundary"].set_index("gate").admit_rate
names = {"G1_fixed_t": "Fixed", "G2_peek_t": "PeekT", "G3_peek_ci": "PeekCI", "G4_sprt": "SPRT", "G5_msprt": "MSPRT",
         "G7_grid": "Bet", "G7_grid_futm": "BetFut", "G6_bet_c050": "BetPrpl", "G6_bet_futm": "BetFutPrpl",
         "G6_bet_c075": "BetHi", "G6_bet_agrapa": "BetAg", "G6_bet_capm0": "BetCapm"}
for g, n in names.items():
    put("NEOne" + n, pct(b[g]))
valid = ["G4_sprt", "G5_msprt", "G6_bet_c050", "G6_bet_c075", "G6_bet_agrapa", "G6_bet_futm", "G6_bet_capm0", "G7_grid", "G7_grid_futm"]
ok = e1[(e1.cell != "stale_baseline_drift") & e1.gate.isin(valid)]
put("NEOneValidMax", pct(ok.admit_rate.max()))
put("NEOneValidMaxSE", pct(ok.loc[ok.admit_rate.idxmax(), "admit_se"], 2))
pk = e1[(e1.cell != "stale_baseline_drift") & e1.gate.isin(["G2_peek_t"])]
put("NEOnePeekTMin", pct(pk.admit_rate.min())); put("NEOnePeekTMax", pct(pk.admit_rate.max()))
pc = e1[(e1.cell != "stale_baseline_drift") & e1.gate.isin(["G3_peek_ci"])]
put("NEOnePeekCIMin", pct(pc.admit_rate.min())); put("NEOnePeekCIMax", pct(pc.admit_rate.max()))
st = e1[e1.cell == "stale_baseline_drift"]
put("NEOneStaleMin", pct(st.admit_rate.min(), 0)); put("NEOneStaleMax", pct(st.admit_rate.max(), 0))
put("NEOneStaleBet", pct(st.set_index("gate").admit_rate[BET], 0))
put("NFixedN", str(int(e1[e1.cell == "boundary"].n_fixed.iloc[0])))
put("NEOnePeekRatio", f"{b['G3_peek_ci'] / v(P, 'alpha'):.1f}")

# ------------------------------------------------ E8 empirical null
e8n = r("E8", "e8_empirical_null_gates").set_index("gate")
for g, n in (("G2_peek_t", "PeekT"), ("G3_peek_ci", "PeekCI"), ("G1_fixed_t", "Fixed"), (BET, "Bet"),
             (BETF, "BetFut"), ("G5_msprt", "MSPRT"), ("G6_bet_c050", "BetPrpl"), ("G6_bet_futm", "BetFutPrpl")):
    put("NEEightNull" + n, pct(e8n.admit_rate[g]))
put("NSweTies", pct(e8n.tie_share.iloc[0], 0)); put("NSweSD", num(e8n.sd_D.iloc[0], 2))

# ------------------------------------------------ E2
e2 = r("E2", "e2_power_cost")
m = e2[e2.config == "main"]
at = lambda g, l, c: float(m[(m.gate == g) & (np.isclose(m.true_lift, l))][c].iloc[0])  # noqa: E731
L1, LC = v(P, "target_lift"), v(P, "curated_lift")
put("NETwoPowFixed", pct(at("G1_fixed_t", L1, "admit_rate"), 0)); put("NETwoPowBet", pct(at(BETF, L1, "admit_rate"), 0))
put("NETwoPowSPRT", pct(at("G4_sprt", L1, "admit_rate"), 0))
put("NETwoNBetLOne", f"{at(BETF, L1, 'mean_n'):.0f}"); put("NETwoNSPRTLOne", f"{at('G4_sprt', L1, 'mean_n'):.0f}")
put("NETwoNBetCur", f"{at(BETF, LC, 'mean_n'):.0f}"); put("NETwoNSPRTCur", f"{at('G4_sprt', LC, 'mean_n'):.0f}")
put("NETwoSaveCur", pct(1 - at(BETF, LC, "mean_n") / at("G1_fixed_t", LC, "mean_n"), 0))
put("NETwoNBetZero", f"{at(BET, 0.0, 'mean_n'):.0f}"); put("NETwoNBetFutZero", f"{at(BETF, 0.0, 'mean_n'):.0f}")
put("NETwoNBetFutNeg", f"{at(BETF, -0.05, 'mean_n'):.0f}"); put("NETwoNSPRTZero", f"{at('G4_sprt', 0.0, 'mean_n'):.0f}")
put("NETwoPowBetHalf", pct(at(BETF, 0.05, "admit_rate"), 0)); put("NETwoPowFixedHalf", pct(at("G1_fixed_t", 0.05, "admit_rate"), 0))
put("NETwoPowPeekZero", pct(at("G3_peek_ci", 0.0, "admit_rate"), 0))
# Nmax 300 ablation: power at L1
n3 = e2[(e2.config == "nmax_300") & np.isclose(e2.true_lift, L1)].set_index("gate")
put("NETwoPowBetThree", pct(n3.admit_rate[BET], 0)); put("NETwoRunsBetThree", f"{n3.mean_runs[BET]:.0f}")
put("NETwoRunsFixed", f"{n3.mean_runs['G1_fixed_t']:.0f}")

# PrPl comparison (revision B: reported next to the grid numbers)
put("NETwoPowBetPrpl", pct(at("G6_bet_futm", L1, "admit_rate"), 0)); put("NETwoNBetLOnePrpl", f"{at('G6_bet_futm', L1, 'mean_n'):.0f}")
put("NETwoNBetCurPrpl", f"{at('G6_bet_futm', LC, 'mean_n'):.0f}"); put("NETwoPowBetHalfPrpl", pct(at("G6_bet_futm", 0.05, "admit_rate"), 0))
put("NETwoNullBet", pct(at(BETF, 0.0, "admit_rate"), 1)); put("NETwoNullBetPrpl", pct(at("G6_bet_futm", 0.0, "admit_rate"), 1))

# ------------------------------------------------ E3
e3 = r("E3", "e3_window_fdr").set_index("method")
for k, n in (("M1_per_exam", "PerExam"), ("M2_ebh_stop_1_over_alpha", "StopOne"), ("M3_ebh_stop_m_over_alpha", "Stopm"),
             ("M4_ebh_run_to_nmax", "RunMax"), ("M5_elond_infinite", "ELond"), ("M6_elond_horizon", "ELondH")):
    put("NEThreeFdr" + n, pct(e3.fdr[k])); put("NEThreePow" + n, pct(e3.power[k], 0)); put("NEThreeN" + n, f"{e3.mean_n[k]:.0f}")
rs = r("E3", "e3_resubmission")
K = int(v(P, "resubmissions"))
for rule, n in (("per_exam_alpha", "PerExam"), ("peek_ci_per_attempt", "PeekCI"), ("lineage_spending", "Lineage")):
    x = rs[(rs.rule == rule) & (rs.attempt == K)].p_admitted_by.iloc[0]
    put("NResub" + n, pct(x, 1))
    put("NResubFive" + n, pct(rs[(rs.rule == rule) & (rs.attempt == 5)].p_admitted_by.iloc[0], 1))
    if n == "PerExam":   # constant per-exam size that gives the same K-attempt rate (Theorem 1b)
        put("NResubImpliedSize", pct(1 - (1 - x) ** (1 / K), 1))

# ------------------------------------------------ E4
e4 = r("E4", "e4_estimation").set_index("true_lift")
put("NEFourRelBiasFive", pct(e4.rel_bias_bet[0.05], 0)); put("NEFourRelBiasTen", pct(e4.rel_bias_bet[0.10], 0))
put("NEFourBiasFixedFive", pct(e4.bias_fixed[0.05] / 0.05, 0))
put("NEFourCoverMin", pct(e4.time_uniform_cover.min(), 1)); put("NEFourLcbMax", pct(e4.mean_lcb_admitted.max(), 1))

# ------------------------------------------------ E5
e5 = r("E5", "e5_library_scenarios")
sw = e5[e5.scenario == "swe_like"].set_index("gate")
for g, n in (("fixed_t_uncorrected", "Fixed"), ("peek_ci", "PeekCI"), ("grid_futility_margin", "Bet"),
             ("grid_elond", "ELond"), ("sprt_discordant", "SPRT")):
    put("NEFiveSweFalse" + n, pct(sw.false_admit_share_of_admitted[g], 0))
    put("NEFiveSweRuns" + n, f"{sw.runs_per_true_admission[g]:,.0f}".replace(",", "{,}"))
cu = e5[e5.scenario == "curated_like"].set_index("gate")
put("NEFiveCurRunsBet", f"{cu.runs_per_true_admission['grid_futility_margin']:.0f}")
put("NEFiveCurRunsFixed", f"{cu.runs_per_true_admission['fixed_t_uncorrected']:.0f}")
put("NEFiveCurRunsELond", f"{cu.runs_per_true_admission['grid_elond']:.0f}")
put("NEFiveFutSave", pct(1 - sw.mean_runs_per_candidate["betting_futility_margin"] / sw.mean_runs_per_candidate["betting_per_exam"], 0))

# ------------------------------------------------ E6, E7
e6 = r("E6", "e6_leakage")
cl = e6[e6.kind == "clean"].set_index("detector")
put("NEsixFPLiteral", pct(cl.recall["literal"], 0)); put("NEsixFPDecoded", pct(cl.recall["decoded"], 0))
l5 = e6[(e6.table_share == 0.05) & (e6.detector == "literal")].set_index("kind")
put("NEsixLitSpaced", pct(l5.recall["spaced"], 0)); put("NEsixLitSplit", pct(l5.recall["split"], 0))
put("NEsixCleanCover", num(100 * cl.mean_cover_clean["literal"], 3))
e7 = r("E7", "e7_verification").set_index("n_variants")
nm = e7.index.max()
put("NESevenN", str(int(nm))); put("NESevenVerify", num(e7.verify_s[nm], 2)); put("NESevenProctor", num(e7.proctor_s[nm], 2))
put("NESevenMB", f"{e7.transcript_mb[nm]:.0f}"); put("NESevenLedger", f"{e7.ledger_bytes[nm]:.0f}")

# ------------------------------------------------ E8 SWE-Skills-Bench
s8 = r("E8", "e8_summary").set_index("rule")
put("NSweCells", str(int(s8.cells["all_cells"]))); put("NSweGolden", str(int(s8.admitted_cells["golden_source_rule"])))
put("NSweGoldenSkills", str(int(s8.skills["golden_source_rule"]))); put("NSweHarmful", str(int(s8.admitted_cells["harmful_source_rule"])))
put("NSweOneSided", str(int(s8.admitted_cells["one_sided"]))); put("NSweBH", str(int(s8.admitted_cells["bh"])))
put("NSweBet", str(int(s8.admitted_cells["bet_c075"]))); put("NSweEBH", str(int(s8.admitted_cells["ebh"])))
put("NSweRepLift", pct(s8.mean_lift_other_configs["golden_source_rule"], 1)); put("NSweRepPos", pct(s8.share_other_configs_positive["golden_source_rule"], 0))
put("NSweAllPos", pct(s8.share_other_configs_positive["all_cells"], 0))
put("NSwePooled", str(int(s8.admitted_cells["pooled_one_sided"]))); put("NSwePooledBH", str(int(s8.admitted_cells["pooled_bh"])))
put("NSwePooledBet", str(int(s8.admitted_cells["pooled_grid"]))); put("NSwePooledBetPrpl", str(int(s8.admitted_cells["pooled_betting"])))
put("NSweBetGrid", str(int(s8.admitted_cells["bet_grid"])))
cells = r("E8", "e8_cells")
from nixigate import gates as _g
import numpy as _np
put("NSweMaxWealth", f"{_np.exp(_g.betting(_np.ones((1, 10)), 0, 0.05, None, 0.5, admit_stop=False).log_wealth[0]):.1f}")
put("NSweMaxWealthHi", f"{_np.exp(_g.betting(_np.ones((1, 10)), 0, 0.05, None, 0.75, admit_stop=False).log_wealth[0]):.1f}")
put("NSweObsWealth", f"{math.exp(cells.logK_final.max()):.1f}")
put("NSweMaxWealthGrid", f"{_np.exp(_g.betting(_np.ones((1, 10)), 0, 0.05, None, rule='grid', admit_stop=False).log_wealth[0]):.0f}")
put("NSweObsWealthGrid", f"{math.exp(cells.logK_final_grid.max()):.1f}")
put("NSweGridMinAllOnes", str(next(n for n in range(1, 40) if _g.betting(_np.ones((1, n)), 0, 0.05, None, rule="grid").admit[0])))
sh = r("E8", "e8_config_shopping_null").iloc[0]
put("NShopPer", pct(sh.per_config_rate, 1)); put("NShopAny", pct(sh.any_of_4_rate, 1))
put("NShopRatio", f"{sh.any_of_4_rate / sh.per_config_rate:.1f}"); put("NShopBonf", pct(sh.bonferroni_any_rate, 2))

# ------------------------------------------------ E9 SkillsBench
f9 = r("E9", "e9_full_rules").set_index("rule")
for k, n in (("t_uncorrected", "T"), ("t_bh", "TBH"), ("betting_grid", "Bet"), ("betting_per_exam", "BetPrpl"), ("betting_c075", "BetHi"),
             ("betting_grid_ebh", "EBH"), ("betting_ebh", "EBHPrpl")):
    put("NSb" + n, str(int(f9.admitted[k])))
put("NSbTasks", str(int(f9.of.iloc[0])))
sp = r("E9", "e9_split_half_replication").set_index("rule")
put("NSbSplitAdm", f"{sp.admitted['t_uncorrected']:.1f}"); put("NSbSplitLift", pct(sp.mean_lift_B_admitted["t_uncorrected"], 0))
put("NSbSplitPos", pct(sp.share_B_positive["t_uncorrected"], 0)); put("NSbSplitNot", pct(sp.mean_lift_B_not_admitted["t_uncorrected"], 0))
put("NSbSplitBet", f"{sp.admitted['betting_c075']:.1f}")
put("NSbSplitBHAdm", f"{sp.admitted['t_bh']:.1f}"); put("NSbSplitBHLift", pct(sp.mean_lift_B_admitted["t_bh"], 0)); put("NSbSplitBHPos", pct(sp.share_B_positive["t_bh"], 0))
cal = r("E9", "e9_calibration").set_index("source")
put("NSbLift", pct(cal.mean_D["skillsbench_released"], 1)); put("NSbSD", num(cal.sd_D["skillsbench_released"], 2))
put("NSbTies", pct(cal.tie_share["skillsbench_released"], 0)); put("NSbModelSD", num(cal.sd_D["outcome_model_r3"], 2))
put("NSbModelTies", pct(cal.tie_share["outcome_model_r3"], 0))
vn = r("E9", "e9_variants_needed")
pos = vn[vn.lift > 1e-9]   # tolerance: one skill has lift 3e-18 (floating-point zero)
put("NSbPos", str(len(pos)))
put("NSbMedNAlpha", f"{np.median(pos.median_n_alpha_grid.replace(np.inf, 1e9)):.0f}")
put("NSbMedNBonf", f"{np.median(pos.median_n_bonf_grid.replace(np.inf, 1e9)):.0f}")
put("NSbMedNAlphaPrpl", f"{np.median(pos.median_n_alpha.replace(np.inf, 1e9)):.0f}")
put("NSbMedNBonfPrpl", f"{np.median(pos.median_n_bonf.replace(np.inf, 1e9)):.0f}")
put("NSbCertSix", pct(pos.p_cert_600_bonf_grid.mean(), 0)); put("NSbCertSixPrpl", pct(pos.p_cert_600_bonf.mean(), 0))
lb = r("E9", "e9_library_rules")
for _rule, _n in (("grid", ""), ("prpl", "Prpl")):
    _x = lb[lb.rule == _rule]
    put("NSbELond" + _n, f"{_x.elond_admitted.mean():.1f}")
    put("NSbELondLo" + _n, f"{_x.elond_admitted.quantile(0.05):.0f}"); put("NSbELondHi" + _n, f"{_x.elond_admitted.quantile(0.95):.0f}")
    put("NSbBonfBet" + _n, str(int(_x.bonferroni_admitted.iloc[0])))


# ------------------------------------------------ E10 fixed-n library procedures
f10 = r("E10", "e10_full_rules").set_index("rule")
for k, n in (("t_bh", "TBH"), ("t_by", "TBY"), ("t_bonferroni", "TBonf"), ("signflip_bh", "SfBH"), ("signflip_by", "SfBY"), ("signflip_bonferroni", "SfBonf")):
    put("NFn" + n, str(int(f10.admitted[k])))
s10 = r("E10", "e10_split_half_replication").set_index("rule")
for k, n in (("t_bh", "TBH"), ("t_by", "TBY"), ("t_bonferroni", "TBonf"), ("signflip_bh", "SfBH"), ("signflip_by", "SfBY"), ("signflip_bonferroni", "SfBonf")):
    put("NFnSplit" + n, f"{s10.admitted[k]:.1f}")

# ------------------------------------------------ E11 price theory
e11 = r("E11", "e11_kinf_and_cost")
put("NThHelpful", str(len(e11)))
put("NThDualGap", f"{(e11.kl_P_Qstar - e11.kinf_dual).abs().max():.0e}".replace("e+00", ""))
for c, n in (("n_oracle", "Oracle"), ("n_prpl", "Prpl"), ("n_agrapa", "Agrapa"), ("n_agrapa_c09", "AgrapaHi")):
    rr = e11[c] / e11.n_ideal
    put("NTh" + n + "Ratio", f"{rr.median():.2f}"); put("NTh" + n + "RatioMax", f"{rr.max():.2f}")
put("NThAllIn", pct((e11.lam_star > 1.999).mean(), 0))
ls = r("E11", "e11_lineage_sim")
med = e11.set_index("task").loc[ls.task.iloc[0]]          # the skill E11 used for the lineage checks
put("NThMedTask", ls.task.iloc[0].replace("-", "\\mbox{-}"))
put("NThMedKinf", f"{med.kinf_dual:.3f}"); put("NThMedIdeal", f"{med.n_ideal:.0f}")
z = ls[ls.schedule.str.startswith("zeta")].set_index("k")
for kk, n in ((2, "Two"), (5, "Five"), (10, "Ten"), (20, "Twenty")):
    put("NThExtra" + n + "Sim", f"{z.extra_vs_k1_sim[kk]:.1f}"); put("NThExtra" + n + "Th", f"{z.extra_vs_k1_theory[kk]:.1f}")
ep = r("E11", "e11_expected_price")
zz = ep[ep.schedule.str.startswith("zeta")].set_index("pi")
put("NThPriceOne", f"{zz.price_variants_median_skill['one_attempt']:.1f}")
put("NThPriceGeoThree", f"{zz.price_variants_median_skill['geometric mean 3']:.1f}")
put("NThFloorGeoThree", f"{zz.floor_variants_median_skill['geometric mean 3']:.1f}")
put("NThPriceZipf", f"{zz.price_variants_median_skill['zipf s=2']:.1f}")
put("NThFloorZipf", f"{zz.floor_variants_median_skill['zipf s=2']:.1f}")
gz = ep[(ep.pi == "zipf s=2")].set_index("schedule").price_variants_median_skill
put("NThGeoOnZipfLoss", f"{gz['geometric (1/2)^k'] - gz['zeta_1.6 (current)']:.1f}")


# ------------------------------------------------ E12 schedule choice
ma = r("E12", "e12_minimax_all").set_index("N")
put("NScLogTwenty", f"{ma.minimax_logN[20]:.2f}"); put("NScZetaWorstTwenty", f"{ma.worst_zeta16[20]:.2f}")
mm = r("E12", "e12_minimax_monotone").set_index("N")
put("NScMonoLoThousand", f"{mm.R_lower_BA[1000]:.4f}"); put("NScMonoHiThousand", f"{mm.R_upper_BA[1000]:.4f}")
assert mm.R_upper_BA[1000] - mm.R_lower_BA[1000] < 1e-3
put("NScLogHThousand", f"{mm.logHN[1000]:.2f}"); put("NScLogThousand", f"{mm.logN[1000]:.2f}")
lk = r("E12", "e12_learned_kl")
assert (lk.E_KL_sim <= lk.bound + 3 * lk.se).all()
g3 = lk[(lk.pi == "geometric mean 3")].set_index(["r_poisoned", "m_completed"])
put("NScKLZero", f"{g3.E_KL_sim[(0, 0)]:.2f}"); put("NScKLHundred", f"{g3.E_KL_sim[(0, 100)]:.3f}")
put("NScKLThousand", f"{g3.E_KL_sim[(0, 1000)]:.3f}")
put("NScKLPoisonHundred", f"{g3.E_KL_sim[(50, 100)]:.2f}"); put("NScBoundPoisonHundred", f"{g3.bound[(50, 100)]:.2f}")
lv = r("E12", "e12_learned_variants").set_index("m_completed")
put("NScVarHundredSim", f"{lv.extra_variants_sim[100]:.1f}"); put("NScVarHundredTh", f"{lv.KL_over_g[100]:.1f}")
put("NScVarFiveHundredSim", f"{lv.extra_variants_sim[500]:.2f}"); put("NScVarFiveHundredTh", f"{lv.KL_over_g[500]:.2f}")
put("NScVarZetaSim", f"{lv.extra_variants_sim[-1]:.1f}")
cl = r("E12", "e12_cost_vs_level")
put("NScSlope", f"{cl.slope_fit.iloc[0]:.1f}"); put("NScOneOverG", f"{cl.one_over_g.iloc[0]:.1f}")


# ------------------------------------------------ live pilot (pilot/ledger.jsonl, signed credential)
import json as _json
_led = [_json.loads(l) for l in (ROOT / "pilot" / "ledger.jsonl").read_text().splitlines()]
_cred = [e for e in _led if e["kind"] == "credential"][-1]["payload"]["body"]
_st = _cred["statistics"]
put("NPilotTasks", str(len(_cred["tasks_run"]))); put("NPilotScored", str(_st["N"]))
put("NPilotExcluded", str(len(_st["tasks_excluded_errors"]))); put("NPilotRuns", str(sum(_cred["n_runs"].values())))
put("NPilotWealth", f"{_st['final_wealth']:.1f}"); put("NPilotDbar", f"{_st['Dbar']:.2f}")
put("NPilotRound", str(_cred["beacon_round"]))

# ------------------------------------------------ write
out = ROOT / "paper" / "numbers.tex"
lines = ["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
         f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]
out.write_text("\n".join(lines) + "\n")

# ------------------------------------------------ tables
T = ROOT / "paper" / "tables"
T.mkdir(exist_ok=True)
GN = {"G1_fixed_t": "Fixed-$n$ paired $t$", "G2_peek_t": "$t$-test re-checked each variant",
      "G3_peek_ci": "95\\% CI re-checked every 10", "G4_sprt": "SPRT, discordant pairs", "G5_msprt": "mSPRT",
      "G6_bet_capm0": "Betting, cap $\\lambda\\le m_0$", "G6_bet_c050": "Betting, $c=1/2$",
      "G6_bet_c075": "Betting, $c=3/4$", "G6_bet_agrapa": "Betting, aGRAPA", "G6_bet_futm": "Betting, $c=1/2$, futility margin",
      "G7_grid": "Betting, grid (deployed)", "G7_grid_futm": "Betting, grid, futility margin"}
cols = [("boundary", "Boundary"), ("two_point_worst", "Worst case"), ("sd_theta_3.0", "$s_\\theta=3$"),
        ("rho_0", "$\\rho=0$"), ("r_3", "$r=3$"), ("margin_0.05", "$\\delta=0.05$"), ("stale_baseline_drift", "Stale baseline")]
rows = []
for g, gn in GN.items():
    vals = []
    for c, _ in cols:
        x = e1[(e1.cell == c) & (e1.gate == g)]
        vals.append(pct(x.admit_rate.iloc[0]) if len(x) else "--")
    emp = pct(e8n.admit_rate[g]) if g in e8n.index else "--"
    rows.append(gn + " & " + " & ".join(vals) + f" & {emp} \\\\")
hdr = " & ".join(h for _, h in cols)
(T / "t_false_admission.tex").write_text(
    "\\begin{tabular}{l" + "r" * (len(cols) + 1) + "}\n\\toprule\nGate & " + hdr + " & SWE null \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

MN = {"M1_per_exam": "Betting at $\\alpha$ per exam, no multiplicity control",
      "M2_ebh_stop_1_over_alpha": "Stop at $1/\\alpha$, then e-BH per window",
      "M3_ebh_stop_m_over_alpha": "Stop at $m/\\alpha$, then e-BH per window",
      "M4_ebh_run_to_nmax": "No early admission, e-BH per window",
      "M5_elond_infinite": "e-LOND, $\\gamma_t\\propto t^{-1.6}$",
      "M6_elond_horizon": "e-LOND, uniform $\\gamma_t$ over planned horizon"}
rows = [f"{MN[k]} & {pct(e3.fdr[k])} & {pct(e3.power[k], 0)} & {e3.mean_n[k]:.0f} \\\\" for k in MN]
(T / "t_library.tex").write_text("\\begin{tabular}{lrrr}\n\\toprule\nRule & FDR (\\%) & Power (\\%) & Mean variants \\\\\n\\midrule\n"
                                  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

SN = {"curated_like": "Curated-like", "selfgen_like": "Self-generated-like", "swe_like": "SWE-Skills-like", "webdev_like": "WebDev-like"}
GN5 = {"fixed_t_uncorrected": "Fixed-$n$ $t$", "peek_ci": "CI re-checked", "sprt_discordant": "SPRT",
       "grid_futility_margin": "Betting + margin", "grid_elond": "Betting + e-LOND"}
rows = []
for s, sn in SN.items():
    x = e5[e5.scenario == s].set_index("gate")
    for i, (g, gn) in enumerate(GN5.items()):
        rows.append(f"{sn if i == 0 else ''} & {gn} & {pct(x.admitted_share[g])} & {pct(x.false_admit_share_of_admitted[g], 0)} & "
                    f"{x.mean_runs_per_candidate[g]:.0f} & {x.runs_per_true_admission[g]:,.0f} \\\\".replace(",", "{,}"))
    rows.append("\\addlinespace")
(T / "t_scenarios.tex").write_text(
    "\\begin{tabular}{llrrrr}\n\\toprule\nScenario & Gate & Admitted (\\%) & False share (\\%) & Runs/candidate & Runs/true admission \\\\\n\\midrule\n"
    + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

KN = {"plain": "Plain table", "spaced": "Spaced characters", "split": "Split string literals", "base64": "Base64 blob",
      "hex": "Hex blob", "computed": "Generator code copied", "clean": "Clean skills (false-positive rate)"}
x = e6[(e6.table_share == 0.05) | (e6.kind == "clean")]
rows = []
for k, kn in KN.items():
    y = x[x.kind == k].set_index("detector").recall
    rows.append(f"{kn} & {pct(y['literal'], 0)} & {pct(y['normalized'], 0)} & {pct(y['decoded'], 0)} \\\\")
(T / "t_leakage.tex").write_text("\\begin{tabular}{lrrr}\n\\toprule\nSkill (table covers 5\\% of the space) & Literal & Normalized & Normalized + decoded \\\\\n\\midrule\n"
                                  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
rows = [f"{int(n)} & {row.transcript_mb:.0f} & {row.proctor_s:.2f} & {row.verify_s:.2f} & {row.ledger_bytes:.0f} \\\\" for n, row in e7.iterrows()]
(T / "t_verification.tex").write_text("\\begin{tabular}{rrrrr}\n\\toprule\nVariants $N$ & Transcripts (MB) & Proctor bookkeeping (s) & Verify (s) & Log bytes \\\\\n\\midrule\n"
                                       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print(f"{len(M)} macros, 5 tables from {RES.name}")

# ------------------------------------------------ E14 referee checks M5 to M7 (appended; rewrites numbers.tex)
nm0 = len(M)
e2s = r("E2", "e2_power_cost")
s2 = lambda cfg, g, lift, col: float(e2s[(e2s.config == cfg) & (e2s.gate == g) & np.isclose(e2s.true_lift, lift)][col].iloc[0])  # noqa: E731
L1v = v(P, "target_lift")
put("NRvMarginPowBet", pct(s2("margin_0.05", BET, L1v, "admit_rate"), 0))
put("NRvMarginPowFixed", pct(s2("margin_0.05", "G1_fixed_t", L1v, "admit_rate"), 0))
put("NRvMarginNFixed", f"{s2('margin_0.05', 'G1_fixed_t', L1v, 'mean_n'):.0f}")
put("NRvMarginNBet", f"{s2('margin_0.05', BET, L1v, 'mean_n'):.0f}")
put("NRvRThreeNBet", f"{s2('r_3', BET, L1v, 'mean_n'):.0f}")
put("NRvRThreeNFixed", f"{s2('r_3', 'G1_fixed_t', L1v, 'mean_n'):.0f}")
put("NRvRThreePowFixed", pct(s2("r_3", "G1_fixed_t", L1v, "admit_rate"), 0))
assert s2("margin_0.05", BET, L1v, "admit_rate") < s2("margin_0.05", "G1_fixed_t", L1v, "admit_rate")
assert s2("r_3", BET, L1v, "mean_n") > s2("r_3", "G1_fixed_t", L1v, "mean_n")

mp = r("E14", "e14_matched_power")
q = lambda lift, g, st, col: float(mp[np.isclose(mp.true_lift, lift) & (mp.gate == g) & np.isclose(mp.setting, st)][col].iloc[0])  # noqa: E731
put("NRvSprtPowLow", pct(q(L1v, "sprt", 0.20, "power"), 0)); put("NRvSprtNLow", f"{q(L1v, 'sprt', 0.20, 'mean_n'):.0f}")
put("NRvBetPowLow", pct(q(L1v, "bet_futm", 300, "power"), 0)); put("NRvBetNLow", f"{q(L1v, 'bet_futm', 300, 'mean_n'):.0f}")
put("NRvSprtPowHigh", pct(q(L1v, "sprt", 0.01, "power"), 0)); put("NRvSprtNHigh", f"{q(L1v, 'sprt', 0.01, 'mean_n'):.0f}")
put("NRvBetPowHigh", pct(q(L1v, "bet_futm", 600, "power"), 0)); put("NRvBetNHigh", f"{q(L1v, 'bet_futm', 600, 'mean_n'):.0f}")
put("NRvFixedPowHigh", pct(q(L1v, "fixed_t", 0.99, "power"), 0)); put("NRvFixedNHigh", f"{q(L1v, 'fixed_t', 0.99, 'mean_n'):.0f}")
put("NRvBetNMaxLow", "300")
assert abs(q(L1v, "sprt", 0.20, "power") - q(L1v, "bet_futm", 300, "power")) < 0.05
assert abs(q(L1v, "sprt", 0.01, "power") - q(L1v, "bet_futm", 600, "power")) < 0.01
assert q(L1v, "sprt", 0.20, "mean_n") < q(L1v, "bet_futm", 300, "mean_n")

mf = r("E14", "e14_moment_fit").iloc[0]
import ast  # noqa: E402
tg = ast.literal_eval(mf.target)
put("NRvFitSdTheta", num(mf.sd_theta, 2)); put("NRvFitSdLift", num(mf.sd_lift, 2)); put("NRvFitRho", num(mf.rho, 2))
put("NRvFitTie", pct(mf.tie, 0)); put("NRvFitSD", num(mf.sd_within, 2)); put("NRvFitBig", pct(mf.big, 0))
put("NRvObsTie", pct(tg["tie"], 0)); put("NRvObsSD", num(tg["sd_within"], 2)); put("NRvObsBig", pct(tg["big"], 0))
cs = r("E14", "e14_cost_sensitivity")
cs1 = cs[(cs.r == 1) & np.isclose(cs.true_lift, L1v)]
csc = cs[(cs.r == 1) & np.isclose(cs.true_lift, v(P, "curated_lift"))]
fit_ = cs1[cs1.model.str.startswith("fitted")]
put("NRvSensNBetLo", f"{cs1.mean_n_bet.min():.0f}"); put("NRvSensNBetHi", f"{cs1.mean_n_bet.max():.0f}")
put("NRvSensNFixedLo", f"{cs1.n_fixed.min():.0f}"); put("NRvSensNFixedHi", f"{cs1.n_fixed.max():.0f}")
put("NRvSensPowBetLo", pct(cs1.power_bet.min(), 0))
put("NRvSensNCurLo", f"{csc.mean_n_bet.min():.0f}"); put("NRvSensNCurHi", f"{csc.mean_n_bet.max():.0f}")
c3 = cs[(cs.r == 3) & np.isclose(cs.true_lift, L1v)].set_index("model")
put("NRvFitRThreeNBet", f"{c3.mean_n_bet['fitted_kappa_+0.0']:.0f}"); put("NRvDefRThreeNBet", f"{c3.mean_n_bet['default']:.0f}")
put("NRvFitRThreeNFixed", f"{c3.n_fixed['fitted_kappa_+0.0']:.0f}")
nul = cs[np.isclose(cs.true_lift, 0.0)]
put("NRvSensNullMax", pct(nul.power_bet.max(), 1))
put("NRvSensSaveLo", pct((1 - csc.mean_n_bet.values / csc.n_fixed.values).min(), 0))
put("NRvSensSaveHi", pct((1 - csc.mean_n_bet.values / csc.n_fixed.values).max(), 0))
assert nul.power_bet.max() < 0.05 and (nul.power_fixed <= 0.065).all()

lp = r("E14", "e14_library_procedure").set_index("procedure")
put("NRvLibELond", f"{lp.median_of_skill_medians['elond_zeta']:.0f}"); put("NRvLibELondCert", pct(lp.share_cert_600_pos["elond_zeta"], 0))
put("NRvLibHorizon", f"{lp.median_of_skill_medians['elond_horizon']:.0f}"); put("NRvLibHorizonCert", pct(lp.share_cert_600_pos["elond_horizon"], 0))
put("NRvLibBonf", f"{lp.median_of_skill_medians['bonferroni']:.0f}"); put("NRvLibPerExam", f"{lp.median_of_skill_medians['per_exam']:.0f}")
assert lp.median_of_skill_medians["per_exam"] < lp.median_of_skill_medians["elond_zeta"] <= lp.median_of_skill_medians["bonferroni"]
ss = r("E14", "e14_selection_split").groupby("resample_from")
put("NRvSelSame", f"{ss.median_of_medians.median()['same_half']:.0f}"); put("NRvSelHeld", f"{ss.median_of_medians.median()['held_out_half']:.0f}")
put("NRvSelSameFin", pct(ss.share_finite.mean()["same_half"], 0)); put("NRvSelHeldFin", pct(ss.share_finite.mean()["held_out_half"], 0))
put("NRvSelN", f"{ss.selected.mean()['same_half']:.0f}")
put("NSbPZero", pct(cal.p0["skillsbench_released"], 1))
assert ss.median_of_medians.median()["held_out_half"] > ss.median_of_medians.median()["same_half"]
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"E14 block: {len(M) - nm0} macros added, {len(M)} in all")

# ------------------------------------------------ E13 level-tuned grid (Theorem thm:adaptive)
nm1 = len(M)
ag = r("E13", "e13_per_skill")
aq = ag[ag.level == "alpha*q1"]
assert (ag.n_grid_tuned <= ag.bound_grid_tuned).all()
put("NAgGridQone", f"{aq.n_grid_tuned.median():.0f}"); put("NAgKellyQone", f"{aq.n_kelly.median():.0f}")
put("NAgAgrapaQone", f"{aq.n_agrapa.median():.0f}"); put("NAgPrplQone", f"{aq.n_prpl.median():.0f}")
put("NAgBoundQone", f"{aq.bound_grid_tuned.median():.0f}")
put("NAgGridBeatsAgrapa", pct((aq.n_grid_tuned < aq.n_agrapa).mean(), 0))
al = r("E13", "e13_levels")
assert (al.excess_grid_tuned <= al.excess_bound).all()
put("NAgExcessMin", num(al.excess_grid_tuned.min())); put("NAgExcessMax", num(al.excess_grid_tuned.max()))
put("NAgFixedMin", num(al.excess_grid_fixed_half.min())); put("NAgFixedMax", num(al.excess_grid_fixed_half.max()))
put("NAgKellyExMax", num(al.excess_kelly.max()))
va = r("E13", "e13_validity")
put("NAgMaxReject", pct(va.reject_rate_Qstar.max(), 1)); put("NAgMeanReject", pct(va.reject_rate_Qstar.mean(), 1))
put("NAgLevelQone", pct(va.a.iloc[0], 2))
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"E13 block: {len(M) - nm1} macros added, {len(M)} in all")

# ------------------------------------------------ universal-schedule normaliser c_u (Corollary cor:universal)
_k = np.arange(1, 10**6 + 1, dtype=float)
put("NCuValue", f"{1 / (np.sum(1 / ((_k + 1) * np.log(_k + 1) ** 2)) + 1 / np.log(10**6 + 1.5)):.3f}")
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"c_u block: {len(M)} macros in all")

# ------------------------------------------------ E12b learned schedule with overlapping lineages
ob = r("E12b", "e12b_overlap"); oc = r("E12b", "e12b_cumulative")
assert (oc.cum_KL_fixed_delay <= oc.bound_c).all()
u25 = ob[(ob.pi == "uniform") & (ob.lineage == 25)].iloc[0]
put("NObUnifQoneNaive", f"{u25.q1_naive:.3f}"); put("NObUnifQoneDelay", f"{u25.q1_fixed_delay:.3f}"); put("NObUnifPiOne", f"{u25.pi_1:.3f}")
inc = oc[(oc.pi == "increasing") & (oc.n == 400)].iloc[0]; geo = oc[(oc.pi == "geometric mean 3") & (oc.n == 400)].iloc[0]
put("NObIncNaive", num(inc.cum_KL_naive)); put("NObIncDelay", num(inc.cum_KL_fixed_delay))
put("NObGeoNaive", num(geo.cum_KL_naive)); put("NObGeoDelay", num(geo.cum_KL_fixed_delay)); put("NObBound", f"{geo.bound_c:.0f}")
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"E12b block: {len(M)} macros in all")

# ------------------------------------------------ E3b long library history
eb = r("E3b", "e3b_summary").set_index("method"); bb = r("E3b", "e3b_power_by_block")
lv = r("E3b", "e3b_levels_history0")
def _p(x): return f"{100 * x:.1f}"
put("NEbT", f"{len(lv)}")
for key, name in (("Inf", "elond_infinite"), ("Hor", "elond_horizon"), ("Per", "per_exam_alpha")):
    put(f"NEb{key}Fdr", f"{eb.loc[name, 'fdr']:.3f}"); put(f"NEb{key}Pow", _p(eb.loc[name, "power"]))
bi = bb[bb.method == "elond_infinite"].set_index("block").power
put("NEbInfPowFirst", _p(bi.iloc[0])); put("NEbInfPowLast", _p(bi.iloc[-1]))
put("NEbElondBinds", _p(float((lv.elond_level <= 0.05 * lv.lineage_first).mean())))
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"E3b block: {len(M)} macros in all")

# ------------------------------------------------ E15 static security screen
es = r("E15", "e15_summary").iloc[0]
import ast as _ast
rec = _ast.literal_eval(es.skills_by_recommendation)
put("NSecSkills", f"{int(es.n_skills)}"); put("NSecSafe", f"{rec['SAFE']}"); put("NSecCaution", f"{rec['CAUTION']}")
put("NSecDNI", f"{rec['DO_NOT_INSTALL']}"); put("NSecTasksFlag", f"{int(es.tasks_flagged_do_not_install)}")
put("NSecFlagHelpful", f"{int(es.tasks_flagged_and_helpful)}"); put("NSecTasksHelpful", f"{int(es.tasks_helpful)}")
put("NSecRho", f"{float(es.spearman_risk_vs_lift):.2f}"); put("NSecP", f"{float(es.spearman_p):.2f}")
adj = r("E15", "e15_flag_adjudication_v2")
put("NSecGenuine", f"{int(((adj.first_reading == 'genuine') & (adj.second_reading == 'genuine') & (adj.third_reading_openai == 'genuine')).sum())}")
put("NSecGenuineAny", f"{int(((adj.first_reading == 'genuine') | (adj.second_reading == 'genuine') | (adj.third_reading_openai == 'genuine')).sum())}")
put("NSecCatAgree", f"{int((adj.first_reading == adj.second_reading).sum())}")
put("NSecFlagN", f"{len(adj)}")
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"E15 block: {len(M)} macros in all")

# ------------------------------------------------ live study 2 (generated hidden variants), from the signed ledger entry
import json as _json
_led = [_json.loads(l) for l in (ROOT / "pilot" / "ledger.jsonl").read_text().splitlines()]
_c2 = [e for e in _led if e["kind"] == "study2_credential"][-1]["payload"]["body"]
_cal = [e for e in _led if e["kind"] == "study2_calibration"][-1]["payload"]
_ex = {e["skill"]: e for e in _c2["exams"]}
_W = {"placebo": "Plac", "leak": "Leak", "harmful": "Harm", "helpful": "Help"}
for sk, w in _W.items():
    e = _ex[sk]
    put(f"NLs{w}N", str(e["decision_pair"])); put(f"NLs{w}Run", str(e["pairs_run"])); put(f"NLs{w}Excl", str(e["pairs_excluded"]))
    put(f"NLs{w}With", str(e["pass_with"])); put(f"NLs{w}Without", str(e["pass_without"]))
    put(f"NLs{w}Lcb", f"{e['LCB_at_decision']:.3f}"); put(f"NLs{w}Level", f"{e['level']:.4f}")
put("NLsCost", f"{_c2['cost_usd']:.2f}"); put("NLsRound", str(_c2["beacon_round"]))
put("NLsPairs", str(sum(e["pairs_run"] for e in _c2["exams"])))
put("NLsCalCost", f"{_cal['cost_usd']:.2f}")
_nb = sum(e["pass_without"] for e in _c2["exams"]); _nn = sum(e["pairs_run"] - e["pairs_excluded"] for e in _c2["exams"])
put("NLsBasePct", f"{100 * _nb / _nn:.0f}")
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"Live study 2 block: {len(M)} macros in all")

# ------------------------------------------------ live study 1 (87 SkillsBench tasks, OpenHands), from the signed ledger entry
_c1 = [e for e in _led if e["kind"] == "study_credential"][-1]["payload"]["body"]
_s1 = _c1["statistics"]
put("NLoTasks", "87"); put("NLoScored", str(_s1["N"])); put("NLoExcluded", str(len(_s1["tasks_excluded_errors"]))); put("NLoMissing", str(len(_s1["tasks_missing"])))
put("NLoDbar", f"{_s1['Dbar']:.2f}"); put("NLoPos", str(_s1["n_positive"])); put("NLoNeg", str(_s1["n_negative"]))
put("NLoWealth", f"{_s1['final_wealth']:.2f}"); put("NLoLcb", f"{_s1['LCB_final']:.2f}")
put("NLoCost", f"{sum(_s1['cost_usd'].values()):.0f}"); put("NLoRound", str(_c1["beacon_round"]))
put("NLoRuns", str(sum(_c1["n_runs"].values())) if isinstance(_c1["n_runs"], dict) else str(_c1["n_runs"]))
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"Live study 1 block: {len(M)} macros in all")

# ------------------------------------------------ live study 3 (family B + resubmission lineage), from the signed ledger entry
_c3s = [e for e in _led if e["kind"] == "study3_credential"]
if _c3s:
    from nixigate import gates as _g
    _c3 = _c3s[-1]["payload"]["body"]
    _e = {(x["lineage"], x["attempt"]): x for x in _c3["exams"]}
    for key, w in ((("D", 1), "BPlac"), (("E", 1), "BHarm"), (("D", 2), "BHelp")):
        x = _e[key]
        put(f"NLs{w}N", str(x["decision_pair"])); put(f"NLs{w}With", str(x["pass_with"])); put(f"NLs{w}Without", str(x["pass_without"]))
        put(f"NLs{w}Run", str(x["pairs_run"])); put(f"NLs{w}Lcb", f"{x['LCB_at_decision']:.3f}"); put(f"NLs{w}Level", f"{x['level']:.4f}")
        put(f"NLs{w}Dec", x["decision"])
    # descriptive: anytime-valid upper bound on Delta for the harmful skill (lower bound on -D), at its decision
    _tr = [_json.loads(l) for l in (ROOT / "pilot2" / "transcripts3" / "exam2.jsonl").read_text().splitlines()]
    _Dh = [int(r["with_skill"]["pass"]) - int(r["no_skill"]["pass"]) for r in _tr if r["with_skill"]["ok"] and r["no_skill"]["ok"]]
    _nh = _e[("E", 1)]["decision_pair"]
    put("NLsBHarmUcb", f"{-float(_g.lcb_path(-np.array(_Dh[:_nh], float), 0.05, 0.5)[-1]):.3f}")
    _G = [_e[("G", k)] for k in range(1, 6)]
    put("NLsResubAdmit", str(sum(x["decision"] == "admit" for x in _G)))
    put("NLsResubAdmitPer", str(sum(x["per_exam_alpha_decision"] == "admit" for x in _G)))
    put("NLsResubNmin", str(min(x["decision_pair"] for x in _G))); put("NLsResubNmax", str(max(x["decision_pair"] for x in _G)))
    put("NLsResubPairs", str(sum(x["pairs_run"] for x in _G)))
    put("NLsResubLevelFirst", f"{_G[0]['level']:.4f}"); put("NLsResubLevelLast", f"{_G[-1]['level']:.5f}")
    put("NLsThreeCost", f"{_c3['cost_usd']:.2f}"); put("NLsThreeRound", str(_c3["beacon_round"]))
    _nb3 = sum(x["pass_without"] for x in _c3["exams"] if x["family"] == "B")
    _nn3 = sum(x["pairs_run"] - x["pairs_excluded"] for x in _c3["exams"] if x["family"] == "B")
    put("NLsBBasePct", f"{100 * _nb3 / _nn3:.0f}")
    out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                              f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
    print(f"Live study 3 block: {len(M)} macros in all")

# ------------------------------------------------ Nemotron arm (live studies 2N and 3N), from the signed ledger entries
_c2n = [e for e in _led if e["kind"] == "study2n_credential"]
_c3n = [e for e in _led if e["kind"] == "study3n_credential"]
if _c2n and _c3n:
    from nixigate import gates as _g
    _b2, _b3 = _c2n[-1]["payload"]["body"], _c3n[-1]["payload"]["body"]
    _th2 = [e for e in _led if e["kind"] == "study2n_commit"][-1]["payload"]["theta"]
    put("NLnModel", _th2["model"].replace("_", r"\_")); put("NLnMaxTok", f"{_th2['max_tokens']:,}".replace(",", "{,}"))
    _ex = {e["skill"]: e for e in _b2["exams"]}
    for sk, w in {"placebo": "Plac", "leak": "Leak", "harmful": "Harm", "helpful": "Help"}.items():
        e = _ex[sk]
        put(f"NLn{w}N", str(e["decision_pair"])); put(f"NLn{w}Run", str(e["pairs_run"])); put(f"NLn{w}Excl", str(e["pairs_excluded"]))
        put(f"NLn{w}With", str(e["pass_with"])); put(f"NLn{w}Without", str(e["pass_without"])); put(f"NLn{w}Level", f"{e['level']:.4f}")
        put(f"NLn{w}Dec", e["decision"])
    _e3 = {(x["lineage"], x["attempt"]): x for x in _b3["exams"]}
    for key, w in ((("D", 1), "BPlac"), (("E", 1), "BHarm"), (("D", 2), "BHelp")):
        x = _e3[key]
        put(f"NLn{w}N", str(x["decision_pair"])); put(f"NLn{w}With", str(x["pass_with"])); put(f"NLn{w}Without", str(x["pass_without"]))
        put(f"NLn{w}Run", str(x["pairs_run"])); put(f"NLn{w}Level", f"{x['level']:.4f}"); put(f"NLn{w}Dec", x["decision"])
    _G = [_e3[("G", k)] for k in range(1, 6)]
    put("NLnResubAdmit", str(sum(x["decision"] == "admit" for x in _G)))
    put("NLnResubAdmitPer", str(sum(x["per_exam_alpha_decision"] == "admit" for x in _G)))
    put("NLnResubNmin", str(min(x["decision_pair"] for x in _G))); put("NLnResubNmax", str(max(x["decision_pair"] for x in _G)))
    put("NLnResubLevelFirst", f"{_G[0]['level']:.4f}"); put("NLnResubLevelLast", f"{_G[-1]['level']:.5f}")
    # transcripts: discordant resubmission pairs, harmful-skill upper bound and sign test, cut-offs, failed calls
    _T2, _T3 = ROOT / "pilot2" / "transcripts_n2", ROOT / "pilot2" / "transcripts_n3"
    def _recs(p):
        return [_json.loads(l) for l in p.read_text().splitlines()]
    _disc = 0
    for k in range(4, 9):
        _disc += sum(r["with_skill"]["pass"] != r["no_skill"]["pass"] for r in _recs(_T3 / f"exam{k}.jsonl")
                     if r["with_skill"]["ok"] and r["no_skill"]["ok"])
    put("NLnResubDisc", str(_disc))
    _rh = [r for r in _recs(_T3 / "exam2.jsonl") if r["with_skill"]["ok"] and r["no_skill"]["ok"]]
    _Dh = [int(r["with_skill"]["pass"]) - int(r["no_skill"]["pass"]) for r in _rh]
    _nh = _e3[("E", 1)]["decision_pair"]
    put("NLnBHarmUcb", f"{-float(_g.lcb_path(-np.array(_Dh[:_nh], float), 0.05, 0.5)[-1]):.3f}")
    _neg, _pos = sum(d < 0 for d in _Dh), sum(d > 0 for d in _Dh)
    put("NLnBHarmNeg", str(_neg)); put("NLnBHarmDisc", str(_neg + _pos))
    put("NLnBHarmSignP", f"{0.5 ** (_neg + _pos) * sum(math.comb(_neg + _pos, i) for i in range(_neg, _neg + _pos + 1)):.4f}")
    _calls = [r[a] for p in (_T2, _T3) for f in sorted(p.glob("exam*.jsonl")) for r in _recs(f) for a in ("with_skill", "no_skill")]
    _cut = sum(c.get("stop_reason") in ("length", "max_tokens") for c in _calls)
    put("NLnCalls", f"{len(_calls):,}".replace(",", "{,}")); put("NLnCut", str(_cut)); put("NLnCutPct", f"{100 * _cut / len(_calls):.0f}")
    put("NLnFailed", str(sum(not c["ok"] for c in _calls)))
    put("NLnProviders", ", ".join(sorted({c["id"].split("|")[1] for c in _calls if c["ok"]})))
    put("NLnPairs", str(sum(e["pairs_run"] for e in _b2["exams"]) + sum(x["pairs_run"] for x in _b3["exams"])))
    put("NLnCost", f"{_b2['cost_usd']:.2f}"); put("NLnThreeCost", f"{_b3['cost_usd']:.2f}")
    put("NLnTotalCost", f"{_b2['cost_usd'] + _b3['cost_usd']:.2f}"); put("NLnRound", str(_b2["beacon_round"]))
    _nb = sum(e["pass_without"] for e in _b2["exams"]); _nn = sum(e["pairs_run"] - e["pairs_excluded"] for e in _b2["exams"])
    put("NLnBasePct", f"{100 * _nb / _nn:.0f}")
    _nb3 = sum(x["pass_without"] for x in _b3["exams"] if x["family"] == "B")
    _nn3 = sum(x["pairs_run"] - x["pairs_excluded"] for x in _b3["exams"] if x["family"] == "B")
    put("NLnBBasePct", f"{100 * _nb3 / _nn3:.0f}")
    out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                              f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
    print(f"Nemotron block: {len(M)} macros in all")

# ------------------------------------------------ E2b grid bet vs PrPl (referee item M2)
gb = r("E2b", "e2b_grid_vs_prpl")
def _gb(lift, bet, col):
    return float(gb[(gb.lift.round(4) == round(lift, 4)) & (gb.bet == bet)][col].iloc[0])
_l1, _lc = v(P, "target_lift"), v(P, "curated_lift")
put("NGbDesignGrid", f"{_gb(_l1, 'grid', 'mean_N'):.0f}"); put("NGbDesignPrpl", f"{_gb(_l1, 'prpl_c050', 'mean_N'):.0f}")
put("NGbDesignPowGrid", pct(_gb(_l1, 'grid', 'admit_rate'), 1)); put("NGbDesignPowPrpl", pct(_gb(_l1, 'prpl_c050', 'admit_rate'), 1))
put("NGbCurGrid", f"{_gb(_lc, 'grid', 'mean_N'):.0f}"); put("NGbCurPrpl", f"{_gb(_lc, 'prpl_c050', 'mean_N'):.0f}")
put("NGbCurSave", f"{100 * (1 - _gb(_lc, 'grid', 'mean_N') / _gb(_lc, 'prpl_c050', 'mean_N')):.0f}")
put("NGbNullGrid", pct(float(gb[(gb.lift <= 0) & (gb.bet == 'grid')].admit_rate.max()), 1))
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"E2b block: {len(M)} macros in all")

# ------------------------------------------------ E16 adaptive attempt index (Theorem thm:price robustness)
e16 = r("E16", "e16_adaptive_attempts")
_adm = e16[e16.case != "C_clairvoyant"]
put("NAdSkills", str(e16.task.nunique()))
put("NAdAdmissibleHold", "all" if (_adm.lower_holds.all() and _adm.upper_holds.all()) else "not all")
_cl = e16[e16.case == "C_clairvoyant"]
put("NAdClairBelowMax", f"{100 * (1 - (_cl.C / _cl.lower)).max():.0f}")
put("NAdClairViol", str(int((~_cl.lower_holds).sum())))
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"E16 block: {len(M)} macros in all")

# ------------------------------------------------ Revision B: Table 3 at tau, post hoc grid decisions, E20
sys.path.insert(0, str(ROOT / "tools"))
import live_counts as _lc  # noqa: E402
_TR = {"study2_credential": ("NLs", "transcripts", "A"), "study3_credential": ("NLs", "transcripts3", "B"),
       "study2n_credential": ("NLn", "transcripts_n2", "A"), "study3n_credential": ("NLn", "transcripts_n3", "B")}
_WA = {"placebo": "Plac", "leak": "Leak", "harmful": "Harm", "helpful": "Help"}
_WB = {("D", 1): "BPlac", ("E", 1): "BHarm", ("D", 2): "BHelp"}
for _kind, (_pre, _dir, _fam) in _TR.items():
    _cr = [e for e in _led if e["kind"] == _kind]
    if not _cr:
        continue
    for x in _cr[-1]["payload"]["body"]["exams"]:
        w = _WA[x["skill"]] if _fam == "A" else _WB.get((x["lineage"], x["attempt"]))
        if w is None:
            continue
        _p = ROOT / "pilot2" / _dir / f"exam{x['t']}.jsonl"
        c = _lc.counts(_p, x["decision_pair"])
        put(f"{_pre}{w}TauWith", str(c["with_tau"])); put(f"{_pre}{w}TauWithout", str(c["without_tau"]))
        put(f"{_pre}{w}ExtraWith", str(c["with_all"] - c["with_tau"])); put(f"{_pre}{w}ExtraWithout", str(c["without_all"] - c["without_tau"]))
        put(f"{_pre}{w}ExtraN", str(c["n_ok"] - x["decision_pair"]))
        gd, gn = _lc.decide(_p, x["level"], rule="grid")
        put(f"{_pre}{w}GridDec", gd); put(f"{_pre}{w}GridN", str(gn))
for _rule in ("grid", "prpl"):
    try:
        _e19 = r("E20", f"e20_library_{_rule}").set_index("method")
    except FileNotFoundError:
        continue
    for _m, _w in (("elond", "Elond"), ("stopped_ebh", "Sebh"), ("ebh_suprema", "Sup"), ("ebh_nmax", "Nmax")):
        _R = _rule.capitalize()
        put(f"NEbh{_R}{_w}Fdr", f"{100 * _e19.loc[_m, 'fdr']:.1f}")
        put(f"NEbh{_R}{_w}Pow", f"{100 * _e19.loc[_m, 'power']:.0f}")
        put(f"NEbh{_R}{_w}N", f"{_e19.loc[_m, 'mean_n']:.0f}")
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"Revision B block: {len(M)} macros in all")

# ------------------------------------------------ Revision B: grid-bet reruns of E3 and E3b (NIXI_RULE=grid)
try:
    _g3 = r("E3", "e3_window_fdr_grid").set_index("method")
    put("NEThreeGridPowELond", f"{100 * _g3.loc['M5_elond_infinite', 'power']:.0f}")
    put("NEThreeGridFdrELond", f"{100 * _g3.loc['M5_elond_infinite', 'fdr']:.2f}")
    put("NEThreeGridNELond", f"{_g3.loc['M5_elond_infinite', 'mean_n']:.0f}")
    _p3 = r0("E3", "e3_window_fdr").set_index("method")
    put("NEThreePrplPowELond", f"{100 * _p3.loc['M5_elond_infinite', 'power']:.0f}")
    put("NEThreePrplNELond", f"{_p3.loc['M5_elond_infinite', 'mean_n']:.0f}")
    _g3b = r("E3b", "e3b_summary_grid").set_index("method")
    put("NEbGridInfPow", f"{100 * _g3b.loc['elond_infinite', 'power']:.0f}")
    put("NEbGridInfFdr", f"{_g3b.loc['elond_infinite', 'fdr']:.4f}")
except FileNotFoundError:
    pass
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"Grid rerun block: {len(M)} macros in all")

# ------------------------------------------------ Live study 4 (real third-party skills), from signed ledger entries
_S4 = {"4": ("Four", "transcripts4", "transcripts_n4", "transcripts4/calibration.jsonl"),
       "4b": ("FourB", "transcripts4b", "transcripts_n4b", "transcripts4/calibration_cand1_newsroom.jsonl"),
       "4c": ("FourC", "transcripts4c", "transcripts_n4c", "transcripts4/calibration_cand2_commit.jsonl")}
for _t, (_tag, _dh, _dn, _cal) in _S4.items():
    _cf = ROOT / "pilot2" / _cal
    if _cf.exists():
        _cr = [_json.loads(l) for l in _cf.read_text().splitlines()]
        for _sk, _w in (("real", "Real"), ("placebo", "Plac")):
            _rs = [x for x in _cr if x["skill"] == _sk]
            put(f"NCal{_tag}{_w}With", str(sum(x["with_skill"]["pass"] for x in _rs)))
            put(f"NCal{_tag}{_w}Without", str(sum(x["no_skill"]["pass"] for x in _rs)))
        put(f"NCal{_tag}N", str(len([x for x in _cr if x["skill"] == "real"])))
        _nr = len([x for x in _cr if x["skill"] == "real"])
        put(f"NCal{_tag}Lift", f"{(int(M[f'NCal{_tag}RealWith']) - int(M[f'NCal{_tag}RealWithout'])) / _nr:.2f}")
    for _pre, _kind, _dir in (("NLs", f"study{_t}_credential", _dh), ("NLn", f"study{_t}n_credential", _dn)):
        _cr = [e for e in _led if e["kind"] == _kind]
        if not _cr:
            continue
        _b = _cr[-1]["payload"]["body"]
        put(f"{_pre}{_tag}Cost", f"{_b['cost_usd']:.2f}"); put(f"{_pre}{_tag}Round", str(_b["beacon_round"]))
        _cut = 0
        for x in _b["exams"]:
            _w = "Real" if x["skill"] == "real" else "Plac"
            _p = ROOT / "pilot2" / _dir / f"exam{x['t']}.jsonl"
            c = _lc.counts(_p, x["decision_pair"])
            put(f"{_pre}{_tag}{_w}Dec", {"nmax": "cap"}.get(x["decision"], x["decision"])); put(f"{_pre}{_tag}{_w}N", str(x["decision_pair"]))
            put(f"{_pre}{_tag}{_w}Level", f"{x['level']:.4f}")
            put(f"{_pre}{_tag}{_w}TauWith", str(c["with_tau"])); put(f"{_pre}{_tag}{_w}TauWithout", str(c["without_tau"]))
            put(f"{_pre}{_tag}{_w}ExtraWith", str(c["with_all"] - c["with_tau"]))
            put(f"{_pre}{_tag}{_w}ExtraWithout", str(c["without_all"] - c["without_tau"]))
            put(f"{_pre}{_tag}{_w}Excl", str(x["pairs_excluded"]))
            _all = [_json.loads(l) for l in _p.read_text().splitlines()]
            put(f"{_pre}{_tag}{_w}Run", str(len(_all)))
            put(f"{_pre}{_tag}{_w}FailWith", str(sum(not d["with_skill"]["ok"] for d in _all)))
            put(f"{_pre}{_tag}{_w}FailWithout", str(sum(not d["no_skill"]["ok"] for d in _all)))
            for l in _p.read_text().splitlines():
                d = _json.loads(l)
                _cut += sum(d[a].get("stop_reason") in ("length", "max_tokens") for a in ("with_skill", "no_skill"))
        put(f"{_pre}{_tag}Cut", str(_cut))
_cn = ROOT / "pilot2" / "transcripts_n4" / "calibration.jsonl"
if _cn.exists():
    _cr = [_json.loads(l) for l in _cn.read_text().splitlines()]
    for _sk, _w in (("real", "Real"), ("placebo", "Plac")):
        _rs = [x for x in _cr if x["skill"] == _sk]
        put(f"NCalNFour{_w}With", str(sum(x["with_skill"]["pass"] for x in _rs)))
        put(f"NCalNFour{_w}Without", str(sum(x["no_skill"]["pass"] for x in _rs)))
for _pre in ("NLs", "NLn"):
    if f"{_pre}FourRealN" in M:
        put(f"{_pre}FourRealLiftTau", f"{(int(M[f'{_pre}FourRealTauWith']) - int(M[f'{_pre}FourRealTauWithout'])) / int(M[f'{_pre}FourRealN']):.2f}")
_bc = [int(M[k]) for k in ("NLsFourBRealN", "NLnFourBRealN", "NLsFourCRealN", "NLnFourCRealN") if k in M]
if _bc:
    put("NLFourBCMinN", str(min(_bc))); put("NLFourBCMaxN", str(max(_bc)))
_pp = ROOT / "pilot2" / "power_plan_study4.json"
if _pp.exists():
    _pl = _json.loads(_pp.read_text())
    _c = [x for x in _pl["curve"] if x["n_max"] == _pl["n_max"] and x["exams"] == _pl["exams"]][0]
    put("NPowCons", f"{_c['power_conservative']:.2f}"); put("NPowNmax", str(_pl["n_max"]))
    put("NPowNoise", f"{_pl['calibration']['placebo_noise_per_side']:.2f}")
    _v1 = _json.loads((ROOT / "pilot2" / "power_plan_study4_v1_no_noise.json").read_text())
    _c1 = [x for x in _v1["curve"] if x["n_max"] == _v1["n_max"] and x["exams"] == _v1["exams"]][0]
    put("NPowConsVone", f"{_c1['power_conservative']:.2f}")
_all4 = [e["payload"]["body"]["cost_usd"] for e in _led if e["kind"] in
         ("study4_credential", "study4n_credential", "study4b_credential", "study4bn_credential", "study4c_credential", "study4cn_credential")]
put("NLFourTotalCost", f"{sum(_all4):.2f}")
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"Live study 4 block: {len(M)} macros in all")

# ------------------------------------------------ Study 4b placebo: per-rule pass counts (why a placebo was admitted)
for _pre, _d in (("NLs", "transcripts4b"), ("NLn", "transcripts_n4b")):
    _p = ROOT / "pilot2" / _d / "exam1.jsonl"
    if not _p.exists():
        continue
    _rs = [_json.loads(l) for l in _p.read_text().splitlines()]
    _rs = [x for x in _rs if x["with_skill"]["ok"] and x["no_skill"]["ok"]]
    for _k, _w in (("over", "Over"), ("money", "Money"), ("time", "Time")):
        _n = [x for x in _rs if _k in (x["no_skill"]["parsed"] or {})]
        put(f"{_pre}FourBPlac{_w}N", str(len(_n)))
        put(f"{_pre}FourBPlac{_w}With", str(sum(bool(x["with_skill"]["parsed"][_k]) for x in _n)))
        put(f"{_pre}FourBPlac{_w}Without", str(sum(bool(x["no_skill"]["parsed"][_k]) for x in _n)))
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"4b placebo block: {len(M)} macros in all")

# ------------------------------------------------ E24 intervals on headline numbers
_e24 = RES / "E24" / "e24_intervals.csv"
if _e24.exists():
    _iv = pd.read_csv(_e24).set_index("quantity")
    for _q, _fmt in (("NThPrplRatio", "{:.2f}"), ("NSbMedNAlpha", "{:.0f}"), ("NSbMedNBonf", "{:.0f}")):
        put(_q + "Lo", _fmt.format(_iv.lo[_q])); put(_q + "Hi", _fmt.format(_iv.hi[_q]))
    put("NIvEbhStopHi", pct(_iv.hi["E3:M2_ebh_stop_1_over_alpha"], 1))
    if "E3b_h600:elond_infinite" in _iv.index:
        put("NIvEbInfFdr", pct(_iv.estimate["E3b_h600:elond_infinite"], 2)); put("NIvEbInfFdrHi", pct(_iv.hi["E3b_h600:elond_infinite"], 2))
        put("NIvEbHorFdr", pct(_iv.estimate["E3b_h600:elond_horizon"], 1)); put("NIvEbHorFdrHi", pct(_iv.hi["E3b_h600:elond_horizon"], 1))
# ------------------------------------------------ E21 conditional-null stress tests
_e21 = RES / "E21" / "e21_null_stress_grid.csv"
if _e21.exists():
    _s = pd.read_csv(_e21)
    _g = _s[_s.gate == "grid"].set_index("cell").false_admit
    put("NNsControl", pct(_g["control"], 1))
    put("NNsMildMax", pct(max(_g[c] for c in ("snapshot_both", "repeat_0.01", "repeat_0.05", "exclude_0.01", "exclude_0.05", "block_5_0.5", "block_10_0.5")), 1))
    put("NNsSkillArm", pct(_g["snapshot_skill"], 1)); put("NNsBlock", pct(_g["block_10_1.0"], 1))
    put("NNsRepeat", pct(_g["repeat_0.20"], 1)); put("NNsExclude", pct(_g["exclude_0.20"], 1)); put("NNsCoin", pct(_g["coin_shift_0.05"], 1))
# ------------------------------------------------ E22 FDR unit
_e22 = RES / "E22" / "e22_summary.csv"
if _e22.exists():
    _u = pd.read_csv(_e22).iloc[0]
    put("NFuFdr", pct(_u.fdr_exam, 2)); put("NFuNullLin", pct(_u.null_lineage_error, 2))
    put("NFuHonFalse", pct(_u.honest_lineage_false_admit, 2)); put("NFuHonAdm", pct(_u.honest_helpful_admitted, 0))
    put("NFuAdm", f"{_u.mean_admissions:.1f}")
# ------------------------------------------------ E18 matched power
_e18 = RES / "E18" / "e18_matched_power.csv"
if _e18.exists():
    _m = pd.read_csv(_e18)
    _d = _m[(_m.model == "default") & (_m.r == 1)].set_index(["lift", "target", "gate"])
    for _l, _ln in ((0.1, "Des"), (0.166, "Cur")):
        for _t, _tn in ((0.8, "Eighty"), (0.95, "NinetyFive")):
            for _g, _gn in (("fixed_t", "Fixed"), ("grid", "Grid"), ("sprt", "Sprt"), ("prpl", "Prpl")):
                put(f"NMp{_ln}{_tn}{_gn}", f"{_d.variants_alt[(_l, _t, _g)]:.0f}")
                put(f"NMp{_ln}{_tn}{_gn}Null", f"{_d['variants_null_0.0'][(_l, _t, _g)]:.0f}")
    put("NMpNullMaxGrid", pct(_m[_m.gate == "grid"]["admit_null_0.0"].max(), 1))
# ------------------------------------------------ E19 total lineage cost (Sai's E19)
_e19c = RES / "E19" / "e19_lineage_cost.csv"
if _e19c.exists():
    _c = pd.read_csv(_e19c)
    _err = ((_c.variants_helpful - _c.predicted_helpful) / _c.predicted_helpful).abs()
    put("NLcPredErrMean", pct(_err.mean(), 1)); put("NLcPredErrMax", pct(_err.max(), 0))
    put("NLcBindMax", pct(_c.share_lineage_binds.max(), 2))
    put("NLcNullMax", pct(_c.null_lineage_admitted.max(), 1)); put("NLcHonFalseMax", pct(_c.honest_false_admit.max(), 2))
    _z = _c[(_c.delta_useless == 0) & (_c.age == 500) & (_c.schedule == "zeta1.6") & (_c.delta_helpful == 0.1)].set_index("pi")
    put("NLcGeoUseless", f"{_z.variants_useless['geometric_mean3']:.0f}"); put("NLcGeoHelp", f"{_z.variants_helpful['geometric_mean3']:.0f}")
    put("NLcUniUseless", f"{_z.variants_useless['uniform20']:.0f}"); put("NLcUniHelp", f"{_z.variants_helpful['uniform20']:.0f}")
    _a = _c[_c.delta_helpful == 0.1].groupby("age")
    put("NLcHelpAgeFifty", f"{_a.variants_helpful.mean()[50]:.0f}"); put("NLcHelpAgeTwoK", f"{_a.variants_helpful.mean()[2000]:.0f}")
    put("NLcAdmAgeFifty", pct(_a.honest_admitted.mean()[50], 0)); put("NLcAdmAgeTwoK", pct(_a.honest_admitted.mean()[2000], 0))
    _s = _c[(_c.delta_helpful == 0.1)].groupby("schedule").variants_helpful.mean()
    put("NLcSchedSpread", pct(_s.max() / _s.min() - 1, 1))
    _u = _c.groupby("delta_useless").variants_useless.mean()
    put("NLcUselessNeg", f"{_u[-0.05]:.0f}"); put("NLcUselessZero", f"{_u[0.0]:.0f}")
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")

# ------------------------------------------------ Live study 5 (E25): DeepMind mathematics_dataset, from signed ledger entries
_T5 = {"math_teacher": "Teach", "harmful": "Harm", "placebo": "Plac", "math_help": "Help", "math_skill": "Skill"}
_c5 = {}
for _pre, _kind, _dir in (("NLs", "study5_credential", "transcripts5"), ("NLn", "study5n_credential", "transcripts_n5")):
    _cr = [e for e in _led if e["kind"] == _kind]
    if not _cr:
        continue
    _b = _cr[-1]["payload"]["body"]
    _c5[_pre] = _b["cost_usd"]
    put(f"{_pre}FiveCost", f"{_b['cost_usd']:.2f}"); put(f"{_pre}FiveRound", str(_b["beacon_round"]))
    _cut = _calls = 0
    for x in _b["exams"]:
        _w = _T5[x["skill"]]
        put(f"{_pre}Five{_w}N", str(x["decision_pair"])); put(f"{_pre}Five{_w}Dec", {"nmax": "cap"}.get(x["decision"], x["decision"]))
        put(f"{_pre}Five{_w}Lift", f"{100 * x['Dbar_at_decision']:+.1f}".replace("-", "$-$"))
        put(f"{_pre}Five{_w}Level", f"{x['level']:.4f}")
        _rs = [_json.loads(l) for l in (ROOT / "pilot2" / _dir / f"exam{x['t']}.jsonl").read_text().splitlines()]
        for _a, _s in (("with_skill", "With"), ("no_skill", "Without")):
            _k = sum(d[_a].get("stop_reason") in ("length", "max_tokens") for d in _rs)
            put(f"{_pre}Five{_w}Cut{_s}", str(_k)); put(f"{_pre}Five{_w}CutPct{_s}", f"{100 * _k / len(_rs):.0f}")
            _cut += _k; _calls += len(_rs)
        put(f"{_pre}Five{_w}Run", str(len(_rs)))
        if x["skill"] == "harmful":
            put(f"{_pre}FiveHarmLcb", f"{100 * x['LCB_at_decision']:+.1f}".replace("-", "$-$"))
    put(f"{_pre}FiveCutPct", f"{100 * _cut / _calls:.0f}")
if _c5:
    put("NLFiveTotalCost", f"{sum(_c5.values()):.2f}")
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
print(f"Live study 5 block: {len(M)} macros in all")

# ------------------------------------------------ Study 5N harmful exam: how many variants its observed lift needs (bootstrap of its pairs)
_p = ROOT / "pilot2" / "transcripts_n5" / "exam2.jsonl"
if _p.exists():
    from nixigate import gates as _gates
    _rs = [_json.loads(l) for l in _p.read_text().splitlines()]
    _D = np.array([int(r["with_skill"]["pass"]) - int(r["no_skill"]["pass"]) for r in _rs
                   if r["with_skill"]["ok"] and r["no_skill"]["ok"]], float)
    _X = _D[np.random.default_rng(7).integers(0, len(_D), (2000, 6000))]
    _lv = [x for x in [e for e in _led if e["kind"] == "study5n_credential"][-1]["payload"]["body"]["exams"] if x["skill"] == "harmful"][0]["level"]
    _res = _gates.betting(_X, 0.0, _lv, 0.10, rule="grid", delta_f=0.05)
    _adm = np.where(_res.admit, _res.n, np.inf)
    put("NLnFiveHarmBootSix", f"{100 * (_adm <= 600).mean():.0f}")
    put("NLnFiveHarmBootMed", f"{np.median(_adm):,.0f}".replace(",", "{,}"))
    put("NLnFiveHarmBootTwoK", f"{100 * (_adm <= 2000).mean():.0f}")
    _disc = (_D != 0).mean()                                  # same discordance, a true lift of 4 points
    _pp, _pm = (_disc + 0.04) / 2, (_disc - 0.04) / 2
    _X4 = np.random.default_rng(8).choice([1.0, -1.0, 0.0], size=(2000, 6000), p=[_pp, _pm, 1 - _disc])
    _r4 = _gates.betting(_X4, 0.0, _lv, 0.10, rule="grid", delta_f=0.05)
    put("NLnFiveHarmFourTwoK", f"{100 * (np.where(_r4.admit, _r4.n, np.inf) <= 2000).mean():.0f}")
    put("NLnFiveHarmValid", str(len(_D)))
out.write_text("\n".join(["% Generated by tools/make_numbers.py from results/ and params.yaml. Do not edit by hand.",
                          f"% Source: {RES.name}/"] + [rf"\newcommand{{\{k}}}{{{v_}}}" for k, v_ in M.items()]) + "\n")
