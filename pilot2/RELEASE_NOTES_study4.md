# Live study 4 (published skills): notes for the code release

The committed runners `run4.py`, `run4n.py`, `run4b.py`, `run4bn.py`, `run4c.py` and `run4cn.py` are hashed in
their commitments and are released unchanged, so that `python pilot2/run4*.py verify` still passes. Known quirks
of these committed files:

1. **Labels.** The descriptive label in the committed parameters of `run4bn.py` and `run4cn.py` reads
   "P9 live study 4bbN" and "4ccN" (a substitution error in the generator). Ledger kinds (`study4bn_`,
   `study4cn_`), seeds and transcript folders are correct.
2. **Docstrings.** All six runners carry the docstring written for the AP-style family ("every graded AP slot
   correct"). The family actually used is the one imported at the top of each runner (`family_s4_apa`,
   `family_s4` or `family_s4_commit`).
3. **`LCB_at_decision`.** This credential field is computed with the PrPl bet (`gates.lcb_path`), while the
   decision used the grid bet. It is not the bound behind the decision, and on admissions it can be negative.
   The paper quotes it once, for the study 5N harmful exam, and labels it there as the PrPl bound.
4. **N_max.** `decide()` runs the bet on all valid pairs, which can exceed `N_max` by up to one batch (the 4N
   placebo exam had 603 valid pairs for N_max 600). No decision fell beyond N_max.
5. **Reveal order.** `build()` does not check that the exams are closed before it logs the reveal. In study 4N
   the reveal was logged before the exams ran (paper, Appendix B.2, deviation iii). `verify` does not test the
   order of reveal and exams.

The generator `make_study4_runners.py` now emits runners with these fixes (tests/test_v_release_runner.py):
`decide()` truncates at N_max; `run` writes `EXAMS_CLOSED` when the exams end or the budget stops them, and
`build()` refuses to reveal before it exists; the bound field is renamed `LCB_at_decision_prpl_bound`.

Excluded pairs: only study 4N excluded pairs, all for HTTP 429 rate-limit errors from the pinned provider
(paper, Appendix B.2).
