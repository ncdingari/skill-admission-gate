# Live study 5 (E25): externally built task family. Plan fixed 2026-11-20, before any model call on this family

**Why.** Reviewers (Sai's Plan 2, E25) object that every generated-variant family so far was designed by us. Chari decided on 2026-11-20 to run a small study on a family built by a major company.

## Family
- **Generator:** Google DeepMind `mathematics_dataset` (Apache-2.0, PyPI 1.0.1), used unmodified through `dm_generate.py` in a pinned venv (`requirements_dm_frozen.txt`, Python 3.11, sympy 1.4, numpy 1.26.4, PYTHONHASHSEED=0).
- **Variant:** a variant is (module, regime, seed). Only modules whose draws are independent of earlier draws are eligible: 35 training modules ("train_hard" = hardest third of the entropy range) and 14 extrapolation modules.
- **Task and grading:** see `family_dm.py`. Exact answer in `<answer>` tags, normalized string equality or exact-rational equality. Frozen before calibration.

## Selection rule (fixed now)
1. **Calibration step 1** (`e25_calibrate.py`): 16 public-seed variants per eligible module and regime, no-skill arm only, claude-haiku-5-5. Not scored, logged.
2. **Eligible family members:** modules with a no-skill pass rate in [0.30, 0.60].
   - Pick up to three, those closest to 0.45 (ties broken alphabetically), at most one per top-level area (algebra, arithmetic, comparison, measurement, numbers, polynomials, probability).
   - If fewer than two qualify, widen the band to [0.20, 0.70] with the same rule.
3. **The family:** the uniform mixture over the chosen (module, regime) pairs. A variant's module is chosen by its seed.

## Skills (fixed before calibration step 2)
- **Published third-party math skills, used byte for byte:** at least three, from public repositories not written for this family. Each licence and commit hash is verified and recorded before use.
  - Candidates: Wholiver/Math.Skill (MIT), jamesrochabrun/skills `math-teacher` (MIT), parcadei/Continuous-Claude-v3 `math` (MIT).
- **Placebo:** general study-skills advice of matched length, no mathematical method.
- **Harmful variant:** matched length, confidently wrong instructions for the chosen modules (for example, sampling with replacement for without-replacement probability questions).

## Design (as study 4)
- **Calibration step 2:** 20 public-seed paired variants per skill, to set N_max by the same power simulation as study 4.
- **Gate:**
  - grid bet;
  - δ = 0, futility δ_f = 0.05, α_f = 0.10;
  - e-LOND with lineage spending γ_j ∝ j^-1.6;
  - both arms concurrent;
  - new registry history;
  - each published skill as its own lineage; placebo and harmful as their own lineages.
- **Models:** claude-haiku-5-5, then the same commitment structure with open-weight Nemotron 3.5 Super VL (as in studies 2n–4n).
- **Commitment:** generator version, module list, skills (SHA-256), α, δ, δ_f, N_max, the bet, the exclusion rule (API failure after retries in either arm) and analysis code, all hashed before a drand round.

## Budget
Ask Chari only if the estimated total exceeds US$300. Expected: under US$40 for both models.

## Calibration step 1 result and one deviation (2026-11-20)

**Calibration run:** 784 no-skill calls, US$0.20, no API failures (`transcripts5/calibration_noskill.jsonl`).

**Deviation: grader v1 → v2.** This was decided after step 1 and before any skill was run, so it cannot favour a skill. Grader v1 failed mathematically correct answers in two ways:
- surds with terms in another order (`arithmetic__simplify_surd`: every one of its 9 failures was correct);
- multiple-choice answers giving the right letter with its value (`comparison__kth_biggest*`).

Under v1 those modules looked hard only because of answer formatting, which no math skill should be credited for fixing. Grader v2 (`family_dm.grade`, v1 kept as `grade_v1`) accepts SymPy-equal surd expressions and the leading letter for multiple choice. The step-1 transcripts were re-graded offline (`calibration_noskill_regraded_v2.json`), with no new calls.

**Selection (rule above, applied to v2 rates):** two modules qualify in [0.30, 0.60], from different areas, so the band was not widened.
- `numbers__is_prime`, train_hard: 7/16 = 0.44. Failures are wrong primality calls.
- `arithmetic__div_big`, extrapolate: 9/16 = 0.56. Failures are decimal approximations where the exact fraction is asked.

**Family:** each variant's module is chosen by its seed, with probability 1/2 each.
