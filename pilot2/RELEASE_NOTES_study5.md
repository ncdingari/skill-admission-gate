# Live study 5 (E25, external task family): notes for the code release

`run5.py` (Claude Haiku 5.5) and `run5n.py` (Nemotron 3.5 Super VL) are hashed in their commitments and are
released unchanged, so `python pilot2/run5.py verify` and `python pilot2/run5n.py verify` still pass.
`run5n.py` is generated from `run5.py` by `make_study5n_runner.py`: counted substitutions only.

Known quirks of the committed files (both runners derive from `run4.py`):

1. **Docstring and one message.** The docstring of `run5.py` was carried over from study 4. Its usage lines
   say `python pilot2/run4.py ...`, and it says "budget 15 USD" and "every graded AP slot correct". The values
   that apply are in the code and in the committed parameters: commands `run5.py` / `run5n.py`, budget 60 USD
   (Haiku) and 40 USD (Nemotron), and pass = `family_dm.grade`. In `run5.py` the refusal to commit twice
   prints "study 4 already committed".
2. **`LCB_at_decision`, N_max, reveal order.** Items 3 to 5 of `RELEASE_NOTES_study4.md` apply unchanged. In
   both studies `build` ran only after `run` had finished. No decision fell beyond N_max: the one exam that
   reached the cap (5N, exam 2) was decided at valid pair 600.
3. **Restarts.**
   - Haiku: the first `run` call stopped with HTTP 425 (beacon round not yet published) before any variant
     was drawn or any API call made.
   - Nemotron: the first `run` call failed on its first chunk because the folder `transcripts_n5/` did not
     exist. The Haiku study's calibration step had created its folder; the Nemotron study has no calibration.
     That chunk's responses were never written, nothing was logged to the ledger, and the rerun started
     exam 1 from pair 0 with the same committed seeds. The same failure occurred for studies 4bN and 4cN
     (`logs_4bn_run_attempt1_nodir.txt`).
4. **Pass counts in the 5N harmful exam.** The credential's `pass_with` / `pass_without` (294 / 259) count all
   606 rows, including the 4 excluded pairs (one with a with-skill pass). Over the 602 valid pairs the counts are
   293 / 259; the decision and `Dbar_at_decision` use the valid pairs only. The paper quotes neither count.
5. **Power plan.** `power_plan_study5.json` comes from the Haiku paired calibration
   (`transcripts5/calibration.jsonl`). The Nemotron study uses the same plan, as in studies 2N to 4N. Every
   skill's calibrated lift was negative, so the pre-registered rule set N_max to 600.
6. **Grader.** `family_dm.grade` is grader v2. The frozen v1 grader and the reason for the change are in
   `E25_PLAN.md`; the change was made on the no-skill calibration only, before any skill was run.

Third-party material: `skills_s5/` holds three MIT-licensed skills at pinned commits (see `MANIFEST.json`). The
task family is DeepMind `mathematics_dataset` 1.0.1 (Apache-2.0), installed from PyPI, not vendored.
