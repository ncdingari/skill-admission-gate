# Sequential admission tests for agent skills: code and data

Code, results and live-study records for the paper "Certifying That an Agent Skill Helps: Anytime-Valid, Resubmission-Proof and Verifiable Admission for Skill Registries" (JMLR submission). Released under the MIT
license for code (`LICENSE`) and CC BY 4.0 for our data and transcripts (`LICENSE-DATA`); third-party material keeps its own license (below).

## Install
    python3.12 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    export PYTHONPATH=src
    pytest -q tests                      # 79 tests, under a minute

## Verify the live studies (no API key needed)
Each committed study recomputes every grade and decision from the hash-chained ledger (`pilot/ledger.jsonl`),
the transcripts and the public drand beacon, checks the commitment was logged before the beacon round existed,
and checks the Ed25519 credential:

    for s in 2 2n 3 3n 4 4n 4b 4bn 4c 4cn 5 5n; do python pilot2/run$s.py verify; done

Study 5 regenerates DeepMind `mathematics_dataset` variants. Install it in a Python 3.11 env from
`pilot2/requirements_dm_frozen.txt` and point `dm_generate.py` at that interpreter (see `pilot2/E25_PLAN.md`).
Each study's quirks are in `pilot2/RELEASE_NOTES_study4.md` and `pilot2/RELEASE_NOTES_study5.md`.

## Reproduce the numbers
- `params.yaml` holds every simulation parameter, with its source.
- `experiments/eNN_*.py` writes `results/ENN/`. Set `NIXI_QUICK=1` for a smoke run and `NIXI_RULE=grid` for
  the deployed bet. Full runs take minutes to a few hours each.
- `python tools/make_numbers.py` writes `paper/numbers.tex`; every number in the paper's results and discussion
  comes from it.
- `python figures/make_figs.py` writes `figures/out/`.

Experiment map: E1 false admission; E2 power and cost; E3 and E3b library FDR; E4 winner's curse; E5 library
scenarios; E6 leakage; E7 credential cost; E8 and E9 benchmark reanalyses; E10 to E16 theory checks and
sensitivity; E18 matched power; E19 total lineage cost; E20 stopped e-BH; E21 conditional-null stress; E22 unit
of error; E23 wider fit grid; E24 intervals; E25 live study 5.

## Not included
- Raw SkillsBench job folders from the first live study (`pilot/jobs`, `pilot/jobs_study`, `pilot/jobs_study_aborted`;
  about 8 GB unpacked), released as a separate data archive, `p9-skillsbench-jobs.tar.gz` (link and SHA-256 in
  `DATA.md`). Unpack it in the repository root to restore those folders. The live-study verifiers do not need it.
- Proctor private keys and commitment secrets; the secrets are revealed in the ledger.

## Third-party material
- `data/external/`: released per-skill outcomes of SWE-Skills-Bench (MIT) and SkillsBench (AGPL-3.0, redistributed
  unmodified under that license), with source commits in `PROVENANCE.md`.
- `pilot2/skills_s5/`: three MIT-licensed skills at pinned commits (`MANIFEST.json`).
- DeepMind `mathematics_dataset` (Apache-2.0) is installed from PyPI, not vendored.
