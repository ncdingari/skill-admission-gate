"""Release fixes in the study-4 runner generator (final review, 18 Nov 2026): N_max truncation, no reveal
before the exams close, PrPl label on the bound field. Generated in memory; committed runners are untouched."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "pilot2"
sys.path.insert(0, str(ROOT))
import make_study4_runners as G  # noqa: E402


def _gen():
    R4 = [(o, n.replace("{MOD}", "family_s4_apa"), c) for o, n, c in G.R4]
    s = (ROOT / "run2.py").read_text()
    s = G.DOC4 + s[s.index('"""\n', 3) + 4:]
    s4 = G.apply(s, R4, "run2.py")
    return s4, G.apply(s4, G.R4N, "run4.py")


def test_release_runner_has_fixes_and_compiles():
    for src in _gen():
        compile(src, "gen", "exec")
        assert "D = list(D)[:NMAX]" in src
        assert 'refusing to reveal the committed secret' in src
        assert src.index('EXAMS_CLOSED").exists()') < src.index('rev = append(')
        assert '"LCB_at_decision_prpl_bound"' in src and '"LCB_at_decision":' not in src


def test_committed_runners_unchanged():
    assert "D = list(D)[:NMAX]" not in (ROOT / "run4.py").read_text()
