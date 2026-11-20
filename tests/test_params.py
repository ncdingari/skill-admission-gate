import pytest
from nixigate.params import MissingSourceError, load_params


def test_repo_params_load_and_all_sourced():
    P = load_params()
    assert len(P) > 30
    for name, p in P.items():
        assert p.source or p.design, name


def test_missing_source_rejected(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("a: {value: 1, unit: x, source: TODO}")
    with pytest.raises(MissingSourceError):
        load_params(f)
    f.write_text("a: {value: 1, unit: x}")
    with pytest.raises(MissingSourceError):
        load_params(f)
