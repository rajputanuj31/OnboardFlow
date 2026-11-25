from __future__ import annotations

import pytest

from onboardflow_evals.metrics import (
    bootstrap_mean,
    hit_at_k,
    paired_difference,
    reciprocal_rank,
)
from onboardflow_evals.paths import extract_paths, first_hit_rank, same_file


def test_bootstrap_mean_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.0, 1.0] * 50
    a, b = bootstrap_mean(values), bootstrap_mean(values)
    assert a == b
    assert a.mean == pytest.approx(0.5)
    assert a.low < 0.5 < a.high
    assert a.n == 100


def test_empty_estimate_formats_as_na() -> None:
    assert bootstrap_mean([]).fmt() == "n/a"


def test_paired_difference_detects_consistent_improvement() -> None:
    a = [50.0, 60.0, 70.0, 80.0] * 10
    b = [x + 5 for x in a]
    diff = paired_difference(a, b)
    assert diff.mean == pytest.approx(5.0)
    assert diff.low == pytest.approx(5.0)  # zero variance in the differences


def test_paired_difference_requires_same_length() -> None:
    with pytest.raises(ValueError, match="same items"):
        paired_difference([1.0], [1.0, 2.0])


def test_rank_metrics() -> None:
    assert hit_at_k(3, 5) == 1.0
    assert hit_at_k(6, 5) == 0.0
    assert hit_at_k(None, 10) == 0.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def test_extract_paths_in_order_without_noise() -> None:
    text = (
        "The fix belongs in `requests/sessions.py` (see Session.request), and maybe "
        "requests/adapters.py. Version 2.31.0 is affected, e.g. on Python 3.12. "
        "Also check requests/sessions.py again and ./src/requests/utils.py."
    )
    assert extract_paths(text) == [
        "requests/sessions.py",
        "requests/adapters.py",
        "src/requests/utils.py",
    ]


@pytest.mark.parametrize(
    ("predicted", "gold", "expected"),
    [
        ("src/requests/models.py", "requests/models.py", True),  # repo layout moved to src/
        ("requests/models.py", "src/requests/models.py", True),
        ("flask/app.py", "src/flask/app.py", True),
        ("models.py", "requests/models.py", False),  # generic basename alone is not enough
        ("blueprints.py", "src/flask/blueprints.py", True),  # distinctive basename is
        ("other/models.py", "requests/models.py", False),
        ("a/requests/api.py", "requests/api.py", True),  # diff prefix stripped
    ],
)
def test_same_file(predicted: str, gold: str, expected: bool) -> None:
    assert same_file(predicted, gold) is expected


def test_first_hit_rank() -> None:
    predicted = ["docs/index.rst", "src/flask/helpers.py", "src/flask/app.py"]
    assert first_hit_rank(predicted, ["flask/app.py", "flask/helpers.py"]) == 2
    assert first_hit_rank(predicted, ["flask/cli.py"]) is None
