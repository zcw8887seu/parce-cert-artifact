"""Regression tests for statistical summaries and unit labels."""

import numpy as np
import pandas as pd
import pytest

from parce_exp.report import (
    cp_confidence_wording,
    format_runtime,
    h4_main_cells,
    h4_status_from_data,
    pointwise_family_rate_range,
)

DISTS = ["shifted_lognormal", "gamma", "lognormal_mixture"]
REGIMES = ["far", "near"]
PROCEDURES = ["bonferroni_simultaneous", "split_select_certify"]
POINTWISE = "pointwise_95_then_select"


def _selval(family_fail_rate_by_cell, reps=200, pointwise_rate=0.0):
    """Synthetic selection-validity frame; callables decide failures."""
    rows = []
    rng = np.random.default_rng(1234)
    for d in DISTS:
        for rg in REGIMES:
            for proc in PROCEDURES:
                rate = family_fail_rate_by_cell(d, rg, proc)
                fails = rng.random(reps) < rate
                for rep in range(reps):
                    rows.append(dict(
                        experiment_scope="main", distribution_id=d,
                        gate_regime=rg, procedure=proc, J=4, R=3, n=3000,
                        repetition=rep, family_failure=bool(fails[rep]),
                        selected_content_failure=False))
            fails = rng.random(reps) < pointwise_rate
            for rep in range(reps):
                rows.append(dict(
                    experiment_scope="main", distribution_id=d,
                    gate_regime=rg, procedure=POINTWISE, J=4, R=3, n=3000,
                    repetition=rep, family_failure=bool(fails[rep]),
                    selected_content_failure=False))
    return pd.DataFrame(rows)


def test_h4_supported_comes_from_data(rec):
    status, cells = h4_status_from_data(_selval(lambda *a: 0.0))
    assert status == "SUPPORTED"
    assert len(cells) == 12
    assert all(c["status"] == "SUPPORTED" for c in cells)
    assert all(c["failures"] == 0 for c in cells)
    rec("h4_dynamic_supported", True, cells=12)


def test_h4_not_supported_comes_from_data(rec):
    # force a high failure rate in every cell of one distribution
    status, cells = h4_status_from_data(_selval(
        lambda d, rg, p: 0.9 if d == "gamma" else 0.0))
    assert status == "NOT_SUPPORTED"
    gamma = [c for c in cells if c["d"] == "gamma"]
    assert len(gamma) == 4 and all(c["status"] == "NOT_SUPPORTED"
                                   for c in gamma)
    rec("h4_dynamic_not_supported", True, cells=len(gamma))


def test_h4_inconclusive_comes_from_data(rec):
    # a borderline cell with 11/200 failures has CP upper above 0.05 but
    # lower below it -> INCONCLUSIVE, and H4 follows dynamically
    def rate(d, rg, p):
        return 0.055 if (d, rg, p) == ("gamma", "far",
                                       "bonferroni_simultaneous") else 0.0
    status, cells = h4_status_from_data(_selval(rate))
    assert status == "INCONCLUSIVE"
    cell = next(c for c in cells if (c["d"], c["rg"], c["m"])
                == ("gamma", "far", "bonferroni_simultaneous"))
    assert cell["lo"] <= 0.05 < cell["hi"]
    rec("h4_dynamic_inconclusive", True)


def test_h4_ignores_pointwise_and_sensitivity_rows(rec):
    df = _selval(lambda d, rg, p: 0.0)
    extra = df.head(100).copy()
    extra["procedure"] = POINTWISE
    extra["family_failure"] = True
    sens = df.head(100).copy()
    sens["experiment_scope"] = "sensitivity"
    sens["J"] = 8
    sens["family_failure"] = True
    status, cells = h4_status_from_data(pd.concat(
        [df, extra, sens], ignore_index=True))
    assert status == "SUPPORTED"  # polluted rows must not count
    assert len(cells) == 12
    rec("h4_scope_filter", True)


def test_pointwise_range_computed_from_data(rec):
    df = _selval(lambda *a: 0.0, pointwise_rate=0.0)
    lo, hi, cells = pointwise_family_rate_range(df)
    assert cells == 6 and lo == 0.0 and hi == 0.0
    rng = np.random.default_rng(7)
    mask = df["procedure"] == POINTWISE
    rates = []
    for d in DISTS:
        for rg in REGIMES:
            sel = mask & (df["distribution_id"] == d) & (
                df["gate_regime"] == rg)
            r = float(rng.random() * 0.5)
            df.loc[sel, "family_failure"] = rng.random(int(sel.sum())) < r
            rates.append(df.loc[sel, "family_failure"].mean())
    lo, hi, cells = pointwise_family_rate_range(df)
    assert cells == 6
    assert lo == pytest.approx(min(rates))
    assert hi == pytest.approx(max(rates))
    rec("pointwise_range_from_data", True, cells=6)


def test_format_runtime_units(rec):
    assert format_runtime(500) == "500 ns"
    assert format_runtime(1_500.0) == "1.5 us"
    assert format_runtime(240_000.0) == "240 us"
    assert format_runtime(1_500_000.0) == "1.5 ms"
    assert format_runtime(2_000_000_000.0) == "2 s"
    assert "ns" not in format_runtime(1_500.0)  # never mislabels us as ns
    rec("runtime_units", True, cases=5)


def test_cp_wording_never_calls_99pct_95(rec):
    w = cp_confidence_wording(0.01)
    assert "99%" in w and "95% CP" not in w
    assert "alpha_row=0.01" in w
    w5 = cp_confidence_wording(0.05)
    assert w5 == "one-sided 95% CP"
    wodd = cp_confidence_wording(0.02)
    assert "98%" in wodd and "alpha_row=0.02" in wodd
    rec("cp_confidence_wording", True, cases=3)


def test_h4_main_cell_shape(rec):
    cells = h4_main_cells(_selval(lambda *a: 0.0))
    keys = {(c["d"], c["rg"], c["m"]) for c in cells}
    assert len(keys) == 12
    assert keys == {(d, rg, p) for d in DISTS for rg in REGIMES
                    for p in PROCEDURES}
    rec("h4_cell_coverage", True, cells=12)
