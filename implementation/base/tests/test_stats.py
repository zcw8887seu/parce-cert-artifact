"""Statistical function tests (spec 7-P0): 6.2 k-table, Clopper-Pearson
known values, and event-log schema defect detection."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom

from parce_exp.schemas import EVENT_COLUMNS, check_event_log
from parce_exp.tolerance import InfeasibleSampleSize, alpha_row, required_order_statistic
from parce_exp.validate import clopper_pearson, verdict


def test_k_table_6_2(rec):
    # n=3000, alpha_row = 0.05/12: mandatory expected orders
    a = alpha_row(12, 0.05)
    got = {
        0.002: required_order_statistic(3000, 0.002, a),
        0.003: required_order_statistic(3000, 0.003, a),
        0.004: required_order_statistic(3000, 0.004, a),
    }
    expected = {0.002: 3000, 0.003: 2999, 0.004: 2997}
    assert got == expected, got
    # boundary condition holds at k but fails at k-1
    for eps, k in got.items():
        assert binom.sf(k - 1, 3000, 1 - eps) <= a
        assert binom.sf(k - 2, 3000, 1 - eps) > a
    # monotone in epsilon
    ks = [required_order_statistic(3000, e, a) for e in (0.002, 0.003, 0.004, 0.01)]
    assert all(ks[i] >= ks[i + 1] for i in range(3))
    try:
        required_order_statistic(10, 0.002, a)
        raise AssertionError("expected InfeasibleSampleSize")
    except InfeasibleSampleSize:
        pass
    rec("statistics_k_table", True, cases=3, k_table=expected)


def test_clopper_pearson(rec):
    max_err = 0.0
    # closed forms: m=0 upper = 1 - alpha^(1/n); m=n lower = alpha^(1/n)
    for n, alpha in ((3500, 0.01), (20, 0.05), (100, 0.05)):
        lo, hi = clopper_pearson(0, n, alpha)
        max_err = max(max_err, abs(hi - (1.0 - alpha ** (1.0 / n))))
        assert lo == 0.0
        lo, hi = clopper_pearson(n, n, alpha)
        max_err = max(max_err, abs(lo - alpha ** (1.0 / n)))
        assert hi == 1.0
    # independent numeric inversion of the binomial CDF
    cases = [(3, 3500, 0.01), (5, 3500, 0.01), (1, 20, 0.05), (10, 100, 0.05), (37, 3500, 0.01)]
    for m, n, alpha in cases:
        lo, hi = clopper_pearson(m, n, alpha)
        upper_root = brentq(lambda p: binom.cdf(m, n, p) - alpha, 1e-12, 1 - 1e-12)
        lower_root = brentq(lambda p: binom.sf(m - 1, n, p) - alpha, 1e-12, 1 - 1e-12)
        max_err = max(max_err, abs(hi - upper_root), abs(lo - lower_root))
    assert max_err < 1e-10, max_err
    # verdict mapping (spec 6.4)
    assert verdict(0.0, 0.002, 0.004) == "SUPPORTED"
    assert verdict(0.005, 0.006, 0.004) == "NOT_SUPPORTED"
    assert verdict(0.001, 0.005, 0.004) == "INCONCLUSIVE"
    rec("statistics_clopper_pearson", True, cases=len(cases) + 6, max_abs_err=max_err)


def _base_frame():
    rows = []
    for run, seqs in (("r1", [1, 2, 3]), ("r2", [1, 2])):
        for seq in seqs:
            cause = 1_000_000_000_000 + run_id_offset(run) + seq * 1_000_000
            rows.append(
                {
                    "experiment_id": "e01",
                    "run_id": run,
                    "session_id": "s1",
                    "split": "calibration",
                    "instance_id": seq,
                    "seq": seq,
                    "is_warmup": False,
                    "selected_single": seq == 1,
                    "phase_ns": 0,
                    "service_demand_ns": 50_000,
                    "t_cause_ns": cause,
                    "t_compute_end_ns": cause + 100_000,
                    "t_gate_ingress_ns": cause + 105_000,
                    "t_gate_target1_ns": cause + 400_000,
                    "t_gate_service_start_ns": cause + 400_000,
                    "t_gate_send_ns": cause + 450_000,
                    "t_rx_ns": cause + 460_000,
                    "target_cycle_id": 0,
                    "target_window_id": 0,
                    "service_cycle_id": 0,
                    "service_window_id": 0,
                    "reschedule_count": 0,
                    "packet_lost": False,
                    "out_of_order": False,
                    "duplicate": False,
                    "valid_event": True,
                    "invalid_reason": None,
                }
            )
    return pd.DataFrame(rows)


def run_id_offset(run):
    return 0 if run == "r1" else 10_000_000


def test_schema_detection(rec):
    clean = _base_frame()
    issues = check_event_log(clean)
    assert issues == [], issues

    defects = {}

    dup = _base_frame()
    dup = pd.concat([dup, dup.iloc[[1]]], ignore_index=True)
    defects["duplicate"] = ("DUPLICATE_KEY", dup)

    missing_col = _base_frame().drop(columns=["t_rx_ns"])
    defects["missing_column"] = ("MISSING_COLUMN", missing_col)

    missing_field = _base_frame()
    missing_field.loc[2, "t_gate_ingress_ns"] = np.nan
    defects["missing_field"] = ("MISSING_FIELD", missing_field)

    out_of_order = _base_frame()
    out_of_order.loc[1, "seq"] = 5  # breaks strict increase within r1
    defects["out_of_order"] = ("OUT_OF_ORDER", out_of_order)

    for stage, later, earlier in (
        ("X1", "t_compute_end_ns", "t_cause_ns"),
        ("X2", "t_gate_ingress_ns", "t_compute_end_ns"),
        ("X3", "t_gate_service_start_ns", "t_gate_target1_ns"),
    ):
        neg = _base_frame()
        neg.loc[0, later] = neg.loc[0, earlier] - 1
        defects[f"negative_{stage}"] = (f"NEGATIVE_STAGE_DURATION:{stage}", neg)

    early_send = _base_frame()
    early_send.loc[0, "t_gate_send_ns"] = early_send.loc[0, "t_gate_service_start_ns"] + 10_000
    defects["service_order"] = ("SERVICE_ORDER_VIOLATION", early_send)

    undetected = []
    for name, (marker, frame) in defects.items():
        got = " ".join(check_event_log(frame))
        if marker not in got:
            undetected.append((name, marker, got))
    rec(
        "statistics_schema",
        not undetected,
        cases=len(defects) + 1,
        defects=len(defects),
        undetected=len(undetected),
    )
    assert not undetected, undetected
