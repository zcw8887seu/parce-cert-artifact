"""Supplement e01-s1 regression tests (supplement.MD 6.4).

The split_select_certify procedure must: freeze the risk indices chosen on
selection data, draw an independent certification dataset of the same size,
re-compute only q at alpha_row = 0.05/J, and never let certification revisit
the selected indices. These tests verify each property directly on the
corrected simulation code, plus seed derivation and pairing acceptance.
"""

import json

import numpy as np
import pytest

from parce_exp.allocator import pareto_dp
from parce_exp.selection_sim import (
    ALL_METHODS,
    DISTRIBUTIONS,
    PAIRING_FIELDS,
    SIMULTANEOUS_METHOD,
    SPLIT_METHOD,
    STAGE_SCALES_NS,
    cell_id,
    certify_frozen,
    derive_seed,
    gates_for,
    h4_disposition,
    make_gate_ops,
    menu_from_samples,
    _order_stats,
    regime_phase,
    run_cell,
)
from parce_exp.tolerance import required_order_statistic

PERIOD = 1_000_000
WINDOWS = [(400_000, 700_000)]
DEMAND = 50_000

CELL = dict(
    dist_name="shifted_lognormal", regime="far", J=2, R=2, n=200, reps=5,
    risks=[0.10, 0.20], budget=0.50, demand=DEMAND, period=PERIOD,
    windows=WINDOWS, master_seed=20260817, scope="test",
    supplement_id="e01-s1-test",
)


def _phase():
    return regime_phase("shifted_lognormal", "far", PERIOD, WINDOWS, DEMAND)


def _rows(methods=None, **over):
    kw = dict(CELL)
    kw.update(over)
    kw["methods"] = methods or ALL_METHODS
    kw.setdefault("phase", _phase())
    kw.pop("reps", None)
    reps = over.get("reps", CELL["reps"])
    return run_cell(reps=reps, **kw)


def test_split_branch_entered_and_certified(rec):
    rows = _rows()
    split = [r for r in rows if r["procedure"] == SPLIT_METHOD]
    assert split, "split rows missing entirely"
    completed = [r for r in split if r["status"] == "COMPLETED"]
    assert completed, "no split row completed certification"
    assert all(r["certification_seed"] is not None for r in completed)
    assert all(r["selected_menu_ids"] for r in completed)
    rec("split_branch_entered", True, rows=len(split))


def test_selection_and_certification_seeds_differ(rec):
    cid = cell_id("test", CELL["dist_name"], CELL["regime"], CELL["J"],
                  CELL["R"], CELL["n"])
    collisions = 0
    for rep in range(50):
        s = derive_seed(CELL["master_seed"], cid, rep, "selection")
        c = derive_seed(CELL["master_seed"], cid, rep, "certification")
        if s == c:
            collisions += 1
    assert collisions == 0
    rec("seed_purposes_distinct", True, reps=50)


def test_derived_seeds_survive_int64_roundtrip(rec):
    seen = set()
    for scope in ("main", "sens", "test"):
        for rep in range(20):
            for purpose in ("selection", "certification"):
                cid = cell_id(scope, "gamma", "near", 4, 3, 3000)
                s = derive_seed(CELL["master_seed"], cid, rep, purpose)
                assert 0 <= s < 2**62  # fits losslessly in parquet int64
                assert s not in seen
                seen.add(s)
    rec("seed_collision_free", True, derived=len(seen))


def test_two_independent_datasets_of_size_n(rec):
    cid = cell_id("test", CELL["dist_name"], CELL["regime"], CELL["J"],
                  CELL["R"], CELL["n"])
    rep = 0
    n = CELL["n"]
    dists = [DISTRIBUTIONS[CELL["dist_name"]](s)
             for s in STAGE_SCALES_NS[:CELL["J"]]]
    sel = np.sort(np.stack([d.rvs(np.random.default_rng(
        derive_seed(CELL["master_seed"], cid, rep, "selection")), n)
        for d in dists]), axis=1)
    cert = np.sort(np.stack([d.rvs(np.random.default_rng(
        derive_seed(CELL["master_seed"], cid, rep, "certification")), n)
        for d in dists]), axis=1)
    assert sel.shape == cert.shape == (CELL["J"], n)
    assert not np.allclose(sel, cert), "certification == selection data"
    rows = _rows(reps=1)
    split = next(r for r in rows if r["procedure"] == SPLIT_METHOD)
    assert split["selection_n_units"] == n
    assert split["certification_n_units"] == n
    rec("two_datasets_size_n", True, n=n)


def test_certification_does_not_change_frozen_indices(rec):
    rows = _rows()
    phase = _phase()
    J, n = CELL["J"], CELL["n"]
    gate_after = gates_for(J)
    gate_ops = make_gate_ops(J, DEMAND, PERIOD, WINDOWS, phase)
    dists = [DISTRIBUTIONS[CELL["dist_name"]](s)
             for s in STAGE_SCALES_NS[:J]]
    checked = 0
    for row in (r for r in rows if r["procedure"] == SPLIT_METHOD
                and r["status"] == "COMPLETED"):
        cid = cell_id("test", CELL["dist_name"], CELL["regime"],
                      CELL["J"], CELL["R"], CELL["n"])
        rng = np.random.default_rng(
            derive_seed(CELL["master_seed"], cid, row["repetition"],
                        "selection"))
        sel_samples = np.sort(np.stack(
            [d.rvs(rng, n) for d in dists]), axis=1)
        menus = menu_from_samples(sel_samples, CELL["risks"],
                                  _order_stats(n, CELL["risks"], 0.05), J)
        dp_sel = pareto_dp(menus, gate_after, DEMAND, PERIOD, WINDOWS,
                           CELL["budget"], base_ns=phase, gate_ops=gate_ops)
        assert dp_sel is not None
        assert json.loads(row["selected_menu_ids"]) == list(dp_sel.menu_ids), \
            "certification must not revisit the selected risk indices"
        checked += 1
    assert checked >= 3
    rec("frozen_indices_preserved", True, rows=checked)


def test_certification_q_comes_from_certification_data(rec):
    phase = _phase()
    J, n = CELL["J"], CELL["n"]
    dists = [DISTRIBUTIONS[CELL["dist_name"]](s)
             for s in STAGE_SCALES_NS[:J]]
    gate_after = gates_for(J)
    gate_ops = make_gate_ops(J, DEMAND, PERIOD, WINDOWS, phase)
    rng = np.random.default_rng(7)
    sel_samples = np.sort(np.stack([d.rvs(rng, n) for d in dists]), axis=1)
    menus = menu_from_samples(sel_samples, CELL["risks"],
                              _order_stats(n, CELL["risks"], 0.05), J)
    sel = pareto_dp(menus, gate_after, DEMAND, PERIOD, WINDOWS,
                    CELL["budget"], base_ns=phase, gate_ops=gate_ops)
    cert_a = np.sort(np.stack([d.rvs(np.random.default_rng(11), n)
                               for d in dists]), axis=1)
    cert_b = np.sort(np.stack([d.rvs(np.random.default_rng(12), n)
                               for d in dists]), axis=1)
    menus_a, eps_a = certify_frozen(sel, cert_a, n, J)
    menus_b, eps_b = certify_frozen(sel, cert_b, n, J)
    assert eps_a == eps_b  # same frozen risks
    assert [m[0].menu_id for m in menus_a] == [m[0].menu_id for m in menus_b]
    qa = [m[0].q_ns for m in menus_a]
    qb = [m[0].q_ns for m in menus_b]
    assert qa != qb, "q must track the certification draw"
    # q equals the alpha_row = 0.05/J order statistic of the cert sample
    for j in range(J):
        eps = eps_a[j]
        k = required_order_statistic(n, eps, 0.05 / J)
        assert menus_a[j][0].q_ns == int(cert_a[j][k - 1])
    rec("certification_q_from_cert_data", True, stages=J)


def test_split_alpha_row_is_0_05_over_J(rec):
    # alpha_row = 0.05/J must be stricter than 0.05 for J > 1
    n, eps = 3000, 0.004
    for J in (2, 4, 8):
        k_strict = required_order_statistic(n, eps, 0.05 / J)
        k_loose = required_order_statistic(n, eps, 0.05)
        assert k_strict >= k_loose
    rec("split_alpha_row_stricter", True, levels=(2, 4, 8))


def test_sample_cost_units(rec):
    rows = _rows()
    n = CELL["n"]
    for r in rows:
        if r["procedure"] == SPLIT_METHOD:
            assert r["sample_cost_units"] == 2 * n
            assert r["certification_n_units"] == n
        else:
            assert r["sample_cost_units"] == n
            assert r["certification_n_units"] == 0
            assert r["certification_seed"] is None
    rec("sample_cost_units", True, methods=len(set(
        r["procedure"] for r in rows)))


def test_deterministic_replay(rec):
    a = _rows()
    b = _rows()
    ka = [(r["repetition"], r["procedure"], r["selection_seed"],
           r["selected_menu_ids"], r["bound_ns"]) for r in a]
    kb = [(r["repetition"], r["procedure"], r["selection_seed"],
           r["selected_menu_ids"], r["bound_ns"]) for r in b]
    assert ka == kb, "re-running a cell must reproduce identical rows"
    rec("deterministic_replay", True, rows=len(ka))


def test_split_not_rowwise_identical_to_simultaneous(rec):
    rows = _rows(reps=10)
    identical = 0
    cells = {}
    for r in rows:
        cells.setdefault(r["repetition"], {})[r["procedure"]] = r
    for rep, by in cells.items():
        s, b = by[SPLIT_METHOD], by[SIMULTANEOUS_METHOD]
        same = all(
            (s[f] == b[f]) or (
                s[f] is None and b[f] is None) for f in PAIRING_FIELDS)
        identical += int(same and s["status"] == b["status"] == "COMPLETED")
    assert identical < len(cells), (
        "split and simultaneous must not be row-wise identical everywhere")
    rec("split_pairs_differ", True, reps=len(cells), identical=identical)


def test_h4_disposition_logic(rec):
    assert h4_disposition(["SUPPORTED"] * 12) == "SUPPORTED"
    assert h4_disposition(["SUPPORTED"] * 11 + ["NOT_SUPPORTED"]) \
        == "NOT_SUPPORTED"
    assert h4_disposition(["SUPPORTED"] * 11 + ["INCONCLUSIVE"]) \
        == "INCONCLUSIVE"
    assert h4_disposition(["NOT_SUPPORTED", "INCONCLUSIVE"]) == "NOT_SUPPORTED"
    rec("h4_disposition_rules", True, cases=4)


def test_infeasible_rows_are_recorded_not_dropped(rec):
    # budget 0 forces NO_FEASIBLE_SELECTION rows to persist in the output
    rows = _rows(reps=2, budget=0.0, risks=[0.30])
    assert rows, "rows vanished on infeasibility"
    assert all(r["status"] == "INFEASIBLE" for r in rows)
    assert all(r["infeasible_reason"] for r in rows)
    assert all(r["family_failure"] is None for r in rows)
    rec("infeasible_rows_recorded", True, rows=len(rows))
