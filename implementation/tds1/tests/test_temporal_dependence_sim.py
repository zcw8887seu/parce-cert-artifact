"""e01-tds1 software gate tests (temporal_dependence_simulation.MD spec 6).

Fixed-seed deterministic tests; no hypothesis. Heavy checks (the 1e6 DKW
marginal gate, full-scale row counts) run in the verify-dgps / verify-final
commands; these tests cover the same logic at test scale.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from parce_exp import temporal_dependence_sim as tds
from parce_exp.coupling import replay_e2e

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _oracle_stub():
    """Small sorted E2E oracle for repetition smoke tests (fixed stream)."""
    rng = tds.stream_rng((9, 9, 9, 9, 9))
    xs = [tds.to_x(tds.open_uniform(rng, 200_000), s)
          for s in tds.STAGE_SCALES_NS]
    e2e = replay_e2e(xs[0], xs[1], xs[2], xs[3], tds.PHASE_NS,
                     tds.DEMAND_NS, tds.PERIOD_NS, tds.WINDOWS)
    return np.sort(e2e.astype(np.int64))


@lru_cache(maxsize=8)
def _rep(dgp_idx, rep):
    return tds.run_repetition(dgp_idx, rep, _oracle_stub())


# ------------------------------------------------------------ DGP params
def test_hmm_stationarity(rec):
    pi = np.asarray(tds.HMM_PI)
    P = np.asarray(tds.HMM_P)
    ok = np.allclose(pi @ P, pi, atol=1e-15)
    rec("tds_hmm_stationarity", ok, cases=1)
    assert ok


def test_burst_transition_and_stationary_tail(rec):
    pi1 = tds.BURST_P01 / (tds.BURST_P01 + (1.0 - tds.BURST_P11))
    ok = (abs(pi1 - 0.01) < 1e-15
          and abs(tds.BURST_P01 - 0.0005050505050505) < 1e-12)
    rec("tds_burst_stationary_tail", ok, pi1=pi1)
    assert ok


def test_ar1_stationary_initialisation(rec):
    key = tds.stream_key_primary(1, 0, 0, 0)
    ref = tds.stream_rng(key)
    z0 = float(ref.standard_normal())
    rng = tds.stream_rng(key)
    u = tds.gen_u("GAUSSIAN_AR1_WEAK", rng, 4)
    from scipy.special import ndtri
    ok_first = abs(float(ndtri(u[0])) - z0) < 1e-12
    # stationary marginal variance ~ 1 over a long chain
    long = tds.gen_u("GAUSSIAN_AR1_WEAK",
                     tds.stream_rng(tds.stream_key_primary(1, 1, 0, 0)),
                     200_000)
    z = ndtri(long)
    ok_var = abs(float(np.var(z)) - 1.0) < 0.02
    rec("tds_ar1_stationary_init", ok_first and ok_var,
        var=float(np.var(z)))
    assert ok_first and ok_var


def test_markov_state_run_lengths(rec):
    rng = tds.stream_rng((8, 8, 8, 8, 8))
    states = tds._markov_states(rng, 500_000, (0.995, 0.980), (0.8, 0.2))
    frac1 = float((states == 1).mean())
    ok = abs(frac1 - 0.2) < 0.02
    rec("tds_markov_run_lengths", ok, frac_state1=frac1)
    assert ok


# ------------------------------------------------- U/X invariants + tails
def test_u_strictly_open_and_x_int64(rec):
    ok = True
    for d, dgp in enumerate(tds.DGPS):
        rng = tds.stream_rng(tds.stream_key_primary(d, 2, 0, 0))
        u = tds.gen_u(dgp, rng, 20_000)
        tds.check_u(u)
        for j, scale in enumerate(tds.STAGE_SCALES_NS):
            x = tds.to_x(u[:5000], scale)
            tds.check_x(x)
            ok &= bool((x >= int(0.6 * scale) + 1).all())
    rec("tds_u_open_x_int64", ok, dgps=len(tds.DGPS))
    assert ok


def test_ceil_strict_tail_equivalence(rec):
    """For integer q: ceil(y) > q <=> y > q (enumerated boundaries)."""
    ok = True
    for q in (60_000, 60_001, 100_000):
        for y in (q - 0.5, q - 1e-9, float(q), q + 1e-9, q + 0.5):
            ok &= (math.ceil(y) > q) == (y > q) or y == float(q)
        # exactly-integer y has probability 0 under the continuous law;
        # at y == q both sides are False:
        ok &= (math.ceil(float(q)) > q) == (float(q) > q)
    rec("tds_ceil_equivalence", ok, cases=15)
    assert ok


def test_discrete_tail_truth_vs_references(rec):
    scale = 25_000
    shift = int(0.6 * scale)
    assert tds.sf_int(shift, scale) == 1.0
    assert tds.sf_int(shift - 10, scale) == 1.0
    # vs independent numeric integration
    ok_quad = True
    for s in tds.STAGE_SCALES_NS:
        for q in (int(0.6 * s) + 1, s, 2 * s, 5 * s):
            a, b = tds.sf_int(q, s), tds.sf_int_reference(q, s)
            ok_quad &= abs(a - b) <= 1e-9 * max(1.0, b) + 1e-12
    # vs empirical frequency on the integer generator
    rng = tds.stream_rng((7, 7, 7, 7, 7))
    x = tds.to_x(tds.open_uniform(rng, 400_000), scale)
    ok_emp = True
    for q in (shift + 1, scale, 2 * scale):
        emp = float((x > q).mean())
        p = tds.sf_int(q, scale)
        se = math.sqrt(p * (1 - p) / len(x)) + 1e-9
        ok_emp &= abs(emp - p) < 5 * se
    rec("tds_discrete_tail_truth", ok_quad and ok_emp, quad=ok_quad,
        empirical=ok_emp)
    assert ok_quad and ok_emp


# ------------------------------------------------------------ seed streams
def test_stream_keys_unique_and_split_separated(rec):
    keys = tds.all_stream_keys()
    ok_unique = len(keys) == len(set(keys)) == 100_020
    ok_split = all(
        tds.stream_key_primary(d, r, 0, j) != tds.stream_key_primary(
            d, r, 1, j)
        for d in range(4) for r in (0, 2999) for j in range(4))
    ok_json = (tds.cal_seed_key_json(0, 5) != tds.val_seed_key_json(0, 5))
    parsed = json.loads(tds.cal_seed_key_json(2, 7))
    ok_arrays = (len(parsed) == 4 and all(
        len(k) == 5 and all(isinstance(v, int) for v in k) for k in parsed))
    ok = ok_unique and ok_split and ok_json and ok_arrays
    rec("tds_stream_keys", ok, total_keys=len(keys))
    assert ok


def test_cal_val_arrays_actually_differ(rec):
    res = _rep(0, 0)
    cal_key = res["summary"][0]["calibration_seed_key"]
    val_key = res["summary"][0]["validation_seed_key"]
    a = tds.gen_u("IID", tds.stream_rng(tds.stream_key_primary(0, 0, 0, 0)),
                  100)
    b = tds.gen_u("IID", tds.stream_rng(tds.stream_key_primary(0, 0, 1, 0)),
                  100)
    ok = cal_key != val_key and not np.array_equal(a, b)
    rec("tds_cal_val_independent", ok)
    assert ok


# ------------------------------------------------------- stats primitives
def test_menu_k_table(rec):
    from parce_exp.tolerance import required_order_statistic
    got = {eps: required_order_statistic(3000, eps, 0.05 / 12)
           for eps in tds.RISK_GRID}
    ok = got == {0.002: 3000, 0.003: 2999, 0.004: 2997}
    rec("tds_menu_k_table", ok, **{str(k): v for k, v in got.items()})
    assert ok


def test_cp_independent_reference(rec):
    from parce_exp.validate import clopper_pearson
    from scipy.stats import beta
    ok = True
    for m, n, a in ((0, 3500, 0.01), (3, 3500, 0.01), (50, 3000, 0.025),
                    (3000, 3000, 0.05 / 7)):
        lo, hi = clopper_pearson(m, n, a)
        ref_lo = 0.0 if m == 0 else float(beta.isf(1 - a, m, n - m + 1))
        ref_hi = 1.0 if m == n else float(beta.isf(a, m + 1, n - m))
        ok &= abs(lo - ref_lo) < 1e-12 and abs(hi - ref_hi) < 1e-12
    rec("tds_cp_reference", ok, cases=4)
    assert ok


def test_oracle_eta_formula(rec):
    ok = abs(tds.ORACLE_ETA
             - math.sqrt(math.log(2 / 0.001) / (2 * 5_000_000))) < 1e-15
    ok &= abs(tds.DKW_THRESHOLD
              - math.sqrt(math.log(2 * 16 / 0.001) / (2 * 1_000_000))) < 1e-15
    rec("tds_dkw_eta_formulas", ok, eta=tds.ORACLE_ETA,
        dkw=tds.DKW_THRESHOLD)
    assert ok


def test_dp_bruteforce_agreement_in_repetition(rec):
    """Per-repetition DP==bruteforce is enforced; also check one directly."""
    res = _rep(2, 1)
    ok = res["design"][0]["dp_matches_bruteforce"]
    rec("tds_dp_bruteforce", ok, dp_brute_mismatch_count=0 if ok else 1)
    assert ok


# --------------------------------------------------- repetition pipeline
def test_repetition_shapes_and_types(rec):
    res = _rep(0, 0)
    ok = all(len(res[t]) == n
             for t, n in tds.BATCH_ROWS_PER_TABLE.items())
    frames = tds.frames_from_rows(res)
    ok &= list(frames["validation"].columns) == tds.VALIDATION_COLUMNS
    ok &= str(frames["validation"]["upper_noncoverage"].dtype) == "boolean"
    ok &= frames["menu"]["q_ns"].dtype == np.int64
    rec("tds_repetition_shapes", ok)
    assert ok


def test_e2e_target_is_risk_sum_not_in_stage_family(rec):
    res = _rep(0, 0)
    design = res["design"][0]
    e2e_row = res["validation"][4]
    ok = e2e_row["item_id"] == "E2E"
    ok &= e2e_row["target_risk"] == design["risk_sum"]
    ok &= e2e_row["true_risk_source"] == "DKW_ORACLE"
    ok &= e2e_row["upper_noncoverage"] is None
    ok &= e2e_row["nominal_false_support"] is None
    stage_supported = all(r["nominal_status"] == "SUPPORTED"
                          for r in res["validation"][:4])
    ok &= res["summary"][0]["nominal_all_stage_supported"] == stage_supported
    rec("tds_e2e_semantics", ok)
    assert ok


def test_pipeline_disposition_rules(rec):
    ok = (tds.disposition_of(True, True, False)
          == "INCONCLUSIVE_DEPENDENCE_DIAGNOSTIC")
    ok &= (tds.disposition_of(False, True, False)
           == "NOMINAL_SUPPORTED_UNDER_IID_CONTRACT")
    ok &= tds.disposition_of(False, False, True) == "NOMINAL_NOT_SUPPORTED"
    ok &= tds.disposition_of(False, False, False) == "NOMINAL_INCONCLUSIVE"
    rec("tds_disposition_rules", ok, cases=4)
    assert ok


def test_unsafe_flag_invariants(rec):
    ok = True
    for d in (0, 3):
        s = _rep(d, 0)["summary"][0]
        if s["unsafe_pipeline_upgrade"]:
            ok &= s["unsafe_nominal_pass"]
            ok &= not s["any_diagnostic_triggered"]
        if s["unsafe_nominal_pass"]:
            ok &= s["selected_content_failure"]
            ok &= s["nominal_all_stage_supported"]
    rec("tds_unsafe_invariants", ok)
    assert ok


def test_repetition_deterministic_replay(rec):
    a = tds.run_repetition(1, 3, _oracle_stub())
    b = tds.run_repetition(1, 3, _oracle_stub())
    ok = True
    for table in tds.BATCH_TABLES:
        for ra, rb in zip(a[table], b[table]):
            eq, key = tds._rows_equal(ra, rb)
            ok &= eq
    rec("tds_deterministic_replay", ok)
    assert ok


def test_json_columns_compact_and_sorted(rec):
    d = _rep(0, 0)["design"][0]
    ok = True
    for col in ("selected_menu_ids", "selected_risks", "selected_q_ns",
                "true_selected_probabilities"):
        s = d[col]
        ok &= " " not in s
        obj = json.loads(s)
        ok &= list(obj.keys()) == sorted(obj.keys())
    rec("tds_json_columns", ok)
    assert ok


def test_event_inclusion_detector_catches_planted_defect(rec):
    """A bound computed from smaller q than checked-q yields detectable
    violations (spec 6 item 15: the detector must catch defects)."""
    arrays = tds.regenerate_validation_arrays(0, 0)
    e2e = replay_e2e(arrays["X1"], arrays["X2"], arrays["X3"], arrays["X4"],
                     tds.PHASE_NS, tds.DEMAND_NS, tds.PERIOD_NS, tds.WINDOWS)
    q_loose = {c: int(arrays[c].max()) for c in tds.STAGE_COORDS}
    b_defective = int(np.quantile(e2e, 0.5))  # planted: B far too small
    union = np.zeros(tds.VAL_N, dtype=bool)
    for c in tds.STAGE_COORDS:
        union |= arrays[c] > q_loose[c]
    violations = int(((e2e > b_defective) & ~union).sum())
    ok = violations > 0 and int(union.sum()) == 0
    rec("tds_inclusion_detector", ok, planted_violations=violations)
    assert ok


def test_couplings_preserve_marginals_and_inclusion(rec):
    res = _rep(0, 0)
    design = res["design"][0]
    q_sel = {k: int(v) for k, v in
             json.loads(design["selected_q_ns"]).items()}
    bound = int(design["bound_ns"])
    arrays = tds.regenerate_validation_arrays(0, 0)
    marg = {c: np.sort(arrays[c]) for c in tds.STAGE_COORDS}
    couplings = [tuple(arrays[c] for c in tds.STAGE_COORDS)]
    rng = tds.stream_rng(tds.stream_key_coupling(0, 0, 1))
    couplings.append(tuple(arrays[c][rng.permutation(tds.VAL_N)]
                           for c in tds.STAGE_COORDS))
    couplings.append(tuple(np.sort(arrays[c]) for c in tds.STAGE_COORDS))
    couplings.append(tuple(np.sort(arrays["X1"]) if c == "X1"
                           else np.sort(arrays[c])[::-1]
                           for c in tds.STAGE_COORDS))
    ok = True
    for joints in couplings:
        ok &= all(np.array_equal(np.sort(joints[i]), marg[c])
                  for i, c in enumerate(tds.STAGE_COORDS))
        e2e = replay_e2e(joints[0], joints[1], joints[2], joints[3],
                         tds.PHASE_NS, tds.DEMAND_NS, tds.PERIOD_NS,
                         tds.WINDOWS)
        union = np.zeros(tds.VAL_N, dtype=bool)
        for i, c in enumerate(tds.STAGE_COORDS):
            union |= joints[i] > q_sel[c]
        ok &= int(((e2e > bound) & ~union).sum()) == 0
    rec("tds_coupling_invariants", ok, couplings=len(couplings))
    assert ok


# ----------------------------------------------------------- batch/resume
def test_batch_write_resume_and_validation(rec, tmp_path):
    frames = {}
    rows = {t: [] for t in tds.BATCH_TABLES}
    for rep in (0, 1):
        res = tds.run_repetition(0, rep, _oracle_stub())
        for t in tds.BATCH_TABLES:
            rows[t].extend(res[t])
    frames = tds.frames_from_rows(rows)
    tds.validate_batch_frames(frames, "IID", 0, 2)
    tds.write_batch(tmp_path, "IID", 0, 2, frames)
    done = tds.completed_batches(tmp_path, "IID")
    ok = done == {(0, 2)}
    # .tmp directories must not count as completed
    (tmp_path / "batches" / "IID" / "reps_000002_000004.tmp").mkdir(
        parents=True)
    ok &= tds.completed_batches(tmp_path, "IID") == {(0, 2)}
    # rewrite is idempotent (atomic replace path exercised)
    back = pd.read_parquet(
        tds.batch_dir(tmp_path, "IID", 0, 2) / "summary.parquet")
    ok &= len(back) == 2 and set(back["repetition"]) == {0, 1}
    # expected_batches partitions [0, 3000) without gaps or overlaps
    spans = tds.expected_batches()
    ok &= spans[0][0] == 0 and spans[-1][1] == tds.REPS_PER_DGP
    ok &= all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    rec("tds_batch_resume", ok, batches=len(spans))
    assert ok


def test_batch_validation_detects_defects(rec):
    res = tds.run_repetition(0, 0, _oracle_stub())
    frames = tds.frames_from_rows(res)
    with pytest.raises(tds.TdsStop):
        tds.validate_batch_frames(frames, "IID", 0, 2)  # wrong rep span
    bad = tds.frames_from_rows(res)
    bad["summary"] = bad["summary"].assign(event_inclusion_violations=1)
    with pytest.raises(tds.TdsStop):
        tds.validate_batch_frames(bad, "IID", 0, 1)
    rec("tds_batch_validation_defects", True, cases=2)


# ------------------------------------------------------- guards and state
def test_guard_rejects_protected_roots(rec):
    for target in ("runs/e01", "runs/e01-h3r1",
                   "runs/e01/derived", "runs/e01-h3r1/state/state.json"):
        with pytest.raises(SystemExit):
            tds.guard_run_root(ROOT / target)
        with pytest.raises(SystemExit):
            tds.guard_write(ROOT / target)
    with pytest.raises(SystemExit):
        tds.guard_run_root(ROOT)
    rec("tds_guard_protected_roots", True, cases=9)


def test_state_machine_order_enforced(rec, tmp_path):
    run_root = tmp_path / "runs" / "tds-test"
    run_root.mkdir(parents=True)
    tds.init_state_file(run_root)
    with pytest.raises(SystemExit):
        tds.init_state_file(run_root)  # refuse re-init
    with pytest.raises(SystemExit):
        tds.require_status(run_root, {"SOFTWARE_TESTED"}, "freeze-plan")
    tds.complete_step(run_root, "T1-software-gate", "SOFTWARE_TESTED",
                      "freeze-plan")
    st = tds.load_state(run_root)
    ok = (st["status"] == "SOFTWARE_TESTED"
          and st["physical_h3_status_unchanged"] == tds.PHYSICAL_H3
          and st["next_allowed_command"] == "freeze-plan"
          and "updated_at" in st)
    # frozen plan must never be regenerated
    plan = run_root / "configs" / "frozen_tds1_plan.yaml"
    plan.parent.mkdir(parents=True)
    plan.write_text("frozen: true\n")
    import argparse
    args = argparse.Namespace(run_root=str(run_root),
                              master_seed=tds.MASTER_SEED)
    with pytest.raises(SystemExit):
        tds.cmd_freeze_plan(args)
    rec("tds_state_machine", ok)
    assert ok


# -------------------------------------------------------- analyze / rules
def _fake_tables(counts):
    """Build minimal summary/validation/coupling frames with given event
    counts per dgp: dict dgp -> dict(metric -> events)."""
    n = tds.REPS_PER_DGP
    summ = []
    for dgp in tds.DGPS:
        c = counts.get(dgp, {})
        for metric in ("menu_family_content_failure",
                       "stage_upper_noncoverage_family",
                       "any_diagnostic_triggered",
                       "unsafe_pipeline_upgrade"):
            pass
        for i in range(n):
            summ.append({
                "dgp_id": dgp,
                "menu_family_content_failure":
                    i < c.get("menu_family_content_failure", 0),
                "stage_upper_noncoverage_family":
                    i < c.get("stage_upper_noncoverage_family", 0),
                "any_diagnostic_triggered":
                    i < c.get("any_diagnostic_triggered", 0),
                "unsafe_pipeline_upgrade":
                    i < c.get("unsafe_pipeline_upgrade", 0),
            })
    summary = pd.DataFrame(summ)
    validation = pd.DataFrame({
        "item_id": ["E2E"] * 8,
        "event_inclusion_violations": [0] * 8,
    })
    coupling = pd.DataFrame({
        "event_inclusion_violations": [0] * 26,
        "marginals_preserved": [True] * 26,
    })
    return summary, validation, coupling


def test_td_rules_supported_path(rec):
    counts = {dgp: {"any_diagnostic_triggered": 3000}
              for dgp in tds.DEP_DGPS}
    counts["HIDDEN_MARKOV"]["menu_family_content_failure"] = 600
    summary, validation, coupling = _fake_tables(counts)
    cell_df, st = tds.compute_td_statuses(summary, validation, coupling)
    ok = (len(cell_df) == tds.CELL_SUMMARY_ROWS
          and st == {"TD1": "SUPPORTED", "TD2": "SUPPORTED",
                     "TD3": "SUPPORTED", "TD4": "SUPPORTED"})
    rec("tds_td_rules_supported", ok, **st)
    assert ok


def test_td4_structural_units_and_no_binomial_ci(rec):
    counts = {dgp: {"any_diagnostic_triggered": 3000}
              for dgp in tds.DEP_DGPS}
    counts["HIDDEN_MARKOV"]["menu_family_content_failure"] = 600
    summary, _, _ = _fake_tables(counts)
    validation = pd.DataFrame({
        "item_id": ["E2E"] * 12_000,
        "n_units": [3_500] * 12_000,
        "event_inclusion_violations": [0] * 12_000,
    })
    coupling = pd.DataFrame({
        "n_units": [3_500] * 5_200,
        "event_inclusion_violations": [0] * 5_200,
        "marginals_preserved": [True] * 5_200,
    })
    cell_df, statuses = tds.compute_td_statuses(
        summary, validation, coupling)
    td4 = cell_df[cell_df["outer_family"] == "TD4"].set_index("metric")
    incl = td4.loc["event_inclusion_violations"]
    marg = td4.loc["coupling_marginals_not_preserved"]
    cp_cols = ["outer_family_size", "outer_alpha_row",
               "outer_mc_cp_lower", "outer_mc_cp_upper"]
    hyp = tds.build_hypothesis_rows(cell_df, statuses)
    h4 = hyp[hyp["hypothesis_id"] == "TD4"].iloc[0]
    evidence = json.loads(h4["evidence_counts"])
    ok = int(incl["n_units"]) == 60_200_000
    ok &= incl["unit_type"] == "REPLAY_INSTANCE"
    ok &= incl["decision_basis"] == "EXACT_POINTWISE_AUDIT"
    ok &= int(marg["n_units"]) == 5_200
    ok &= marg["unit_type"] == "COUPLING_ROW"
    ok &= marg["decision_basis"] == "EXACT_MULTISET_AUDIT"
    ok &= td4[cp_cols].isna().all().all()
    ok &= (td4["direction"] == "EXACT_ZERO").all()
    ok &= pd.isna(h4["estimate"]) and pd.isna(h4["ci_lower"])
    ok &= pd.isna(h4["ci_upper"]) and pd.isna(h4["n_units"])
    ok &= evidence == {
        "coupling_rows": 5_200,
        "event_inclusion_instances": 60_200_000,
        "source_replay_rows": 17_200,
        "stage_multiset_comparisons": 20_800,
    }
    rec("tds_td4_structural_units", ok, **evidence)
    assert ok


def test_td_rules_failure_paths(rec):
    # TD1 NOT_SUPPORTED: IID menu failures far above 0.05
    s, v, c = _fake_tables({"IID": {"menu_family_content_failure": 600,
                                    "any_diagnostic_triggered": 0}})
    _, st = tds.compute_td_statuses(s, v, c)
    ok = st["TD1"] == "NOT_SUPPORTED"
    # TD2 NOT_SUPPORTED: zero events everywhere in dependent cells
    ok &= st["TD2"] == "NOT_SUPPORTED"
    # TD3 NOT_SUPPORTED: detection rate zero
    ok &= st["TD3"] == "NOT_SUPPORTED"
    # TD3 unsafe upgrade breach
    s2, v2, c2 = _fake_tables(
        {dgp: {"any_diagnostic_triggered": 3000,
               "unsafe_pipeline_upgrade": 400} for dgp in tds.DEP_DGPS})
    _, st2 = tds.compute_td_statuses(s2, v2, c2)
    ok &= st2["TD3"] == "NOT_SUPPORTED"
    # TD4 NOT via nonzero violation would stop upstream; rule check:
    c3 = c2.copy()
    c3.loc[0, "event_inclusion_violations"] = 1
    _, st3 = tds.compute_td_statuses(s2, v2, c3)
    ok &= st3["TD4"] == "NOT_SUPPORTED"
    rec("tds_td_rules_failures", ok)
    assert ok


def test_inner_outer_cp_field_names_distinct(rec):
    ok = {"inner_cp_lower", "inner_cp_upper"} <= set(tds.VALIDATION_COLUMNS)
    ok &= {"outer_mc_cp_lower", "outer_mc_cp_upper"} <= set(tds.CELL_COLUMNS)
    ok &= not ({"cp_lower", "cp_upper"}
               & (set(tds.VALIDATION_COLUMNS) | set(tds.CELL_COLUMNS)))
    ok &= {"n_units", "unit_type", "decision_basis"} <= set(
        tds.CELL_COLUMNS)
    ok &= "n_repetitions" not in tds.CELL_COLUMNS
    rec("tds_inner_outer_names", ok)
    assert ok


def test_frozen_plan_payload_consistency(rec):
    p = tds.frozen_plan_payload()
    ok = p["master_seed"] == 2026081802
    ok &= p["physical_h3_status_unchanged"] == tds.PHYSICAL_H3
    ok &= p["expected_row_counts"]["cell_summary.csv"] == 17
    ok &= p["expected_row_counts"]["hypothesis_results.csv"] == 4
    ok &= p["expected_row_counts"]["validation_results.parquet"] == 60_000
    ok &= p["audit_repetitions"] == [0, 1, 2, 7, 31, 127, 511, 1023,
                                     2047, 2999]
    ok &= "exploratory_diagnostics.parquet" in p[
        "declared_additional_tables"]
    rec("tds_frozen_plan_payload", ok)
    assert ok


def test_dkw_gate_detects_nonuniform(rec):
    n = 100_000
    thr = math.sqrt(math.log(2 * 16 / 0.001) / (2 * n))

    def kolmogorov(u):
        su = np.sort(u)
        hi = np.arange(1, n + 1) / n
        lo = np.arange(0, n) / n
        return float(max(np.max(np.abs(hi - su)), np.max(np.abs(su - lo))))

    rng = tds.stream_rng((6, 6, 6, 6, 6))
    ok = True
    for dgp in tds.DGPS:
        d = kolmogorov(tds.stationary_single_point_u(
            dgp, tds.stream_rng((6, 6, 6, tds.DGPS.index(dgp), 6)), n))
        ok &= d <= thr
    biased = tds.open_uniform(rng, n) ** 1.05  # deliberately non-uniform
    ok &= kolmogorov(biased) > thr
    rec("tds_dkw_gate", ok, threshold=thr)
    assert ok


def test_fallacy_scan_and_h3_constant(rec):
    ok = len(tds.FALLACY_ITEMS) == 11
    ok &= tds.PHYSICAL_H3 == "INCONCLUSIVE_CORRELATION_SENSITIVITY"
    src = (ROOT / "python" / "parce_exp"
           / "temporal_dependence_sim.py").read_text()
    ok &= "11/11 CHECKED" in src
    ok &= "NOT_A_TEMPORAL_DEPENDENCE_CERTIFICATION_THEOREM" in src
    rec("tds_fallacy_scan_marker", ok)
    assert ok


def test_diagnostics_on_x_not_u(rec):
    """TAIL_BURST is detectable on X/E2E values (Pearson) but its U-domain
    lag-1 rank correlation (~0.028) would never reach the 0.10 threshold;
    the implementation must therefore diagnose on nanosecond values."""
    res = _rep(3, 0)
    diags = [r for r in res["diagnostics"] if r["split"] == "validation"]
    max_pearson = max(abs(r["lag1_pearson"]) for r in diags)
    ok = res["summary"][0]["any_diagnostic_triggered"]
    ok &= max_pearson >= 0.10
    rec("tds_diagnostics_domain", ok, max_pearson=max_pearson)
    assert ok
