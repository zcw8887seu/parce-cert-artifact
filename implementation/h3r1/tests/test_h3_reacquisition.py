"""e01-h3r1 reacquisition guardrails (h3_reacquisition.MD R0).

Covers: e01 write-guard, DESIGN_FROZEN gating, sample isolation,
session-aware lag pairs, session-id validity, plan shape, CP wording,
zero-failure block degeneracy, state/report consistency and POSIX
artifact indexing.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from parce_exp import h3_reacquisition as h3r  # noqa: E402

COORDS = h3r.COORDS


@pytest.fixture()
def run_root(tmp_path, monkeypatch):
    """A prepared e01-h3r1 run root outside runs/e01."""
    root = tmp_path / "runs" / "e01-h3r1"
    h3r.guard_run_root(root)
    for sub in ("state", "meta", "configs", "derived", "figures",
                "reports", "raw/events/calibration", "raw/events/validation"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    h3r.init_state_file(root)
    # a legit plans pair for tests that need them
    sess, runs = h3r.generate_plans()
    sess.to_csv(root / "configs" / "session_plan.csv", index=False)
    runs.to_csv(root / "configs" / "run_plan.csv", index=False)
    return root


# ---------------------------------------------------------------- 1
def test_run_root_resolving_into_e01_is_rejected(tmp_path):
    direct = h3r.PROJECT_ROOT / "runs" / "e01"
    with pytest.raises(SystemExit):
        h3r.guard_run_root(direct)
    with pytest.raises(SystemExit):
        h3r.guard_run_root(h3r.PROJECT_ROOT / "runs" / "e01-h3r1"
                           / ".." / "e01")
    nested = tmp_path / "link"
    os.symlink(direct, nested)
    with pytest.raises(SystemExit):
        h3r.guard_run_root(nested)
    ok = h3r.guard_run_root(tmp_path / "runs" / "e01-h3r1")
    assert "e01-h3r1" in str(ok)


def test_guard_write_refuses_e01_targets():
    with pytest.raises(SystemExit):
        h3r.guard_write(h3r.E01_ROOT / "derived" / "menu.parquet")
    with pytest.raises(SystemExit):
        h3r.guard_write(h3r.E01_ROOT / "raw" / "events" / "batch_x.parquet")
    assert h3r.guard_write(h3r.E01_ROOT.parent / "e01-h3r1" / "x") is not None


# ---------------------------------------------------------------- 2
def test_validation_rejected_before_design_frozen(run_root):
    h3r.complete_step(run_root, "R1", status="PLAN_FROZEN")
    with pytest.raises(SystemExit, match="DESIGN_FROZEN"):
        h3r.acquire_split(run_root, "validation", resume=True)
    h3r.complete_step(run_root, "R2", status="CALIBRATION_COMPLETE")
    h3r.complete_step(run_root, "R3",
                      status="CALIBRATION_DIAGNOSTICS_COMPLETE")
    with pytest.raises(SystemExit, match="DESIGN_FROZEN"):
        h3r.acquire_split(run_root, "validation", resume=True)


def test_calibration_rejected_before_plan_frozen(run_root):
    with pytest.raises(SystemExit):
        h3r.acquire_split(run_root, "calibration", resume=True)


# ---------------------------------------------------------------- 3
def test_old_e01_samples_cannot_enter_new_stage_samples(run_root):
    plan = h3r.load_plans(run_root)[1]
    plan["actual_session_id"] = plan["actual_session_id"].astype(object)
    plan["invalid_reason"] = plan["invalid_reason"].astype(object)
    plan.loc[plan.index[:5], "status"] = "DONE"
    plan.loc[plan.index[:5], "actual_session_id"] = "e01-h3r1-calibration" \
                                                   "-s01-X"
    # rows from e01 (status PLANNED here) and foreign experiment ids
    ev = pd.DataFrame({
        "experiment_id": ["e01", "e01-h3r1", "e01-h3r1"],
        "split": ["calibration"] * 3,
        "run_id": [plan["run_id"].iloc[0], plan["run_id"].iloc[1],
                   plan["run_id"].iloc[2]],
        "seq": [32, 32, 33],
        "is_warmup": [False, False, False],
        "selected_single": [True, True, False],
        "valid_event": [True, True, True],
        "phase_ns": [455509] * 3,
        "t_cause_ns": [0, 0, 0],
        "t_compute_end_ns": [100, 100, 100],
        "t_gate_ingress_ns": [200, 200, 200],
        "t_gate_target1_ns": [300, 300, 300],
        "t_gate_service_start_ns": [400, 400, 400],
        "service_demand_ns": [50000] * 3,
        "t_rx_ns": [50500.0, 50500.0, 50500.0],
        "packet_lost": [False, False, False],
    })
    events = ev
    samples = h3r.build_stage_samples(events, plan, "calibration")
    # only run_id 2 (selected_single, valid, DONE) contributes 5 rows;
    # the e01-tagged row and the non-selected row are excluded
    assert set(samples["run_id"]) == {int(plan["run_id"].iloc[1])}
    assert len(samples) == 5
    assert (samples["experiment_id"] == "e01-h3r1").all()


# ---------------------------------------------------------------- 4
def test_no_lag_pairs_across_sessions():
    values = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0]
    keys = [1, 1, 1, 1, 2, 2, 2]
    a, b = h3r.session_pairs(values, keys, lag=1)
    pairs = list(zip(a.tolist(), b.tolist()))
    assert (3.0, 10.0) not in pairs  # cross-session pair forbidden
    assert pairs == [(1, 2), (2, 3), (3, 4), (10, 20), (20, 30)]
    a5, b5 = h3r.session_pairs(values, keys, lag=5)
    assert len(a5) == 0  # every lag-5 pair would cross the boundary


def test_session_aware_diagnostics_do_not_flag_independent_sessions():
    # within-session monotone trend (strong correlation) but sessions
    # independent of each other: diagnostics must see only inner pairs
    rng = np.random.default_rng(7)
    rows = []
    for idx in (1, 2, 3):
        base = rng.normal(0, 1)
        vals = base + np.linspace(0, 5, 40) + rng.normal(0, 0.01, 40)
        for order, v in enumerate(vals, start=1):
            for coord in COORDS:
                rows.append({
                    "experiment_id": "e01-h3r1", "split": "validation",
                    "planned_session_index": idx, "session_order": order,
                    "actual_session_id": f"s{idx}",
                    "run_id": idx * 1000 + order,
                    "coordinate_id": coord, "value_ns": v,
                    "is_infinite": False, "selected_single": True,
                    "phase_ns": 455509, "calendar_id": "C1",
                    "workload_id": "S1",
                })
    samples = pd.DataFrame(rows)
    diag, sess = h3r.run_order_diagnostics(samples, None)
    for r in diag.itertuples():
        assert int(r.n_lag_pairs) == 3 * 39
        assert abs(r.lag1_pearson - 1.0) < 0.05  # inner trend still visible
        assert r.correlation_triggered  # correctly detected within sessions
    assert set(sess["planned_session_index"]) == {1, 2, 3}


# ---------------------------------------------------------------- 5
@pytest.mark.parametrize("bad", ["PLANNED", "", "   ", None])
def test_session_id_validator_rejects_bad_ids(bad):
    assert not h3r.valid_session_id(bad)


def test_session_id_validator_accepts_real_ids():
    sid = h3r.actual_session_id("calibration", 3)
    assert h3r.valid_session_id(sid)
    assert sid.startswith("e01-h3r1-calibration-s03-")


def test_duplicate_session_ids_rejected_in_final_checks(run_root):
    acq = {
        "split": "calibration",
        "sessions": {str(i): {"valid": [i], "actual_session_id":
                              "e01-h3r1-calibration-s01-X"}
                     for i in range(1, 21)},
    }
    with pytest.raises(SystemExit, match="session ids"):
        h3r._final_acquisition_checks(run_root, "calibration", acq)


# ---------------------------------------------------------------- 6
def test_plan_rows_and_per_session_counts_exact():
    sess, runs = h3r.generate_plans()
    assert len(sess) == 40
    assert len(runs) == 6500
    cal = sess[sess["split"] == "calibration"]
    val = sess[sess["split"] == "validation"]
    assert len(cal) == 20 and (cal["planned_valid_runs"] == 150).all()
    assert len(val) == 20 and (val["planned_valid_runs"] == 175).all()
    for split, n in (("calibration", 3000), ("validation", 3500)):
        r = runs[runs["split"] == split]
        assert len(r) == n
        for _idx, grp in r.groupby("planned_session_index"):
            assert sorted(grp["session_order"]) == list(
                range(1, n // 20 + 1))
    assert not runs["run_id"].duplicated().any()
    assert not runs["seed"].duplicated().any()
    assert runs["selected_instance_index"].between(0, 7).all()
    assert not sess["session_seed"].duplicated().any()
    assert (runs["status"] == "PLANNED").all()
    assert h3r.validate_plans(sess, runs) == []


def test_plan_regeneration_is_deterministic():
    s1, r1 = h3r.generate_plans()
    s2, r2 = h3r.generate_plans()
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------- 7
def test_cp_alpha_row_and_wording():
    assert h3r.ALPHA_ROW_VALIDATION == 0.01
    lo, hi = h3r.clopper_pearson(0, 3500, 0.01)
    assert lo == 0.0
    assert abs(hi - float(beta.ppf(0.99, 1, 3500))) < 1e-12
    assert abs(hi - 0.001315) < 5e-5  # the value e01 reported
    wording = h3r.cp_wording(0.01)
    assert "one-sided 99%" in wording
    assert "95% family" in wording
    assert "95% CP per item" not in wording


# ---------------------------------------------------------------- 8
def test_zero_failure_block_sensitivity_flagged_degenerate():
    flags = {
        "X1": [False] * 200,
        "_session_keys": [1] * 100 + [2] * 100,
        "_run_ids": list(range(200)),
    }
    rows = h3r.block_sensitivity(flags["X1"], flags, "X1")
    assert len(rows) == 3
    for r in rows:
        assert r["failed_blocks"] == 0
        assert r["run_level_failures"] == 0
        assert r["degenerate_flag"] == \
            "ZERO_EVENT_EMPIRICAL_BOOTSTRAP_DEGENERATE"
        assert r["cp_lower"] == 0.0  # CP math still reported
    rows_nz = h3r.block_sensitivity([True] + [False] * 199, flags, "X1")
    assert all(r["degenerate_flag"] == "" for r in rows_nz)
    assert rows_nz[0]["run_level_failures"] == 1


def test_blocks_never_cross_sessions():
    flags = {
        "X1": [True] * 60,
        "_session_keys": [1] * 50 + [2] * 10,  # session 1: 50, session 2: 10
        "_run_ids": list(range(60)),
    }
    rows = h3r.block_sensitivity(flags["X1"], flags, "X1")
    by_size = {r["block_size"]: r for r in rows}
    # block 50: session1 -> 1 block of 50; session2 -> 1 block of 10
    assert by_size[50]["n_blocks"] == 2
    # block 25: session1 -> 2; session2 -> 1 (partial, allowed within)
    assert by_size[25]["n_blocks"] == 3
    assert by_size[100]["n_blocks"] == 2  # 50 + 10, no merge across


# ---------------------------------------------------------------- 9
def test_h3_disposition_cascade():
    mm = "MODEL_OR_IMPLEMENTATION_ERROR"
    assert h3r.h3_disposition(1, 0, False, False, ["SUPPORTED"] * 5) == mm
    assert h3r.h3_disposition(0, 1, False, False, ["SUPPORTED"] * 5) == mm
    assert h3r.h3_disposition(0, 0, False, False, ["SUPPORTED"] * 5) == \
        "SUPPORTED"
    assert h3r.h3_disposition(0, 0, False, False, [
        "NOT_SUPPORTED"] + ["SUPPORTED"] * 4) == "NOT_SUPPORTED"
    assert h3r.h3_disposition(0, 0, False, False, [
        "INCONCLUSIVE"] + ["SUPPORTED"] * 4) == "INCONCLUSIVE"
    assert h3r.h3_disposition(0, 0, False, True, ["SUPPORTED"] * 5) == \
        "INCONCLUSIVE_CORRELATION_SENSITIVITY"
    # calibration flag caps SUPPORTED but keeps refutation
    assert h3r.h3_disposition(0, 0, True, False, ["SUPPORTED"] * 5) == \
        "INCONCLUSIVE_CORRELATION_SENSITIVITY"
    assert h3r.h3_disposition(0, 0, True, False, [
        "NOT_SUPPORTED"] + ["SUPPORTED"] * 4) == "NOT_SUPPORTED"


def test_state_and_report_consistency(run_root):
    h3r.complete_step(run_root, "R0", status="SOFTWARE_VERIFIED",
                      h3_status="NOT_RUN")
    st = h3r.load_state(run_root)
    assert st["status"] == "SOFTWARE_VERIFIED"
    assert st["h3_status"] == "NOT_RUN"
    assert st["last_completed_step"] == "R0"
    # stopped experiments refuse further steps
    h3r.write_state(run_root, status="STOPPED", stop_reason="unit-test")
    with pytest.raises(SystemExit, match="STOPPED"):
        h3r.complete_step(run_root, "R1", status="PLAN_FROZEN")


# --------------------------------------------------------------- 10
def test_artifact_index_uses_posix_separators(run_root):
    (run_root / "derived" / "menu.parquet").write_bytes(b"x")
    nested = run_root / "reports" / "sub"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "04_VALIDATION_AND_H3.md").write_text("x")
    idx = h3r.artifact_index(run_root)
    assert all("\\" not in p for p in idx)
    assert all(not os.path.isabs(p) for p in idx)
    assert "derived/menu.parquet" in idx
    assert "reports/sub/04_VALIDATION_AND_H3.md" in idx


def test_manifest_paths_are_posix(run_root, tmp_path):
    """Zip entries use / on any host platform (Windows-safe)."""
    import zipfile

    (run_root / "derived" / "software_tests.json").write_text("{}")
    (run_root / "state" / "state.json").write_text("{}")
    out = tmp_path / "pkg.zip"
    args = type("A", (), {
        "run_root": str(run_root), "output": str(out)})()
    # only the minimal files exist; the rest are recorded MISSING, which
    # forces execution_status=STOPPED but must still produce a valid zip
    h3r.cmd_build_handoff(args)
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "PACKAGE_MANIFEST.json" in names
        manifest = json.loads(zf.read("PACKAGE_MANIFEST.json"))
        for f in manifest["files"]:
            assert "\\" not in f["path"]
            assert not f["path"].startswith("/")
        assert manifest["execution_status"] in {"COMPLETE", "STOPPED"}
