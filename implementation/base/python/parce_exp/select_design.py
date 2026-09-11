"""Design selection from the calibration menu (spec 6.3, 7-P3).

Simultaneous-menu procedure: certify all menu points on calibration only,
run the Pareto DP under the system risk budget, freeze the selected design,
verify it with an independent §5.6 replay, and freeze the B1-B4 baselines
before any validation data is read.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .allocator import (MenuItem, Selection, brute_force_selection,
                        chain_bound_spec56, gate_admission_offset,
                        next_feasible_start_vec, pareto_dp)
from .calendar_ref import load_calendar
from .tolerance import alpha_row, required_order_statistic

GATE_AFTER = [False, True, True, False]  # spec 5.6: gates after X2 and X3
COORDS = ["X1", "X2", "X3", "X4"]


def calendar_signature(cal, phase_ns):
    return [
        cal["calendar_id"], int(cal["period_ns"]),
        int(cal["eligibility_offset_ns"]), int(cal["service_demand_ns"]),
        [[int(o), int(c)] for o, c in cal["windows"]], int(phase_ns),
    ]


def gate_signature(sig_list):
    return hashlib.sha256(
        json.dumps(sig_list, separators=(",", ":")).encode()).hexdigest()


def build_menus(menu_df):
    menus = {}
    for coord in COORDS:
        pts = menu_df[(menu_df["coordinate_id"] == coord)
                      & (menu_df["status"] == "CERTIFIED")
                      & (~menu_df["q_is_infinite"].astype(bool))]
        menus[coord] = [MenuItem(menu_id=str(r.menu_id), risk=float(r.risk),
                                 q_ns=int(r.q_ns))
                        for r in pts.itertuples()]
    return menus


def main(argv=None):
    p = argparse.ArgumentParser(prog="parce_exp.select_design")
    p.add_argument("--run-root", default="runs/e01")
    args = p.parse_args(argv)
    run_root = Path(args.run_root)
    root = Path(__file__).resolve().parents[2]
    cal = load_calendar(root / "configs" / "calendar.yaml")
    risk = yaml.safe_load((root / "configs" / "risk.yaml").read_text())
    frozen = yaml.safe_load((root / "configs" / "frozen_plan.yaml").read_text())
    phase = int(frozen["binding"]["phase_ns"])
    demand = int(cal["service_demand_ns"])
    period = int(cal["period_ns"])
    offset = int(cal["eligibility_offset_ns"])
    windows = [(int(o) + offset, int(c) + offset) for o, c in cal["windows"]]
    budget = float(risk["system_risk_budget"])
    menu_df = pd.read_parquet(run_root / "derived" / "menu.parquet")
    menus = build_menus(menu_df)

    sig = calendar_signature(cal, phase)
    design = {
        "procedure": "SIMULTANEOUS_MENU",
        "calendar_id": cal["calendar_id"],
        "phase_ns": phase,
        "risk_budget": budget,
        "selected_menu_ids": {},
        "selected_risks": {},
        "selected_q_ns": {},
        "risk_sum": 0.0,
        "bound_ns": 0,
        "calendar_signature": sig,
        "algorithm": "PARETO_FRONTIER_DP",
    }
    notes = {"gate_signature": gate_signature(sig), "frozen_at":
             datetime.now().astimezone().isoformat(timespec="seconds")}

    empty = [c for c in COORDS if not menus[c]]
    if empty:
        design["status"] = "NOT_SUPPORTED"
        design["not_supported_reason"] = (
            f"no finite certified menu point for {empty}")
        (run_root / "derived" / "selected_design.json").write_text(
            json.dumps(design, indent=2))
        print(json.dumps(design, indent=2))
        return

    last_gate = max(j for j, g in enumerate(GATE_AFTER) if g)

    def gate_op(j):
        def op(r):
            out = gate_admission_offset(r, demand, period, windows, phase)
            if j == last_gate:
                out += demand
            return out
        return op

    gate_ops = [gate_op(j) if GATE_AFTER[j] else None for j in range(4)]
    menu_list = [menus[c] for c in COORDS]
    sel = pareto_dp(menu_list, GATE_AFTER, demand, period, windows, budget,
                    base_ns=phase, gate_ops=gate_ops)
    brute = brute_force_selection(menu_list, GATE_AFTER, demand, period,
                                  windows, budget, base_ns=phase,
                                  gate_ops=gate_ops)
    dp_matches_brute = ((sel is None and brute is None)
                        or (sel is not None and brute is not None
                            and (sel.bound_ns, sel.risk_sum, sel.menu_ids)
                            == (brute.bound_ns, brute.risk_sum, brute.menu_ids)))
    if sel is None:
        design["status"] = "NOT_SUPPORTED"
        design["not_supported_reason"] = (
            f"no menu combination satisfies risk_sum <= {budget}")
        (run_root / "derived" / "selected_design.json").write_text(
            json.dumps(design, indent=2))
        print(json.dumps(design, indent=2))
        return

    by_id = {c: {it.menu_id: it for it in menus[c]} for c in COORDS}
    qs = [by_id[c][mid].q_ns for c, mid in zip(COORDS, sel.menu_ids)]
    replay_bound = chain_bound_spec56(qs, GATE_AFTER, demand, period, windows,
                                      base_ns=phase)
    if replay_bound != sel.bound_ns:
        raise SystemExit(f"design replay mismatch: DP={sel.bound_ns} "
                         f"replay={replay_bound}")

    design["selected_menu_ids"] = {c: m for c, m in zip(COORDS, sel.menu_ids)}
    design["selected_risks"] = {
        c: by_id[c][m].risk for c, m in zip(COORDS, sel.menu_ids)}
    design["selected_q_ns"] = {c: by_id[c][m].q_ns
                               for c, m in zip(COORDS, sel.menu_ids)}
    design["risk_sum"] = float(sel.risk_sum)
    design["bound_ns"] = int(sel.bound_ns)
    design["status"] = "DESIGN_FROZEN"
    design["replay_bound_ns"] = int(replay_bound)
    design["dp_matches_brute_force"] = bool(dp_matches_brute)
    (run_root / "derived" / "selected_design.json").write_text(
        json.dumps(design, indent=2))

    # ---------------- baselines (calibration only, frozen pre-validation)
    samples = pd.read_parquet(run_root / "derived" / "stage_samples.parquet")
    calib = samples[samples["split"] == "calibration"]

    b1_q, b1_risk = {}, 0.0
    for c in COORDS:
        cand = [it for it in menus[c] if it.risk <= 0.0025]
        best = max(cand, key=lambda it: (it.q_ns, -it.risk))
        b1_q[c] = best.q_ns
        b1_risk += best.risk
    b1_bound = chain_bound_spec56([b1_q[c] for c in COORDS], GATE_AFTER,
                                  demand, period, windows, base_ns=phase)

    from .extract import max_wait_ns
    w_max = max_wait_ns(period, demand, windows)
    b2_bound = (sum(qs) + demand + 2 * w_max)

    e2e_vals = sorted(
        float(v) if not inf else float("inf")
        for v, inf in zip(calib[calib["coordinate_id"] == "E2E"]["value_ns"],
                          calib[calib["coordinate_id"] == "E2E"]["is_infinite"]))
    a_row = alpha_row(12, risk["calibration_family_alpha"])
    eps_eff = float(sel.risk_sum)
    k3 = required_order_statistic(len(e2e_vals), eps_eff, a_row)
    b3_q = e2e_vals[k3 - 1]
    b3 = {"epsilon": eps_eff, "alpha_row": a_row, "order_k": int(k3),
          "n_units": len(e2e_vals),
          "bound_ns": None if b3_q == float("inf") else int(b3_q),
          "status": "INFEASIBLE_PACKET_LOSS" if b3_q == float("inf")
          else "CERTIFIED"}

    rng = np.random.default_rng(np.random.SeedSequence(entropy=(20260816, 5)))
    n_resample = 200_000
    marg = {}
    for c in COORDS:
        col = calib[calib["coordinate_id"] == c]
        vals = col["value_ns"].to_numpy(dtype=np.float64)
        marg[c] = vals[np.isfinite(vals)]
    x1 = rng.choice(marg["X1"], size=n_resample, replace=True)
    x2 = rng.choice(marg["X2"], size=n_resample, replace=True)
    x3 = rng.choice(marg["X3"], size=n_resample, replace=True)
    x4 = rng.choice(marg["X4"], size=n_resample, replace=True)
    t2 = np.int64(phase) + x1.astype(np.int64) + x2.astype(np.int64)
    s1 = next_feasible_start_vec(t2, demand, period, windows)[0]
    t3 = s1 + x3.astype(np.int64)
    s2 = next_feasible_start_vec(t3, demand, period, windows)[0]
    e2e_sim = (s2 - np.int64(phase)).astype(np.int64) + demand \
        + x4.astype(np.int64)
    order = int(np.ceil((1.0 - eps_eff) * n_resample))
    b4 = {
        "procedure": "INDEPENDENT_RESAMPLING_EMPIRICAL_CONVOLUTION",
        "flag": "MODEL_DEPENDENT_NON_CERTIFYING",
        "n_resamples": n_resample,
        "epsilon": eps_eff,
        "exceedance_probability_estimate": float(
            (e2e_sim > sel.bound_ns).mean()),
        "quantile_bound_ns": int(np.sort(e2e_sim)[order - 1]),
        "frozen_before_validation": True,
    }

    baselines = {
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "B1_equal_risk_phase_aware": {
            "description": "per stage the largest certified menu with risk "
                           "<= 0.0025, phase-aware propagation",
            "q_ns": b1_q, "risk_sum": float(b1_risk),
            "bound_ns": int(b1_bound)},
        "B2_same_q_max_wait": {
            "description": "selected q, each admission replaced by the legal "
                           "max wait (two gates)",
            "bound_ns": int(b2_bound), "w_max_ns": int(w_max)},
        "B3_calibration_e2e_order_statistic": b3,
        "B4_marginal_resampling": b4,
        **notes,
    }
    (run_root / "derived" / "baselines.json").write_text(
        json.dumps(baselines, indent=2))
    print(json.dumps({"design": design, "baselines": {
        "B1_bound_ns": b1_bound, "B2_bound_ns": b2_bound,
        "B3": b3, "B4_exceedance": b4["exceedance_probability_estimate"],
        "B4_quantile_ns": b4["quantile_bound_ns"]}}, indent=2))


if __name__ == "__main__":
    main()
