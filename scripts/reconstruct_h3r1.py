#!/usr/bin/env python3
"""Recompute the published H3R1 numerical disposition from reduced data.

The input intentionally contains one selected statistical unit per run, a
neutral session label, within-session order, and derived coordinate durations.
It contains neither absolute timestamps nor host- or process-identifying data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTIFACT_ROOT / "src"))

from parce_cert_core import (
    clopper_pearson,
    next_feasible_start,
    required_order_statistic,
    validation_verdict,
)


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "split", "session_label", "run_order_in_session",
        "statistical_unit", "coordinate_id", "value_ns", "is_infinite",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError(f"unexpected reduced-data columns in {path}")
    for row in rows:
        row["run_order_in_session"] = int(row["run_order_in_session"])
        row["is_infinite"] = row["is_infinite"].lower() == "true"
        row["value_ns"] = None if row["value_ns"] == "" else int(row["value_ns"])
    return rows


def coordinate_rows(rows, split, coordinate):
    return sorted(
        (r for r in rows if r["split"] == split and r["coordinate_id"] == coordinate),
        key=lambda r: (r["session_label"], r["run_order_in_session"]),
    )


def numeric(row):
    return math.inf if row["is_infinite"] else float(row["value_ns"])


def pairs_within_sessions(series, lag=1):
    left, right = [], []
    for index in range(len(series) - lag):
        if series[index][0] == series[index + lag][0]:
            left.append(series[index][1])
            right.append(series[index + lag][1])
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


def pearson(left, right):
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left, right):
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return math.nan
    return float(spearmanr(left, right).statistic)


def guard_threshold(n_lag_pairs):
    return max(0.10, 3.0 / math.sqrt(max(n_lag_pairs, 1)))


def diagnostics(rows, config):
    result, session_rows = {}, []
    q = config["selected_design"]["selected_q_ns"]
    bound = config["selected_design"]["candidate_bound_ns"]
    for split in ("calibration", "validation"):
        result[split] = {}
        for coordinate in config["coordinates"]:
            ordered = coordinate_rows(rows, split, coordinate)
            series = [(r["session_label"], numeric(r)) for r in ordered]
            left, right = pairs_within_sessions(series)
            p, s = pearson(left, right), spearman(left, right)
            threshold = guard_threshold(len(left))
            triggered = bool(abs(p) >= threshold or abs(s) >= threshold)
            result[split][coordinate] = {
                "n_units": len(series), "n_lag_pairs": len(left),
                "lag1_pearson": p, "lag1_spearman": s,
                "trigger_threshold": threshold, "correlation_triggered": triggered,
            }
            by_session = defaultdict(list)
            for session, value in series:
                by_session[session].append(value)
            for session, values in sorted(by_session.items()):
                values = np.asarray(values, dtype=float)
                threshold_value = None if split == "calibration" else q.get(coordinate, bound)
                failures = int(np.sum(~np.isfinite(values))) if threshold_value is None else int(
                    np.sum(~np.isfinite(values) | (values > threshold_value))
                )
                sp_left, sp_right = pairs_within_sessions([(session, x) for x in values])
                session_rows.append({
                    "split": split, "session_label": session, "coordinate_id": coordinate,
                    "n_units": len(values), "median_ns": float(np.nanmedian(values)),
                    "maximum_ns": float(np.nanmax(values)), "failure_count": failures,
                    "lag1_pearson": pearson(sp_left, sp_right),
                    "lag1_spearman": spearman(sp_left, sp_right),
                })
    return result, session_rows


def vectors(rows, split, coordinates):
    out = defaultdict(dict)
    for row in rows:
        if row["split"] == split:
            out[(row["session_label"], row["run_order_in_session"])][row["coordinate_id"]] = numeric(row)
    if any(set(vector) != set(coordinates) for vector in out.values()):
        raise ValueError("every released statistical unit must contain X1--X4 and E2E")
    return dict(sorted(out.items()))


def selected_order_statistics(rows, config):
    alpha = config["calibration"]["alpha_row"]
    risks = config["selected_design"]["selected_risks"]
    out = {}
    for coordinate, risk in risks.items():
        values = sorted(numeric(r) for r in coordinate_rows(rows, "calibration", coordinate))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("the selected calibration order statistic is infinite")
        # Compute k from the frozen risk and row error; the expected q is used
        # only by verify(), never to choose an order statistic.
        k = required_order_statistic(len(values), risk, alpha)
        out[coordinate] = {"risk": risk, "order_k": k, "q_ns": int(values[k - 1]), "alpha_row": alpha}
    return out


def candidate_bound(config):
    q = config["selected_design"]["selected_q_ns"]
    calendar = config["calendar"]
    eligible = calendar["phase_ns"] + q["X1"] + q["X2"]
    start, cycle, window, reason = next_feasible_start(
        eligible, calendar["service_demand_ns"], calendar["period_ns"],
        [tuple(window) for window in calendar["windows_ns"]],
    )
    if reason != 0:
        raise ValueError("selected demand has no feasible service window")
    wait = start - eligible
    bound = q["X1"] + q["X2"] + wait + q["X3"] + calendar["service_demand_ns"] + q["X4"]
    return {
        "eligible_ns": eligible, "service_start_ns": start, "cycle_id": cycle,
        "window_id": window, "gate_wait_ns": wait, "candidate_bound_ns": bound,
    }


def validation(rows, config):
    design = config["selected_design"]
    items = {**design["selected_q_ns"], "E2E": design["candidate_bound_ns"]}
    risks = {**design["selected_risks"], "E2E": design["risk_sum"]}
    out = []
    for coordinate in config["coordinates"]:
        values = [vector[coordinate] for vector in vectors(rows, "validation", config["coordinates"]).values()]
        failures = sum(not math.isfinite(value) or value > items[coordinate] for value in values)
        lower, upper = clopper_pearson(failures, len(values), config["validation"]["alpha_row"])
        out.append({
            "item_id": coordinate, "target_risk": risks[coordinate], "n_units": len(values),
            "failures": failures, "alpha_row": config["validation"]["alpha_row"],
            "cp_lower": lower, "cp_upper": upper,
            "status": validation_verdict(lower, upper, risks[coordinate]),
        })
    return out


def h3_state(rows, config, diag, validation_rows):
    design = config["selected_design"]
    validation_vectors = vectors(rows, "validation", config["coordinates"])
    inclusion_violations = sum(
        vector["E2E"] > design["candidate_bound_ns"] and not any(
            vector[coordinate] > design["selected_q_ns"][coordinate] for coordinate in ("X1", "X2", "X3", "X4")
        ) for vector in validation_vectors.values()
    )
    calibration_triggered = any(row["correlation_triggered"] for row in diag["calibration"].values())
    validation_triggered = any(row["correlation_triggered"] for row in diag["validation"].values())
    statuses = [row["status"] for row in validation_rows]
    if inclusion_violations:
        status = "MODEL_OR_IMPLEMENTATION_ERROR"
    elif "NOT_SUPPORTED" in statuses:
        status = "NOT_SUPPORTED"
    elif validation_triggered or (calibration_triggered and all(value == "SUPPORTED" for value in statuses)):
        status = "INCONCLUSIVE_CORRELATION_SENSITIVITY"
    elif all(value == "SUPPORTED" for value in statuses):
        status = "SUPPORTED"
    else:
        status = "INCONCLUSIVE"
    return {
        "coordinate_level_inclusion_violations": inclusion_violations,
        "calibration_correlation_triggered": calibration_triggered,
        "validation_correlation_triggered": validation_triggered,
        "reference_mismatch_count": config["frozen_non_trace_checks"]["reference_mismatch_count"],
        "reference_mismatch_recomputed": False,
        "h3_state": status,
    }


def verify(result, config):
    errors = []
    if result["candidate_gate_trace"]["candidate_bound_ns"] != config["selected_design"]["candidate_bound_ns"]:
        errors.append("candidate bound differs")
    for coordinate, row in result["selected_order_statistics"].items():
        if row["q_ns"] != config["selected_design"]["selected_q_ns"][coordinate]:
            errors.append(f"selected q differs for {coordinate}")
    for row in result["validation"]:
        if row["failures"] != config["expected"]["validation_failures"][row["item_id"]]:
            errors.append(f"validation failures differ for {row['item_id']}")
    for split, by_coordinate in result["diagnostics"].items():
        for coordinate, row in by_coordinate.items():
            expected_p, expected_s, expected_trigger = config["expected"]["diagnostics"][split][coordinate]
            if abs(row["lag1_pearson"] - expected_p) > 1e-12 or abs(row["lag1_spearman"] - expected_s) > 1e-12 or row["correlation_triggered"] != expected_trigger:
                errors.append(f"diagnostic differs for {split}/{coordinate}")
    if result["h3"]["h3_state"] != config["expected"]["h3_state"]:
        errors.append("H3 disposition differs")
    result["verification"] = {"passed": not errors, "errors": errors}


def write_session_diagnostics(path: Path, rows):
    path.mkdir(parents=True, exist_ok=True)
    target = path / "h3r1_reconstructed_session_diagnostics.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ARTIFACT_ROOT / "data" / "h3r1_reduced_runs.csv")
    parser.add_argument("--config", type=Path, default=ARTIFACT_ROOT / "data" / "h3r1_reconstruction_config.json")
    parser.add_argument("--write-dir", type=Path, help="optional directory for the reconstructed session diagnostics and JSON summary")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = load_csv(args.data)
    diag, session_rows = diagnostics(rows, config)
    result = {
        "selected_order_statistics": selected_order_statistics(rows, config),
        "candidate_gate_trace": candidate_bound(config),
        "validation": validation(rows, config),
        "diagnostics": diag,
    }
    result["h3"] = h3_state(rows, config, diag, result["validation"])
    verify(result, config)
    if args.write_dir:
        args.write_dir.mkdir(parents=True, exist_ok=True)
        result["session_diagnostics_file"] = str(write_session_diagnostics(args.write_dir, session_rows))
        (args.write_dir / "h3r1_reconstruction_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["verification"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
