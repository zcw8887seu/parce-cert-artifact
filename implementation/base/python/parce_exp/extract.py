"""Stage-coordinate extraction and phase-model helpers (spec 5.5, 5.6, 7-P2)."""

import numpy as np
from pathlib import Path
import pandas as pd

from .allocator import next_feasible_start_vec
from .calendar_ref import next_feasible_start

COORDINATES = ["X1", "X2", "X3", "X4", "E2E"]


def event_coordinates(events):
    """Per-event X1..X4 and E2E in nanoseconds (spec 5.5).

    X4 and E2E are null when the packet was lost (the receiver never saw it);
    callers represent that as value_ns=null, is_infinite=true downstream.
    """
    df = events
    x1 = (df["t_compute_end_ns"].astype("Int64")
          - df["t_cause_ns"].astype("Int64"))
    x2 = (df["t_gate_ingress_ns"].astype("Int64")
          - df["t_compute_end_ns"].astype("Int64"))
    x3 = (df["t_gate_service_start_ns"].astype("Int64")
          - df["t_gate_target1_ns"].astype("Int64"))
    completion = (df["t_gate_service_start_ns"].astype("Int64")
                  + df["service_demand_ns"].astype("Int64"))
    x4 = (df["t_rx_ns"].astype("Int64") - completion)
    e2e = (df["t_rx_ns"].astype("Int64") - df["t_cause_ns"].astype("Int64"))
    out = pd.DataFrame({
        "run_id": df["run_id"], "seq": df["seq"],
        "phase_ns": df["phase_ns"], "is_warmup": df["is_warmup"],
        "selected_single": df["selected_single"], "X1": x1, "X2": x2, "X3": x3,
        "X4": x4, "E2E": e2e, "packet_lost": df["packet_lost"].astype(bool),
    })
    return out


def stage_samples(events, split):
    """Long-format stage sample rows (spec 8.4) for one split.

    Only valid, non-warmup, selected instances contribute one row per
    coordinate; X4/E2E of lost packets get value_ns=null, is_infinite=true.
    """
    df = events[(events["split"] == split) & (~events["is_warmup"].astype(bool))
                & events["valid_event"].astype(bool)
                & events["selected_single"].astype(bool)]
    coords = event_coordinates(df)
    analysis_min_seq = df.groupby("run_id")["seq"].min().to_dict()
    rows = []
    for _, r in coords.iterrows():
        for coord in COORDINATES:
            value = r[coord]
            lost = bool(r["packet_lost"]) and coord in ("X4", "E2E")
            rows.append({
                "run_id": int(r["run_id"]), "split": split,
                "coordinate_id": coord,
                "value_ns": None if (pd.isna(value) or lost) else int(value),
                "is_infinite": bool(lost),
                "selected_instance_index": int(r["seq"]) - analysis_min_seq[int(r["run_id"])],
                "valid_sample": True, "invalid_reason": None,
            })
    return pd.DataFrame(rows, columns=[
        "run_id", "split", "coordinate_id", "value_ns", "is_infinite",
        "selected_instance_index", "valid_sample", "invalid_reason"])


def max_wait_ns(period_ns, demand_ns, windows):
    """Largest gate waiting time S(e)-e over eligibility positions [0, P)."""
    e = np.arange(int(period_ns), dtype=np.int64)
    start, _cyc, _win, reason = next_feasible_start_vec(
        e, demand_ns, period_ns, windows)
    ok = reason == 0
    if not ok.any():
        return None
    wait = start[ok] - e[ok]
    return int(wait.max())


def phase_wait_ns(position_ns, demand_ns, period_ns, windows):
    """Gate waiting S(e)-e for one eligibility position (periodic in e)."""
    e = int(position_ns) % int(period_ns)
    start, _cyc, _win, reason = next_feasible_start(
        e, demand_ns, period_ns, windows)
    if reason != 0:
        return None
    return start - e


def phase_bound_ns(x1, x2, x3, x4, demand_ns, period_ns, windows):
    """Phase-aware E2E bound for one observed stage vector (spec 5.6 replay).

    E2E = X1 + X2 + (S(X1+X2) - (X1+X2)) + X3 + d + X4; the wait term is the
    only phase-dependent part, and the expression is exact for observed x.
    Returns (bound_ns, wait_ns).
    """
    arrival = int(x1) + int(x2)
    start, _cyc, _win, reason = next_feasible_start(
        arrival, demand_ns, period_ns, windows)
    if reason != 0:
        return None, None
    wait = start - arrival
    return int(x1) + int(x2) + wait + int(x3) + int(demand_ns) + int(x4), wait


def _load_events(run_root):
    import os
    root = Path(run_root) / "raw" / "events"
    frames = [pd.read_parquet(p) for p in sorted(root.glob("batch_*.parquet"))]
    pending_dir = root / "pending"
    if pending_dir.exists():
        frames += [pd.read_parquet(p)
                   for p in sorted(pending_dir.glob("run_*.parquet"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    import argparse
    import json
    import os
    from pathlib import Path

    p = argparse.ArgumentParser(prog="parce_exp.extract")
    p.add_argument("--run-root", default="runs/e01")
    p.add_argument("--split", choices=["calibration", "validation"],
                   required=True)
    args = p.parse_args(argv)
    run_root = Path(args.run_root)
    state_path = run_root / "derived" / f"{args.split}_state.json"
    state = json.loads(state_path.read_text())
    valid_ids = set(state["valid"])
    events = _load_events(run_root)
    ev = events[(events["split"] == args.split)
                & events["run_id"].isin(valid_ids)]
    samples = stage_samples(ev, args.split)
    out = run_root / "derived" / "stage_samples.parquet"
    if out.exists():
        old = pd.read_parquet(out)
        old = old[old["split"] != args.split]
        samples = pd.concat([old, samples], ignore_index=True)
    tmp = out.with_suffix(".tmp.parquet")
    samples.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    summary = samples.groupby(["split", "coordinate_id"]).agg(
        n=("value_ns", "size"),
        n_infinite=("is_infinite", "sum"),
    )
    print(summary.to_string())
    print(f"wrote {out} ({len(samples)} rows for split={args.split})")


if __name__ == "__main__":
    main()
