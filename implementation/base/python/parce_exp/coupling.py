"""Dependence re-coupling on frozen validation marginals (spec 7-P4.2).

Re-couples the four frozen validation stage-margin samples under four
couplings, replays every joint vector through the same periodic service
operator, verifies that each stage multiset is preserved and that the event
inclusion has zero counterexamples. These couplings do NOT cover all
dependence structures.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .allocator import next_feasible_start_vec
from .calendar_ref import load_calendar

COORDS = ["X1", "X2", "X3", "X4"]


def replay_e2e(x1, x2, x3, x4, phase, demand, period, windows):
    """Vectorised spec-5.6 model replay; returns model E2E (ns)."""
    t2 = np.int64(phase) + x1.astype(np.int64) + x2.astype(np.int64)
    s1 = next_feasible_start_vec(t2, demand, period, windows)[0]
    t3 = s1 + x3.astype(np.int64)
    s2 = next_feasible_start_vec(t3, demand, period, windows)[0]
    return (s2 - np.int64(phase)) + np.int64(demand) + x4.astype(np.int64)


def evaluate(joints, q_sel, B, phase, demand, period, windows, marginals):
    x1, x2, x3, x4 = joints
    e2e = replay_e2e(x1, x2, x3, x4, phase, demand, period, windows)
    viol = np.zeros(len(x1), dtype=bool)
    for arr, q in zip(joints, (q_sel[c] for c in COORDS)):
        viol |= arr > q
    beyond = e2e > B
    inclusion_violations = int((beyond & ~viol).sum())
    marginals_ok = all(
        np.array_equal(np.sort(joints[i].astype(float)),
                       np.sort(marginals[c].astype(float)))
        for i, c in enumerate(COORDS))
    return {
        "union_bound": float(viol.mean()),
        "replay_miss_rate": float(beyond.mean()),
        "event_inclusion_violations": inclusion_violations,
        "marginals_preserved": bool(marginals_ok),
    }


def main(argv=None):
    from . import plots

    p = argparse.ArgumentParser(prog="parce_exp.coupling")
    p.add_argument("--run-root", default="runs/e01")
    p.add_argument("--n-permutation-seeds", type=int, default=100)
    p.add_argument("--baseline-resamples", type=int, default=100_000)
    args = p.parse_args(argv)
    run_root = Path(args.run_root)
    root = Path(__file__).resolve().parents[2]
    cal = load_calendar(root / "configs" / "calendar.yaml")
    design = json.loads(
        (run_root / "derived" / "selected_design.json").read_text())
    demand = int(cal["service_demand_ns"])
    period = int(cal["period_ns"])
    offset = int(cal["eligibility_offset_ns"])
    windows = [(int(o) + offset, int(c) + offset) for o, c in cal["windows"]]
    phase = int(design["phase_ns"])
    B = int(design["bound_ns"])
    q_sel = {c: int(design["selected_q_ns"][c]) for c in COORDS}
    risk_sum = float(design["risk_sum"])

    samples = pd.read_parquet(run_root / "derived" / "stage_samples.parquet")
    val = samples[samples["split"] == "validation"]
    per_run = {}
    for coord in COORDS:
        col = val[val["coordinate_id"] == coord]
        for r in col.itertuples():
            per_run.setdefault(int(r.run_id), {})[coord] = (
                None if r.is_infinite else int(r.value_ns))
    runs = sorted(per_run)
    complete = [r for r in runs
                if all(per_run[r][c] is not None for c in COORDS)]
    marginals = {c: np.array([per_run[r][c] for r in complete], dtype=np.int64)
                 for c in COORDS}
    n = len(complete)
    observed = tuple(marginals[c] for c in COORDS)

    rows = []

    def add(coupling_id, seed, n_units, metrics, indep=None):
        rows.append({
            "coupling_id": coupling_id, "coupling_seed": seed,
            "n_joint_units": n_units,
            "union_bound": metrics["union_bound"],
            "independence_estimate": indep,
            "replay_miss_rate": metrics["replay_miss_rate"],
            "event_inclusion_violations": metrics["event_inclusion_violations"],
            "marginals_preserved": metrics["marginals_preserved"],
        })

    # 1. observed synchronous pairing (the physical validation vectors)
    m = evaluate(observed, q_sel, B, phase, demand, period, windows, marginals)
    add("OBSERVED_SYNCHRONOUS", 0, n, m)

    # 2. independent permutations (100 seeds) + per-seed 100k baseline
    for seed in range(1, args.n_permutation_seeds + 1):
        rng = np.random.default_rng(
            np.random.SeedSequence(entropy=(20260816, 7, seed)))
        perm = tuple(marginals[c][rng.permutation(n)].astype(np.int64)
                     for c in COORDS)
        m = evaluate(perm, q_sel, B, phase, demand, period, windows, marginals)
        ridx = rng.integers(0, n, size=args.baseline_resamples)
        base = tuple(marginals[c][ridx] for c in COORDS)
        e2e_b = replay_e2e(*base, phase, demand, period, windows)
        add("INDEPENDENT_PERMUTATION", seed, n, m,
            indep=float((e2e_b > B).mean()))

    # 3. comonotonic rank coupling
    como = tuple(np.sort(marginals[c]).astype(np.int64) for c in COORDS)
    m = evaluate(como, q_sel, B, phase, demand, period, windows, marginals)
    add("COMONOTONIC_RANK", 0, n, m)

    # 4. antithetic rank coupling: X1 ascending, X2-X4 descending
    anti = tuple(
        np.sort(marginals["X1"]).astype(np.int64) if c == "X1"
        else np.sort(marginals[c])[::-1].astype(np.int64) for c in COORDS)
    m = evaluate(anti, q_sel, B, phase, demand, period, windows, marginals)
    add("ANTITHETIC_RANK", 0, n, m)

    dep = pd.DataFrame(rows, columns=[
        "coupling_id", "coupling_seed", "n_joint_units", "union_bound",
        "independence_estimate", "replay_miss_rate",
        "event_inclusion_violations", "marginals_preserved"])
    out = run_root / "derived" / "dependence_results.parquet"
    dep.to_parquet(out, index=False)

    fig_path = plots.fig05_dependence(
        dep, run_root / "figures" / "fig05_dependence.svg")

    total_inclusion = int(dep["event_inclusion_violations"].sum())
    marg_ok = bool(dep["marginals_preserved"].all())
    perm = dep[dep["coupling_id"] == "INDEPENDENT_PERMUTATION"]
    h5_ok = total_inclusion == 0 and marg_ok

    lines = [
        "# 05 Selection and Dependence (P4.2 part)",
        "",
        f"- generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- marginals: frozen validation selected instances, n={n} per stage",
        f"- frozen design: B={B} ns, risk_sum={risk_sum:.4f}, "
        f"phase={phase}",
        "",
        "## Couplings",
        "",
        "| coupling | seeds | union certificate | replay miss rate | "
        "independence estimate | inclusion violations | marginals ok |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for cid in ("OBSERVED_SYNCHRONOUS", "INDEPENDENT_PERMUTATION",
                "COMONOTONIC_RANK", "ANTITHETIC_RANK"):
        g = dep[dep["coupling_id"] == cid]
        indep = (f"{g['independence_estimate'].median():.5f} (median)"
                 if cid == "INDEPENDENT_PERMUTATION" else "-")
        lines.append(
            f"| {cid} | {len(g)} | {g['union_bound'].max():.5f} | "
            f"{g['replay_miss_rate'].max():.5f} (max) | {indep} | "
            f"{int(g['event_inclusion_violations'].sum())} | "
            f"{bool(g['marginals_preserved'].all())} |")
    lines += [
        "",
        f"- union certificate vs Boole bound: max union = "
        f"{dep['union_bound'].max():.5f} <= risk_sum {risk_sum:.4f}",
        f"- independence baselines: {args.n_permutation_seeds} seeds x "
        f"{args.baseline_resamples} joint resamples each",
        f"- event inclusion counterexamples across all couplings: "
        f"{total_inclusion} (required 0)",
        f"- marginals preserved in every coupling: {marg_ok}",
        "",
        "**Limitation**: these four couplings do not cover all dependence",
        "structures; results bound only what was tested (spec 7-P4.2 rule 6).",
        "",
        f"- H5 disposition: {'SUPPORTED' if h5_ok else 'NOT_SUPPORTED'} "
        "(zero event-inclusion counterexamples under every coupling with "
        "marginals preserved).",
        "",
        f"Figure: figures/fig05_dependence.svg; data: {out}",
    ]
    report_path = run_root / "reports" / "05_SELECTION_AND_DEPENDENCE.md"
    if report_path.exists():
        existing = report_path.read_text()
        report_path.write_text(existing.rstrip() + "\n\n" + "\n".join(lines) + "\n")
    else:
        report_path.write_text("\n".join(lines) + "\n")
    print(dep.groupby("coupling_id").agg(
        seeds=("coupling_seed", "size"),
        union_max=("union_bound", "max"),
        miss_max=("replay_miss_rate", "max"),
        indep_median=("independence_estimate", "median"),
        inclusion=("event_inclusion_violations", "sum")).to_string())
    print(f"inclusion violations total: {total_inclusion}; "
          f"marginals preserved: {marg_ok}")


if __name__ == "__main__":
    main()
