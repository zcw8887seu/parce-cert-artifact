"""Figures for the experiment reports (P2 fig01; more added in later phases)."""

from pathlib import Path


def fig01_phase_staircase(sweep_rows, aggregates, demo_wait_curve, w_max_ns,
                          out_path, boundary_phase_ns):
    """Phase-sweep staircase figure.

    - observed median gate wait per phase (scatter, coarse + fine)
    - theoretical wait staircase from the P1-frozen demo vector (line)
    - legal max-wait horizontal line
    - median phase-aware bound vs max-wait bound per phase (second axis)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coarse = aggregates[aggregates["sweep"] == "coarse"]
    fine = aggregates[aggregates["sweep"] == "fine"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    ax1.scatter(coarse["phase_ns"], coarse["median_observed_wait_ns"], s=22,
                color="#1f77b4", label="observed median wait (coarse)", zorder=3)
    ax1.scatter(fine["phase_ns"], fine["median_observed_wait_ns"], s=22,
                color="#d62728", label="observed median wait (fine)", zorder=3)
    ax1.plot(fine["phase_ns"], fine["median_wait_ns"], color="#ff7f0e", lw=1.2,
             ls="-", label="reference replay wait (fine)", zorder=2)
    if demo_wait_curve is not None and len(demo_wait_curve):
        phases, waits = zip(*demo_wait_curve)
        ax1.plot(phases, waits, color="#2ca02c", lw=1.0, alpha=0.8,
                 label="demo-vector staircase (P1-frozen)", zorder=2)
    ax1.axhline(w_max_ns, color="k", ls="--", lw=1,
                label=f"legal max wait = {w_max_ns} ns")
    ax1.axvline(boundary_phase_ns, color="gray", ls=":", lw=1,
                label=f"eligibility boundary phase {boundary_phase_ns}")
    ax1.set_ylabel("gate wait [ns]")
    ax1.set_title("fig01: periodic-gating staircase and phase-aware bound\n"
                  "(observed wait includes late-wake reschedules; timer slack "
                  "~50 us shifts the effective boundary left)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3)

    for agg, color, marker in ((coarse, "#1f77b4", "o"), (fine, "#d62728", "s")):
        ax2.plot(agg["phase_ns"], agg["median_phase_bound_ns"], marker=marker,
                 ms=3, lw=1, color=color, label="median phase bound")
        ax2.plot(agg["phase_ns"], agg["median_maxwait_bound_ns"], marker=None,
                 lw=1, ls="--", color=color, alpha=0.6,
                 label="median max-wait bound")
    ax2.set_xlabel("phase [ns]")
    ax2.set_ylabel("E2E bound [ns]")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def fig02_stage_validation(coord_values, q_sel, coord_results, out_path):
    """Per-stage tail scatter with the frozen envelope and failure count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = ["X1", "X2", "X3", "X4"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, c in zip(axes.flat, coords):
        vals = sorted(v for v in coord_values[c] if v != float("inf"))
        tail = min(len(vals), 400)
        y = vals[-tail:]
        ax.scatter(range(len(y)), y, s=4, color="#1f77b4")
        ax.axhline(q_sel[c], color="#d62728", lw=1.2,
                   label=f"q = {q_sel[c]} ns")
        n_fail = coord_results[c]["failures"]
        n = coord_results[c]["n_units"]
        verdict_s = coord_results[c]["status"]
        ax.set_title(f"{c}: failures {n_fail}/{n} -> {verdict_s}", fontsize=9)
        ax.set_ylabel("ns")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("fig02: stage tails vs frozen envelopes (validation)")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def fig03_e2e_validation(e2e_vals, bound_ns, baselines, risk_sum, out_path):
    """E2E tail with the frozen bound and baseline markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = sorted(v for v in e2e_vals if v != float("inf"))
    tail = min(len(finite), 400)
    y = finite[-tail:]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(len(y)), y, s=5, color="#1f77b4", label="E2E tail")
    ax.axhline(bound_ns, color="#d62728", lw=1.5,
               label=f"frozen B = {bound_ns} ns")
    b1 = baselines["B1_equal_risk_phase_aware"]["bound_ns"]
    b2 = baselines["B2_same_q_max_wait"]["bound_ns"]
    b3 = baselines["B3_calibration_e2e_order_statistic"]["bound_ns"]
    b4q = baselines["B4_marginal_resampling"]["quantile_bound_ns"]
    ax.axhline(b1, color="#9467bd", lw=1, ls="--", label=f"B1 = {b1} ns")
    if b3 is not None:
        ax.axhline(b3, color="#8c564b", lw=1, ls="-.",
                   label=f"B3 = {b3} ns")
    ax.axhline(b4q, color="#7f7f7f", lw=1, ls=":",
               label=f"B4 q = {b4q} ns (non-certifying)")
    ax.axhline(b2, color="#e377c2", lw=1, ls=":", label=f"B2 = {b2} ns")
    n_inf = sum(1 for v in e2e_vals if v == float("inf"))
    ax.set_title(f"fig03: validation E2E tail vs bounds "
                 f"(risk_sum {risk_sum:.4f}; lost={n_inf})", fontsize=10)
    ax.set_xlabel("rank (tail)")
    ax.set_ylabel("ns")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def fig04_selection_validity(feas_df, analysis_df, out_path):
    """Family-failure CP uppers per procedure/config vs the 0.05 threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    an = analysis_df[analysis_df["J"] == 4].copy()
    labels = (an["distribution_id"] + "/" + an["gate_regime"] + " " +
              an["procedure"])
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(an))
    colors = {"bonferroni_simultaneous": "#1f77b4",
              "split_select_certify": "#2ca02c"}
    ax.bar(x, an["cp_upper"], color=[colors.get(p, "#7f7f7f")
                                     for p in an["procedure"]])
    ax.scatter(x, an["family_rate"], color="k", s=18, zorder=3,
               label="observed family rate")
    ax.axhline(0.05, color="#d62728", ls="--", lw=1.2, label="threshold 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel("one-sided 95% CP upper of family failure")
    ax.set_title("fig04: post-selection validity (main config; bars = CP upper)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def fig05_dependence(dep_df, out_path):
    """Replay miss rates per coupling with union bound and baselines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 5))
    groups = ["OBSERVED_SYNCHRONOUS", "INDEPENDENT_PERMUTATION",
              "COMONOTONIC_RANK", "ANTITHETIC_RANK"]
    data = [dep_df[dep_df["coupling_id"] == g]["replay_miss_rate"].to_numpy()
            for g in groups]
    parts = ax.violinplot([d if len(d) else np.array([0.0]) for d in data],
                          positions=range(1, len(groups) + 1), showmeans=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    union = dep_df["union_bound"].astype(float)
    ax.axhline(union.max(), color="#d62728", ls="--", lw=1,
               label=f"max union certificate = {union.max():.4f}")
    indep = dep_df[dep_df["coupling_id"] == "INDEPENDENT_PERMUTATION"][
        "independence_estimate"]
    if len(indep):
        ax.axhline(float(np.median(indep)), color="#1f77b4", ls=":", lw=1.2,
                   label=f"median independence estimate = "
                         f"{float(np.median(indep)):.4f}")
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([g.replace("_", "\n") for g in groups], fontsize=7)
    ax.set_ylabel("replay miss rate P(model E2E > B)")
    ax.set_title("fig05: dependence re-coupling replay miss rates "
                 "(marginals preserved in every coupling)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def fig06_allocator(syn_df, out_path):
    """DP runtime and frontier growth vs chain length."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dp = syn_df[syn_df["method"] == "pareto_dp"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for menu_size, color in ((3, "#1f77b4"), (6, "#d62728")):
        g = dp[dp["menu_size"] == menu_size].groupby("stages")["runtime_ns"]
        mean = g.mean() / 1000.0
        ax1.plot(mean.index, mean.values, marker="o", ms=4, color=color,
                 label=f"menu={menu_size} (mean)")
        ax2.plot(mean.index,
                 dp[dp["menu_size"] == menu_size].groupby("stages")[
                     "frontier_peak"].mean().values,
                 marker="s", ms=4, color=color, label=f"menu={menu_size}")
    ax1.set_yscale("log")
    ax1.set_xlabel("stages")
    ax1.set_ylabel("DP runtime [us]")
    ax1.set_title("Pareto DP runtime vs chain length")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("stages")
    ax2.set_ylabel("frontier peak (states)")
    ax2.set_title("Pareto frontier peak vs chain length")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle("fig06: allocator scaling (synthetic menus, 20 seeds/cell)",
                 fontsize=10)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def main(argv=None):
    """Regenerate every figure from the derived data (spec 10, P6)."""
    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import yaml

    from .extract import event_coordinates, max_wait_ns, phase_bound_ns
    from .validate import clopper_pearson

    p = argparse.ArgumentParser(prog="parce_exp.plots")
    p.add_argument("--run-root", default="runs/e01")
    args = p.parse_args(argv)
    run_root = Path(args.run_root)
    root = Path(__file__).resolve().parents[2]
    fig_dir = run_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    # fig01: phase staircase
    sweep = pd.read_parquet(run_root / "derived" / "phase_sweep.parquet")
    agg = sweep[sweep["phase_bound_ns"].notna()].groupby(
        ["sweep", "phase_ns"]).agg(
        median_wait_ns=("wait_ns", "median"),
        median_observed_wait_ns=("observed_wait_ns", "median"),
        median_phase_bound_ns=("phase_bound_ns", "median"),
        median_maxwait_bound_ns=("maxwait_bound_ns", "median")).reset_index()
    frozen = yaml.safe_load((root / "configs" / "frozen_plan.yaml").read_text())
    cal = yaml.safe_load((root / "configs" / "calendar.yaml").read_text())
    phase_f = int(frozen["binding"]["phase_ns"])
    margin = float(frozen["pilot"]["final_margin_median_ns"])
    boundary = int(round(phase_f + margin)) % int(cal["period_ns"])
    pilot_summary = json.loads(
        (run_root / "derived" / "pilot_summary.json").read_text())
    frames = [pd.read_parquet(p) for p in
              sorted((run_root / "raw" / "events").glob("batch_*.parquet"))]
    events = pd.concat(frames, ignore_index=True)
    pc = event_coordinates(
        events[(events["split"] == "pilot")
               & events["run_id"].isin(pilot_summary["stages"]["C"]["valid"])
               & (~events["is_warmup"].astype(bool))
               & events["valid_event"].astype(bool)
               & events["t_rx_ns"].notna()])
    demo = {c: int(pc[c].median()) for c in ("X1", "X2", "X3", "X4")}
    windows = [(int(w["open_ns"]), int(w["close_ns"]))
               for w in cal["windows"]]
    grid = sorted(sweep["phase_ns"].unique().tolist())
    demo_curve = [(p, phase_bound_ns(demo["X1"], demo["X2"], demo["X3"],
                                     demo["X4"],
                                     int(cal["service_demand_ns"]),
                                     int(cal["period_ns"]), windows)[1])
                  for p in grid]
    made.append(fig01_phase_staircase(
        sweep, agg, demo_curve,
        max_wait_ns(int(cal["period_ns"]), int(cal["service_demand_ns"]),
                    windows),
        fig_dir / "fig01_phase_staircase.svg", boundary))

    # fig02 / fig03: validation
    design = json.loads((run_root / "derived" / "selected_design.json").read_text())
    baselines = json.loads((run_root / "derived" / "baselines.json").read_text())
    samples = pd.read_parquet(run_root / "derived" / "stage_samples.parquet")
    val = samples[samples["split"] == "validation"]
    coords = ["X1", "X2", "X3", "X4"]
    per_run = {}
    for coord in coords + ["E2E"]:
        col = val[val["coordinate_id"] == coord]
        for r in col.itertuples():
            per_run.setdefault(int(r.run_id), {})[coord] = (
                float("inf") if r.is_infinite else float(r.value_ns))
    results = pd.read_parquet(run_root / "derived" / "validation_results.parquet")
    q_sel = {c: int(design["selected_q_ns"][c]) for c in coords}
    made.append(fig02_stage_validation(
        {c: [per_run[r][c] for r in per_run] for c in coords}, q_sel,
        {c: dict(results[results.item_id == c].iloc[0]) for c in coords},
        fig_dir / "fig02_stage_validation.svg"))
    made.append(fig03_e2e_validation(
        [per_run[r]["E2E"] for r in per_run], int(design["bound_ns"]),
        baselines, float(design["risk_sum"]),
        fig_dir / "fig03_e2e_validation.svg"))

    # fig04: selection validity
    sel = pd.read_parquet(run_root / "derived" / "selection_validity.parquet")
    feas = sel[sel["selected_content_failure"].notna()]
    analysis = []
    for (d, rg, m, J, R, n), g in feas.groupby(
            ["distribution_id", "gate_regime", "procedure", "J", "R", "n"]):
        if m == "pointwise_95_then_select":
            continue
        mm = int(g["family_failure"].astype(bool).sum())
        lo, hi = clopper_pearson(mm, len(g), 0.05)
        analysis.append({"distribution_id": d, "gate_regime": rg,
                         "procedure": m, "J": J, "R": R, "n": n,
                         "reps": len(g), "family_failures": mm,
                         "family_rate": mm / len(g), "cp_lower": lo,
                         "cp_upper": hi})
    made.append(fig04_selection_validity(
        feas, pd.DataFrame(analysis), fig_dir / "fig04_selection_validity.svg"))

    # fig05: dependence
    dep = pd.read_parquet(run_root / "derived" / "dependence_results.parquet")
    made.append(fig05_dependence(dep, fig_dir / "fig05_dependence.svg"))

    # fig06: allocator
    alloc = pd.read_parquet(run_root / "derived" / "allocator_results.parquet")
    made.append(fig06_allocator(alloc[alloc["source"] == "synthetic"],
                                fig_dir / "fig06_allocator.svg"))
    for m in made:
        print(f"regenerated {m}")


if __name__ == "__main__":
    main()
