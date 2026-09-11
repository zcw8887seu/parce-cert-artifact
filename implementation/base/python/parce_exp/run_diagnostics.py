"""Run-order and batch diagnostics for calibration/validation (e01-s1 S3).

Reads only existing data (configs/run_plan.csv, derived/stage_samples.parquet,
derived/selected_design.json), orders each split by run_plan.csv run_order and
tests for lag-1 correlation; on trigger, runs a moving-block bootstrap and
flags H3 for correlation sensitivity (supplement.MD 8).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .selection_sim import write_stop_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COORDS = ["X1", "X2", "X3", "X4", "E2E"]
SPLITS = ["calibration", "validation"]
BATCH_SIZE = 25
MBB_BLOCKS = (25, 50, 100)
MBB_REPS = 10_000
MBB_SEED = 20260817

DIAG_COLUMNS = [
    "split", "coordinate_id", "n_units", "lag1_pearson", "lag1_spearman",
    "trigger_threshold", "correlation_triggered", "batch_size",
    "superblock_sensitivity", "status",
]
BATCH_COLUMNS = [
    "split", "coordinate_id", "batch_index", "run_order_start",
    "run_order_end", "n_values", "n_infinite", "batch_median_ns",
    "batch_maximum_ns", "envelope_failures",
]


def lag1_correlations(values):
    """Lag-1 Pearson/Spearman on the finite subsequence (NaN-safe)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan"), float("nan")
    a, b = v[:-1], v[1:]
    pearson = float("nan")
    if np.std(a) > 0 and np.std(b) > 0:
        pearson = float(np.corrcoef(a, b)[0, 1])
    spearman = float("nan")
    if np.std(a) > 0 and np.std(b) > 0:
        spearman = float(stats.spearmanr(a, b).statistic)
    return pearson, spearman


def mbb_failure_upper(fail_indicator, block, reps, seed):
    """Moving-block bootstrap one-sided 95% upper bound on the failure rate."""
    f = np.asarray(fail_indicator, dtype=float)
    N = len(f)
    n_blocks = N // block
    if n_blocks == 0:
        return float("nan")
    rng = np.random.default_rng(np.random.SeedSequence(entropy=(seed, block)))
    starts = rng.integers(0, N - block + 1, size=(reps, n_blocks))
    idx = starts[..., None] + np.arange(block)
    rates = f[idx].mean(axis=(1, 2))
    return float(np.quantile(rates, 0.95))


def make_fig07(diag, batches, raw, thresholds, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(COORDS), len(SPLITS),
                             figsize=(12.5, 13.0), sharex=False)
    for c_i, coord in enumerate(COORDS):
        for s_i, split in enumerate(SPLITS):
            ax = axes[c_i, s_i]
            r = raw[(raw["split"] == split) & (raw["coordinate_id"] == coord)]
            ax.plot(r["run_order"], r["value_ns"], lw=0.4, alpha=0.6,
                    label="raw value")
            b = batches[(batches["split"] == split)
                        & (batches["coordinate_id"] == coord)]
            mid = (b["run_order_start"] + b["run_order_end"] - 1) / 2
            ax.plot(mid, b["batch_median_ns"], drawstyle="steps-mid", lw=1.0,
                    label="25-run batch median")
            ax.plot(mid, b["batch_maximum_ns"], drawstyle="steps-mid", lw=1.0,
                    label="25-run batch maximum")
            ax.axhline(thresholds[coord], color="red", ls="--", lw=1.0,
                       label=("frozen q" if coord != "E2E" else "E2E bound B"))
            d = diag[(diag["split"] == split) & (diag["coordinate_id"] == coord)]
            if len(d) and bool(d["correlation_triggered"].iloc[0]):
                ax.set_facecolor("#fff2f2")
            ax.set_title(f"{split}/{coord}", fontsize=9)
            if c_i == 0 and s_i == 0:
                ax.legend(fontsize=6)
    for s_i, split in enumerate(SPLITS):
        axes[-1, s_i].set_xlabel("run order")
    for c_i in range(len(COORDS)):
        axes[c_i, 0].set_ylabel("ns")
    fig.suptitle("fig07: run-order diagnostics (raw values, 25-run batch "
                 "median/maximum, frozen envelope q / E2E bound B)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(prog="parce_exp.run_diagnostics")
    p.add_argument("--run-root", default="runs/e01")
    p.add_argument("--supplement-id", default="e01-s1")
    args = p.parse_args(argv)

    run_root = Path(args.run_root)
    plan = pd.read_csv(PROJECT_ROOT / "configs" / "run_plan.csv")
    samples = pd.read_parquet(run_root / "derived" / "stage_samples.parquet")
    design = json.loads((run_root / "derived" / "selected_design.json").read_text())

    thresholds = {k: int(v) for k, v in design["selected_q_ns"].items()}
    thresholds["E2E"] = int(design["bound_ns"])

    df = samples[samples["split"].isin(SPLITS)
                 & samples["coordinate_id"].isin(COORDS)].merge(
        plan[["run_id", "run_order"]], on="run_id", how="inner",
        validate="many_to_one")
    missing = (set(plan[plan["split"].isin(SPLITS)]["run_id"])
               - set(samples["run_id"]))
    if missing:
        raise SystemExit(f"stage_samples missing {len(missing)} planned runs")

    diag_rows, batch_rows, raw_rows = [], [], []
    for split in SPLITS:
        sub = df[df["split"] == split].sort_values("run_order")
        for coord in COORDS:
            g = sub[sub["coordinate_id"] == coord]
            values = g["value_ns"].to_numpy(dtype=float)
            infs = g["is_infinite"].to_numpy(dtype=bool)
            finite = np.where(np.isfinite(values), values, np.nan)
            n_units = int(len(g))
            thr = max(0.10, 3.0 / np.sqrt(n_units))
            pearson, spearman = lag1_correlations(finite)
            triggered = bool(
                (np.isfinite(pearson) and abs(pearson) >= thr)
                or (np.isfinite(spearman) and abs(spearman) >= thr))
            threshold = thresholds[coord]
            exceeds = np.where(infs, True,
                               np.where(np.isfinite(values),
                                        values > threshold, True))
            raw_rows.append(pd.DataFrame({
                "split": split, "coordinate_id": coord,
                "run_order": g["run_order"].to_numpy(),
                "value_ns": np.where(infs, np.nan, values),
                "envelope_failure": exceeds,
            }))
            for b_i in range(0, n_units, BATCH_SIZE):
                seg_v = finite[b_i:b_i + BATCH_SIZE]
                seg_f = exceeds[b_i:b_i + BATCH_SIZE]
                batch_rows.append({
                    "split": split, "coordinate_id": coord,
                    "batch_index": b_i // BATCH_SIZE,
                    "run_order_start": int(g["run_order"].iloc[b_i]),
                    "run_order_end": int(
                        g["run_order"].iloc[min(b_i + BATCH_SIZE, n_units) - 1]),
                    "n_values": int(np.isfinite(seg_v).sum()),
                    "n_infinite": int(infs[b_i:b_i + BATCH_SIZE].sum()),
                    "batch_median_ns": (float(np.nanmedian(seg_v))
                                        if np.isfinite(seg_v).any() else None),
                    "batch_maximum_ns": (float(np.nanmax(seg_v))
                                         if np.isfinite(seg_v).any() else None),
                    "envelope_failures": int(seg_f.sum()),
                })
            if triggered:
                parts = []
                for block in MBB_BLOCKS:
                    up = mbb_failure_upper(exceeds, block, MBB_REPS,
                                           MBB_SEED + SPLITS.index(split)
                                           * 100 + COORDS.index(coord))
                    parts.append(f"block{block}:95upper={up:.5f}")
                sb = "TRIGGERED_MBB[" + "; ".join(parts) + "]"
                status = "CORRELATION_TRIGGERED"
            else:
                sb = "NOT_TRIGGERED"
                status = "OK"
            diag_rows.append({
                "split": split, "coordinate_id": coord, "n_units": n_units,
                "lag1_pearson": pearson, "lag1_spearman": spearman,
                "trigger_threshold": float(thr),
                "correlation_triggered": triggered,
                "batch_size": BATCH_SIZE, "superblock_sensitivity": sb,
                "status": status,
            })

    diag = pd.DataFrame(diag_rows, columns=DIAG_COLUMNS)
    batches = pd.DataFrame(batch_rows, columns=BATCH_COLUMNS)
    raw = pd.concat(raw_rows, ignore_index=True)

    supp = run_root / "supplement"
    derived = supp / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    diag.to_parquet(derived / "run_order_diagnostics.parquet", index=False)
    batches.to_parquet(derived / "run_order_batches.parquet", index=False)

    fig_path = supp / "figures" / "fig07_run_order_diagnostics.svg"
    make_fig07(diag, batches, raw, thresholds, fig_path)

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    L = ["# 03 Run-Order and Batch Diagnostics (supplement e01-s1, S3)", "",
         f"- generated: {now}",
         f"- inputs: configs/run_plan.csv (run_order), derived/"
         "stage_samples.parquet, derived/selected_design.json",
         f"- trigger rule: |lag1 corr| >= max(0.10, 3/sqrt(n_units))",
         f"- batch size: {BATCH_SIZE} runs; envelope failures use the frozen "
         "q (X1-X4) or the frozen E2E bound B; infinite values count as "
         "failures",
         "",
         "| split | coordinate | n | lag1 pearson | lag1 spearman | "
         "threshold | triggered | superblock sensitivity | status |",
         "|---|---|---:|---:|---:|---:|---|---|---|"]
    for _, r in diag.iterrows():
        L.append(
            f"| {r['split']} | {r['coordinate_id']} | {int(r['n_units'])} | "
            f"{r['lag1_pearson']:.4f} | {r['lag1_spearman']:.4f} | "
            f"{r['trigger_threshold']:.4f} | "
            f"{bool(r['correlation_triggered'])} | "
            f"{r['superblock_sensitivity']} | {r['status']} |")
    L += ["", "## Batch statistics summary (25-run batches)", "",
          "| split | coordinate | batches | batch median range [ns] | "
          "max of batch maxima [ns] | total envelope failures |", "|---|---|"
          "---:|---|---:|---:|"]
    for split in SPLITS:
        for coord in COORDS:
            b = batches[(batches["split"] == split)
                        & (batches["coordinate_id"] == coord)]
            med = b["batch_median_ns"].dropna()
            L.append(
                f"| {split} | {coord} | {len(b)} | "
                f"[{med.min():.0f}, {med.max():.0f}] | "
                f"{b['batch_maximum_ns'].dropna().max():.0f} | "
                f"{int(b['envelope_failures'].sum())} |")
    any_trigger = bool(diag["correlation_triggered"].any())
    if any_trigger:
        L += ["",
              "**H3 status: INCONCLUSIVE_CORRELATION_SENSITIVITY** - at least "
              "one coordinate shows run-order correlation above the trigger "
              "threshold. Moving-block bootstrap results are in the "
              "superblock_sensitivity column. No physical re-acquisition is "
              "performed automatically; a decision on new split-session "
              "acquisition is required.",
              ""]
    else:
        L += ["",
              "superblock_sensitivity = NOT_TRIGGERED for every split/"
              "coordinate: no moving-block bootstrap was required and the "
              "original Clopper-Pearson validation stands as reported.",
              "",
              "Data: derived/run_order_diagnostics.parquet, "
              "derived/run_order_batches.parquet; figure: figures/"
              "fig07_run_order_diagnostics.svg", ""]
    (supp / "reports").mkdir(parents=True, exist_ok=True)
    (supp / "reports" / "03_RUN_ORDER_DIAGNOSTICS.md").write_text(
        "\n".join(L) + "\n")

    print(diag.to_string(index=False))
    if any_trigger:
        reason = ("run-order correlation triggered for: "
                  + ", ".join(f"{r['split']}/{r['coordinate_id']}"
                              for _, r in diag[diag["correlation_triggered"]]
                              .iterrows())
                  + "; H3=INCONCLUSIVE_CORRELATION_SENSITIVITY; awaiting "
                  "user decision on new split-session acquisition")
        write_stop_report(run_root, args.supplement_id, reason, "S3")
        raise SystemExit(f"STOPPED (supplement.MD 8.2): {reason}")
    print("diagnostics: no correlation trigger; H3 unchanged")


if __name__ == "__main__":
    import sys
    sys.exit(main())
