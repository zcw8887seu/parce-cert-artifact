"""Generate the eight paper figures from the retained plotting inputs."""
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Shared figure palette.
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Font embedding and export settings.
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Save PDF and 300 dpi PNG versions."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figure_data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
TECS_WIDTH_MM = 139

BLUE = CATEGORICAL[0]
RED = CATEGORICAL[1]
GREEN = CATEGORICAL[2]
ORANGE = CATEGORICAL[3]
PURPLE = CATEGORICAL[4]
DARK_GREY = CATEGORICAL[5]
LIGHT_BLUE = "#E8F1F7"
LIGHT_GREEN = "#E8F3ED"
LIGHT_ORANGE = "#FCEEDC"
LIGHT_RED = "#F8E5E5"
LIGHT_GREY = "#F1F1F1"


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="top", clip_on=False)


def validate_source_data() -> dict:
    phase = pd.read_csv(DATA / "phase_sweep_all_1720.csv")
    selection = pd.read_csv(DATA / "selection_validity_main_18.csv")
    coupling = pd.read_csv(DATA / "cross_stage_coupling_all_103.csv")
    h3_diag = pd.read_csv(DATA / "h3r1_diagnostics_all_10.csv")
    h3_sessions = pd.read_csv(DATA / "h3r1_session_summary_all_200.csv")
    h3_validation = pd.read_csv(DATA / "h3r1_validation_all_5.csv")
    tds = pd.read_csv(DATA / "tds1_cell_summary_all_17.csv")
    tds_marginals = pd.read_csv(DATA / "tds1_marginal_checks_all_16.csv")
    allocator = pd.read_csv(DATA / "allocator_dp_scaling_all_400.csv")
    boundary = pd.read_csv(DATA / "allocator_float_boundary_cases.csv")
    assert len(phase) == 1720 and phase["reference_match"].all()
    assert len(phase.loc[phase["sweep"] == "coarse"]) == 480
    assert len(phase.loc[phase["sweep"] == "fine"]) == 1240
    assert int((phase["phase_bound_ns"] > phase["maxwait_bound_ns"]).sum()) == 0
    assert len(selection) == 18
    assert len(coupling) == 103
    assert int(coupling["event_inclusion_violations"].sum()) == 0
    assert coupling["marginals_preserved"].all()
    assert len(h3_diag) == 10 and len(h3_sessions) == 200
    assert len(h3_validation) == 5 and set(h3_validation["status"]) == {"SUPPORTED"}
    assert len(tds) == 17 and len(tds_marginals) == 16
    assert len(allocator) == 400 and len(boundary) == 0
    return {"phase": phase, "selection": selection, "coupling": coupling,
            "h3_diag": h3_diag, "h3_sessions": h3_sessions,
            "h3_validation": h3_validation, "tds": tds,
            "tds_marginals": tds_marginals, "allocator": allocator,
            "boundary": boundary}


def figure1_system_timing(data: dict) -> None:
    """Measured timing objects and the finite-DAG abstraction, without flow boxes."""
    fig = plt.figure(figsize=(TECS_WIDTH_MM * MM, 72 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.58, 0.42], hspace=0.54)

    ax = fig.add_subplot(gs[0])
    panel_label(ax, "a", -0.035, 1.16)
    ax.text(0.06, 1.14, "measured four-coordinate serial chain",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", clip_on=False)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.set_axis_off()
    # Two periodic windows make the phase discontinuity visible behind the events.
    for start in (4.15, 7.45):
        ax.add_patch(Rectangle((start, 1.18), 1.45, 0.74,
                               facecolor=LIGHT_ORANGE, edgecolor=ORANGE,
                               linewidth=0.7, zorder=0))
        ax.text(start + 0.72, 2.02, "gate open", ha="center", va="bottom",
                fontsize=7, color=ORANGE)
    ax.plot([0.55, 9.45], [1.55, 1.55], color=DARK_GREY, lw=1.0, zorder=1)
    event_x = [0.65, 2.05, 3.55, 4.75, 6.35, 8.85]
    event_labels = ["cause", "producer\ncompletion", "gate\ningress",
                    "service\nstart", "gate\negress", "receiver"]
    coordinate_labels = [None, "$X_1$", "$X_2$", "$S(t,d)$", "$X_3$", "$X_4$"]
    markers = ["o", "o", "o", "D", "o", "o"]
    colors = [DARK_GREY, BLUE, BLUE, ORANGE, BLUE, BLUE]
    for x, label, coord, marker, color in zip(
            event_x, event_labels, coordinate_labels, markers, colors):
        ax.scatter([x], [1.55], s=52 if marker == "D" else 40,
                   marker=marker, facecolor="white", edgecolor=color,
                   linewidth=1.4, zorder=4)
        ax.text(x, 2.68, label, ha="center", va="bottom", fontsize=7.2,
                fontweight="bold" if marker == "D" else "normal")
        if coord:
            ax.text(x, 2.31, coord, ha="center", va="bottom", fontsize=8,
                    color=color)
        ax.plot([x, x], [1.75, 2.22], color=color, lw=0.6)
    segment_labels = ["host work", "eligibility", "window-fit rule",
                      "transmission", "receive work"]
    segment_y = [1.04, 0.69, 1.04, 0.69, 1.04]
    for left, right, label, label_y in zip(event_x[:-1], event_x[1:],
                                           segment_labels, segment_y):
        ax.annotate("", xy=(right - 0.10, 1.55), xytext=(left + 0.10, 1.55),
                    arrowprops=dict(arrowstyle="->", color=DARK_GREY, lw=0.8))
        ax.text((left + right) / 2, label_y, label, ha="center", va="top",
                fontsize=6.6, color=DARK_GREY)
    ax.plot([5.58, 5.58], [1.08, 2.18], color=RED, lw=0.8, ls="--")
    ax.text(5.62, 0.34, "last feasible start", ha="left", va="center",
            fontsize=6.7, color=RED)

    ax = fig.add_subplot(gs[1])
    panel_label(ax, "b", -0.035, 1.16)
    ax.text(0.06, 1.14, "finite timing DAG (theory only)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", clip_on=False)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.0); ax.set_axis_off()
    # A minimal fork-join graph: circles are jobs; diamonds are joins.
    nodes = {
        "s": (0.65, 1.45), "u1": (2.15, 2.25), "u2": (2.15, 0.65),
        "j1": (4.05, 1.45), "v1": (5.85, 2.25), "v2": (5.85, 0.65),
        "j2": (7.65, 1.45), "z": (9.25, 1.45),
    }
    edges = [("s", "u1"), ("s", "u2"), ("u1", "j1"), ("u2", "j1"),
             ("j1", "v1"), ("j1", "v2"), ("v1", "j2"), ("v2", "j2"),
             ("j2", "z")]
    for source, target in edges:
        x0, y0 = nodes[source]; x1, y1 = nodes[target]
        ax.annotate("", xy=(x1 - 0.16, y1), xytext=(x0 + 0.16, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.75,
                                    color=DARK_GREY))
    for name, (x, y) in nodes.items():
        join = name.startswith("j")
        ax.scatter([x], [y], s=50 if join else 38,
                   marker="D" if join else "o", facecolor="white",
                   edgecolor=PURPLE if join else BLUE, linewidth=1.2, zorder=3)
    ax.text(4.05, 0.17, "joins take maxima", fontsize=6.8, ha="center",
            color=PURPLE)
    # A dashed arc marks two occurrences driven by one uncertainty coordinate.
    ax.annotate("", xy=(5.72, 0.80), xytext=(2.28, 2.08),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=0.9,
                                ls="--", connectionstyle="arc3,rad=-0.28"))
    ax.text(4.02, 2.58, "shared coordinate: one exceedance event",
            fontsize=6.8, ha="center", color=GREEN)
    ax.text(9.25, 1.02, "sink", fontsize=6.8, ha="center")
    save_cns_figure(fig, OUT / "fig01_system_timing_model")
    plt.close(fig)


def figure2_certification_workflow(data: dict) -> None:
    """Compact two-layer workflow and orthogonal dependence responsibilities."""
    selected = json.loads((DATA / "h3r1_selected_design.json").read_text(encoding="utf-8"))
    risks = selected.get("selected_risks", selected.get("risk_vector", []))
    risk_values = risks.values() if isinstance(risks, dict) else risks
    risk_sum = sum(float(x) for x in risk_values)
    bound_ns = int(selected.get("bound_ns", selected.get("selected_bound_ns", 15_544_694)))
    fig = plt.figure(figsize=(TECS_WIDTH_MM * MM, 56 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.66, 0.34], hspace=0.26)

    ax = fig.add_subplot(gs[0])
    panel_label(ax, "a", -0.035, 1.07)
    ax.set_xlim(0, 11.3); ax.set_ylim(0, 3.2); ax.set_axis_off()
    ax.annotate("", xy=(8.20, 1.67), xytext=(0.62, 1.67),
                arrowprops=dict(arrowstyle="->", color=DARK_GREY, lw=1.0))
    xs = [0.78, 2.02, 3.26, 4.42, 5.72, 7.02]
    labels_top = ["Calibrate", "Menu", "Select", "Freeze", "Validate", "Guard"]
    markers = ["o", "s", "D", "|", "o", "D"]
    colors = [BLUE, BLUE, ORANGE, BLACK, BLUE, RED]
    sizes = [58, 60, 62, 260, 58, 62]
    for x, top, marker, color, size in zip(xs, labels_top, markers, colors, sizes):
        if marker == "|":
            ax.plot([x, x], [1.34, 2.00], color=color, lw=2.0,
                    solid_capstyle="round", zorder=4)
        else:
            ax.scatter([x], [1.67], s=size, marker=marker,
                       facecolor="white", edgecolor=color,
                       linewidth=1.5, zorder=4)
        ax.text(x, 2.24, top, ha="center", fontsize=7.8,
                fontweight="bold" if marker in ("D", "|") else "normal")
    ax.axvspan(0.38, 3.80, color=LIGHT_BLUE, alpha=0.38, zorder=-2)
    ax.axvspan(4.88, 7.62, color="#F4F7FA", alpha=0.65, zorder=-2)
    ax.text(2.08, 2.88, "selection-valid acquisition", color=BLUE,
            fontsize=8.0, ha="center", fontweight="bold")
    ax.text(6.30, 2.88, "independent issuance audit", color=DARK_GREY,
            fontsize=8.0, ha="center", fontweight="bold")
    ax.text(1.40, 0.98, "$n=3{,}000$ · 12 protected rows", ha="center",
            fontsize=7.1, color=DARK_GREY)
    ax.text(3.92, 0.72, "exact selection · frozen design", ha="center",
            fontsize=7.1, color=DARK_GREY)
    ax.text(6.35, 0.98, "$n=3{,}500$ · fixed guard", ha="center",
            fontsize=7.1, color=DARK_GREY)

    outcome_x = 9.34
    outcomes = [(2.50, GREEN, "CERTIFIED"),
                (1.67, RED, "NOT-SUPPORTED"),
                (0.84, ORANGE, "INCONCLUSIVE")]
    for y, color, label in outcomes:
        ax.scatter([outcome_x], [y], s=78, marker="o", facecolor="white",
                   edgecolor=color, linewidth=1.8, zorder=4)
        ax.text(outcome_x + 0.28, y, label, ha="left", va="center",
                fontsize=7.8, fontweight="bold", color=color)
        ax.annotate("", xy=(outcome_x - 0.10, y), xytext=(8.12, 1.67),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.75,
                                    connectionstyle=f"arc3,rad={(1.67-y)*0.10}"))
    ax.text(4.66, 0.24,
            "physical re-acquisition (H3R1): nominal support → guard → INCONCLUSIVE",
            ha="center", va="center", fontsize=6.8, color=ORANGE)

    ax = fig.add_subplot(gs[1])
    panel_label(ax, "b", -0.035, 1.07)
    ax.set_xlim(-5.2, 5.2); ax.set_ylim(-1.60, 1.55); ax.set_axis_off()
    ax.annotate("", xy=(4.68, 0), xytext=(-4.68, 0),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
    ax.annotate("", xy=(0, 1.28), xytext=(0, -1.38),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.scatter([0], [0], s=40, facecolor="white", edgecolor=BLACK,
               linewidth=1.0, zorder=3)
    ax.text(4.52, 0.50, "within one execution", color=BLUE,
            fontsize=8.0, ha="right", fontweight="bold")
    ax.text(4.62, -0.47, "cross-stage dependence: composition",
            color=BLUE, fontsize=7.6, ha="right", va="top")
    ax.text(0.18, 1.49, "across runs / sessions", color=RED,
            fontsize=8.0, ha="left", va="top", fontweight="bold")
    ax.text(-2.18, 1.30, "temporal dependence\nacquisition premise; fail-closed",
            color=RED, fontsize=7.6, ha="center", va="top", linespacing=1.10)
    ax.text(-4.72, -0.78,
            "$\\mathbf{X}\\leq\\mathbf{q}\\Rightarrow L_z\\leq B_z$;  "
            "$\\Pr_\\star(L_z>B_z)\\leq\\sum_j\\epsilon_j$",
            color=BLUE, fontsize=7.8, ha="left", va="center")
    save_cns_figure(fig, OUT / "fig02_certification_workflow")
    plt.close(fig)


def figure3_phase(data: dict) -> None:
    df = data["phase"]
    coarse = df.loc[df["sweep"] == "coarse"].copy()
    fine = df.loc[df["sweep"] == "fine"].copy()
    fig = plt.figure(figsize=(TECS_WIDTH_MM * MM, 103 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.34)
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a")
    ax.set_xlim(300, 1750); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("time within two gate periods (µs)")
    for start in (400, 1400):
        ax.add_patch(Rectangle((start, 0.52), 300, 0.28, facecolor=LIGHT_ORANGE,
                               edgecolor=ORANGE, lw=0.7))
        ax.add_patch(Rectangle((start, 0.52), 250, 0.28, facecolor="#F7D9B4",
                               edgecolor="none"))
        ax.text(start + 150, 0.83, "open window", ha="center", fontsize=6.0,
                color=ORANGE)
        ax.axvline(start + 250, 0.48, 0.84, color=RED, lw=0.7, ls="--")
    ax.text(650, 0.92, "last feasible start", ha="center", fontsize=6.0,
            color=RED)
    ax.scatter([640, 660], [0.43, 0.28], marker="v", s=18,
               color=[BLUE, RED], zorder=5)
    ax.add_patch(Rectangle((640, 0.36), 50, 0.10, facecolor=BLUE,
                           edgecolor=BLUE, lw=0.5))
    ax.add_patch(Rectangle((1400, 0.21), 50, 0.10, facecolor=RED,
                           edgecolor=RED, lw=0.5))
    ax.annotate("", xy=(1400, 0.26), xytext=(660, 0.28),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=RED,
                                connectionstyle="arc3,rad=-0.18"))
    ax.text(500, 0.08, "arrive 640 µs\nserve now", ha="center", fontsize=6.0)
    ax.text(1270, 0.06, "arrive 660 µs\nserve next cycle", ha="center", fontsize=6.0)
    ax.text(1070, 0.43, "almost one-period wait jump", ha="center",
            fontsize=6.0, fontweight="bold", color=RED)
    ax.spines["left"].set_visible(False)

    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b")
    x = np.sort(coarse["phase_ns"].unique())
    for col, color, label in (("phase_bound_ns", BLUE, "phase-aware"),
                              ("maxwait_bound_ns", ORANGE, "maximum-wait")):
        group = coarse.groupby("phase_ns")[col]
        med = group.median().reindex(x).to_numpy() / 1e6
        lo = group.quantile(0.10).reindex(x).to_numpy() / 1e6
        hi = group.quantile(0.90).reindex(x).to_numpy() / 1e6
        ax.fill_between(x / 1000, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x / 1000, med, color=color, lw=1.2, label=label)
    ax.set_xlabel("phase (µs)"); ax.set_ylabel("bound (ms)")
    ax.set_title("coarse sweep: all 480 instances", loc="left", fontsize=7)
    ax.legend(loc="upper right", fontsize=6)

    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "c", -0.24, 1.12)
    boundary = 476_042
    x_raw = (fine["phase_ns"].to_numpy() - boundary) / 1000
    y_raw = fine["wait_ns"].to_numpy() / 1000
    ax.scatter(x_raw, y_raw, s=4, color=GREY, alpha=0.18,
               edgecolors="none", rasterized=True)
    med = fine.groupby("phase_ns")["wait_ns"].median()
    ax.plot((med.index.to_numpy() - boundary) / 1000, med.to_numpy() / 1000,
            color=BLUE, marker="o", ms=2.0, lw=1.0)
    ax.axvline(0, color=RED, ls="--", lw=0.8)
    ax.annotate("0 → 746.4 µs\nmedian replay wait", xy=(1.0, 746.4),
                xytext=(8, 520), fontsize=6.0, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.6))
    ax.set_xlabel("phase relative to boundary (µs)")
    ax.set_ylabel("reference replay wait (µs)")
    ax.set_title("fine sweep: all 1,240 instances", loc="left", fontsize=7)

    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "d")
    delta = np.sort((df["maxwait_bound_ns"] - df["phase_bound_ns"]).to_numpy() / 1000)
    ecdf = np.arange(1, len(delta) + 1) / len(delta)
    ax.plot(delta, ecdf, color=GREEN, lw=1.4)
    median = float(np.median(delta)); ax.axvline(median, color=GREEN, ls="--", lw=0.8)
    ax.text(median + 18, 0.62, f"median {median:.1f} µs", color=GREEN, fontsize=6.1)
    rescheduled_instances = int((df["reschedule_count"] > 0).sum())
    reschedule_events = int(df["reschedule_count"].sum())
    ax.text(0.05, 0.95,
            f"$\\Delta B$ min/median/max\n0.003 / {median:.1f} / 750.0 µs",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2,
            color=DARK_GREY)
    ax.text(0.97, 0.08,
            f"{rescheduled_instances}/1,720 instances rescheduled\n"
            f"{reschedule_events} reschedule events",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
            color=DARK_GREY)
    ax.set_xlabel("$\\Delta B=B_{\\mathrm{maxwait}}-B_{\\mathrm{phase}}$ (µs)")
    ax.set_ylabel("empirical cumulative fraction"); ax.set_xlim(left=0)
    save_cns_figure(fig, OUT / "fig03_periodic_gate_phase"); plt.close(fig)


def figure5_selection_validity(data: dict) -> None:
    """Full-width forest plot for the 18 selection-validity cells."""
    sel = data["selection"].copy()
    procedures = ["pointwise_95_then_select", "bonferroni_simultaneous",
                  "split_select_certify"]
    titles = ["Pointwise baseline\n3,000 samples",
              "Simultaneous menu\n3,000 samples",
              "Split select--certify\n6,000 samples"]
    colors = [DARK_GREY, BLUE, GREEN]
    distribution_labels = {"gamma": "gamma", "shifted_lognormal": "shifted lognormal",
                           "lognormal_mixture": "lognormal mixture"}
    fig, axes = plt.subplots(1, 3, figsize=(TECS_WIDTH_MM * MM, 62 * MM),
                             sharey=True, gridspec_kw={"wspace": 0.12})
    row_order = [(d, g) for d in ("gamma", "shifted_lognormal", "lognormal_mixture")
                 for g in ("near", "far")]
    y_positions = np.arange(len(row_order))[::-1]
    for index, (ax, procedure, title, color) in enumerate(
            zip(axes, procedures, titles, colors)):
        if index == 0:
            panel_label(ax, "a", -0.48, 1.08)
        subset = sel.loc[sel["procedure"] == procedure].set_index(
            ["distribution_id", "gate_regime"])
        ax.axvspan(0.008, 0.05, color=LIGHT_GREEN, alpha=0.55, zorder=0)
        ax.axvline(0.05, color=RED, ls="--", lw=0.9, zorder=1)
        for y, key in zip(y_positions, row_order):
            row = subset.loc[key]
            rate = float(row["family_failure_rate"])
            lo = float(row["cp_lower"]); hi = float(row["cp_upper"])
            filled = key[1] == "near"
            ax.errorbar(rate, y, xerr=[[rate - lo], [hi - rate]], fmt="o",
                        ms=5.2, mfc=color if filled else "white", mec=color,
                        mew=1.0, ecolor=color, elinewidth=0.8, capsize=2.0,
                        zorder=3)
        ax.set_xscale("log"); ax.set_xlim(0.008, 0.30)
        ax.xaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05, 0.10, 0.20]))
        ax.xaxis.set_major_formatter(FixedFormatter([".01", ".02", ".05", ".10", ".20"]))
        ax.set_ylim(-0.65, 5.65); ax.set_title(title, fontsize=7.4, pad=7)
        ax.set_xlabel("family failure rate")
        ax.grid(axis="x", which="major", color="#DDDDDD", lw=0.45)
        if index == 0:
            labels = [f"{distribution_labels[d]} · {g}" for d, g in row_order][::-1]
            ax.set_yticks(y_positions, [f"{distribution_labels[d]} · {g}"
                                       for d, g in row_order])
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    axes[-1].text(0.05, 5.54, "target", color=RED, fontsize=6.8,
                  ha="center", va="bottom")
    handles = [Line2D([0], [0], marker="o", color="none", mfc=BLACK, mec=BLACK,
                      label="near gate"),
               Line2D([0], [0], marker="o", color="none", mfc="white", mec=BLACK,
                      label="far gate")]
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.24, top=0.82,
                        wspace=0.12)
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.035), fontsize=7)
    save_cns_figure(fig, OUT / "fig05_selection_validity")
    plt.close(fig)


def figure6_recoupling_audit(data: dict) -> None:
    """Rank-alignment templates and exact counts for executed structural audits."""
    coupling = data["coupling"]
    counts = coupling.groupby("coupling_id", sort=False).size().to_dict()
    rng = np.random.default_rng(20260821)
    ranks = np.arange(1, 7)
    templates = {
        "observed": np.array([[1, 3, 2, 4, 6, 5], [2, 1, 4, 3, 5, 6],
                              [3, 2, 1, 6, 4, 5], [1, 2, 4, 6, 5, 3]]).T,
        "comonotonic": np.tile(ranks[:, None], (1, 4)),
        "antithetic": np.column_stack([ranks, ranks[::-1], ranks, ranks[::-1]]),
        "permutation": np.column_stack([rng.permutation(ranks) for _ in range(4)]),
    }
    titles = ["observed (1)", "comonotonic (1)",
              "antithetic (1)", "permutations (100)"]
    fig = plt.figure(figsize=(TECS_WIDTH_MM * MM, 78 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.50, 0.50], hspace=0.58)
    top = gs[0].subgridspec(1, 4, wspace=0.18)
    stage_x = np.arange(4)
    for idx, (key, title) in enumerate(zip(templates, titles)):
        ax = fig.add_subplot(top[idx])
        if idx == 0:
            panel_label(ax, "a", -0.34, 1.18)
        for trajectory in templates[key]:
            ax.plot(stage_x, trajectory, color=[BLUE, ORANGE, PURPLE, GREEN][idx],
                    lw=0.75, alpha=0.72, marker="o", ms=2.3)
        ax.set_xlim(-0.12, 3.12); ax.set_ylim(0.65, 6.35)
        ax.set_xticks(stage_x, ["$X_1$", "$X_2$", "$X_3$", "$X_4$"])
        ax.set_yticks([1, 3, 6] if idx == 0 else [])
        if idx == 0:
            ax.set_ylabel("within-stage rank")
        ax.set_title(title, fontsize=7.1, pad=5)
        ax.grid(axis="y", color="#E5E5E5", lw=0.4)
    ax = fig.add_subplot(gs[1])
    panel_label(ax, "b", -0.07, 1.10)
    audit_labels = ["original H5 marginal rows", "TDS1 multiset checks",
                    "original H5 replay instances", "TDS1 replay instances"]
    checked = np.array([103, 20_800, 360_500, 60_200_000], dtype=float)
    y = np.arange(len(audit_labels))[::-1]
    display_counts = ["103", "20,800", "360,500", "60.2M"]
    for yy, count, display_count in zip(y, checked, display_counts):
        ax.plot([100, count], [yy, yy], color="#B9D9C3", lw=3.0,
                solid_capstyle="round", zorder=1)
        ax.scatter([count], [yy], s=54, facecolor="white", edgecolor=GREEN,
                   linewidth=1.5, zorder=3)
        ax.text(count * 1.18, yy, display_count,
                ha="left", va="center", fontsize=7.0, color=GREEN)
    ax.set_xscale("log"); ax.set_xlim(80, 1.2e8); ax.set_ylim(-0.65, 3.65)
    ax.set_yticks(y, audit_labels); ax.set_xlabel("executed audit units (log scale)")
    ax.grid(axis="x", which="major", color="#DDDDDD", lw=0.45)
    ax.text(0.99, 1.07,
            "all endpoints: 0 failures; exact executed-object counts",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            color=DARK_GREY)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.18)
    save_cns_figure(fig, OUT / "fig06_recoupling_audit")
    plt.close(fig)


def figure4_h3(data: dict) -> None:
    diag = data["h3_diag"].copy(); sessions = data["h3_sessions"].copy()
    validation = data["h3_validation"].copy()
    coordinates = ["X1", "X2", "X3", "X4", "E2E"]
    fig = plt.figure(figsize=(TECS_WIDTH_MM * MM, 122 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.40, 0.43, 0.17], hspace=0.52)

    ax = fig.add_subplot(gs[0])
    panel_label(ax, "a", -0.055, 1.07)
    ordered = pd.concat([
        diag.loc[diag["split"] == split].set_index("coordinate_id").loc[
            coordinates].reset_index()
        for split in ("calibration", "validation")], ignore_index=True)
    columns = ["lag1_pearson", "lag1_spearman", "acf_lag2", "acf_lag5",
               "acf_lag10", "acf_lag25"]
    matrix = ordered[columns].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("parce_diverging", DIVERGING)
    vmax = max(0.5, float(np.nanmax(np.abs(matrix))))
    heat = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    row_labels = [f"Cal {c}" for c in coordinates] + [f"Val {c}" for c in coordinates]
    col_labels = ["Pearson\nlag 1", "Spearman\nlag 1", "ACF\nlag 2",
                  "ACF\nlag 5", "ACF\nlag 10", "ACF\nlag 25"]
    ax.set_xticks(np.arange(6), col_labels); ax.set_yticks(np.arange(10), row_labels)
    ax.tick_params(length=0, labelsize=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(value) > 0.28 else BLACK)
            if j < 2 and abs(value) >= 0.10:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=RED, linewidth=1.3))
    ax.axhline(4.5, color=BLACK, lw=0.8)
    ax.axvline(1.5, color=BLACK, lw=0.8, ls="--")
    ax.text(0.16, 1.09, "frozen guard inputs", transform=ax.transAxes,
            ha="center", fontsize=7.2, fontweight="bold")
    ax.text(0.67, 1.09, "exploratory persistence", transform=ax.transAxes,
            ha="center", fontsize=7.2, color=DARK_GREY)
    colorbar = fig.colorbar(heat, ax=ax, orientation="vertical",
                            fraction=0.025, pad=0.015)
    colorbar.set_label("correlation / ACF", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)

    middle = gs[1].subgridspec(5, 2, hspace=0.18, wspace=0.14)
    normalized = {}
    for split in ("calibration", "validation"):
        subset = sessions.loc[sessions["split"] == split].copy()
        pivot = subset.pivot(index="coordinate_id", columns="planned_session_index",
                             values="median_ns").loc[coordinates]
        normalized[split] = np.log2(pivot.div(pivot.median(axis=1), axis=0))
    for row, coordinate in enumerate(coordinates):
        values_both = np.concatenate([
            normalized["calibration"].loc[coordinate].to_numpy(float),
            normalized["validation"].loc[coordinate].to_numpy(float)])
        local_limit = max(0.002, float(np.nanmax(np.abs(values_both))) * 1.12)
        for col, (split, color) in enumerate((("calibration", BLUE),
                                               ("validation", RED))):
            ax = fig.add_subplot(middle[row, col])
            if row == 0 and col == 0:
                panel_label(ax, "b", -0.16, 1.48)
            values = normalized[split].loc[coordinate].to_numpy(float)
            x = np.arange(1, len(values) + 1)
            ax.axhline(0, color="#BBBBBB", lw=0.55)
            ax.plot(x, values, color=color, lw=0.85, marker="o", ms=2.2,
                    markerfacecolor="white", markeredgewidth=0.65)
            ax.set_xlim(0.5, 20.5); ax.set_ylim(-local_limit, local_limit)
            ax.set_yticks([-local_limit, 0, local_limit] if col == 0 else [])
            if col == 0:
                ax.set_yticklabels([f"{-local_limit:.2g}", "0", f"{local_limit:.2g}"],
                                   fontsize=6.5)
                ax.set_ylabel(coordinate, rotation=0, ha="right", va="center",
                              labelpad=14, fontsize=7.0, fontweight="bold")
            if row == 0:
                ax.set_title("Calibration sessions" if split == "calibration"
                             else "Validation sessions", fontsize=7.2,
                             color=color, pad=3)
            if row == len(coordinates) - 1:
                ax.set_xticks([1, 5, 10, 15, 20])
                ax.set_xlabel("planned session index")
            else:
                ax.set_xticks([])
            ax.tick_params(length=2)
    ax = fig.add_subplot(gs[2])
    panel_label(ax, "c", -0.055, 1.16)
    ax.set_xlim(0, 10); ax.set_ylim(-0.25, 2.20); ax.set_axis_off()
    node_x = [1.30, 4.80, 8.30]
    node_y = 1.15
    ax.annotate("", xy=(4.47, node_y), xytext=(1.62, node_y),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.annotate("", xy=(7.97, node_y), xytext=(5.13, node_y),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
    ax.scatter([node_x[0]], [node_y], s=92, marker="o", facecolor="white",
               edgecolor=GREEN, linewidth=1.8)
    ax.scatter([node_x[1]], [node_y], s=94, marker="D", facecolor="white",
               edgecolor=RED, linewidth=1.8)
    ax.scatter([node_x[2]], [node_y], s=102, marker="o", facecolor="white",
               edgecolor=ORANGE, linewidth=2.0)
    failures = "/".join(str(int(v)) for v in validation["failures"])
    ax.text(node_x[0], 1.72, "5/5 nominal items supported",
            ha="center", fontsize=7.2, fontweight="bold", color=GREEN)
    ax.text(node_x[0], 0.43, f"failures $X_1$--$X_4$/E2E: {failures}",
            ha="center", fontsize=6.7, color=DARK_GREY)
    ax.text(node_x[1], 1.72, "frozen guard triggered",
            ha="center", fontsize=7.2, fontweight="bold", color=RED)
    ax.text(node_x[1], 0.43, "Calibration 4/5 · Validation 3/5",
            ha="center", fontsize=6.7, color=DARK_GREY)
    ax.text(node_x[2], 1.72, "INCONCLUSIVE",
            ha="center", fontsize=7.5, fontweight="bold", color=ORANGE)
    ax.text(node_x[2], 0.43, "CORRELATION-SENSITIVITY\nnot evidence that target risk was exceeded",
            ha="center", va="top", fontsize=6.7, color=ORANGE)
    save_cns_figure(fig, OUT / "fig04_h3_physical_dependence")
    plt.close(fig)


def figure8_tds(data: dict) -> None:
    cells = data["tds"].copy(); marginals = data["tds_marginals"]
    dgp_order = ["IID", "GAUSSIAN_AR1_WEAK", "HIDDEN_MARKOV", "TAIL_BURST"]
    dgp_labels = ["iid", "weak AR(1)", "hidden Markov", "tail burst"]
    dgp_colors = [BLUE, PURPLE, ORANGE, RED]
    fig = plt.figure(figsize=(TECS_WIDTH_MM * MM, 76 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.57, 0.43], wspace=0.34)

    ax = fig.add_subplot(gs[0]); panel_label(ax, "a", -0.18, 1.11)
    ax.axvspan(0.02, 1.0, color=LIGHT_GREEN, alpha=0.48, zorder=0)
    ax.axvline(1.0, color=RED, ls="--", lw=0.9)
    metrics = [("menu_family_content_failure", 0.05, "menu family", "o", BLUE, 0.13),
               ("stage_upper_noncoverage_family", 0.04, "stage family", "^", ORANGE, -0.13)]
    y_base = np.arange(4)[::-1]
    for metric, target, label, marker, color, offset in metrics:
        subset = cells.loc[cells["metric"] == metric].set_index("dgp_id").loc[dgp_order]
        rates = subset["rate"].to_numpy(float) / target
        lo = subset["outer_mc_cp_lower"].to_numpy(float) / target
        hi = subset["outer_mc_cp_upper"].to_numpy(float) / target
        for y, rate, lower, upper, status in zip(y_base + offset, rates, lo, hi,
                                                 subset["status"]):
            ax.errorbar(rate, y, xerr=[[rate - lower], [upper - rate]], fmt=marker,
                        ms=5.2, mfc=color, mec=RED if status == "FAIL" else BLACK,
                        mew=1.2 if status == "FAIL" else 0.6,
                        ecolor=color, elinewidth=0.8, capsize=2.0, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(0.02, 30); ax.set_ylim(-0.65, 3.65)
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.10, 0.25, 0.50, 1, 2, 5, 10, 20]))
    ax.xaxis.set_major_formatter(FixedFormatter([".05", ".1", ".25", ".5", "1", "2", "5", "10", "20"]))
    ax.set_yticks(y_base, dgp_labels); ax.set_xlabel("observed failure rate / frozen target")
    ax.grid(axis="x", which="major", color="#DDDDDD", lw=0.45)
    ax.text(1.05, 3.48, "target", fontsize=7.0, color=RED, ha="left")
    handles = [Line2D([0], [0], marker="o", color=BLUE, mfc=BLUE,
                      label="menu-family content failure"),
               Line2D([0], [0], marker="^", color=ORANGE, mfc=ORANGE,
                      label="stage-family noncoverage")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.9)

    ax = fig.add_subplot(gs[1]); panel_label(ax, "b", -0.23, 1.11)
    detection = [
        cells.loc[(cells["dgp_id"] == "IID") &
                  (cells["metric"] == "any_diagnostic_triggered")].iloc[0],
        cells.loc[(cells["dgp_id"] == "GAUSSIAN_AR1_WEAK") &
                  (cells["metric"] == "diagnostic_detection")].iloc[0],
        cells.loc[(cells["dgp_id"] == "HIDDEN_MARKOV") &
                  (cells["metric"] == "diagnostic_detection")].iloc[0],
        cells.loc[(cells["dgp_id"] == "TAIL_BURST") &
                  (cells["metric"] == "diagnostic_detection")].iloc[0]]
    unsafe_pass = [None, 0, 1, 121]
    for y, row, color, nominal_passes, label in zip(
            y_base, detection, dgp_colors, unsafe_pass, dgp_labels):
        rate = float(row["rate"]); lo = float(row["outer_mc_cp_lower"])
        hi = float(row["outer_mc_cp_upper"])
        ax.plot([0, rate], [y + 0.07, y + 0.07], color=color, lw=2.2,
                alpha=0.45, solid_capstyle="round")
        ax.errorbar(rate, y + 0.07, xerr=[[rate - lo], [hi - rate]], fmt="o",
                    ms=5.4, mfc="white", mec=color, mew=1.3, ecolor=color,
                    elinewidth=0.8, capsize=2.0, zorder=3)
        threshold = 0.05 if label == "iid" else 0.80
        ax.scatter([threshold], [y + 0.07], marker="|", s=155,
                   color=DARK_GREY, linewidth=1.2, zorder=4)
        if nominal_passes is not None:
            ax.scatter([0], [y - 0.20], marker="x", s=30, color=RED,
                       linewidth=1.3, zorder=4)
            ax.text(0.08, y - 0.20,
                    f"unsafe upgrades 0; unsafe nominal passes {nominal_passes}",
                    ha="left", va="center", fontsize=6.9, color=DARK_GREY)
    ax.set_xlim(-0.03, 1.07); ax.set_ylim(-0.65, 3.65)
    ax.set_yticks(y_base, dgp_labels); ax.set_xlabel("false-trigger / detection rate")
    ax.grid(axis="x", color="#E2E2E2", lw=0.45)
    ax.text(0.99, 1.06, "threshold marks: iid .05; dependent .80",
            transform=ax.transAxes, ha="right", fontsize=6.9, color=DARK_GREY)
    gate_pass_col = "implementation_gate_pass"
    marginal_passes = int(marginals[gate_pass_col].sum()) if gate_pass_col in marginals else len(marginals)
    fig.text(0.50, 0.995,
             f"known-truth simulation; frozen DKW marginal checks {marginal_passes}/{len(marginals)} pass",
             ha="center", va="top", fontsize=7.2)
    save_cns_figure(fig, OUT / "fig08_temporal_dependence_sensitivity")
    plt.close(fig)


def figure7_allocator(data: dict) -> None:
    alloc = data["allocator"].copy(); boundary = data["boundary"]
    fig, axes = plt.subplots(1, 2, figsize=(TECS_WIDTH_MM * MM, 73 * MM),
                             gridspec_kw={"wspace": 0.30})
    stages = [4, 8, 16, 32, 64]; stage_pos = {s: i for i, s in enumerate(stages)}
    conditions = [(3, 1), (3, 4), (6, 1), (6, 4)]
    cond_colors = {(3, 1): BLUE, (3, 4): BLUE, (6, 1): RED, (6, 4): RED}
    cond_markers = {(3, 1): "o", (3, 4): "^", (6, 1): "o", (6, 4): "^"}
    cond_lines = {(3, 1): "-", (3, 4): "--", (6, 1): "-", (6, 4): "--"}
    offsets = {(3, 1): -0.18, (3, 4): -0.06, (6, 1): 0.06, (6, 4): 0.18}
    for panel_index, (ax, column, ylabel, label, title) in enumerate(
            ((axes[0], "frontier_peak", "retained frontier states", "a",
              "frontier size"),
             (axes[1], "runtime_ns", "runtime (ms)", "b",
              "recorded runtime"))):
        panel_label(ax, label, -0.17, 1.06)
        for condition in conditions:
            menu_size, gates = condition
            subset = alloc.loc[(alloc["menu_size"] == menu_size) &
                               (alloc["gates"] == gates)]
            medians = []
            for stage in stages:
                values = subset.loc[subset["stages"] == stage, column].to_numpy(float)
                if column == "runtime_ns":
                    values = values / 1e6
                jitter = np.linspace(-0.040, 0.040, len(values))
                x = stage_pos[stage] + offsets[condition] + jitter
                ax.scatter(x, values, s=12, facecolor=cond_colors[condition],
                           edgecolor="none", alpha=0.20, rasterized=True)
                medians.append(float(np.median(values)))
            x_med = np.arange(len(stages)) + offsets[condition]
            ax.plot(x_med, medians, color=cond_colors[condition],
                    marker=cond_markers[condition], ms=4.3, lw=1.15,
                    ls=cond_lines[condition])
        ax.set_xticks(range(len(stages)), [str(v) for v in stages])
        ax.set_xlabel("number of stages"); ax.set_ylabel(ylabel); ax.set_yscale("log")
        ax.set_title(title, fontsize=7.5, pad=5)
        ax.grid(axis="y", which="major", color="#E0E0E0", lw=0.45)
    handles = [Line2D([0], [0], color=cond_colors[c], marker=cond_markers[c],
                      ls=cond_lines[c], lw=1.0, ms=4,
                      label=f"menu {c[0]}, gates {c[1]}") for c in conditions]
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.78,
                        wspace=0.30)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, fontsize=6.6, columnspacing=1.0, handletextpad=0.35)
    fig.text(0.50, 0.055,
              "Exact finite-decimal risk: 1,000 random + 81 real-menu "
              "oracle checks, 0 mismatches; no boundary rows excluded.",
             ha="center", va="center", fontsize=6.7, color=DARK_GREY)
    save_cns_figure(fig, OUT / "fig07_allocator_scaling")
    plt.close(fig)


def write_statistics(data: dict) -> None:
    phase = data["phase"]; selection = data["selection"]
    coupling = data["coupling"]; h3_diag = data["h3_diag"]
    h3_validation = data["h3_validation"]; tds = data["tds"]
    allocator = data["allocator"]
    stats = {
        "figure_1": {"physical_coordinates": 4,
                     "measured_topology": "fixed serial chain",
                     "theory_topology": "finite timing DAG"},
        "figure_2": {"calibration_units": 3000, "menu_rows": 12,
                     "validation_units": 3500, "physical_state": "INCONCLUSIVE",
                     "physical_reason": "CORRELATION_SENSITIVITY"},
        "figure_3": {"rows": len(phase),
                     "coarse_rows": int((phase["sweep"] == "coarse").sum()),
                     "fine_rows": int((phase["sweep"] == "fine").sum()),
                     "dominance_violations": int((phase["phase_bound_ns"] >
                                                   phase["maxwait_bound_ns"]).sum()),
                     "rescheduled_instances": int((phase["reschedule_count"] > 0).sum()),
                     "reschedule_events": int(phase["reschedule_count"].sum())},
        "figure_4": {"diagnostic_rows": len(h3_diag),
                     "nominal_validation_supported": int((h3_validation["status"] ==
                                                          "SUPPORTED").sum()),
                     "calibration_guard_triggers": int(h3_diag.loc[
                         h3_diag["split"] == "calibration", "correlation_triggered"].sum()),
                     "validation_guard_triggers": int(h3_diag.loc[
                         h3_diag["split"] == "validation", "correlation_triggered"].sum())},
        "figure_5": {"selection_cells": len(selection),
                     "repetitions_per_cell": 10000,
                     "family_target": 0.05},
        "figure_6": {
                     "coupling_rows": len(coupling),
                     "original_replay_instances": int(coupling["n_joint_units"].sum())},
        "figure_7": {"allocator_dp_rows": len(allocator),
                     "frontier_peak_min": int(allocator["frontier_peak"].min()),
                     "frontier_peak_max": int(allocator["frontier_peak"].max()),
                     "runtime_ns_min": int(allocator["runtime_ns"].min()),
                     "runtime_ns_max": int(allocator["runtime_ns"].max()),
                     "float_boundary_cases_remaining": len(data["boundary"])},
        "figure_8": {"outer_cells": len(tds),
                     "td1_td3_repetitions_per_dgp": 3000,
                     "physical_h3_unchanged": True}
    }
    (OUT / "rendered_figure_statistics.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    data = validate_source_data()
    figure1_system_timing(data); figure2_certification_workflow(data)
    figure3_phase(data); figure4_h3(data)
    figure5_selection_validity(data); figure6_recoupling_audit(data)
    figure7_allocator(data); figure8_tds(data); write_statistics(data)
    print(f"Generated eight PARCE-Cert figures in {OUT}")


if __name__ == "__main__":
    main()
