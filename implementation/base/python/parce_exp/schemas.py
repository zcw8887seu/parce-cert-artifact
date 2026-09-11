"""Event-log schema checks (spec 8.2, 9.1, 9.3)."""

import pandas as pd

EVENT_COLUMNS = [
    "experiment_id",
    "run_id",
    "session_id",
    "split",
    "instance_id",
    "seq",
    "is_warmup",
    "selected_single",
    "phase_ns",
    "service_demand_ns",
    "t_cause_ns",
    "t_compute_end_ns",
    "t_gate_ingress_ns",
    "t_gate_target1_ns",
    "t_gate_service_start_ns",
    "t_gate_send_ns",
    "t_rx_ns",
    "target_cycle_id",
    "target_window_id",
    "service_cycle_id",
    "service_window_id",
    "reschedule_count",
    "packet_lost",
    "out_of_order",
    "duplicate",
    "valid_event",
    "invalid_reason",
]


def check_event_log(df):
    """Return a list of issue strings; empty list means the log is clean.

    Detects: missing columns, missing (null) timestamp fields, duplicate
    (run_id, seq) keys, out-of-order seq within a run, negative stage
    durations and sends that precede service completion.
    """
    issues = []
    missing_cols = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append("MISSING_COLUMN:" + ",".join(missing_cols))
        return issues

    for col in EVENT_COLUMNS:
        if col.startswith("t_"):
            n_null = int(pd.isna(df[col]).sum())
            if n_null:
                issues.append(f"MISSING_FIELD:{col}:{n_null}")

    if len(df):
        dup = df.duplicated(subset=["run_id", "seq"], keep="first")
        if bool(dup.any()):
            issues.append(f"DUPLICATE_KEY:{int(dup.sum())}")

    out_of_order = 0
    for _run, group in df.groupby("run_id", sort=False):
        seq = group["seq"].to_numpy()
        if len(seq) > 1 and bool((seq[1:] <= seq[:-1]).any()):
            out_of_order += int((seq[1:] <= seq[:-1]).sum())
    if out_of_order:
        issues.append(f"OUT_OF_ORDER:{out_of_order}")

    pairs = [
        ("X1", "t_compute_end_ns", "t_cause_ns"),
        ("X2", "t_gate_ingress_ns", "t_compute_end_ns"),
        ("X3", "t_gate_service_start_ns", "t_gate_target1_ns"),
    ]
    for name, later, earlier in pairs:
        mask = df[later].notna() & df[earlier].notna()
        n_neg = int((df.loc[mask, later] < df.loc[mask, earlier]).sum())
        if n_neg:
            issues.append(f"NEGATIVE_STAGE_DURATION:{name}:{n_neg}")

    demand = df["service_demand_ns"]
    mask_x4 = (
        (~df["packet_lost"].astype(bool))
        & df["t_rx_ns"].notna()
        & df["t_gate_service_start_ns"].notna()
        & demand.notna()
    )
    n_x4 = int(
        (
            df.loc[mask_x4, "t_rx_ns"]
            < df.loc[mask_x4, "t_gate_service_start_ns"] + df.loc[mask_x4, "service_demand_ns"]
        ).sum()
    )
    if n_x4:
        issues.append(f"NEGATIVE_STAGE_DURATION:X4:{n_x4}")

    mask_send = (
        df["t_gate_send_ns"].notna()
        & df["t_gate_service_start_ns"].notna()
        & demand.notna()
    )
    n_send = int(
        (
            df.loc[mask_send, "t_gate_send_ns"]
            < df.loc[mask_send, "t_gate_service_start_ns"] + df.loc[mask_send, "service_demand_ns"]
        ).sum()
    )
    if n_send:
        issues.append(f"SERVICE_ORDER_VIOLATION:{n_send}")

    return issues
