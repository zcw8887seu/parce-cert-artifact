#pragma once

#include <cstdint>

namespace parce {

// Column names of raw/events/*.parquet (spec 8.2). The C++ processes emit
// JSONL with the per-process subset; the runner merges them into this schema.
inline constexpr const char* EVENT_COLUMNS[] = {
    "experiment_id", "run_id",    "session_id",      "split",
    "instance_id",   "seq",       "is_warmup",       "selected_single",
    "phase_ns",      "service_demand_ns",            "t_cause_ns",
    "t_compute_end_ns",           "t_gate_ingress_ns",
    "t_gate_target1_ns",          "t_gate_service_start_ns",
    "t_gate_send_ns",             "t_rx_ns",
    "target_cycle_id",            "target_window_id",
    "service_cycle_id",           "service_window_id",
    "reschedule_count",           "packet_lost",
    "out_of_order",  "duplicate", "valid_event",     "invalid_reason",
};
inline constexpr int EVENT_COLUMN_COUNT = 27;

}  // namespace parce
