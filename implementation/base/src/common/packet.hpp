#pragma once

#include <cstdint>

namespace parce {

inline constexpr uint32_t PACKET_MAGIC = 0x50433101u;

inline constexpr uint32_t PACKET_FLAG_FIN = 0x1u;

// Fixed 256-byte datagram (workload packet_payload_bytes: 256).
struct Packet {
  uint32_t magic;
  uint32_t run_id;
  uint32_t seq;
  uint32_t flags;
  int64_t t_cause_ns;
  int64_t t_compute_end_ns;
  int64_t t_gate_ingress_ns;
  int64_t t_gate_target1_ns;
  int64_t t_gate_service_start_ns;
  int64_t t_gate_send_ns;
  int32_t target_cycle_id;
  int32_t target_window_id;
  int32_t service_cycle_id;
  int32_t service_window_id;
  int32_t reschedule_count;
  int32_t reserved;
  uint8_t pad[256 - (4 * 4 + 6 * 8 + 6 * 4)];
};

static_assert(sizeof(Packet) == 256, "packet payload must be exactly 256 bytes");

}  // namespace parce
