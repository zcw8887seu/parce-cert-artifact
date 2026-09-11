#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace parce {

struct Window {
  int64_t open_ns;
  int64_t close_ns;
};

enum ReasonCode : int32_t {
  REASON_OK = 0,
  REASON_INFEASIBLE_NO_WINDOW = 1,
};

struct NextStartResult {
  int64_t start_ns;
  int64_t cycle_id;
  int64_t window_id;
  int32_t reason_code;
};

inline int64_t floor_div_ns(int64_t a, int64_t b) {
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) --q;
  return q;
}

// Reference periodic-service operator (spec agent.MD 5.2). Window instances
// repeat at k*period_ns + open_ns for every integer k; half-open [open, close).
// Admission: s >= eligible, s >= window_open, s + demand <= window_close.
// Returns the earliest feasible start; ties on identical start are broken by
// (window_id, cycle_id). All arithmetic is integer nanoseconds.
inline NextStartResult next_feasible_start(int64_t eligible_ns, int64_t demand_ns,
                                           int64_t period_ns,
                                           const std::vector<Window>& windows) {
  bool have = false;
  NextStartResult best{0, 0, 0, REASON_INFEASIBLE_NO_WINDOW};
  for (size_t w = 0; w < windows.size(); ++w) {
    const int64_t open = windows[w].open_ns;
    const int64_t close = windows[w].close_ns;
    if (close - open < demand_ns) continue;
    int64_t k = floor_div_ns(eligible_ns - open, period_ns);
    int64_t wopen = k * period_ns + open;
    int64_t cand = eligible_ns > wopen ? eligible_ns : wopen;
    if (cand + demand_ns > wopen + (close - open)) {
      ++k;
      cand = k * period_ns + open;
    }
    if (!have || cand < best.start_ns ||
        (cand == best.start_ns &&
         ((int64_t)w < best.window_id ||
          ((int64_t)w == best.window_id && k < best.cycle_id)))) {
      best = NextStartResult{cand, k, (int64_t)w, REASON_OK};
      have = true;
    }
  }
  if (!have) return NextStartResult{0, 0, 0, REASON_INFEASIBLE_NO_WINDOW};
  return best;
}

}  // namespace parce
