#include <arpa/inet.h>
#include <errno.h>
#include <sched.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/calendar.hpp"
#include "common/packet.hpp"

namespace {

int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

void sleep_abs(int64_t target) {
  while (true) {
    timespec req{target / 1000000000LL, target % 1000000000LL};
    int rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &req, nullptr);
    if (rc == 0) return;
    if (rc == EINTR) continue;  // retry the same absolute target
    std::fprintf(stderr, "gate: clock_nanosleep rc=%d\n", rc);
    std::exit(3);
  }
}

bool pin_cpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;
}

int run_calcheck() {
  std::string line;
  unsigned long n = 0;
  while (std::getline(std::cin, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    long long e = 0, d = 0, p = 0, nw = 0;
    if (!(ss >> e >> d >> p >> nw) || nw < 0 || p <= 0) {
      std::fprintf(stderr, "calcheck: malformed line %lu\n", n);
      return 1;
    }
    std::vector<parce::Window> wins((size_t)nw);
    bool ok = true;
    for (long long i = 0; i < nw; ++i) {
      long long o = 0, c = 0;
      if (!(ss >> o >> c)) { ok = false; break; }
      wins[(size_t)i] = parce::Window{o, c};
    }
    if (!ok) {
      std::fprintf(stderr, "calcheck: malformed windows on line %lu\n", n);
      return 1;
    }
    parce::NextStartResult r = parce::next_feasible_start(e, d, p, wins);
    std::cout << r.start_ns << ' ' << r.cycle_id << ' ' << r.window_id << ' '
              << r.reason_code << '\n';
    ++n;
  }
  std::fprintf(stderr, "calcheck cases=%lu\n", n);
  return 0;
}

int run_gate(int argc, char** argv) {
  uint32_t run_id = 0;
  int64_t period = 1000000, offset = 0, demand = 50000;
  int instances = 40, cpu = 2;
  int rx_port = 57001, tx_port = 57002;
  const char* tx_addr = "127.0.0.1";
  const char* windows_spec = "400000:700000";
  const char* out_path = nullptr;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--run-id") run_id = uint32_t(std::strtoul(v.c_str(), nullptr, 10));
    else if (k == "--period") period = std::atoll(v.c_str());
    else if (k == "--offset") offset = std::atoll(v.c_str());
    else if (k == "--demand") demand = std::atoll(v.c_str());
    else if (k == "--windows") windows_spec = argv[i + 1];
    else if (k == "--instances") instances = std::atoi(v.c_str());
    else if (k == "--cpu") cpu = std::atoi(v.c_str());
    else if (k == "--rx-port") rx_port = std::atoi(v.c_str());
    else if (k == "--tx-port") tx_port = std::atoi(v.c_str());
    else if (k == "--tx-addr") tx_addr = argv[i + 1];
    else if (k == "--out") out_path = argv[i + 1];
    else { std::fprintf(stderr, "gate: unknown arg %s\n", k.c_str()); return 2; }
  }
  if (!out_path || instances <= 0) { std::fprintf(stderr, "gate: missing args\n"); return 2; }
  if (!pin_cpu(cpu)) { std::fprintf(stderr, "gate: pin cpu %d failed\n", cpu); return 4; }

  std::vector<parce::Window> wins;
  {
    std::string spec(windows_spec);
    size_t pos = 0;
    while (pos <= spec.size()) {
      size_t comma = spec.find(',', pos);
      std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      if (!item.empty()) {
        size_t colon = item.find(':');
        if (colon == std::string::npos) { std::fprintf(stderr, "gate: bad windows spec\n"); return 2; }
        int64_t o = std::atoll(item.substr(0, colon).c_str()) + offset;
        int64_t c = std::atoll(item.substr(colon + 1).c_str()) + offset;
        wins.push_back(parce::Window{o, c});
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }

  int rx = socket(AF_INET, SOCK_DGRAM, 0);
  int one = 1;
  setsockopt(rx, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  int buf = 4194304;
  setsockopt(rx, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
  timeval tv{1, 0};  // 1s inactivity timeout
  setsockopt(rx, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(uint16_t(rx_port));
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(rx, (sockaddr*)&addr, sizeof(addr)) < 0) {
    std::fprintf(stderr, "gate: bind %d errno=%d\n", rx_port, errno);
    return 5;
  }

  int tx = socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in dst{};
  dst.sin_family = AF_INET;
  dst.sin_port = htons(uint16_t(tx_port));
  dst.sin_addr.s_addr = inet_addr(tx_addr);

  FILE* out = std::fopen(out_path, "w");
  long log_drops = 0;
  long handled = 0;

  while (handled < instances) {
    parce::Packet p{};
    sockaddr_in from{};
    socklen_t flen = sizeof(from);
    ssize_t rc = recvfrom(rx, &p, sizeof(p), 0, (sockaddr*)&from, &flen);
    if (rc < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;  // inactivity timeout
      ++log_drops;
      continue;
    }
    if (rc != (ssize_t)sizeof(p) || p.magic != parce::PACKET_MAGIC || p.run_id != run_id) {
      ++log_drops;
      continue;
    }
    if (p.flags & parce::PACKET_FLAG_FIN) break;
    const int64_t t_ingress = now_ns();

    const parce::NextStartResult r1 =
        parce::next_feasible_start(t_ingress, demand, period, wins);
    if (r1.reason_code != parce::REASON_OK) {
      ++log_drops;
      continue;  // cannot happen for C1; drop defensively
    }
    const int64_t target1 = r1.start_ns;
    const int32_t target_cycle = int32_t(r1.cycle_id);
    const int32_t target_window = int32_t(r1.window_id);

    int64_t target = target1;
    int32_t reschedules = 0;
    int64_t service_start = 0;
    int32_t service_cycle = 0, service_window = 0;
    while (true) {
      sleep_abs(target);
      const int64_t now = now_ns();
      if (now < target) continue;  // spurious early wake, keep waiting
      const parce::NextStartResult r2 =
          parce::next_feasible_start(now, demand, period, wins);
      if (r2.reason_code != parce::REASON_OK) break;  // defensive
      if (r2.start_ns <= now) {
        service_start = now;
        service_cycle = int32_t(r2.cycle_id);
        service_window = int32_t(r2.window_id);
        break;
      }
      if (r2.start_ns != target) ++reschedules;
      target = r2.start_ns;
    }
    if (service_start == 0) { ++log_drops; continue; }

    sleep_abs(service_start + demand);
    p.t_gate_ingress_ns = t_ingress;
    p.t_gate_target1_ns = target1;
    p.t_gate_service_start_ns = service_start;
    p.target_cycle_id = target_cycle;
    p.target_window_id = target_window;
    p.service_cycle_id = service_cycle;
    p.service_window_id = service_window;
    p.reschedule_count = reschedules;
    if (sendto(tx, &p, sizeof(p), 0, (sockaddr*)&dst, sizeof(dst)) < 0) ++log_drops;
    const int64_t t_send = now_ns();
    std::fprintf(out,
        "{\"seq\":%u,\"t_gate_ingress_ns\":%lld,\"t_gate_target1_ns\":%lld,"
        "\"t_gate_service_start_ns\":%lld,\"t_gate_send_ns\":%lld,"
        "\"target_cycle_id\":%d,\"target_window_id\":%d,"
        "\"service_cycle_id\":%d,\"service_window_id\":%d,"
        "\"reschedule_count\":%d}\n",
        p.seq, (long long)t_ingress, (long long)target1, (long long)service_start,
        (long long)t_send, target_cycle, target_window, service_cycle, service_window,
        reschedules);
    std::fflush(out);
    ++handled;
  }
  usleep(2000);
  {
    parce::Packet fin{};
    fin.magic = parce::PACKET_MAGIC;
    fin.run_id = run_id;
    fin.seq = uint32_t(instances);
    fin.flags = parce::PACKET_FLAG_FIN;
    sendto(tx, &fin, sizeof(fin), 0, (sockaddr*)&dst, sizeof(dst));
  }
  if (log_drops) std::fprintf(stderr, "gate: log_drops=%ld handled=%ld\n", log_drops, handled);
  std::fclose(out);
  close(rx);
  close(tx);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--calcheck") return run_calcheck();
  if (argc > 1) return run_gate(argc, argv);
  std::fprintf(stderr,
               "software_gate: modes: --calcheck (batch reference check) or "
               "runtime args (--rx-port, --windows, ...)\n");
  return 2;
}
