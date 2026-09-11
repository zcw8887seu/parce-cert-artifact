#include <arpa/inet.h>
#include <errno.h>
#include <sched.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
    std::fprintf(stderr, "sender: clock_nanosleep rc=%d\n", rc);
    std::exit(3);
  }
}

bool pin_cpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t run_id = 0;
  int64_t period = 1000000, phase = 0, compute_target = 100000;
  int instances = 40, warmup = 32, cpu = 0, grace_ms = 20;
  const char* gate_addr = "127.0.0.1";
  int gate_port = 57001;
  const char* out_path = nullptr;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--run-id") run_id = uint32_t(std::strtoul(v.c_str(), nullptr, 10));
    else if (k == "--period") period = std::atoll(v.c_str());
    else if (k == "--phase") phase = std::atoll(v.c_str());
    else if (k == "--compute-target") compute_target = std::atoll(v.c_str());
    else if (k == "--instances") instances = std::atoi(v.c_str());
    else if (k == "--warmup") warmup = std::atoi(v.c_str());
    else if (k == "--cpu") cpu = std::atoi(v.c_str());
    else if (k == "--grace-ms") grace_ms = std::atoi(v.c_str());
    else if (k == "--gate-port") gate_port = std::atoi(v.c_str());
    else if (k == "--gate-addr") gate_addr = argv[i + 1];
    else if (k == "--out") out_path = argv[i + 1];
    else { std::fprintf(stderr, "sender: unknown arg %s\n", k.c_str()); return 2; }
  }
  if (!out_path || instances <= 0) { std::fprintf(stderr, "sender: missing args\n"); return 2; }
  if (!pin_cpu(cpu)) { std::fprintf(stderr, "sender: pin cpu %d failed\n", cpu); return 4; }

  const int kArr = 4096;
  std::vector<int32_t> arr(kArr);
  {
    uint64_t s = 88172645463325252ULL;
    for (int i = 0; i < kArr; ++i) arr[i] = i;
    for (int i = kArr - 1; i > 0; --i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      std::swap(arr[i], arr[int(s % uint64_t(i + 1))]);
    }
  }

  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in dst{};
  dst.sin_family = AF_INET;
  dst.sin_port = htons(uint16_t(gate_port));
  dst.sin_addr.s_addr = inet_addr(gate_addr);

  FILE* out = std::fopen(out_path, "w");

  const int64_t t0 = now_ns();
  const int64_t base = ((t0 + int64_t(grace_ms) * 1000000LL) / period + 1) * period;

  for (int k = 0; k < instances; ++k) {
    const int64_t release = base + int64_t(k) * period + phase;
    sleep_abs(release);
    const int64_t t_cause = now_ns();
    int32_t idx = 0;
    while (true) {
      for (int j = 0; j < 64; ++j) idx = arr[idx];
      if (now_ns() - t_cause >= compute_target) break;
    }
    const int64_t t_end = now_ns();
    parce::Packet p{};
    p.magic = parce::PACKET_MAGIC;
    p.run_id = run_id;
    p.seq = uint32_t(k);
    p.flags = 0;
    p.t_cause_ns = t_cause;
    p.t_compute_end_ns = t_end;
    if (sendto(fd, &p, sizeof(p), 0, (sockaddr*)&dst, sizeof(dst)) < 0)
      std::fprintf(stderr, "sender: sendto seq=%d errno=%d\n", k, errno);
    std::fprintf(out,
                 "{\"seq\":%d,\"is_warmup\":%s,\"t_cause_ns\":%lld,\"t_compute_end_ns\":%lld}\n",
                 k, k < warmup ? "true" : "false", (long long)t_cause, (long long)t_end);
    std::fflush(out);
  }
  {
    parce::Packet p{};
    p.magic = parce::PACKET_MAGIC;
    p.run_id = run_id;
    p.seq = uint32_t(instances);
    p.flags = parce::PACKET_FLAG_FIN;
    sendto(fd, &p, sizeof(p), 0, (sockaddr*)&dst, sizeof(dst));
  }
  std::fclose(out);
  close(fd);
  return 0;
}
