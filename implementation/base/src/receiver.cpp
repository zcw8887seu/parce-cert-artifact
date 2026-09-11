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

bool pin_cpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t run_id = 0;
  int instances = 40, cpu = 4, port = 57002;
  const char* out_path = nullptr;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--run-id") run_id = uint32_t(std::strtoul(v.c_str(), nullptr, 10));
    else if (k == "--instances") instances = std::atoi(v.c_str());
    else if (k == "--cpu") cpu = std::atoi(v.c_str());
    else if (k == "--port") port = std::atoi(v.c_str());
    else if (k == "--out") out_path = argv[i + 1];
    else { std::fprintf(stderr, "receiver: unknown arg %s\n", k.c_str()); return 2; }
  }
  if (!out_path || instances <= 0) { std::fprintf(stderr, "receiver: missing args\n"); return 2; }
  if (!pin_cpu(cpu)) { std::fprintf(stderr, "receiver: pin cpu %d failed\n", cpu); return 4; }

  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  int buf = 4194304;
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
  timeval tv{0, 500000};  // 500ms inactivity timeout
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(uint16_t(port));
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
    std::fprintf(stderr, "receiver: bind %d errno=%d\n", port, errno);
    return 5;
  }

  FILE* out = std::fopen(out_path, "w");
  std::vector<uint8_t> seen(instances + 2, 0);
  int64_t max_seq = -1;
  long unique = 0;
  long log_drops = 0;

  while (unique < instances) {
    parce::Packet p{};
    sockaddr_in from{};
    socklen_t flen = sizeof(from);
    ssize_t rc = recvfrom(fd, &p, sizeof(p), 0, (sockaddr*)&from, &flen);
    if (rc < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;
      ++log_drops;
      continue;
    }
    if (rc != (ssize_t)sizeof(p) || p.magic != parce::PACKET_MAGIC || p.run_id != run_id) {
      ++log_drops;
      continue;
    }
    if (p.flags & parce::PACKET_FLAG_FIN) break;
    const int64_t t_rx = now_ns();
    const int64_t seq = p.seq;
    const bool out_of_order = seq < max_seq;
    const bool duplicate = seen[size_t(seq)] != 0;
    if (!duplicate) {
      seen[size_t(seq)] = 1;
      ++unique;
      if (seq > max_seq) max_seq = seq;
      std::fprintf(out,
                   "{\"seq\":%lld,\"t_rx_ns\":%lld,\"out_of_order\":%s,\"duplicate\":false}\n",
                   (long long)seq, (long long)t_rx, out_of_order ? "true" : "false");
      std::fflush(out);
    } else {
      std::fprintf(out,
                   "{\"seq\":%lld,\"t_rx_ns\":%lld,\"out_of_order\":%s,\"duplicate\":true}\n",
                   (long long)seq, (long long)t_rx, out_of_order ? "true" : "false");
      std::fflush(out);
    }
  }
  if (log_drops) std::fprintf(stderr, "receiver: log_drops=%ld unique=%ld\n", log_drops, unique);
  std::fclose(out);
  close(fd);
  return 0;
}
