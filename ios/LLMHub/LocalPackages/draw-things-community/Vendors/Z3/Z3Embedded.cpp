#include "Z3Embedded.h"

#include <cstring>
#include <limits>

namespace z3_embedded {
namespace {
std::atomic<invocation*> active_invocation{nullptr};
}

std::timed_mutex& runtime_mutex() {
  static std::timed_mutex mutex;
  return mutex;
}

void set_invocation(invocation* value) {
  active_invocation.store(value, std::memory_order_release);
}

bool is_active() {
  return active_invocation.load(std::memory_order_acquire) != nullptr;
}

std::istream& in() {
  auto* run = active_invocation.load(std::memory_order_acquire);
  return run ? *run->input : std::cin;
}

std::ostream& out() {
  auto* run = active_invocation.load(std::memory_order_acquire);
  return run ? *run->output : std::cout;
}

std::ostream& err() {
  auto* run = active_invocation.load(std::memory_order_acquire);
  return run ? *run->diagnostic : std::cerr;
}

[[noreturn]] void exit(int status) { throw command_exit(status); }

void set_timeout(long seconds) {
  auto* run = active_invocation.load(std::memory_order_acquire);
  if (!run || seconds <= 0) return;
  auto now = std::chrono::steady_clock::now();
  auto maximum = std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::steady_clock::time_point::max() - now)
                     .count();
  if (seconds < maximum) run->deadline = now + std::chrono::seconds(seconds);
}

bool cancelled() {
  auto* run = active_invocation.load(std::memory_order_acquire);
  return run && run->cancelled();
}

bool invocation::cancelled() {
  if (stop_status.load(std::memory_order_relaxed)) return true;
  if (is_cancelled && is_cancelled(cancellation_context)) {
    stop_status.store(130, std::memory_order_relaxed);
    return true;
  }
  if (deadline != std::chrono::steady_clock::time_point::max() &&
      std::chrono::steady_clock::now() >= deadline) {
    stop_status.store(124, std::memory_order_relaxed);
    return true;
  }
  return false;
}

void checkpoint() {
  if (cancelled()) {
    auto* run = active_invocation.load(std::memory_order_acquire);
    throw command_exit(run->stop_status.load(std::memory_order_relaxed));
  }
}

std::string resolve_path(const char* path, bool writing) {
  auto* run = active_invocation.load(std::memory_order_acquire);
  return run && run->resolve_path && *run->resolve_path
             ? (*run->resolve_path)(path, writing)
             : path;
}

FILE* fopen(const char* path, const char* mode) {
  return std::fopen(
      resolve_path(path, mode[0] != 'r' || std::strchr(mode, '+')).c_str(),
      mode);
}

}  // namespace z3_embedded
