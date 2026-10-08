#ifndef Z3_EMBEDDED_H
#define Z3_EMBEDDED_H

#include <atomic>
#include <chrono>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>

namespace z3_embedded {

// All app entry points share this lock because Z3 CLI parameters are global.
std::timed_mutex& runtime_mutex();

struct command_exit : std::exception {
  int status;
  explicit command_exit(int value) : status(value) {}
  const char* what() const noexcept override { return "Z3 command exited"; }
};

struct invocation {
  std::istream* input;
  std::ostream* output;
  std::ostream* diagnostic;
  int (*is_cancelled)(const void*);
  const void* cancellation_context;
  const std::function<std::string(const char*, bool)>* resolve_path;
  std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::time_point::max();
  std::atomic<int> stop_status{0};
  bool cancelled();
};

// Set only while holding runtime_mutex; solver workers finish before clearing
// it.
void set_invocation(invocation* value);
bool is_active();
std::istream& in();
std::ostream& out();
std::ostream& err();
[[noreturn]] void exit(int status);
void set_timeout(long seconds);
bool cancelled();
void checkpoint();
std::string resolve_path(const char* path, bool writing);
FILE* fopen(const char* path, const char* mode);

// Resolve both frontend files and files opened by SMT-LIB commands/parameters.
// The host snapshots its paths before entering Z3, so workers need no session
// TLS.
template <typename Stream, std::ios::openmode DefaultMode>
class file_stream : public Stream {
 public:
  file_stream() = default;
  explicit file_stream(const std::string& path,
                       std::ios::openmode mode = DefaultMode) {
    open(path, mode);
  }
  void open(const std::string& path, std::ios::openmode mode = DefaultMode) {
    Stream::open(
        resolve_path(path.c_str(), ((mode | DefaultMode) & std::ios::out) != 0),
        mode);
  }
};
using ifstream = file_stream<std::ifstream, std::ios::in>;
using ofstream = file_stream<std::ofstream, std::ios::out>;
using fstream = file_stream<std::fstream, std::ios::in | std::ios::out>;

}  // namespace z3_embedded

#endif
