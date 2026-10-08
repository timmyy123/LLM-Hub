#include "Z3Shell.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <streambuf>

#include "Z3Embedded.h"
#include "util/env_params.h"
#include "util/gparams.h"
#include "util/memory_manager.h"
#include "util/util.h"
#include "util/warning.h"

int z3_upstream_main(int argc, char** argv);

namespace {

// These descriptors belong to the command. Nonblocking pipes let cancellation
// unwind even if a producer never closes stdin or a consumer stops reading.
class DescriptorScope {
  int descriptor, flags, noSignal;

 public:
  explicit DescriptorScope(FILE* file)
      : descriptor(fileno(file)), flags(-1), noSignal(-1) {
    struct stat info{};
    if (fstat(descriptor, &info) == 0 &&
        (S_ISFIFO(info.st_mode) || S_ISSOCK(info.st_mode) ||
         isatty(descriptor))) {
      flags = fcntl(descriptor, F_GETFL);
      if (flags >= 0) fcntl(descriptor, F_SETFL, flags | O_NONBLOCK);
    }
#ifdef F_GETNOSIGPIPE
    noSignal = fcntl(descriptor, F_GETNOSIGPIPE);
    if (noSignal >= 0) fcntl(descriptor, F_SETNOSIGPIPE, 1);
#endif
  }
  ~DescriptorScope() {
    if (flags >= 0) fcntl(descriptor, F_SETFL, flags);
#ifdef F_SETNOSIGPIPE
    if (noSignal >= 0) fcntl(descriptor, F_SETNOSIGPIPE, noSignal);
#endif
  }
};

class FileInputBuffer : public std::streambuf {
  FILE* file;
  char buffer;

  int_type underflow() override {
    if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
    for (;;) {
      z3_embedded::checkpoint();
      // Use stdio so previously buffered input is retained across invocations.
      int value = fgetc(file);
      if (value != EOF) {
        buffer = static_cast<char>(value);
        setg(&buffer, &buffer, &buffer + 1);
        return traits_type::to_int_type(buffer);
      }
      if (!ferror(file) || (errno != EAGAIN && errno != EINTR))
        return traits_type::eof();
      clearerr(file);
      pollfd descriptor{fileno(file), POLLIN, 0};
      if (poll(&descriptor, 1, 25) < 0 && errno != EINTR)
        return traits_type::eof();
    }
  }

 public:
  explicit FileInputBuffer(FILE* value) : file(value) {}
};

class FileOutputBuffer : public std::streambuf {
  FILE* file;
  z3_embedded::invocation& invocation;
  // Z3's parallel tactics can emit diagnostics from worker threads. Leave the
  // base put area empty so even single-character writes acquire this lock.
  std::mutex mutex;
  char buffer[4096];
  size_t length = 0;

  int_type overflow(int_type value) override {
    std::lock_guard lock(mutex);
    if (length == sizeof(buffer) && flush() != 0) return traits_type::eof();
    if (!traits_type::eq_int_type(value, traits_type::eof())) {
      buffer[length++] = traits_type::to_char_type(value);
    }
    return traits_type::not_eof(value);
  }
  int sync() override {
    std::lock_guard lock(mutex);
    return flush();
  }
  std::streamsize xsputn(const char* data, std::streamsize count) override {
    std::lock_guard lock(mutex);
    std::streamsize accepted = 0;
    while (accepted < count) {
      if (length == sizeof(buffer) && flush() != 0) break;
      size_t chunk = std::min(sizeof(buffer) - length,
                              static_cast<size_t>(count - accepted));
      std::memcpy(buffer + length, data + accepted, chunk);
      length += chunk;
      accepted += chunk;
    }
    return accepted;
  }
  int flush() {
    size_t count = length;
    length = 0;
    size_t offset = 0;
    int descriptor = fileno(file);
    if (descriptor < 0)
      return fwrite(buffer, 1, count, file) == count && fflush(file) == 0 ? 0
                                                                          : -1;
    while (offset < count) {
      ssize_t written = write(descriptor, buffer + offset, count - offset);
      if (written > 0) {
        offset += static_cast<size_t>(written);
        continue;
      }
      if (written < 0 && errno == EINTR) continue;
      if (written >= 0 || errno != EAGAIN || invocation.cancelled()) return -1;
      pollfd pollDescriptor{descriptor, POLLOUT, 0};
      if (poll(&pollDescriptor, 1, 25) < 0 && errno != EINTR) return -1;
    }
    return 0;
  }

 public:
  FileOutputBuffer(FILE* value, z3_embedded::invocation& run)
      : file(value), invocation(run) {}
};

class RuntimeScope {
  void* parameters;
  unsigned verbosity;
  std::ostream* verbose;
  std::ostream* warning;

 public:
  explicit RuntimeScope(z3_embedded::invocation& invocation) {
    memory::initialize(0);
    verbosity = get_verbosity_level();
    verbose = &verbose_stream();
    warning = warning_stream();
    parameters = gparams::begin_scope();
    z3_embedded::set_invocation(&invocation);
    memory::exit_when_out_of_memory(false, nullptr);
    set_verbosity_level(0);
    set_verbose_stream(*invocation.diagnostic);
    set_warning_stream(invocation.diagnostic);
  }

  ~RuntimeScope() {
    // Lift command limits before reclaiming parameters or restoring the host.
    memory::set_max_size(0);
    memory::set_max_alloc_count(0);
    memory::reset_out_of_memory();
    gparams::end_scope(parameters);
    env_params::updt_params();
    set_verbosity_level(verbosity);
    set_verbose_stream(*verbose);
    set_warning_stream(warning);
    z3_embedded::set_invocation(nullptr);
  }
};

}  // namespace

int Z3ShellRun(
    int argc, char** argv, FILE* input, FILE* output, FILE* error,
    int (*isCancelled)(const void*), const void* cancellationContext,
    const std::function<std::string(const char*, bool)>& resolvePath) {
  if (argc < 1 || !argv || !input || !output || !error) return 1;
  std::unique_lock lock(z3_embedded::runtime_mutex(), std::defer_lock);
  while (!lock.try_lock_for(std::chrono::milliseconds(25))) {
    if (isCancelled && isCancelled(cancellationContext)) return 130;
  }
  z3_embedded::invocation invocation{
      nullptr,     nullptr, nullptr, isCancelled, cancellationContext,
      &resolvePath};
  DescriptorScope inputDescriptor(input), outputDescriptor(output),
      errorDescriptor(error);
  if (fflush(output) != 0 || fflush(error) != 0) return 1;
  FileInputBuffer inputBuffer(input);
  FileOutputBuffer outputBuffer(output, invocation),
      errorBuffer(error, invocation);
  std::istream in(&inputBuffer);
  std::ostream out(&outputBuffer), err(&errorBuffer);
  in.tie(&out);
  in.exceptions(std::ios::badbit);
  invocation.input = &in;
  invocation.output = &out;
  invocation.diagnostic = &err;
  int status = 0;
  try {
    RuntimeScope runtime(invocation);
    z3_embedded::checkpoint();
    status = z3_upstream_main(argc, argv);
    z3_embedded::checkpoint();
  } catch (const z3_embedded::command_exit& exit) {
    status = exit.status;
    if (status == 124) out << "timeout\n";
  } catch (const std::exception& exception) {
    err << "ERROR: " << exception.what() << '\n';
    status = 1;
  }
  out.flush();
  err.flush();
  if (invocation.stop_status.load()) status = invocation.stop_status.load();
  if (status == 0 && (!out || !err || ferror(output) || ferror(error)))
    status = 1;
  return status;
}
