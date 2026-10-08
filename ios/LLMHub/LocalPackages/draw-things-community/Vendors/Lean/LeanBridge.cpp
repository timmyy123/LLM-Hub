#include "LeanBridge.h"

#include <fcntl.h>
#include <getopt.h>
#include <lean/lean.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "Vendors/Lean/LeanShellOptions.inc"
#include "runtime/interrupt.h"
#include "runtime/io.h"
#include "runtime/option_ref.h"
#include "util/io.h"

extern "C" void lean_initialize();
extern "C" lean_object* runtime_initialize_CheckLeanBridge(uint8_t builtin);
extern "C" lean_object* check_lean_bridge(lean_object* source,
                                          lean_object* base_root,
                                          lean_object* package_root,
                                          lean_object* artifact_json);
extern "C" lean_object* check_lean_source_imports(lean_object* source);
extern "C" lean_object* check_lean_module_imports(lean_object* olean_path);
extern "C" lean_object* lean_bridge_command(lean_object*, lean_object*,
                                            lean_object*, lean_object*,
                                            lean_object*, lean_object*, uint8_t,
                                            lean_object*);
extern "C" lean_object* lean_bridge_new_cancel_token();
extern "C" lean_object* lean_bridge_cancel_token(lean_object*);
extern "C" lean_object* lean_shell_options_mk(lean_object*);
extern "C" lean_object* lean_shell_options_process(lean_object*, uint32_t,
                                                   lean_object*);
extern "C" uint8_t lean_bridge_options_stdin(lean_object*);
extern "C" uint64_t lean_bridge_options_timeout(lean_object*);
extern "C" lean_object* lean_stream_of_handle(lean_object*);
extern "C" lean_object* lean_get_set_stdin(lean_object*);
extern "C" lean_object* lean_get_set_stdout(lean_object*);
extern "C" lean_object* lean_get_set_stderr(lean_object*);
extern "C" lean_object* lean_io_error_to_string(lean_object*);

namespace {

std::once_flag initialization_flag;
std::mutex evaluation_mutex;

void initialize_lean() {
  lean_initialize();
  lean::consume_io_result(runtime_initialize_CheckLeanBridge(1));
  lean::io_mark_end_initialization();
  lean_init_task_manager();
}

char* copy_string(const char* value, size_t size) {
  char* copy = static_cast<char*>(std::malloc(size));
  if (copy != nullptr) {
    std::memcpy(copy, value, size);
  }
  return copy;
}

template <typename Action>
auto on_lean_thread(Action action) {
  // The upstream shell uses a Lean thread, not Darwin's 512 KiB default.
  // lthread also initializes stack guards and finalizes Lean TLS on exit.
  decltype(action()) result{};
  std::exception_ptr failure;
  lean::lthread worker([&] {
    try {
      result = action();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  worker.join();
  if (failure) std::rethrow_exception(failure);
  return result;
}

class CommandStream {
  struct Descriptor {
    int fd, flags, noSignal;
    std::mutex mutex;
    bool finished = false;
    int (*cancelled)(const void*);
    const void* context;
    FILE* input;
    void finish() {
      std::lock_guard lock(mutex);
      finished = true;
      if (flags >= 0) fcntl(fd, F_SETFL, flags);
      if (noSignal >= 0) fcntl(fd, F_SETNOSIGPIPE, noSignal);
      flags = noSignal = -1;
    }
    bool stopped() {
      std::lock_guard lock(mutex);
      return finished || (cancelled && cancelled(context));
    }
    static int read(void* cookie, char* buffer, int count) {
      auto* state = static_cast<Descriptor*>(cookie);
      for (;;) {
        {
          // Keep buffered session input, as Z3Shell does. Retiring the stream
          // synchronizes with this borrowed FILE access before the caller
          // closes it.
          std::lock_guard lock(state->mutex);
          if (state->finished ||
              (state->cancelled && state->cancelled(state->context))) {
            // Lean's EINTR decoder requires a file name, but stream errors
            // have none. ECANCELED is safe for anonymous command streams.
            errno = ECANCELED;
            return -1;
          }
          size_t n = fread(buffer, 1, count, state->input);
          if (n > 0) return static_cast<int>(n);
          if (!ferror(state->input)) return 0;
          if (errno != EAGAIN && errno != EINTR) return -1;
          clearerr(state->input);
        }
        pollfd descriptor{state->fd, POLLIN, 0};
        poll(&descriptor, 1, 25);
      }
    }
    static int write(void* cookie, const char* buffer, int count) {
      auto* state = static_cast<Descriptor*>(cookie);
      int offset = 0;
      while (offset < count) {
        if (state->stopped()) {
          errno = ECANCELED;
          return offset ? offset : -1;
        }
        ssize_t n = ::write(state->fd, buffer + offset, count - offset);
        if (n > 0) {
          offset += static_cast<int>(n);
          continue;
        }
        if (n == 0 || (errno != EAGAIN && errno != EINTR))
          return offset ? offset : -1;
        pollfd descriptor{state->fd, POLLOUT, 0};
        poll(&descriptor, 1, 25);
      }
      return offset;
    }
    static int close(void* cookie) {
      auto* state = static_cast<Descriptor*>(cookie);
      state->finish();
      int result = ::close(state->fd);
      delete state;
      return result;
    }
  };
  lean_object* previous;
  lean_object* (*replace)(lean_object*);
  FILE* file;
  Descriptor* descriptor;

 public:
  CommandStream(int fd, const char* mode, lean_object* (*replace)(lean_object*),
                int (*cancelled)(const void*) = nullptr,
                const void* context = nullptr, FILE* input = nullptr)
      : replace(replace) {
    int owned = dup(fd);
    if (owned < 0)
      throw std::runtime_error("cannot duplicate Lean command stream");
    descriptor = new Descriptor{owned,   -1,    fcntl(owned, F_GETNOSIGPIPE),
                                {},      false, cancelled,
                                context, input};
    struct stat info{};
    if (fstat(owned, &info) == 0 &&
        (S_ISFIFO(info.st_mode) || S_ISSOCK(info.st_mode) || isatty(owned))) {
      descriptor->flags = fcntl(owned, F_GETFL);
      if (descriptor->flags >= 0)
        fcntl(owned, F_SETFL, descriptor->flags | O_NONBLOCK);
    }
    fcntl(owned, F_SETNOSIGPIPE, 1);
    file = funopen(descriptor, *mode == 'r' ? Descriptor::read : nullptr,
                   *mode == 'w' ? Descriptor::write : nullptr, nullptr,
                   Descriptor::close);
    if (!file) {
      Descriptor::close(descriptor);
      throw std::runtime_error("cannot open Lean command stream");
    }
    // Do not read ahead into a second stdio buffer that would be discarded at
    // command exit. Unconsumed bytes must stay in the borrowed session FILE.
    if (input) setvbuf(file, nullptr, _IONBF, 0);
    previous = replace(lean_stream_of_handle(lean::io_wrap_handle(file)));
  }
  ~CommandStream() {
    fflush(file);
    // A captured Lean stream may outlive this call. Retire the callback before
    // its borrowed cancellation context disappears, even if the handle
    // survives.
    descriptor->finish();
    lean_dec(replace(previous));
  }
  bool flush() { return fflush(file) == 0 && !ferror(file); }
};

class CommandCancellation {
  std::mutex mutex;
  std::condition_variable changed;
  bool finished = false;
  std::thread monitor;

 public:
  lean_object* token;
  CommandCancellation(int (*cancelled)(const void*), const void* context) {
    token = lean_bridge_new_cancel_token();
    lean_mark_mt(token);
    // Lean elaboration and its kernel poll this token. A monitor is necessary
    // because ios_system exposes a polled session flag, not a cancel callback.
    monitor = std::thread([this, cancelled, context] {
      lean::lean_initialize_thread();
      {
        std::unique_lock lock(mutex);
        while (!finished) {
          if (cancelled && cancelled(context)) {
            lean_inc(token);
            lean_dec(lean_bridge_cancel_token(token));
            break;
          }
          changed.wait_for(lock, std::chrono::milliseconds(20));
        }
      }
      lean::lean_finalize_thread();
    });
  }
  ~CommandCancellation() {
    {
      std::lock_guard lock(mutex);
      finished = true;
    }
    changed.notify_one();
    monitor.join();
    // Resolve the token's internal promise also on successful completion.
    lean_dec(lean_bridge_cancel_token(token));
  }
};

template <typename Action>
char* run_string_action(Action action) {
  try {
    std::call_once(initialization_flag, initialize_lean);
    std::lock_guard<std::mutex> evaluation_lock(evaluation_mutex);
    return on_lean_thread([&]() -> char* {
      lean_object* result = action();
      if (lean_io_result_is_error(result)) {
        lean_dec(result);
        return nullptr;
      }
      lean_object* value = lean_io_result_get_value(result);
      char* copy =
          copy_string(lean_string_cstr(value), lean_string_size(value));
      lean_dec(result);
      return copy;
    });
  } catch (...) {
    return nullptr;
  }
}

}  // namespace

extern "C" char* lean_bridge_check(const char* source, const char* base_root,
                                   const char* package_root,
                                   const char* artifact_json) {
  char* result = run_string_action([&] {
    return check_lean_bridge(
        lean_mk_string(source), lean_mk_string(base_root),
        lean_mk_string(package_root),
        lean_mk_string(artifact_json ? artifact_json : ""));
  });
  if (result != nullptr) return result;
  static constexpr char error[] = "status: error\n";
  return copy_string(error, sizeof(error));
}

extern "C" char* lean_bridge_source_imports(const char* source) {
  return run_string_action(
      [&] { return check_lean_source_imports(lean_mk_string(source)); });
}

extern "C" char* lean_bridge_module_imports(const char* olean_path) {
  return run_string_action(
      [&] { return check_lean_module_imports(lean_mk_string(olean_path)); });
}

extern "C" const char* lean_bridge_version() { return LEAN_VERSION_STRING; }

extern "C" void lean_bridge_free(char* result) { std::free(result); }

extern "C" LeanBridgeOptions lean_bridge_parse_options(
    int argc, char** argv, int audit, int output_fd, int error_fd,
    int (*cancelled)(const void*), const void* context) {
  LeanBridgeOptions parsed{nullptr, 1, 0, 1};
  try {
    std::call_once(initialization_flag, initialize_lean);
    std::unique_lock lock(evaluation_mutex, std::defer_lock);
    while (!lock.try_lock()) {
      if (cancelled && cancelled(context)) {
        parsed.status = 130;
        return parsed;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return on_lean_thread([&] {
      CommandStream output(output_fd, "w", lean_get_set_stdout, cancelled,
                           context);
      CommandStream error(error_fd, "w", lean_get_set_stderr, cancelled,
                          context);
      lean::object_ref options(lean_shell_options_mk(lean_box(0)));
      // getopt is process-global on Darwin. Only this serialized embedding uses
      // libc's parser here; ios_system commands use their separate TLS getopt.
      int savedIndex = optind, savedReset = optreset, savedErrors = opterr,
          savedOption = optopt;
      char* savedArgument = optarg;
      struct Restore {
        int index, reset, errors, option;
        char* argument;
        ~Restore() {
          optind = index;
          optreset = reset;
          opterr = errors;
          optopt = option;
          optarg = argument;
        }
      } restore{savedIndex, savedReset, savedErrors, savedOption,
                savedArgument};
      optind = 1;
      optreset = 1;
      opterr = 0;
      for (;;) {
        int c = getopt_long(argc, argv, g_opt_str, g_long_options, nullptr);
        if (c == -1) break;
        if (c == 'h') {
          if (audit) {
            dprintf(output_fd,
                    "Usage: check_lean [options] SOURCE\n"
                    "       check_lean [options] --stdin\n"
                    "SOURCE is Lean source text, not a file name.\n"
                    "Reports valid, conditional (with assumptions), invalid, "
                    "or error.\n"
                    "Exit status: valid/conditional 0; invalid/error 1; "
                    "interrupted 130.\n");
          }
          dprintf(
              output_fd,
              "Embedded Lean supports proof checking with -D, -T/--timeout, "
              "--stdin, -q, --json (lean only), -E/--error, and version/help "
              "options.\n"
              "The upstream help below also lists modes unavailable in this "
              "embedding.\n");
        }
        // Only these handlers have immediate process-global side effects.
        // Let upstream validate other options before rejecting unsupported
        // modes.
        if (strchr("esplB", c)) {
          dprintf(error_fd,
                  "%s: option -%c is not supported by embedded Lean\n", argv[0],
                  c);
          return parsed;
        }
        if (audit && c == 'J') {
          dprintf(error_fd,
                  "check_lean: --json is unavailable; audit responses are "
                  "plain text\n");
          return parsed;
        }
        auto argument = optarg ? lean::mk_option_some(lean_mk_string(optarg))
                               : lean::mk_option_none();
        lean_object* result =
            lean_shell_options_process(options.steal(), c, argument);
        if (lean_io_result_is_error(result)) {
          parsed.status = lean_unbox(lean_io_result_get_error(result));
          lean_dec(result);
          return parsed;
        }
        options = lean::object_ref(lean_io_result_get_value(result), true);
        lean_dec(result);
        if (!strchr("vVghfDITqJE", c)) {
          dprintf(error_fd,
                  "%s: option -%c is not supported by embedded Lean\n", argv[0],
                  c);
          return parsed;
        }
      }
      parsed.argument_index = optind;
      parsed.use_stdin = lean_bridge_options_stdin(options.to_obj_arg());
      parsed.value = options.steal();
      lean_mark_mt(static_cast<lean_object*>(parsed.value));
      parsed.status = -1;
      return parsed;
    });
  } catch (const std::exception& error) {
    dprintf(error_fd, "lean: %s\n", error.what());
  }
  return parsed;
}

extern "C" void lean_bridge_free_options(LeanBridgeOptions options) {
  if (options.value) lean_dec(static_cast<lean_object*>(options.value));
}

extern "C" int lean_bridge_run_command(
    const char* source, const char* file_name, const char* base_root,
    const char* package_root, const char* artifact_json, void* options,
    int audit, FILE* input_file, int output_fd, int error_fd,
    int (*cancelled)(const void*), const void* context) {
  try {
    std::call_once(initialization_flag, initialize_lean);
    std::unique_lock lock(evaluation_mutex, std::defer_lock);
    while (!lock.try_lock()) {
      if (cancelled && cancelled(context)) return 130;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (cancelled && cancelled(context)) return 130;
    return on_lean_thread([&] {
      CommandStream input(fileno(input_file), "r", lean_get_set_stdin,
                          cancelled, context, input_file);
      CommandStream output(output_fd, "w", lean_get_set_stdout, cancelled,
                           context);
      CommandStream error(error_fd, "w", lean_get_set_stderr, cancelled,
                          context);
      CommandCancellation cancellation(cancelled, context);
      lean_inc(static_cast<lean_object*>(options));
      uint64_t timeout =
          lean_bridge_options_timeout(static_cast<lean_object*>(options));
      lean::scope_heartbeat heartbeat(0);
      lean::scope_max_heartbeat limit(
          timeout > SIZE_MAX / 1000 ? SIZE_MAX : timeout * 1000);
      lean_inc(cancellation.token);
      lean_inc(static_cast<lean_object*>(options));
      lean_object* result = lean_bridge_command(
          lean_mk_string(source), lean_mk_string(file_name),
          lean_mk_string(base_root), lean_mk_string(package_root),
          lean_mk_string(artifact_json), static_cast<lean_object*>(options),
          audit ? 1 : 0, cancellation.token);
      int status;
      if (lean_io_result_is_error(result)) {
        lean_object* error = lean_io_result_get_error(result);
        lean_inc(error);
        lean_object* message = lean_io_error_to_string(error);
        if (audit)
          dprintf(output_fd, "status: error\nerror: %s\n",
                  lean_string_cstr(message));
        else
          dprintf(error_fd, "lean: %s\n", lean_string_cstr(message));
        lean_dec(message);
        status = 1;
      } else {
        status = lean_unbox_uint32(lean_io_result_get_value(result));
      }
      lean_dec(result);
      bool streamsOK = output.flush();
      streamsOK = error.flush() && streamsOK;
      if (!streamsOK && status == 0) status = 1;
      return cancelled && cancelled(context) ? 130 : status;
    });
  } catch (const std::exception& error) {
    dprintf(error_fd, "lean: %s\n", error.what());
    return 1;
  } catch (...) {
    dprintf(error_fd, "lean: embedded runtime error\n");
    return 1;
  }
}
