#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "Z3Bridge.h"
#include "Z3Shell.h"
#include "z3++.h"

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

struct File {
  FILE* value = tmpfile();
  File() { require(value != nullptr, "tmpfile"); }
  ~File() { fclose(value); }
  std::string contents() {
    fflush(value);
    rewind(value);
    std::string result;
    char buffer[4096];
    while (size_t count = fread(buffer, 1, sizeof(buffer), value))
      result.append(buffer, count);
    return result;
  }
};

struct Result {
  int status;
  std::string output, error;
};

Result run(std::vector<std::string> arguments, const std::string& input = "",
           FILE* stream = nullptr, std::atomic<bool>* cancel = nullptr) {
  File in, out, err;
  if (!stream) {
    fwrite(input.data(), 1, input.size(), in.value);
    rewind(in.value);
    stream = in.value;
  }
  std::vector<char*> argv;
  for (auto& argument : arguments) argv.push_back(argument.data());
  argv.push_back(nullptr);
  int status = Z3ShellRun(
      static_cast<int>(arguments.size()), argv.data(), stream, out.value,
      err.value,
      [](const void* value) -> int {
        return value && static_cast<const std::atomic<bool>*>(value)->load();
      },
      cancel);
  return {status, out.contents(), err.contents()};
}

void sat() {
  auto result = run({"z3", "-in"}, "(check-sat)\n");
  require(
      result.status == 0 && result.output == "sat\n" && result.error.empty(),
      "clean subsequent run: " + result.output + result.error);
  auto bridge = Z3BridgeCheckSMT("(assert false)", 1000);
  require(bridge.status == Z3BridgeStatusUnsat, "check_smt after CLI");
  Z3BridgeFreeResult(bridge);
}

void signalHandler(int) {}

std::string pigeonhole() {
  std::string input;
  constexpr int count = 24;
  auto variable = [](int pigeon, int hole) {
    return "p" + std::to_string(pigeon) + "h" + std::to_string(hole);
  };
  for (int p = 0; p <= count; ++p) {
    for (int h = 0; h < count; ++h)
      input += "(declare-const " + variable(p, h) + " Bool)\n";
    input += "(assert (or";
    for (int h = 0; h < count; ++h) input += " " + variable(p, h);
    input += "))\n";
  }
  for (int h = 0; h < count; ++h)
    for (int p = 0; p <= count; ++p)
      for (int q = p + 1; q <= count; ++q)
        input += "(assert (not (and " + variable(p, h) + " " + variable(q, h) +
                 ")))\n";
  return input + "(check-sat)\n";
}
}  // namespace

int main() {
  struct sigaction previous{}, installed{}, observed{};
  installed.sa_handler = signalHandler;
  sigemptyset(&installed.sa_mask);
  require(sigaction(SIGINT, &installed, &previous) == 0,
          "set host SIGINT handler");
  auto* coutBuffer = std::cout.rdbuf();

  // A host context stays alive throughout all shell invocations.
  z3::context host;
  z3::solver solver(host);
  solver.add(host.bool_val(true));
  Z3_global_param_set("timeout", "4321");

  for (const char* option : {"-h", "-version", "-p", "-pd", "-pm:smt"}) {
    auto result = run({"z3", option});
    require(result.status == 0 && !result.output.empty(),
            std::string("upstream option ") + option + result.error);
    sat();
  }
  auto invalid = run({"z3", "-not-an-option"});
  require(invalid.status != 0 && !invalid.error.empty(),
          "invalid option does not exit host");
  auto missing = run({"z3", "/missing-z3-test-file.smt2"});
  require(missing.status != 0, "missing file returns error");
  sat();

  auto model = run(
      {"z3", "-in", "-smt2", "-t:1000", "smt.random_seed=42"},
      "(declare-const x Int)\n(assert (= x 7))\n(check-sat)\n(get-value (x))\n"
      "(push)\n(assert (= x "
      "8))\n(check-sat)\n(pop)\n(check-sat)\n(get-model)\n");
  require(model.status == 0 &&
              model.output.find("((x 7))") != std::string::npos &&
              model.output.find("unsat") != std::string::npos &&
              model.output.find("define-fun x") != std::string::npos,
          "incremental SMT-LIB/model: " + model.output + model.error);
  auto statistics = run({"z3", "-in", "-st"}, "(check-sat)\n");
  require(statistics.status == 0 &&
              statistics.output.find(":") != std::string::npos,
          "statistics flag");
  sat();
  auto malformed = run(
      {"z3", "-in"}, "(set-option :exit-on-error true)\n(assert (= bad 1))\n");
  require(malformed.status != 0, "parser exit-on-error is contained");
  sat();

  auto dimacs = run({"z3", "-in", "-dimacs"}, "p cnf 1 2\n1 0\n-1 0\n");
  require(dimacs.output.find("UNSATISFIABLE") != std::string::npos,
          "DIMACS frontend: " + dimacs.output + dimacs.error);
  for (int i = 0; i < 2; ++i) {
    auto opt = run({"z3", "-in", "-wcnf"}, "p wcnf 1 2 10\n10 1 0\n1 -1 0\n");
    require(opt.status == 0 && opt.output == "sat\n   1\n",
            "repeated WCNF frontend: " + opt.output + opt.error);
  }
  sat();

  std::string difficult = pigeonhole();
  auto softTimeout = run({"z3", "-in", "-t:1"}, difficult);
  require(softTimeout.status == 0 && softTimeout.output == "unknown\n",
          "upstream soft timeout");
  auto hardTimeout = run({"z3", "-in", "-T:1"}, difficult);
  require(hardTimeout.status == 124,
          "solver hard timeout: " + hardTimeout.output + hardTimeout.error);
  sat();
  std::atomic<bool> solverCancel{false};
  std::thread canceller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    solverCancel.store(true);
  });
  auto cancelledSolver = run({"z3", "-in", "parallel.enable=true"}, difficult,
                             nullptr, &solverCancel);
  canceller.join();
  require(cancelledSolver.status == 130, "solver cooperative cancellation");
  sat();
  auto memoryLimit = run({"z3", "-in", "-memory:1"}, difficult);
  require(memoryLimit.status != 0, "memory limit returns an error");
  sat();

  // API command IDs from the pinned 4.15.4 generated api_commands.cpp.
  // A replayed reset/finalize must not invalidate the live host context.
  auto replay =
      run({"z3", "-in", "-log"}, "V \"4.15.4.0\"\nR\nC 413\nR\nC 414\n");
  require(replay.status == 0, "API-log frontend: " + replay.error);
  sat();

  // Different callers must not share flags, parameters, or captured streams.
  std::thread other([] {
    for (int i = 0; i < 10; ++i) {
      auto result = run({"z3", "-in"}, "(assert false)\n(check-sat)\n");
      require(result.status == 0 && result.output == "unsat\n",
              "concurrent shell output");
    }
  });
  for (int i = 0; i < 10; ++i) sat();
  other.join();

  // Leave the writer open: timeout/cancellation must interrupt a waiting read.
  for (bool timeout : {true, false}) {
    int descriptors[2];
    require(pipe(descriptors) == 0, "pipe");
    FILE* input = fdopen(descriptors[0], "r");
    std::atomic<bool> cancel{false};
    std::thread canceller;
    if (!timeout)
      canceller = std::thread([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cancel.store(true);
      });
    auto start = std::chrono::steady_clock::now();
    auto result = run(timeout ? std::vector<std::string>{"z3", "-in", "-T:1"}
                              : std::vector<std::string>{"z3", "-in"},
                      "", input, &cancel);
    require(result.status == (timeout ? 124 : 130),
            "blocked input exit status " + std::to_string(result.status));
    require(std::chrono::steady_clock::now() - start < std::chrono::seconds(5),
            "bounded interruption");
    if (canceller.joinable()) canceller.join();
    fclose(input);
    close(descriptors[1]);
    sat();
  }

  // A second caller waiting for the runtime must still be cancellable.
  int heldDescriptors[2];
  require(pipe(heldDescriptors) == 0, "held input pipe");
  FILE* heldInput = fdopen(heldDescriptors[0], "r");
  std::atomic<bool> heldCancel{false};
  std::thread held([&] { run({"z3", "-in"}, "", heldInput, &heldCancel); });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  std::atomic<bool> queuedCancel{true};
  auto queued = run({"z3", "-in"}, "(check-sat)\n", nullptr, &queuedCancel);
  require(queued.status == 130, "cancellation while waiting for runtime");
  heldCancel.store(true);
  held.join();
  fclose(heldInput);
  close(heldDescriptors[1]);

#ifdef __APPLE__
  // Backpressure and a closed reader must neither hang nor send process
  // SIGPIPE.
  for (bool broken : {false, true}) {
    int descriptors[2];
    require(pipe(descriptors) == 0, "output pipe");
    if (broken) close(descriptors[0]);
    FILE* output = fdopen(descriptors[1], "w");
    File input, error;
    std::string echo = "(echo \"" + std::string(1024 * 1024, 'x') + "\")\n";
    fwrite(echo.data(), 1, echo.size(), input.value);
    rewind(input.value);
    int flags = fcntl(descriptors[1], F_GETFL);
    int noSignal = fcntl(descriptors[1], F_GETNOSIGPIPE);
    std::atomic<bool> cancel{false};
    std::thread canceller([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      cancel.store(true);
    });
    char command[] = "z3", option[] = "-in";
    char* argv[] = {command, option, nullptr};
    int status = Z3ShellRun(
        2, argv, input.value, output, error.value,
        [](const void* value) -> int {
          return static_cast<const std::atomic<bool>*>(value)->load();
        },
        &cancel);
    canceller.join();
    require(status == (broken ? 1 : 130),
            "output interruption status " + std::to_string(status));
    // Darwin also reports its internal FWASWRITTEN bit after the first write.
    constexpr int mutableFlags = O_NONBLOCK | O_APPEND | O_ASYNC;
    require((fcntl(descriptors[1], F_GETFL) & mutableFlags) ==
                (flags & mutableFlags),
            "restore pipe flags");
    require(fcntl(descriptors[1], F_GETNOSIGPIPE) == noSignal,
            "restore per-descriptor SIGPIPE setting");
    fclose(output);
    if (!broken) close(descriptors[0]);
    sat();
  }
#endif

  const char* timeout = nullptr;
  require(Z3_global_param_get("timeout", &timeout) &&
              std::string(timeout) == "4321",
          "restore host parameters");
  require(solver.check() == z3::sat, "host context remains usable");
  require(sigaction(SIGINT, nullptr, &observed) == 0 &&
              observed.sa_handler == signalHandler,
          "preserve SIGINT handler");
  require(std::cout.rdbuf() == coutBuffer, "preserve host iostream");
  sigaction(SIGINT, &previous, nullptr);
  puts("Z3 shell tests passed");
}
