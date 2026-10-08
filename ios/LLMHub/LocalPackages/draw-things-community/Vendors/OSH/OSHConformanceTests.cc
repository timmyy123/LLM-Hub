#include "Vendors/OSH/osh_ios.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <string>
#include <thread>

namespace {

std::atomic<bool> gExternalBlocked(false);
std::atomic<bool> gReleaseExternal(false);

void Copy(FILE* input, FILE* output) {
  char bytes[4096];
  while (true) {
    size_t count = fread(bytes, 1, sizeof(bytes), input);
    if (count != 0) {
      fwrite(bytes, 1, count, output);
    }
    if (count != sizeof(bytes)) {
      break;
    }
  }
}

int RunExternal(int argc, const char* const argv[], int envc,
                const char* const environment[], FILE* input, FILE* output,
                FILE* error, void* context) {
  if (argc == 0) {
    return 127;
  }
  if (strcmp(argv[0], "cat") == 0) {
    if (argc == 1) {
      Copy(input, output);
      return 0;
    }
    for (int i = 1; i < argc; ++i) {
      FILE* file = fopen(argv[i], "rb");
      if (file == nullptr) {
        fprintf(error, "cat: %s: %s\n", argv[i], strerror(errno));
        return 1;
      }
      Copy(file, output);
      fclose(file);
    }
    return 0;
  }
  if (strcmp(argv[0], "grep") == 0 && argc == 2) {
    char* line = nullptr;
    size_t capacity = 0;
    int found = 0;
    while (getline(&line, &capacity, input) >= 0) {
      if (strstr(line, argv[1]) != nullptr) {
        fputs(line, output);
        found = 1;
      }
    }
    free(line);
    return found ? 0 : 1;
  }
  if (strcmp(argv[0], "showargv") == 0) {
    for (int i = 1; i < argc; ++i) {
      fprintf(output, "[%s]", argv[i]);
    }
    fputc('\n', output);
    return 0;
  }
  if (strcmp(argv[0], "showenv") == 0 && argc == 2) {
    std::string prefix = std::string(argv[1]) + "=";
    for (int i = 0; i < envc; ++i) {
      if (strncmp(environment[i], prefix.c_str(), prefix.size()) == 0) {
        fprintf(output, "%s\n", environment[i] + prefix.size());
        return 0;
      }
    }
    return 1;
  }
  if (strcmp(argv[0], "big") == 0) {
    char bytes[4096];
    memset(bytes, 'x', sizeof(bytes));
    size_t remaining = argc == 2 ? strtoul(argv[1], nullptr, 10) : 0;
    while (remaining != 0) {
      size_t count = remaining < sizeof(bytes) ? remaining : sizeof(bytes);
      fwrite(bytes, 1, count, output);
      remaining -= count;
    }
    return 0;
  }
  if (strcmp(argv[0], "countbytes") == 0) {
    char bytes[4096];
    size_t total = 0;
    size_t count;
    while ((count = fread(bytes, 1, sizeof(bytes), input)) != 0) {
      total += count;
    }
    fprintf(output, "%zu\n", total);
    return 0;
  }
  if (strcmp(argv[0], "wait_external") == 0) {
    gExternalBlocked.store(true, std::memory_order_release);
    while (!gReleaseExternal.load(std::memory_order_acquire)) {
      usleep(1000);
    }
    return 0;
  }
  fprintf(error, "%s: command not found\n", argv[0]);
  return 127;
}

struct Context {
  bool cancelled;
  osh_ios_signal_handler signal_handlers[NSIG];
};

int IsCancelled(void* context) {
  return static_cast<Context*>(context)->cancelled ? 1 : 0;
}

int SendSignal(int identifier, int signal_number, int process_group,
               void* context) {
  Context* test_context = static_cast<Context*>(context);
  if (identifier == 42 && signal_number > 0 && signal_number < NSIG &&
      test_context->signal_handlers[signal_number] != nullptr) {
    test_context->signal_handlers[signal_number](signal_number);
  }
  return 0;
}

int SetSignalHandler(int signal_number, osh_ios_signal_handler handler,
                     void* context) {
  if (signal_number > 0 && signal_number < NSIG) {
    static_cast<Context*>(context)->signal_handlers[signal_number] = handler;
  }
  return 0;
}

int GetProcessId(void* context) {
  return 42;
}

struct Result {
  int status;
  std::string output;
  std::string error;
};

std::string Read(FILE* file) {
  rewind(file);
  std::string result;
  char bytes[4096];
  while (true) {
    size_t count = fread(bytes, 1, sizeof(bytes), file);
    result.append(bytes, count);
    if (count != sizeof(bytes)) {
      break;
    }
  }
  return result;
}

Result Run(const char* command, const char* input_contents = "",
           bool cancelled = false) {
  FILE* input = tmpfile();
  FILE* output = tmpfile();
  FILE* error = tmpfile();
  fputs(input_contents, input);
  rewind(input);
  const char* environment[] = {"BASE=initial"};
  Context context = {cancelled};
  osh_ios_config config = {RunExternal, IsCancelled, SendSignal,
                            SetSignalHandler, GetProcessId, nullptr, 1, environment,
                            &context};
  int status = osh_ios_run(command, input, output, error, &config);
  Result result = {status, Read(output), Read(error)};
  fclose(input);
  fclose(output);
  fclose(error);
  return result;
}

void Expect(const char* name, const char* command, int expected_status,
            const char* expected_output, const char* expected_error = "",
            const char* input_contents = "", bool cancelled = false) {
  Result result = Run(command, input_contents, cancelled);
  if (result.status != expected_status || result.output != expected_output ||
      result.error.find(expected_error) == std::string::npos) {
    fprintf(stderr,
            "%s failed\nstatus: %d (expected %d)\nstdout: %s\nstderr: %s\n",
            name, result.status, expected_status, result.output.c_str(),
            result.error.c_str());
    exit(1);
  }
}

}  // namespace

int main() {
  Expect("filesystem discovery without virtual callback",
         "command -v /bin/sh", 0, "/bin/sh\n");
  Expect("bash language features",
         "set -u; arr=(one two); [[ ${arr[0]} =~ ^o ]]; "
         "f() { local x=${arr[1]}; echo ${x//w/W}; }; "
         "for ((i=0; i<2; i++)); do printf '%s ' \"${arr[i]}\"; done; f",
         0, "one two tWo\n");
  Expect("command substitution isolation",
         "x=parent; y=$(x=child; echo \"$x\"); printf '%s %s\\n' \"$x\" \"$y\"",
         0, "parent child\n");
  Expect("array isolation",
         "a=(parent); y=$(a[0]=child; echo \"${a[0]}\"); "
         "printf '%s %s\\n' \"${a[0]}\" \"$y\"",
         0, "parent child\n");
  Expect("pipeline input", "printf 'abc\\n' | { read x; echo \"$x\"; }", 0,
         "abc\n");
  Expect("pipeline state isolation",
         "a=(parent); printf 'abc\\n' | { a[0]=child; read x; "
         "echo \"$x ${a[0]}\"; }; echo \"${a[0]}\"",
         0, "abc child\nparent\n");
  Expect("pipeline status",
         "set -o pipefail; false | true; status=$?; echo \"$status\"; "
         "[[ $status == 1 ]]",
         0, "1\n");
  Expect("external pipeline", "printf 'foo\\n' | cat | grep foo", 0,
         "foo\n");
  Expect("exact external argv", "showargv 'a b' '' '*'", 0,
         "[a b][][*]\n");
  Expect("environment", "echo \"$BASE\"; export FOO=bar; showenv FOO", 0,
         "initial\nbar\n");
  Expect("virtual stdin", "read x; echo \"$x\"", 0, "from input\n", "",
         "from input\n");
  Expect("file redirects",
         "echo value >osh-ios-test-input; read x <osh-ios-test-input; "
         "typeset -p x >osh-ios-test-vars; cat osh-ios-test-vars",
         0, "declare -- x=value\n");
  Expect("here document", "cat <<'EOF'\nhello\nEOF", 0, "hello\n");
  Expect("subshell cwd isolation",
         "here=$PWD; (cd /); [[ $PWD == \"$here\" ]] && echo restored", 0,
         "restored\n");
  Expect("stderr copy", "echo error >&2", 0, "", "error\n");
  Expect("cancellation", "echo no", OSH_IOS_STATUS_CANCELLED, "", "", "",
         true);
  Expect("syntax error", "if then", 2, "", "Expected a condition");
  Expect("unknown command", "not-a-command", 127, "", "command not found");
  Expect("background foreground fallback",
         "x=parent; { x=child; echo \"$x\"; } & echo \"$x\"", 0,
         "child\nparent\n",
         "warning: '&' requested a background job; running it in the current "
         "Bash job instead");
  Expect("background fallback status", "false &", 1, "",
         "warning: '&' requested a background job; running it in the current "
         "Bash job instead");
  Expect("process substitution unsupported", "cat <(echo no)",
         OSH_IOS_STATUS_UNSUPPORTED, "",
         "unsupported in Local Code: process substitution");
  Expect("descriptor unsupported", "echo no 3>file",
         OSH_IOS_STATUS_UNSUPPORTED, "",
         "unsupported in Local Code: redirection of file descriptors");
  Expect("descriptor close unsupported", "echo no 1>&-",
         OSH_IOS_STATUS_UNSUPPORTED, "",
         "unsupported in Local Code: move/close file-descriptor redirection");
  Expect("exec primitive", "exec showargv done; echo no", 0, "[done]\n");
  Expect("exec redirect primitive", "exec >osh-ios-exec-output; echo permanent",
         0, "");
  Expect("exec redirect persisted", "cat osh-ios-exec-output", 0,
         "permanent\n");
  Expect("kill primitive", "kill -0 $$; builtin kill -0 $$", 0, "");
  Expect("virtual process and signal primitives",
         "echo $$; trap 'echo signalled' USR1; kill -USR1 $$; echo after", 0,
         "42\nsignalled\nafter\n");
  Expect("exit trap", "trap 'echo trapped' EXIT; echo body", 0,
         "body\ntrapped\n");
  Expect("subshell trap isolation",
         "trap 'echo parent' EXIT; (trap 'echo child' EXIT); echo after", 0,
         "child\nafter\nparent\n");
  Expect("empty job primitives", "jobs; wait", 0, "");
  Expect("pipeline above old limit", "big 16777217 | countbytes",
         0, "16777217\n");
  Expect("pipeline at size limit", "big 268435456 | countbytes",
         0, "268435456\n");
  Expect("pipeline size exceeded", "big 268435457 | showargv not-run",
         OSH_IOS_STATUS_UNSUPPORTED, "",
         "pipeline stage output larger than 256 MiB");
  Expect("pipeline size failure recovery",
         "set -o pipefail; big 268435457 | showargv not-run || "
         "{ echo \"status=$? stages=${PIPESTATUS[*]}\"; }; echo recovered",
         0, "status=125 stages=125\nrecovered\n",
         "pipeline stage output larger than 256 MiB");

  gExternalBlocked.store(false, std::memory_order_release);
  gReleaseExternal.store(false, std::memory_order_release);
  std::atomic<bool> secondDone(false);
  Result firstResult = {};
  Result secondResult = {};
  std::thread first([&firstResult]() {
    firstResult = Run("wait_external; echo first");
  });
  for (int attempt = 0;
       attempt < 5000 && !gExternalBlocked.load(std::memory_order_acquire);
       ++attempt) {
    usleep(1000);
  }
  std::thread second([&secondResult, &secondDone]() {
    secondResult = Run("trap 'echo second' USR1; kill -USR1 $$");
    secondDone.store(true, std::memory_order_release);
  });
  for (int attempt = 0;
       attempt < 5000 && !secondDone.load(std::memory_order_acquire);
       ++attempt) {
    usleep(1000);
  }
  bool ranConcurrently = secondDone.load(std::memory_order_acquire);
  gReleaseExternal.store(true, std::memory_order_release);
  first.join();
  second.join();
  if (!ranConcurrently || firstResult.status != 0 ||
      firstResult.output != "first\n" || secondResult.status != 0 ||
      secondResult.output != "second\n") {
    fprintf(stderr, "concurrent OSH evaluations failed\n");
    return 1;
  }
  return 0;
}
