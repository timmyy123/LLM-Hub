#include "Z3Command.h"
#import <Foundation/Foundation.h>
#include <pthread.h>
#include <array>
#include <filesystem>
#include <stdexcept>
#include "Z3Shell.h"
extern "C" {
#import "ios_system/ios_system.h"
}

// ios_error.h also defines libc macros, which must not affect C++ headers.
extern "C" int ios_resolveDirectoryAlias(const char *, char *, size_t);

int z3_main(int argc, char **argv) {
  // Darwin pthread cancellation does not unwind C++ solver objects. Poll the
  // session request instead and restore cancellation only after cleanup.
  int cancellationState;
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cancellationState);
  int status;
  // All C++ objects must be destroyed before reenabling pthread cancellation.
  try {
    const char *pwd = ios_getenv("PWD");
    std::string directory = pwd ? pwd : std::filesystem::current_path().string();
    std::array<std::pair<std::string, std::string>, 3> aliases;
    const char *names[] = {"/home", "/tmp", "/app_bundle"};
    for (size_t i = 0; i < aliases.size(); ++i) {
      char resolved[PATH_MAX];
      int result = ios_resolveDirectoryAlias(names[i], resolved, sizeof(resolved));
      aliases[i] = {names[i], result > 0 ? resolved : names[i]};
    }
    status = Z3ShellRun(argc, argv, thread_stdin, thread_stdout, thread_stderr,
                        ios_commandCancellationRequested, ios_getCommandCancellationContext(),
                        [&](const char *value, bool writing) {
                          auto path = std::filesystem::path(value);
                          if (path.is_relative()) path = std::filesystem::path(directory) / path;
                          std::string resolved = path.lexically_normal().string();
                          for (const auto &[alias, target] : aliases) {
                            if (resolved == alias || resolved.starts_with(alias + "/")) {
                              if (writing && alias == "/app_bundle")
                                throw std::runtime_error("/app_bundle is read-only");
                              return target + resolved.substr(alias.size());
                            }
                          }
                          return resolved;
                        });
  } catch (const std::exception &error) {
    fprintf(thread_stderr, "z3: %s\n", error.what());
    status = 1;
  }
  pthread_setcancelstate(cancellationState, nullptr);
  return status;
}
