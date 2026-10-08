#include "LeanCommand.h"
#import <Foundation/Foundation.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <array>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include "LeanBridge.h"
#import "Vendors/Lean/LeanResources-Swift.h"
extern "C" {
#import "ios_system/ios_system.h"
int ios_resolveDirectoryAlias(const char *, char *, size_t);
}

namespace {
struct OptionsScope {
  LeanBridgeOptions value;
  ~OptionsScope() { lean_bridge_free_options(value); }
};

struct InputScope {
  FILE *file;
  int flags;
  bool owned;
  InputScope(FILE *file, bool owned)
      : file(file), flags(fcntl(fileno(file), F_GETFL)), owned(owned) {
    if (flags >= 0) fcntl(fileno(file), F_SETFL, flags | O_NONBLOCK);
  }
  ~InputScope() {
    if (flags >= 0) fcntl(fileno(file), F_SETFL, flags);
    if (owned) fclose(file);
  }
};

int run(int argc, char **argv, bool audit) {
  FILE *input = thread_stdin ?: stdin;
  FILE *output = thread_stdout ?: stdout;
  FILE *error = thread_stderr ?: stderr;
  const void *context = ios_getCommandCancellationContext();
  auto cancelled = ^BOOL {
    return ios_commandCancellationRequested(context) != 0;
  };
  if (cancelled()) return 130;
  OptionsScope options{lean_bridge_parse_options(argc, argv, audit, fileno(output), fileno(error),
                                                 ios_commandCancellationRequested, context)};
  if (cancelled()) return 130;
  if (options.value.status >= 0) return options.value.status;
  int operands = argc - options.value.argument_index;
  if (operands > 1 || (operands == 0 && !options.value.use_stdin) ||
      (audit && options.value.use_stdin && operands != 0)) {
    fprintf(error, "%s: expected %s\n", argv[0],
            audit ? "one source string or --stdin" : "exactly one file name (or --stdin)");
    return 1;
  }

  const char *pwd = ios_getenv("PWD");
  std::string directory = pwd ? pwd : std::filesystem::current_path().string();
  std::array<std::pair<std::string, std::string>, 3> aliases;
  const char *names[] = {"/home", "/tmp", "/app_bundle"};
  for (size_t i = 0; i < aliases.size(); ++i) {
    char resolved[PATH_MAX];
    int result = ios_resolveDirectoryAlias(names[i], resolved, sizeof(resolved));
    aliases[i] = {names[i], result > 0 ? resolved : names[i]};
  }
  auto resolve = [&](const char *value) {
    auto path = std::filesystem::path(value);
    if (path.is_relative()) path = std::filesystem::path(directory) / path;
    std::string resolved = path.lexically_normal().string();
    for (const auto &[alias, target] : aliases) {
      if (resolved == alias || resolved.starts_with(alias + "/"))
        return target + resolved.substr(alias.size());
    }
    return resolved;
  };
  NSString *source;
  NSString *fileName = audit ? @"<check_lean>" : @"<stdin>";
  if (audit && !options.value.use_stdin) {
    source = [NSString stringWithUTF8String:argv[options.value.argument_index]];
  } else {
    FILE *sourceFile = input;
    if (operands == 1 && !audit) {
      std::string resolved = resolve(argv[options.value.argument_index]);
      fileName = [NSString stringWithUTF8String:resolved.c_str()];
      if (!options.value.use_stdin) {
        sourceFile = fopen(resolved.c_str(), "rb");
        if (!sourceFile) {
          fprintf(error, "lean: %s: %s\n", resolved.c_str(), strerror(errno));
          return 1;
        }
      }
    }
    InputScope scope(sourceFile, sourceFile != input);
    NSMutableData *data = [NSMutableData data];
    char buffer[4096];
    for (;;) {
      if (cancelled()) return 130;
      size_t count = fread(buffer, 1, sizeof(buffer), sourceFile);
      if (count) [data appendBytes:buffer length:count];
      if (feof(sourceFile)) break;
      if (ferror(sourceFile) && errno != EAGAIN && errno != EINTR) {
        fprintf(error, "%s: input error: %s\n", argv[0], strerror(errno));
        return 1;
      }
      clearerr(sourceFile);
      pollfd descriptor{fileno(sourceFile), POLLIN, 0};
      poll(&descriptor, 1, 25);
    }
    source = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
  }
  if (!source) {
    fprintf(error, "%s: input is not UTF-8\n", argv[0]);
    return 1;
  }

  NSMutableArray<NSString *> *packages = [NSMutableArray array];
  const char *leanPath = ios_getenv("LEAN_PATH");
  if (leanPath && *leanPath) {
    for (NSString *entry in
         [[NSString stringWithUTF8String:leanPath] componentsSeparatedByString:@":"]) {
      if (entry.length)
        [packages addObject:[NSString stringWithUTF8String:resolve(entry.UTF8String).c_str()]];
    }
  }
  [packages addObject:[NSString stringWithUTF8String:resolve(".").c_str()]];

  const char *archivePath = ios_getenv("LEAN_RESOURCE_ARCHIVE");
  NSString *archive = archivePath && *archivePath
                          ? [NSString stringWithUTF8String:resolve(archivePath).c_str()]
                          : [NSBundle.mainBundle pathForResource:@"Lean_Init_Std" ofType:@"zip"];
  if (!archive) {
    // Bazel command-line binaries put data in runfiles, not an app bundle.
    NSString *executable = NSBundle.mainBundle.executablePath;
    NSString *candidate =
        [executable stringByAppendingString:@".runfiles/__main__/Vendors/Lean/Lean_Init_Std.zip"];
    if ([NSFileManager.defaultManager fileExistsAtPath:candidate]) archive = candidate;
  }
  if (!archive) {
    fprintf(error, "%s: Lean data archive unavailable; set LEAN_RESOURCE_ARCHIVE\n", argv[0]);
    return 1;
  }
  NSURL *cache = [[NSFileManager.defaultManager URLsForDirectory:NSCachesDirectory
                                                       inDomains:NSUserDomainMask]
                      .firstObject URLByAppendingPathComponent:@"Lean"
                                                   isDirectory:YES];
  const char *cachePath = ios_getenv("LEAN_RESOURCE_CACHE");
  if (cachePath && *cachePath)
    cache = [NSURL fileURLWithPath:[NSString stringWithUTF8String:resolve(cachePath).c_str()]
                       isDirectory:YES];

  static std::timed_mutex resourcesMutex;
  static NSMutableDictionary<NSString *, LeanRuntimeResources *> *resourcesByPath;
  LeanRuntimeResources *resources;
  {
    std::unique_lock lock(resourcesMutex, std::defer_lock);
    while (!lock.try_lock_for(std::chrono::milliseconds(25)))
      if (cancelled()) return 130;
    if (!resourcesByPath) resourcesByPath = [NSMutableDictionary dictionary];
    NSString *key = [NSString stringWithFormat:@"%@\n%@", archive, cache.path];
    resources = resourcesByPath[key];
    if (!resources) {
      NSError *failure = nil;
      NSURL *url = [NSURL fileURLWithPath:archive];
      resources = [[LeanRuntimeResources alloc]
          initWithArchiveURL:url
                 manifestURL:[[url URLByDeletingLastPathComponent]
                                 URLByAppendingPathComponent:@"Lean_Init_Std.json"]
                    cacheURL:cache
                       error:&failure];
      if (!resources) {
        fprintf(error, "%s: %s\n", argv[0], failure.localizedDescription.UTF8String);
        return 1;
      }
      resourcesByPath[key] = resources;
    }
  }
  return [resources runCommandWithSource:source
                                fileName:fileName
                             packageRoot:[packages componentsJoinedByString:@":"]
                                 options:options.value.value
                                   audit:audit
                                   input:input
                                outputFD:fileno(output)
                                 errorFD:fileno(error)
                               cancelled:cancelled];
}

int command(int argc, char **argv, bool audit) {
  int previous;
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &previous);
  int status;
  @autoreleasepool {
    try {
      status = run(argc, argv, audit);
    } catch (const std::exception &exception) {
      fprintf(thread_stderr ?: stderr, "%s: %s\n", argv[0], exception.what());
      status = 1;
    }
  }
  // All runtime objects and locks have unwound before Darwin may cancel us.
  pthread_setcancelstate(previous, nullptr);
  return status;
}
}  // namespace

int lean_main(int argc, char **argv) { return command(argc, argv, false); }
int check_lean_main(int argc, char **argv) { return command(argc, argv, true); }
