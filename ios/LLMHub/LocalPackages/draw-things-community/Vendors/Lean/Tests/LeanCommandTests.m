#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "Vendors/Lean/LeanBridge.h"
#include "Vendors/Lean/LeanCommand.h"
#import "ios_system/ios_system.h"

static const char session[] = "lean-command-tests";
// No native reference or -u root: release dead stripping should remove this,
// despite its public visibility and Lean-like name outside the Lean archives.
__attribute__((visibility("default"))) int lean_unretained_export_probe(void) { return 42; }

static void require(bool condition, NSString *message) {
  if (!condition) {
    dprintf(STDERR_FILENO, "FAIL: %s\n", message.UTF8String);
    _Exit(1);
  }
}
static NSString *contents(FILE *file) {
  fflush(file);
  rewind(file);
  NSMutableData *data = [NSMutableData data];
  char buffer[4096];
  size_t count;
  while ((count = fread(buffer, 1, sizeof(buffer), file)) > 0)
    [data appendBytes:buffer length:count];
  return [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
}
static NSString *check(const char *command, int expectedStatus, NSString *expected) {
  FILE *input = tmpfile(), *output = tmpfile(), *error = tmpfile();
  require(input && output && error, @"tmpfile");
  ios_setStreams(input, output, error);
  int status = ios_system_osh(command);
  NSString *actual = contents(output), *diagnostic = contents(error);
  dprintf(STDERR_FILENO, "[LeanCommand] %s -> %d\n", command, status);
  require(status == expectedStatus && (expected.length == 0 || [actual containsString:expected]),
          [NSString stringWithFormat:@"%s: status=%d stdout=%@ stderr=%@", command, status, actual,
                                     diagnostic]);
  ios_setStreams(stdin, stdout, stderr);
  fclose(input);
  fclose(output);
  fclose(error);
  return diagnostic;
}
struct CancellationRun {
  FILE *input, *output, *error;
  _Atomic(bool) started;
  int status;
  const char *source;
};
static void *cancelledCommand(void *value) {
  @autoreleasepool {
    struct CancellationRun *run = value;
    ios_switchSession(session);
    ios_setStreams(run->input, run->output, run->error);
    atomic_store(&run->started, true);
    const char *arguments[] = {"check_lean", run->source ? run->source : "--stdin"};
    run->status = ios_system(2, arguments);
    return NULL;
  }
}
int main(int argc, char **argv) {
  @autoreleasepool {
    require(argc == 3 || argc == 4, @"archive, symbol-list, and optional skill examples arguments");
    NSString *skill = nil;
    if (argc == 4) {
      skill = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[3]]
                                       encoding:NSUTF8StringEncoding
                                          error:nil];
      require(skill.length > 0, @"read skill examples");
    }
    NSString *symbols = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]
                                                  encoding:NSUTF8StringEncoding
                                                     error:nil];
    require(symbols.length > 0, @"Lean symbol list");
    NSUInteger symbolCount = 0;
    for (NSString *symbol in [symbols componentsSeparatedByString:@"\n"]) {
      if (symbol.length == 0) continue;
      require([symbol hasPrefix:@"_"] && dlsym(RTLD_MAIN_ONLY, symbol.UTF8String + 1) != NULL,
              [@"missing interpreter export: " stringByAppendingString:symbol]);
      ++symbolCount;
    }
    dprintf(STDERR_FILENO, "[LeanCommand] %lu interpreter exports found after stripping\n",
            (unsigned long)symbolCount);
    for (const char **symbol =
             (const char *[]){"uv_default_loop", "uv_loop_init", "uv_run", "mi_malloc", "mi_free",
                              "mpz_init", "mp_set_memory_functions", NULL};
         *symbol; ++symbol) {
      require(dlsym(RTLD_MAIN_ONLY, *symbol) == NULL,
              [NSString stringWithFormat:@"private dependency leaked: %s", *symbol]);
    }
    Dl_info allocator;
    require(dladdr((const void *)malloc, &allocator) != 0 &&
                strstr(allocator.dli_fname, "/libsystem_malloc.dylib") != NULL,
            @"host malloc must still resolve to libSystem");
#ifdef NDEBUG
    require(dlsym(RTLD_MAIN_ONLY, "lean_unretained_export_probe") == NULL,
            @"unrelated export should be dead stripped");
#endif
    NSString *archive = [[[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath]
        stringByResolvingSymlinksInPath];
    if (![archive hasPrefix:@"/"])
      archive = [NSFileManager.defaultManager.currentDirectoryPath
          stringByAppendingPathComponent:archive];
    ios_switchSession(session);
    ios_setStreams(stdin, stdout, stderr);
    initializeEnvironment();
    require(ios_registerCommand("lean", lean_main) == 1, @"register lean");
    require(ios_registerCommand("check_lean", check_lean_main) == 1, @"register check_lean");
    NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory()
                                             stringByAppendingPathComponent:NSUUID.UUID.UUIDString]
                             isDirectory:YES];
    NSFileManager *manager = NSFileManager.defaultManager;
    require([manager createDirectoryAtURL:root
                withIntermediateDirectories:YES
                                 attributes:nil
                                      error:nil],
            @"root");
    require(ios_setSystemPaths(root, root, root, root, root, root), @"system paths");
    ios_setDirectoryURL(root);
    ios_setenv("LEAN_RESOURCE_ARCHIVE", archive.UTF8String, 1);
    ios_setenv("LEAN_RESOURCE_CACHE",
               [[root URLByAppendingPathComponent:@"cache"] fileSystemRepresentation], 1);

    if (skill) {
      // Execute the documented proof submissions through the actual shell,
      // including quoted heredocs, instead of maintaining a second example copy.
      NSRegularExpression *fences = [NSRegularExpression
          regularExpressionWithPattern:@"(?ms)^```sh\n(.*?)^```$" options:0 error:nil];
      NSArray<NSTextCheckingResult *> *examples = [fences matchesInString:skill options:0
                                                                  range:NSMakeRange(0, skill.length)];
      require(examples.count > 0, @"skill contains executable proof examples");
      for (NSTextCheckingResult *example in examples) {
        NSString *command = [skill substringWithRange:[example rangeAtIndex:1]];
        check(command.UTF8String, 0, @"status: valid");
      }
    }

    check("node -e \"setTimeout(() => console.log('node before lean'), 5)\"", 0,
          @"node before lean");
    check("which lean check_lean", 0, @"/usr/bin/check_lean");
    check("lean --short-version", 0, @"4.34.0");
    check("lean -V", 0, @"4.34.0");
    check("check_lean --help", 0, @"SOURCE is Lean source text");
    check("lean -DmaxHeartbeats=bad --stdin </dev/null", 1, @"");
    check("lean --plugin=/tmp/forbidden", 1, @"");
    check("lean --not-a-lean-option", 1, @"");
    require([check("lean --memory=bad", 1, @"") containsString:@"expected numeric argument"],
            @"upstream numeric-option validation");
    require([check("lean --timeout", 1, @"") containsString:@"argument missing"],
            @"upstream optional long-option argument handling");
    require([check("lean -0", 1, @"") containsString:@"Unknown command line option"],
            @"upstream unknown short-option handling");
    check("check_lean --stdin 'ignored source'", 1, @"");
    check("check_lean 'module\nprelude\npublic import Init.Prelude\npublic theorem proof : True := "
          "True.intro'",
          0, @"status: valid");
    check(
        "check_lean 'module\nprelude\npublic import Init.Prelude\npublic axiom hypothesis : False'",
        0, @"status: conditional\nassumptions:\n  hypothesis");
    check("check_lean 'module\nprelude\npublic import Init.Prelude\npublic theorem hole : False := "
          "by sorry'",
          1, @"status: invalid");
    check("printf 'module\\nprelude\\npublic import Init.Prelude\\npublic theorem proof : True := "
          "True.intro\\n' > 'proof file.lean'; lean 'proof file.lean'",
          0, @"");
    check("lean '/home/proof file.lean'", 0, @"");
    check("check_lean --stdin < 'proof file.lean' > audit.txt; cat audit.txt", 0, @"status: valid");
    check("lean --stdin < 'proof file.lean'", 0, @"");
    check("printf 'theorem hole : False := by sorry\\n' | lean --stdin", 0, @"sorry");
    check("printf 'theorem bad : False := by trivial\\n' | lean --json --stdin", 1,
          @"\"severity\":\"error\"");
    check("printf 'theorem hole : False := by sorry\\n' | lean --error=hasSorry --stdin", 1,
          @"error");
    check("printf '#eval IO.println \"stream-captured\"\\n' | lean --stdin", 0, @"stream-captured");
    check("printf '#eval (IO.Process.exit 0 : IO Unit)\\n' | lean --stdin", 1,
          @"unavailable in embedded Lean");
    check("check_lean 'theorem recovered : True := by trivial'", 0, @"status: valid");
    NSURL *initArtifact = nil;
    for (NSURL *file in [manager enumeratorAtURL:root includingPropertiesForKeys:nil options:0
                                  errorHandler:nil]) {
      // A broad import also extracts Lean/Linter/Init.olean; select core Init.
      if ([file.lastPathComponent isEqualToString:@"Init.olean"] &&
          [manager fileExistsAtPath:[[file URLByDeletingLastPathComponent]
                                       URLByAppendingPathComponent:@"Init/Prelude.olean"].path]) {
        initArtifact = file;
        break;
      }
    }
    require(initArtifact != nil, @"Init was lazily extracted");
    // Exercise native metadata serialization, where a pure result computation
    // used to be reordered after freeing its mapped ModuleData backing store.
    for (int attempt = 0; attempt < 3; ++attempt) {
      char *metadata = lean_bridge_module_imports(initArtifact.fileSystemRepresentation);
      require(metadata != NULL, @"read native module metadata");
      NSData *data = [NSData dataWithBytes:metadata length:strlen(metadata)];
      lean_bridge_free(metadata);
      NSArray *imports = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
      require([imports isKindOfClass:NSArray.class] && [imports containsObject:@"Init.Prelude"],
              @"module dependencies must survive mapped-region release");
    }
    // Lean's process-lifetime UV worker remains alive while Node starts and
    // shuts down command loops. Exercise both timers, then Lean again.
    const char *timerCommand =
        "check_lean 'module\npublic meta import Std.Internal.UV.Timer\n"
        "#eval show IO Unit from do\n"
        "  let timer ← Std.Internal.UV.Timer.mk 5 false\n"
        "  let promise ← timer.next\n"
        "  unless promise.result?.get.isSome do throw (IO.userError \"timer dropped\")\n"
        "  IO.println \"lean timer completed\"'";
    check(timerCommand, 0, @"lean timer completed");
    check("node -e \"setTimeout(() => console.log('node timer completed'), 5)\"", 0,
          @"node timer completed");
    check(timerCommand, 0, @"lean timer completed");
    check("node -e \"setTimeout(() => console.log('node timer restarted'), 5)\"", 0,
          @"node timer restarted");
    check("check_lean '#eval (2^256 : Nat) / 2^192'", 0, @"18446744073709551616");
    // Upstream -T is not a wall-clock timer and does not reject this small proof.
    check("check_lean --timeout=1 'theorem limited : True := by trivial'", 0, @"status: valid");
    check("check_lean -DmaxHeartbeats=1 'example : True := by repeat skip'", 1, @"timeout");
    check("check_lean -D maxHeartbeats=200000 'theorem limit_reset : True := by trivial'", 0,
          @"status: valid");
    check("check_lean -- '-- source comment\ntheorem after_separator : True := by trivial'", 0,
          @"status: valid");
    check("check_lean 'theorem first_bad : False := by trivial\ntheorem last_good : True := by "
          "trivial'",
          1, @"status: invalid");

    int descriptors[2];
    require(pipe(descriptors) == 0, @"pipe");
    struct CancellationRun run = {
        .input = fdopen(descriptors[0], "r"), .output = tmpfile(), .error = tmpfile()};
    atomic_init(&run.started, false);
    pthread_t worker;
    require(pthread_create(&worker, NULL, cancelledCommand, &run) == 0, @"worker");
    while (!atomic_load(&run.started)) usleep(1000);
    usleep(200000);
    ios_switchSession(session);
    require(ios_kill() == 0, @"cancel");
    require(pthread_join(worker, NULL) == 0, @"join");
    require(run.status == 130, [NSString stringWithFormat:@"cancel status %d", run.status]);
    ios_setStreams(stdin, stdout, stderr);
    fclose(run.input);
    fclose(run.output);
    fclose(run.error);
    close(descriptors[1]);
    check("check_lean 'theorem after_cancel : True := by trivial'", 0, @"status: valid");

    struct CancellationRun proofRun = {
        .input = tmpfile(),
        .output = tmpfile(),
        .error = tmpfile(),
        .source = "set_option maxHeartbeats 0\nset_option stderrAsMessages false\n"
                  "#eval show IO Unit from do\n  let out ← IO.getStderr\n  out.putStrLn "
                  "\"ready\"\n  out.flush\n"
                  "example : True := by repeat skip\n"};
    atomic_init(&proofRun.started, false);
    require(pthread_create(&worker, NULL, cancelledCommand, &proofRun) == 0, @"proof worker");
    bool ready = false;
    for (int i = 0; i < 1000; ++i) {
      struct stat info;
      if (fstat(fileno(proofRun.error), &info) == 0 && info.st_size > 0) {
        ready = true;
        break;
      }
      usleep(10000);
    }
    require(ready, @"proof reached tactic after imports");
    ios_switchSession(session);
    require(ios_kill() == 0, @"cancel elaboration");
    require(pthread_join(worker, NULL) == 0, @"join cancelled proof");
    require(proofRun.status == 130,
            [NSString stringWithFormat:@"proof cancel status %d", proofRun.status]);
    ios_setStreams(stdin, stdout, stderr);
    fclose(proofRun.input);
    fclose(proofRun.output);
    fclose(proofRun.error);
    check("check_lean 'theorem after_tactic_cancel : True := by trivial'", 0, @"status: valid");

    require(pipe(descriptors) == 0, @"output pipe");
    struct CancellationRun outputRun = {
        .input = tmpfile(),
        .output = tmpfile(),
        .error = fdopen(descriptors[1], "w"),
        .source =
            "set_option stderrAsMessages false\n"
            "#eval show IO Unit from do\n  let out ← IO.getStderr\n"
            "  out.putStrLn \"ready\"\n  out.flush\n"
            "  for _ in [:100000] do\n    out.putStr (String.ofList (List.replicate 8192 'x'))\n"
            "  out.flush\n"};
    int outputFlags = fcntl(descriptors[1], F_GETFL);
    atomic_init(&outputRun.started, false);
    require(pthread_create(&worker, NULL, cancelledCommand, &outputRun) == 0, @"output worker");
    struct pollfd readable = {.fd = descriptors[0], .events = POLLIN};
    require(poll(&readable, 1, 10000) == 1, @"running IO reached output stream");
    usleep(200000);  // Leave the pipe unread so the writer blocks.
    ios_switchSession(session);
    require(ios_kill() == 0, @"cancel blocked output");
    require(pthread_join(worker, NULL) == 0, @"join blocked output");
    require(outputRun.status == 130,
            [NSString stringWithFormat:@"output cancel status %d", outputRun.status]);
    // ios_system itself sets NOSIGPIPE; only the adapter's NONBLOCK change is scoped.
    require((fcntl(descriptors[1], F_GETFL) & O_NONBLOCK) == (outputFlags & O_NONBLOCK),
            @"restore output descriptor flags");
    ios_setStreams(stdin, stdout, stderr);
    fclose(outputRun.input);
    fclose(outputRun.output);
    fclose(outputRun.error);
    close(descriptors[0]);
    check("check_lean 'theorem after_output_cancel : True := by trivial'", 0, @"status: valid");
    ios_closeSession(session);
    [manager removeItemAtURL:root error:nil];
    puts("Lean command integration tests passed");
  }
  return 0;
}
