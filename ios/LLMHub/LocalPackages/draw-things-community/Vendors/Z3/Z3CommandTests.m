#import <Foundation/Foundation.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include "Z3Bridge.h"
#include "Z3Command.h"
#import "ios_system/ios_system.h"

static const char session[] = "z3-command-tests";

static void require(bool condition, NSString *message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    exit(1);
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

static void check(const char *command, int expectedStatus, NSString *expectedOutput) {
  FILE *input = tmpfile(), *output = tmpfile(), *error = tmpfile();
  require(input && output && error, @"tmpfile");
  ios_setStreams(input, output, error);
  int status = ios_system_osh(command);
  NSString *actual = contents(output);
  NSString *diagnostic = contents(error);
  require(status == expectedStatus && [actual isEqualToString:expectedOutput],
          [NSString stringWithFormat:@"%s: status=%d stdout=%@ stderr=%@", command, status, actual,
                                     diagnostic]);
  ios_setStreams(stdin, stdout, stderr);
  fclose(input);
  fclose(output);
  fclose(error);
}

struct CancellationRun {
  FILE *input, *output, *error;
  _Atomic(bool) started;
  int status;
};

static void *cancelledCommand(void *value) {
  @autoreleasepool {
    struct CancellationRun *run = value;
    ios_switchSession(session);
    ios_setStreams(run->input, run->output, run->error);
    atomic_store(&run->started, true);
    // OSH snapshots stdin before dispatch; use argv execution to exercise
    // cancellation while Z3 itself is waiting for a live input pipe.
    const char *arguments[] = {"z3", "-in"};
    run->status = ios_system(2, arguments);
    return NULL;
  }
}

int main(void) {
  @autoreleasepool {
    ios_switchSession(session);
    ios_setStreams(stdin, stdout, stderr);
    initializeEnvironment();
    require(ios_registerCommand("z3", z3_main) == 1, @"register Z3");
    require([commandsAsArray() containsObject:@"z3"], @"command listing");
    require(ios_executable("z3"), @"command lookup");

    NSFileManager *manager = NSFileManager.defaultManager;
    NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory()
                                             stringByAppendingPathComponent:NSUUID.UUID.UUIDString]
                             isDirectory:YES];
    NSURL *project = [root URLByAppendingPathComponent:@"Project" isDirectory:YES];
    NSURL *documents = [root URLByAppendingPathComponent:@"Documents" isDirectory:YES];
    NSURL *library = [root URLByAppendingPathComponent:@"Library" isDirectory:YES];
    NSURL *caches = [root URLByAppendingPathComponent:@"Caches" isDirectory:YES];
    NSURL *bundle = [root URLByAppendingPathComponent:@"Bundle" isDirectory:YES];
    for (NSURL *url in @[ project, documents, library, caches, bundle ])
      require([manager createDirectoryAtURL:url
                  withIntermediateDirectories:YES
                                   attributes:nil
                                        error:nil],
              @"test directory");
    require(ios_setSystemPaths(documents, library, caches, root, project, bundle), @"system paths");
    ios_setDirectoryURL(project);

    check("which z3", 0, @"/usr/bin/z3\n");
    check("printf '(check-sat)\\n' | z3 -in", 0, @"sat\n");
    check("printf '(assert false)\\n(check-sat)\\n' | /usr/bin/z3 -in", 0, @"unsat\n");
    check("printf '(check-sat)\\n' > 'input file.smt2'; z3 -file:'input file.smt2'", 0, @"sat\n");
    check("z3 /home/Project/'input file.smt2'", 0, @"sat\n");
    check("printf '(include \"input file.smt2\")\\n' | z3 -in", 0, @"sat\n");
    check("printf '(check-sat)\\n' | z3 -in > result.txt; cat result.txt", 0, @"sat\n");
    check("z3 -invalid >/dev/null 2>error.txt; test -s error.txt", 0, @"");
    check("printf '(set-option :regular-output-channel \"solver.txt\")\\n(check-sat)\\n' | z3 -in; "
          "cat solver.txt",
          0, @"sat\n");
    check("printf '(set-option :regular-output-channel \"/app_bundle/forbidden.txt\")\\n' | z3 -in "
          "2>/dev/null",
          1, @"");
    require(
        ![manager fileExistsAtPath:[[bundle URLByAppendingPathComponent:@"forbidden.txt"] path]],
        @"bundle stays read-only");

    const char *arguments[] = {"z3", "-version"};
    require(ios_system(2, arguments) == 0, @"direct argv invocation");

    // The command's pthread has cancellation disabled while C++ owns Z3 state.
    int descriptors[2];
    require(pipe(descriptors) == 0, @"cancellation pipe");
    struct CancellationRun run = {
        .input = fdopen(descriptors[0], "r"), .output = tmpfile(), .error = tmpfile()};
    atomic_init(&run.started, false);
    pthread_t worker;
    require(pthread_create(&worker, NULL, cancelledCommand, &run) == 0, @"command thread");
    while (!atomic_load(&run.started)) usleep(1000);
    usleep(200000);
    ios_switchSession(session);
    require(ios_kill() == 0, @"request shell cancellation");
    require(pthread_join(worker, NULL) == 0, @"join cancelled shell");
    require(run.status == 130, [NSString stringWithFormat:@"cancel status %d", run.status]);
    ios_setStreams(stdin, stdout, stderr);
    fclose(run.input);
    fclose(run.output);
    fclose(run.error);
    close(descriptors[1]);
    check("printf '(check-sat)\\n' | z3 -in", 0, @"sat\n");

    Z3BridgeResult result = Z3BridgeCheckSMT("(assert false)", 1000);
    require(result.status == Z3BridgeStatusUnsat, @"check_smt survives cancellation");
    Z3BridgeFreeResult(result);
    ios_closeSession(session);
    [manager removeItemAtURL:root error:nil];
    puts("Z3 command integration tests passed");
  }
  return 0;
}
