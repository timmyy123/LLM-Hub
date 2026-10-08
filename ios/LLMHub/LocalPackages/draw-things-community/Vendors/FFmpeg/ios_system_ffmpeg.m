#import "FFmpegCommand.h"
#import "ios_system/ios_system.h"

#include <limits.h>
#include <pthread.h>

static int run_command(int argc, char **argv,
    int (*command)(int, char **, const FFmpegCommandContext *))
{
    // Keep pending pthread cancellation disabled until FFmpeg has joined every
    // worker, released its resources and released the process-wide execution lock.
    int previous;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &previous);
    char directory[PATH_MAX];
    const char *workingDirectory = ios_getenv("PWD");
    if (!workingDirectory || strlcpy(directory, workingDirectory, sizeof(directory)) >= sizeof(directory)) {
        pthread_setcancelstate(previous, NULL);
        return 1;
    }
    int status;
    @autoreleasepool {
        // UTF8String pointers remain borrowed by worker threads through return,
        // including in optimized ARC builds and while waiting for admission.
        __attribute__((objc_precise_lifetime)) NSArray<NSString *> *environment = environmentAsArray();
        const char **entries = calloc(environment.count + 1, sizeof(char *));
        if (!entries) {
            pthread_setcancelstate(previous, NULL);
            return 1;
        }
        for (NSUInteger i = 0; i < environment.count; ++i)
            entries[i] = environment[i].UTF8String;
        char home[PATH_MAX] = "", temporary[PATH_MAX] = "", bundle[PATH_MAX] = "";
        ios_resolveDirectoryAlias("/home", home, sizeof(home));
        ios_resolveDirectoryAlias("/tmp", temporary, sizeof(temporary));
        ios_resolveDirectoryAlias("/app_bundle", bundle, sizeof(bundle));
        const FFmpegCommandContext context = {
            .input = thread_stdin,
            .output = thread_stdout,
            .error = thread_stderr,
            .directory = directory,
            .home = home[0] ? home : NULL,
            .temporaryDirectory = temporary[0] ? temporary : NULL,
            .appBundle = bundle[0] ? bundle : NULL,
            .environment = entries,
            .cancellation = ios_getCommandCancellationContext(),
            .isCancelled = ios_commandCancellationRequested,
        };
        status = command(argc, argv, &context);
        free(entries);
    }
    pthread_setcancelstate(previous, NULL);
    return status;
}

int ffmpeg_main(int argc, char **argv)
{
    return run_command(argc, argv, FFmpegCommandRun);
}

int ffprobe_main(int argc, char **argv)
{
    return run_command(argc, argv, FFprobeCommandRun);
}
