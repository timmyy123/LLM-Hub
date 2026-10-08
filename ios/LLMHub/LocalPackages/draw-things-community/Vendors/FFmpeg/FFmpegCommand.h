#ifndef DRAW_THINGS_FFMPEG_COMMAND_H
#define DRAW_THINGS_FFMPEG_COMMAND_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// All pointers are borrowed through return. The command is synchronous and
// serialized process-wide, including worker shutdown and resource cleanup.
typedef struct {
    FILE *input;
    FILE *output;
    FILE *error;
    const char *directory;
    const char *home;
    const char *temporaryDirectory;
    const char *appBundle;
    const char *const *environment;
    const void *cancellation;
    int (*isCancelled)(const void *);
} FFmpegCommandContext;

__attribute__((visibility("default")))
int FFmpegCommandRun(int argc, char **argv, const FFmpegCommandContext *context);

// FFprobe uses the same context and process-wide lock as FFmpeg.
__attribute__((visibility("default")))
int FFprobeCommandRun(int argc, char **argv, const FFmpegCommandContext *context);

#ifdef __cplusplus
}
#endif

#endif
