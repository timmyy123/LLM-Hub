#include "FFmpegCommand.h"

// Keep both production entry points reachable. Never executed on the build host.
int main(int argc, char **argv)
{
    FFmpegCommandContext context = {
        .input = stdin, .output = stdout, .error = stderr, .directory = "/",
    };
    return argc > 1 ? FFmpegCommandRun(argc, argv, &context)
                    : FFprobeCommandRun(argc, argv, &context);
}
