#ifndef Z3_SHELL_H
#define Z3_SHELL_H

#include <stdio.h>

#include <functional>
#include <string>

// The caller owns the streams and cancellation context until this returns.
// Mutable argv follows the upstream CLI contract (the parser edits arguments).
int Z3ShellRun(
    int argc, char** argv, FILE* input, FILE* output, FILE* error,
    int (*isCancelled)(const void*), const void* cancellationContext,
    const std::function<std::string(const char*, bool)>& resolvePath = {});

#endif
