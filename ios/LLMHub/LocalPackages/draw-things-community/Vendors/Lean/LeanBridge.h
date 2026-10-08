#pragma once
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

char* lean_bridge_check(const char* source, const char* base_root,
                        const char* package_root, const char* artifact_json);
// Internal resource metadata, not tool responses. NULL indicates a runtime
// error. Returned strings must be released with lean_bridge_free.
char* lean_bridge_source_imports(const char* source);
char* lean_bridge_module_imports(const char* olean_path);
const char* lean_bridge_version(void);
void lean_bridge_free(char* result);

// Synchronous embedded command. Input is borrowed stdio (including buffered
// bytes); output descriptors are duplicated. Cancellation may be polled from
// a helper thread. No stream or callback context is used after return.
typedef struct {
  void* value;
  int argument_index;
  int use_stdin;
  // -1: parsed; otherwise an early-exit status (e.g. help or invalid options).
  int status;
} LeanBridgeOptions;
LeanBridgeOptions lean_bridge_parse_options(int argc, char** argv, int audit,
                                            int output_fd, int error_fd,
                                            int (*cancelled)(const void*),
                                            const void* context);
void lean_bridge_free_options(LeanBridgeOptions options);
int lean_bridge_run_command(const char* source, const char* file_name,
                            const char* base_root, const char* package_root,
                            const char* artifact_json, void* options, int audit,
                            FILE* input, int output_fd, int error_fd,
                            int (*cancelled)(const void*), const void* context);

#ifdef __cplusplus
}
#endif
