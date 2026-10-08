#ifndef LOCAL_CODE_RIPGREP_SEARCH_H
#define LOCAL_CODE_RIPGREP_SEARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
  LOCAL_CODE_RIPGREP_MODE_TEXT = 0,
  LOCAL_CODE_RIPGREP_MODE_REGEX = 1,
  LOCAL_CODE_RIPGREP_MODE_FILES = 2,
};

enum {
  LOCAL_CODE_RIPGREP_RESULT_FILE = 0,
  LOCAL_CODE_RIPGREP_RESULT_DIRECTORY = 1,
  LOCAL_CODE_RIPGREP_RESULT_LINE = 2,
};

enum {
  LOCAL_CODE_RIPGREP_STATUS_OK = 0,
  LOCAL_CODE_RIPGREP_STATUS_INVALID_REGEX = 1,
  LOCAL_CODE_RIPGREP_STATUS_ERROR = 2,
};

typedef void (*local_code_ripgrep_result_callback)(
    void *context,
    int32_t kind,
    const uint8_t *path,
    size_t path_length,
    uint64_t line_number,
    const uint8_t *line,
    size_t line_length);

typedef void (*local_code_ripgrep_error_callback)(
    void *context,
    const uint8_t *message,
    size_t message_length);

// Searches synchronously and invokes callbacks before returning. Callback
// buffers are only valid for the duration of each callback. path is
// project-relative. line excludes its LF terminator and is null for file results.
int32_t local_code_ripgrep_search(
    const char *project_root,
    const char *root,
    const char *query,
    const char *glob,
    int32_t mode,
    bool case_sensitive,
    size_t context_lines,
    size_t max_results,
    void *context,
    local_code_ripgrep_result_callback result_callback,
    local_code_ripgrep_error_callback error_callback,
    bool *max_results_reached);

#endif
