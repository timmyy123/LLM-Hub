// C interface to BigMoeOnEdge's streaming engine (third_party/bigmoeonedge) for the iOS app: flash streaming for
// MoE GGUF models. The routed experts stay in the model file and are read from storage as tokens need them, into a
// cache sized from the free memory, so a model larger than the device's memory runs. CPU decode.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llmhub_bmoe llmhub_bmoe;

/// Opens `model_path` (and the vision projector `mmproj_path`, or NULL). Returns NULL on failure with the reason in
/// `error` (NUL-terminated, truncated to `error_size`).
llmhub_bmoe * llmhub_bmoe_open(const char * model_path, const char * mmproj_path, int context_size, int threads,
                               char * error, size_t error_size);
void llmhub_bmoe_free(llmhub_bmoe * session);

/// 1 when a vision projector is loaded.
int llmhub_bmoe_vision(const llmhub_bmoe * session);
/// The marker a prompt uses to place an image.
const char * llmhub_bmoe_image_marker(void);

/// Receives each generated piece of text (UTF-8, possibly part of a character). Return nonzero to stop.
typedef int (*llmhub_bmoe_piece_fn)(const char * piece, size_t length, void * user);

/// Generates from a prompt already in the model's chat format, clearing the previous conversation first. `image` may
/// be NULL. Returns 0 on success (including a stop), nonzero on failure with the reason in `error`.
int llmhub_bmoe_generate(llmhub_bmoe * session, const char * prompt,
                         const uint8_t * image, size_t image_size, const char * image_mime,
                         int max_tokens, float temperature, int top_k, float top_p,
                         llmhub_bmoe_piece_fn on_piece, void * user,
                         int * generated_tokens, double * generate_seconds,
                         char * error, size_t error_size);

/// Ends the running generation. Safe from any thread.
void llmhub_bmoe_cancel(llmhub_bmoe * session);

#ifdef __cplusplus
}
#endif
