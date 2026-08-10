#ifndef MAGENTA_LITERT_BRIDGE_H_
#define MAGENTA_LITERT_BRIDGE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Runs signature 0 of a TFLite model with LiteRT's compiled-model API.
/// Inputs are matched by element type, as Magenta's text encoder orders its
/// token and padding inputs differently between exported model revisions.
int32_t MagentaLiteRTRunModel(const char *model_path,
                              const int32_t *int32_input,
                              size_t int32_input_count,
                              const float *float_input,
                              size_t float_input_count,
                              const float *second_float_input,
                              size_t second_float_input_count,
                              void *output,
                              size_t output_capacity,
                              int32_t expected_output_type);

#ifdef __cplusplus
}
#endif

#endif  // MAGENTA_LITERT_BRIDGE_H_
