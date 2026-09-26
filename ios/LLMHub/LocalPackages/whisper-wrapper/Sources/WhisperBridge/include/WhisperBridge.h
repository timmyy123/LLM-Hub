#ifndef LLMHUB_WHISPER_BRIDGE_H
#define LLMHUB_WHISPER_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void *LLMHubWhisperCreate(const char *model_path);
void LLMHubWhisperDestroy(void *context);
int32_t LLMHubWhisperTranscribe(void *context, const float *samples,
                               int32_t sample_count, int32_t thread_count,
                               char **result_text);
void LLMHubWhisperFreeText(char *result_text);

#ifdef __cplusplus
}
#endif

#endif
