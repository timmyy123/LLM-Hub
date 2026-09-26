#include "WhisperBridge.h"

#include "../../../whisper.spm/Sources/whisper/include/whisper.h"

#include <cstdlib>
#include <cstring>
#include <string>

void *LLMHubWhisperCreate(const char *model_path) {
  if (!model_path) return nullptr;
  whisper_context_params params = whisper_context_default_params();
  params.use_gpu = true;
  return whisper_init_from_file_with_params(model_path, params);
}

void LLMHubWhisperDestroy(void *context) {
  if (context) whisper_free(static_cast<whisper_context *>(context));
}

int32_t LLMHubWhisperTranscribe(void *context, const float *samples,
                               int32_t sample_count, int32_t thread_count,
                               char **result_text) {
  if (!context || !samples || sample_count <= 0 || !result_text) return -1;
  *result_text = nullptr;
  whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  params.n_threads = thread_count;
  params.print_realtime = false;
  params.print_progress = false;
  params.print_timestamps = false;
  params.print_special = false;
  params.translate = false;
  params.language = nullptr;
  params.no_context = true;
  params.single_segment = false;
  auto *whisper = static_cast<whisper_context *>(context);
  const int status = whisper_full(whisper, params, samples, sample_count);
  if (status != 0) return status;

  std::string text;
  for (int index = 0; index < whisper_full_n_segments(whisper); ++index) {
    if (const char *segment = whisper_full_get_segment_text(whisper, index)) {
      text += segment;
    }
  }
  char *copy = static_cast<char *>(std::malloc(text.size() + 1));
  if (!copy) return -2;
  std::memcpy(copy, text.c_str(), text.size() + 1);
  *result_text = copy;
  return 0;
}

void LLMHubWhisperFreeText(char *result_text) { std::free(result_text); }
