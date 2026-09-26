#ifndef LLMHUB_WHISPER_MODULE_SHIM_H
#define LLMHUB_WHISPER_MODULE_SHIM_H

// The C++ WhisperBridge links this target but exposes its own small C API.
// Keeping ggml declarations out of this generated Clang module avoids
// collisions with the independent ggml version bundled in llama.cpp.

#endif
