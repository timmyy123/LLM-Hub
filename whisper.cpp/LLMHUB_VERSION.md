# Vendored whisper.cpp

- Upstream: https://github.com/ggml-org/whisper.cpp
- Version: `v1.9.6`
- Commit: `00c1630c14fc8fc187aa4c5ae3975d81ac2e6517`

Only `src/whisper.cpp`, its two private headers, `src/aneforge/` (an Apple Neural Engine encoder that is a stub on other platforms; whisper.cpp always includes it), `include/whisper.h` and the license are vendored. The Android
app compiles them into `libllmhub_whisper.so` against the ggml of the vendored llama.cpp (`../llama.cpp`), so
speech recognition uses the same per-chip CPU backends (`libllmhub_ggml-cpu-*`) as the GGUF chat models.
To upgrade, copy the same files from a newer release and update the version above.

## Local changes

- `src/whisper.cpp`, `whisper_kv_cache_init`: the KV caches go to a CPU buffer when the backend cannot run the
  `GGML_OP_CPY` that writes into them (the Hexagon HTP backend), instead of aborting the scheduler.
