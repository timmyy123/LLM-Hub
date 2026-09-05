# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `v0.4.0` / `b10816`
- Commit: `427291b5b34cd914a31b3fd3b61a68f6184f4b9f`
- Source archive SHA-256: `128a83d8deb5ec83a47d9932ae0458d8bd8fcfe88c1f982919b84ccac269e3c3`
- License: MIT (see `LICENSE`)

Only the source directories required by the Android CPU engine are vendored, including
`tools/mtmd` for GGUF vision-projector support. The fallback is compiled statically into
`libllmhub_llama_cpu.so` so its llama/ggml/mtmd symbols and filenames do not collide with
the shared libraries inside GenieX.

Android Studio builds this source through `app/src/main/cpp/CMakeLists.txt`.
The llama.cpp engine uses the baseline ARMv8 CPU backend with zero GPU-offloaded layers.
