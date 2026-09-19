# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b11048`
- Commit: `59fc5a1ca3842241dd53617ae2ae030c1a015061`
- Source archive SHA-256: `fc0a53035c43453d0ce8d0cb8aa26a046617dce1098138b6416276264ec38f67`
- License: MIT (see `LICENSE`)

Only the source directories required by the Android CPU engine are vendored, including
`tools/mtmd` for GGUF vision-projector support. The fallback is compiled statically into
`libllmhub_llama_cpu.so` so its llama/ggml/mtmd symbols and filenames do not collide with
the shared libraries inside GenieX.

Android Studio builds this source through `app/src/main/cpp/CMakeLists.txt`.
The llama.cpp engine uses the baseline ARMv8 CPU backend with zero GPU-offloaded layers.
