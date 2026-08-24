# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b10603`
- Commit: `c060ca974c773c7c3d17fd1b66dc9d312bc292c0`
- Source archive SHA-256: `7a319224f291d4c533e634aa861ed9407f5287d06424212830ff7444db5a578b`
- License: MIT (see `LICENSE`)

Only the source directories required by the Android CPU fallback are vendored.
The fallback is compiled statically into `libllmhub_llama_cpu.so` so its llama/ggml
symbols and filenames do not collide with the shared libraries inside GenieX.
