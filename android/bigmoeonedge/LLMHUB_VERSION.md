# Vendored BigMoeOnEdge core

- Upstream: https://github.com/Helldez/BigMoeOnEdge
- Version: `v0.47.2`, commit `f33bf03f77333632cfce50de43802f3e3b4a0280`
- License: Apache License 2.0 (see `LICENSE`)
- Copied: `core/` only (the expert-streaming engine, the hardware planner and the Session API). Not copied: the
  CLI, server, desktop app, apps and their third-party code.

## How LLM Hub uses it

Flash streaming for MoE GGUF models: the routed experts stay in the model file and are read from flash per token
into an LRU cache, so a model larger than the phone's free RAM runs. `bmoe_core` is built against LLM Hub's own
vendored llama.cpp (`android/llama.cpp`, b12613 with LLMHub patches) rather than BigMoeOnEdge's llama.cpp fork:
its public-API streaming path needs no fork. The fork's one extra (the CPU expert-ready hook behind `--overlap`)
is not carried, so overlap is unavailable and the engine runs the serial streamer.

Measured on a Galaxy Tab S9+ (SM8550, 11.5 GB RAM) with LFM2.5-24B-A2B QAD Q4_0 (13.6 GB), CPU decode, 4.9 GB
expert cache: 2.2 tok/s, prompt reading 4 to 16 tok/s, output identical to the fork's build.

In the apps (Android `android/app/src/main/cpp/llama_stream_jni.cpp`, iOS `ios/LLMHub/LocalPackages/llama-b12613/bmoe`):
on by default for any GGUF whose header lists more than one expert, with a per-model switch. Raw-prompt mode (the
app formats the chat), CPU only, dense weights pinned, 4 read lanes, O_DIRECT. Android also sets:

- Decode on 2 threads, prompt reading on the app's usual count. On the Tab S9+ 6 decode threads cost 669 ms of
  compute per token (1.1 tok/s) and 2 threads 180 ms (2.3 to 2.4 tok/s).
- Expert cache from the free RAM but at most 30% of the device's RAM. Samsung kills an app whose anonymous memory
  passes 6 GiB on the Tab S9+ whatever is free; the uncapped 5.8 GB cache plus the resident weights reached 6.36 GB.
  With the cap (3.4 GB, 62% hits) the app ran 2.26 tok/s on a short answer and 2.41 tok/s on a paragraph, load 16 s.

## Local changes

Keep these few; record any change here.

- `core/src/io/platform_io.cpp`: on Android, `can_run_self()` is false and `run_self()` starts nothing (as upstream
  already does on iOS). Inside an app the executable is the system's `app_process`, not a bmoe binary, and
  `posix_spawn` needs API 28 while the app supports 27. Only the CLI's isolated device probes use it; the Session
  path the app runs never does.
- `core/src/moe/device_arena.cpp`: the prompt-reading loader threads catch an exception from a device weight write
  (ggml-hexagon throws when an NPU buffer mapping fails) and mark the arena failed, as a failed flash read does.
  Uncaught on those threads it ended the app.
- `core/src/moe/device_arena.cpp`: the expert arena refuses an expert type the prefill device cannot multiply by
  (asked of the device with a MUL_MAT_ID probe), so the load fails instead of the prompt hanging. With Gemma 4
  26B-A4B UD-Q2_K_XL (IQ2_XS and IQ3_XXS gate/up experts) on the Hexagon NPU, that op was left to the CPU reading
  the NPU's slot while the loaders wrote it, and the NPU's queue stopped answering (Snapdragon 8 Elite Gen 6).
