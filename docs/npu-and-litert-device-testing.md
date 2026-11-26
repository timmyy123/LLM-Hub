# NPU and LiteRT device tests

Physical-device results for the chip-specific NPU builds and LiteRT 2.3.0 features on Android. The tests are opt-in
instrumentation tests under `android/app/src/androidTest`. Device Farm specs come from
`android/tools/imagegen_devicefarm/make_test_spec.py`. Suites that download their models on the phone take
`--arg=downloadFixtures=1` (this is also how they ran on BrowserStack), and every download is pinned by SHA-256.

## EmbeddingGemma 2 NPU builds

`MediaSearchNpuFallbackDeviceTest` (suite `embedgemma_fallback`) opens Video Moment Finder as on first launch, with the
740M NPU build for the device's SoC and the standard 740M build downloaded.

Before the app loads an NPU build, `EmbeddingNpuProbe` loads it in the separate `:npu_probe` process. On Samsung's
MediaTek firmware, MediaTek's NeuronAdapter dereferences null when its NPU service is unavailable to apps, which
used to close the app. Now only the probe process crashes. The probe result is cached per model file, system build,
and app version. Media Search and Video Moment Finder then load a downloaded standard build, or say the NPU can't
run the model.

MediaTek builds load their dispatch library and compiler plugin from a folder of their own. LiteRT loads the
first dispatch library in a folder, and in the app's native library folder that is Google Tensor's.

| Device | SoC | Result | First load |
| :-- | :-- | :-- | :-- |
| Galaxy S25 Ultra | Snapdragon 8 Elite (SM8750) | NPU build loads | 4.2 s |
| Pixel 11 Pro | Tensor G6 | NPU build loads; the main and vision encoders run on the TPU | 4.6 s |
| Pixel 10 | Tensor G5 | NPU build loads | 5.3 s |
| Galaxy Tab S11 | Dimensity 9400+ (MT6991) | probe crashes; standard 740M on GPU | 47 s |
| Galaxy Tab S12 Plus | Dimensity 9500 (MT6993) | probe crashes; standard 740M on GPU | 11 s |
| Galaxy S26 | Exynos 2600 | no NPU build published | — |

On Pixel, a small helper subgraph of the EmbeddingGemma file is not precompiled. Pixel's TPU service refuses on-device
compilation for apps it doesn't list ("not in the EdgeTPU allowed list"), so that part runs on the CPU.

## Stable Diffusion 1.5 on the Google Tensor TPU

`LiteRtSdAppDeviceTest` (suite `litert_sd_app`) lays out a Tensor build as the downloader writes it, then loads and
generates through `ImageGeneratorHelper`. The models come from `timmyboy26/stable-diffusion-v1-5-LiteRT-NPU`
(AOT-compiled with the Google Tensor SDK beta; every op runs on the TPU). Generation runs in the separate
`:litert_sd` process (`LiteRtSdService`).

| Device | Build | Result |
| :-- | :-- | :-- |
| Pixel 11 Pro (Tensor G6, Android 17) | `tensor_g6` | 512×512 image, load 3.2–4.0 s, 20 steps in 71 s |
| Pixel 10 (Tensor G5, Android 17) | `tensor_g5` | the TPU driver aborts on the first UNet step (`pending_sink_fds` check in `libedgetpu_litert.so`); the app stays open and reports that the NPU stopped |

The G4 and G3 UNet and decoder fail to compile with the Tensor SDK beta (`INTERNAL`); only their text encoders
compile.

## SoundGen on LiteRT 2.3.0

`SoundGenDeviceTest#quickAndHdGenerateAudio` (suite `soundgen_app`) generates 5 s of audio with SoundGen Quick and
HD through `MusicGeneratorBackend`. Both passed and produced non-silent audio on every phone:

| Device | Quick | HD |
| :-- | :-- | :-- |
| Galaxy S25 Ultra | 11.7 s (LiteRT 2.1.6: 11.5 s) | 65 s |
| Galaxy S26 | 7.4 s | 22 s |
| Galaxy Tab S11 | 13.7 s (LiteRT 2.1.6: 12.6 s) | 75 s |
| Pixel 11 Pro | 20.0 s | 72 s |
| Pixel 10 | 21.5 s | 81 s |

From LiteRT 2.3.0 the GPU accelerator ships in `litert-gpu`. Without it, `CompiledModel`'s GPU option silently runs
on the CPU.

LiteRT 2.3.0 does not compile Quick's audio decoder for the GPU, although 2.1.6 delegated all of it. Its largest
dilated convolutions fall back to TFLite's built-in kernel, whose dilated im2col buffer overflows (`CONV_2D` node
367). The app downloads `autoencoder_model_undilated_1g.tflite` from `timmyboy26/soundgen-quick-litert-gpu`
instead. In that file, the two convolutions over 1 GiB of im2col are rewritten as undilated convolutions with
zero-filled kernels. On the CPU its output is bit-identical to the original.

The decoder compiles with texture storage. Pixel 10's PowerVR OpenCL driver crashes with buffer storage, and texture
storage is as fast or faster on Adreno, Mali, and Xclipse.

`SoundGenGpuOptionsDeviceTest` (suite `soundgen_gpu`) measured one decoder run:

| Device | CPU | GPU (texture storage) | Mean difference from CPU |
| :-- | :-- | :-- | :-- |
| Galaxy S25 Ultra | 32 s | 2.2 s | 0.0012 |
| Galaxy S26 | 14 s | 2.5 s | 0.0012 |
| Galaxy Tab S11 | 34 s | 4.4 s | 0.0012 |
| Pixel 11 Pro | 34 s | 4.9 s | 0.0010 |
| Pixel 10 | 37 s | 10.7 s | 0.0013 |

SoundGen HD Long is killed for memory on these phones with both LiteRT 2.1.6 and 2.3.0 (peak resident memory
7–8.4 GB; `SoundGenDeviceTest#hdLongGeneratesAudio`, suite `soundgen_long`).

## Stable Audio 3 on the CPU

The Stable Audio 3 models (`StableAudio3.kt`) run Stability AI's official LiteRT files from
`stabilityai/stable-audio-3-optimized`. Each DiT and decoder file is a ladder of fixed-length subgraphs ("rungs",
signatures `s<N>`) that share one copy of the weights. A render runs on the smallest rung that fits, with the padding
masked out of attention, so memory stays flat at any length. The DiT takes its adaLN input from a `gcond` signature
in the same file. All files are compiled on the CPU with an XNNPACK weight cache (`<file>.xnnwc` next to the model),
which lets the rungs share packed weights. Building it adds ~2.5 s (Small) or ~4.5 s (Medium) to the first load;
later loads take ~0.15 s.

`StableAudio3DeviceTest` downloads the model through `ModelDownloader` into the app's own folder, generates the
lengths in `-e sa3Seconds`, and keeps the model. Galaxy Tab S9+ (Snapdragon 8 Gen 2), 4 threads:

| Model | Length | Time | Peak resident memory |
| :-- | :-- | :-- | :-- |
| Small Music | 10 s | 6.8 s | 2.0 GB |
| Small Music | 30 s | 11.6 s | 2.5 GB |
| Small Music | 120 s | 45.8 s | 2.5 GB |
| Medium | 30 s | 41 s (first load) | 2.3 GB |
| Medium | 380 s | 25.3 min (~110 s per DiT step) | 4.1 GB |

AWS Device Farm, Small Music (`tools/sa3_devicefarm/make_test_spec.py`, 15 device-minutes for the three phones):

| Device | 10 s | 30 s | Peak resident memory |
| :-- | :-- | :-- | :-- |
| Pixel 10 (Tensor G5) | 8.8 s | 10.5 s | 2.1 GB |
| Galaxy A56 (Exynos 1580) | 11.3 s | 18.1 s | 2.1 GB |
| Galaxy A34 (Dimensity 1080) | 18.5 s | 31.3 s | 2.1 GB |

Six threads were no faster than four on the Tab S9+ (11.9 s for 30 s). Every clip had the exact frame count and the decoder's
built-in limiter peak (0.977). The decoder window seams show no clicks, and the prompt steers the output: a
low-drone prompt gave 1,036 zero crossings per second and a piccolo prompt 6,594.

## Magenta RealTime 2 on the Adreno GPU

Android runs Magenta RealTime 2 Small (230M) through the adreno-llms OpenCL engine (`MagentaRealtime.kt`,
built by `tools/magenta_rt2/build.sh` at commit 2b1aee9). The engine runs as its own process in `--serve` mode:
the app sends one request per 0.4 s of audio (adreno-llms' live settings: `frames=10 chunk=10 pipeline=0
temperature=1.1 topk=50 cfgtok=5,0,4 mctail=6`) and reads back a WAV. `MagentaLiveService` plays it as a
mediaPlayback foreground service with a MediaSession, so it keeps playing with the screen locked and shows a media
card. The model is offered only where live generation is faster than playback.

Engine speed (`--serve`, pipeline off; live = 30 requests of 10 frames; render = one 750-frame request):

| Device | GPU | Live | 30 s render |
| :-- | :-- | :-- | :-- |
| QDC 8 Elite Extreme Gen 6 (SM8975) | Adreno 850 | 1.42–1.65× real time | 20.5 s |
| Galaxy S26 (8 Elite Gen 5) | Adreno 840 | 1.31–1.39× | 22.8 s |
| Galaxy S25 (8 Elite) | Adreno 830 | 0.83–0.95× | 32.2 s |
| Galaxy Tab S9+ (8 Gen 2) | Adreno 740 | not run | 56.4 s (fp16 codec, big cores) |
| Galaxy S24 (8 Gen 3) | Adreno 750 | 0.24× | 125 s (fp32 codec by default) |

On the 8 Elite the transformer takes ~20 ms per 40 ms frame and the codec ~25 ms. The engine uses the fp16 codec
only on Adreno 8xx by default. `MagentaRealtimeDeviceTest` (in the app) passed on the S26 Ultra (Remote Test Lab,
SM8850) and the QDC Gen 6: a 10 s render at the exact length, then live playback through the service. On the S26
Ultra, playback kept exact pace with the wall clock with no AudioTrack underruns. On the QDC unit it kept playing
with the screen locked and dozing, and the lock-screen card followed a prompt change. That rack unit consumes audio
about 1.5× faster than real time (its playback position runs ahead of the wall clock), so it underruns there.

## llama.cpp b11738 on the GPU

`LlamaCppGpuDeviceTest` (suite `llama_gpu`) loads LFM-2.5 1.2B Q4_0 on the GPU through `LlamaCppInferenceService`
with the app's backend rules, at the model's default 128K context, and asks for the capital of France.

| Device | Backend | Result |
| :-- | :-- | :-- |
| Galaxy S25 Ultra (Adreno 830) | OpenCL | "Paris" in 2.8 s |
| Galaxy S26 (Adreno 840) | OpenCL | "Paris" in 1.7 s |
| Galaxy Tab S11 (Mali-G925) | Vulkan | "Paris" in 4.1 s |
| Pixel 10 (PowerVR) | Vulkan | garbled output, as with earlier llama.cpp versions |

Snapdragon SoCs where llama.cpp Vulkan crashes keep using OpenCL. On the Galaxy S25 Ultra, b11738's OpenCL backend
aborted on the first token. Its tiled Q6_K lm_head GEMV is dispatched with 256 work items per group, and Adreno 830
allows 192 for that kernel. The app now ships `libggml-opencl.so` built from the vendored source with that check
added (see `llama.cpp/LLMHUB_VERSION.md`).

Turning flash attention off is not a workaround. At 128K context the attention compute buffer then needs 8.7 GB,
OpenCL cannot allocate it, and the app falls back to the CPU.
