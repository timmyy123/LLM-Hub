# On-device voice cloning candidates for Android

Research checked 2026-10-08. **Implemented choice: PocketTTS.cpp**, as requested. See [the Android integration and validation](pocket-tts.md). The remaining alternatives below are research notes, not additional installed engines. Supertonic 3 remains preset-only: its public release does not include a reference-audio encoder. The external Voice Builder and custom JSON import flow have been removed.

## Alternative: sherpa-onnx with Pocket TTS

The [official sherpa PocketTTS guide](https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html) documents offline cloning from reference audio without a transcript and distributes a model bundle containing the encoder, decoder, text conditioner, and language-model graphs. The [Kotlin example](https://github.com/k2-fsa/sherpa-onnx/blob/master/kotlin-api-examples/test_pocket_tts.kt) and [Kotlin API](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/kotlin-api/Tts.kt) expose `OfflineTtsPocketModelConfig` and `generateWithConfig`, with reference PCM and sample rate passed through `GenerationConfig`. The runtime has an [Android build path](https://k2-fsa.github.io/sherpa/onnx/android/build-sherpa-onnx.html).

This is the best first integration candidate because the complete audio encoder is available and users do not need to type a transcript. The documented January 2026 sherpa example is English; do not assume that export supports every language in newer upstream Pocket TTS releases. [Upstream Pocket TTS](https://github.com/kyutai-labs/pocket-tts) has newer language-specific models. Code is MIT; the [official model card](https://huggingface.co/kyutai/pocket-tts) lists CC-BY-4.0 weights and access conditions. Pin and validate the exact chosen export and its notices before integration.

## Alternative: sherpa-onnx with ZipVoice-Distill INT8

[ZipVoice](https://github.com/k2-fsa/ZipVoice) supports English and Chinese cloning and offers a distilled version for speed. The [sherpa guide](https://k2-fsa.github.io/sherpa/onnx/tts/zipvoice.html) documents reference audio plus its transcript. The same Kotlin generation interface accepts both inputs. This is a practical candidate where Chinese support matters and a transcript is acceptable. The transcript can be entered by the user or generated with a separate local ASR model; cloning must not depend on a cloud transcription service.

## Implemented library: PocketTTS.cpp

[PocketTTS.cpp](https://github.com/VolgaGerm/PocketTTS.cpp) provides local cloning with an ONNX audio encoder, streaming, cached voice state, and a C FFI/shared-library option. Its documented targets are desktop Linux/macOS/Windows. The app now includes that NDK/JNI adaptation, using its existing Android ONNX runtime with a pinned model bundle. Reference encoding and synthesis have passed a JNI smoke test on an ARM64 Android device.

## NeuTTS caveat

[NeuTTS](https://github.com/neuphonic/neutts) supports local reference-audio cloning and GGUF backbones. However, its documented ONNX codec artifacts are decoder-only and require pre-encoded references; the reference encoder is in the PyTorch codecs. A GGUF model plus ONNX decoder alone would recreate the missing-encoder problem. It requires additional encoder deployment work before satisfying fully in-app cloning. Nano models also use a different model license from Air.

## Implemented app flow

Record/import reference audio → decode and resample locally → encode reference on-device → synthesize locally → play audio. Download the complete encoder and synthesis bundle from the model screen, and manage saved voices in Settings. No external voice-building website or uploaded recording is part of this flow.

Before choosing a release, measure full encoder-plus-synthesis performance and peak memory on an Android ARM64 device, verify native library coexistence with the app's ONNX runtime, and check 16 KB page alignment. The PocketTTS.cpp JNI smoke test is documented above; broad device performance benchmarking remains separate.
