# PocketTTS.cpp on Android

The Android app uses [VolgaGerm/PocketTTS.cpp](https://github.com/VolgaGerm/PocketTTS.cpp) for fully local voice cloning and English speech. Supertonic 3 remains a separate multilingual engine with presets and compatible voice JSON import. No iOS files are changed.

## Usage

1. Download **Pocket TTS (English, ONNX)** from the Text-to-Speech section of the model download screen.
2. Select Pocket TTS as the TTS engine in Settings and open **Voice cloning · Pocket TTS**. Voice management appears only in Settings.
3. Choose **Record voice**, grant microphone permission, and speak clearly for 3–30 seconds. Choose **Stop and clone** after at least 3 seconds; recording automatically finishes at 30 seconds. Dismissing the recording dialog, navigating away or backgrounding the app cancels capture, releases the microphone and discards unfinished audio. Alternatively, import a clear WAV, MP3, FLAC, M4A/AAC or OGG speech recording, at least 3 seconds long, maximum 20 MiB. Import detects file contents rather than trusting the filename. WAV, MP3 and FLAC use the native decoder; other supported formats use Android’s local audio decoder to create mono PCM16 WAV. Both paths use at most the first 30 seconds, and the native encoder resamples to 24 kHz locally. Silent, empty, short and malformed references are rejected.
4. The app runs Mimi encoding before saving the voice. Selecting an imported voice selects Pocket TTS across Chat, Agent, Writing Aid, Scam Analysis, VibeVoice readout and Text to Speech. Translator keeps system TTS.
5. Manage/select/rename/delete cloned voices in Settings. Use the edit icon beside any cloned voice to rename it; names are trimmed, must be nonempty and can contain up to 80 characters. Renaming updates the private label atomically while preserving the selected voice, audio and cached voice state. Voice samples, labels, embeddings and KV caches are private app files, separate from model downloads. Deleting a voice deletes its audio, label and both cache files. No reference audio is uploaded.

The pinned model bundle supports English only. No transcript is required. There is no cloud inference, external voice builder, JSON voice import, Python runtime or HTTP server in the Android integration. Import uses Android’s document picker. Microphone capture records mono PCM16 at 24 kHz directly into a private WAV before using the same local voice encoder.

## Pinned sources

- PocketTTS.cpp: `e801e7d6c2692121a39e80ae525cb5265174a495`, MIT.
- SentencePiece: v0.2.1, Apache-2.0, vendored source.
- dr_libs: `dfe8377631000664666519fdb83da193fd8037f4`, MIT-0/public domain, vendored WAV/MP3/FLAC decoders.
- ONNX Runtime Android: 1.24.1, the same AAR already used by the app. Gradle extracts its C++ headers and ARM64 binary for CMake; there is no native dependency download during CMake configuration. Duplicate packaging picks the same AAR runtime; APK processing preserves its build ID and executable code.
- ONNX models: [KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx/tree/58a6d00cf13d239b6748cb0769f35c580a8f606c), revision `58a6d00cf13d239b6748cb0769f35c580a8f606c`, original English root graphs, CC-BY-4.0, Kyutai weights with KevinAHM’s ONNX export. The newer language-specific subdirectories are not used.

Upstream changes are limited to bounded audio decoding/reference validation, propagation of generator and decoder exceptions, and an embedded-build guard excluding server/CLI/FFI code. SentencePiece’s desktop executable targets are disabled on Android. JNI serializes operations around upstream’s global RNG/profiler, reference-counts shared native sessions across speech services and the cloning UI, returns Java exceptions on failures and uses a cancellation callback. Kotlin synchronizes voice deletion against active encoding/synthesis. The ARM64 JNI library uses 16 KiB ELF alignment. Full notices are packaged in `assets/pocket-tts-NOTICE.txt`.

## Download sizes

Verified 2026-10-08 with `curl -fsSIL --retry 2` against each pinned URL, using the final response Content-Length. The downloaded files matched all lengths. Completion checks require every exact file size, including the tokenizer and license.

| File | Bytes |
| --- | ---: |
| [onnx/flow_lm_main_int8.onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/58a6d00cf13d239b6748cb0769f35c580a8f606c/onnx/flow_lm_main_int8.onnx) | 76,341,627 |
| [onnx/flow_lm_flow_int8.onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/58a6d00cf13d239b6748cb0769f35c580a8f606c/onnx/flow_lm_flow_int8.onnx) | 9,962,530 |
| [onnx/mimi_decoder_int8.onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/58a6d00cf13d239b6748cb0769f35c580a8f606c/onnx/mimi_decoder_int8.onnx) | 22,684,077 |
| [onnx/mimi_encoder.onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/58a6d00cf13d239b6748cb0769f35c580a8f606c/onnx/mimi_encoder.onnx) | 73,165,554 |
| [onnx/text_conditioner.onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/58a6d00cf13d239b6748cb0769f35c580a8f606c/onnx/text_conditioner.onnx) | 16,388,363 |
| [tokenizer.model](https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/58a6d00cf13d239b6748cb0769f35c580a8f606c/tokenizer.model) | 59,339 |
| [onnx/LICENSE](https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/58a6d00cf13d239b6748cb0769f35c580a8f606c/onnx/LICENSE) | 18,655 |
| **Total** | **198,620,145** |

## Validation

- `:app:assembleDebug` passed, including Kotlin, resources, all native targets and APK packaging. All seven new strings exist in all 24 Android locale sets. The APK contains exactly one ARM64 ONNX runtime, the Pocket JNI library and its notices.
- ARM64 native JNI library built with NDK 29.0.13113456 and CMake 3.31.6.
- The pinned graph bundle was loaded by the actual C++ runtime. A synthetic English reference produced a voice tensor of `[1, 88, 1024]` and finite, non-silent speech. Cancellation passed.
- The actual JNI library and Android ONNX Runtime 1.24.1 were run on the connected ARM64 Android device using `app_process` and temporary files, without installing or changing the app. Reference encoding took approximately 0.74 seconds in this single test; generated audio contained 3.52 seconds of finite speech at 24 kHz, RMS approximately 0.132. Shared-session reference counting, cancellation before and during generation, and missing/short/silent-reference rejection passed. These are smoke-test observations, not a device benchmark.
- Eight focused `SupertonicTest` regression tests passed. The existing unrelated `GeniexPromptTest` compile failure remains; the same temporary source filter documented in `supertonic-3.md` was used.
- A Java ONNX Runtime environment and PocketTTS.cpp successfully coexisted in the same Android process.
- The native library has 16 KiB alignment for every ELF LOAD segment.
- A repeatable JNI harness is included in `android/tools/pocket-tts-smoke`.
- The Settings microphone flow compiles with six additional strings translated across all 24 locale sets. Hardware microphone capture, physical speaker playback and visual app UI have not been exercised by the JNI smoke test.


- The 30-second cap is restored for recording, converted imports, native decoding and encoding. Whole-reference and windowed inference changes were reverted after Android reported low-memory termination at approximately 7.5 GiB RSS. The format-detection and M4A/AAC/OGG import fix remains.

- After restoration, `:app:assembleDebug` passed and the actual APK importer capped a synthetic 90-second M4A at exactly 30 seconds on the connected ARM64 device. Extensionless/mislabeled audio and short-reference cleanup checks also passed. All 24 locale sets contain the restored cap and recording timer.
