# Supertonic 3 assets

Revision: `3cadd1ee6394adea1bd021217a0e650ede09a323`

Verified with `curl -fsSIL --retry 2 <URL>` on 2026-10-08. Sizes are final HTTP 200 Content-Length values after redirects.

| Asset | Bytes |
| --- | ---: |
| `onnx/duration_predictor.onnx` | 3,700,147 |
| `onnx/text_encoder.onnx` | 36,416,150 |
| `onnx/tts.json` | 8,253 |
| `onnx/unicode_indexer.json` | 277,676 |
| `onnx/vector_estimator.onnx` | 256,534,781 |
| `onnx/vocoder.onnx` | 101,424,195 |
| `voice_styles/F1.json` | 292,046 |
| `voice_styles/F2.json` | 292,423 |
| `voice_styles/F3.json` | 290,794 |
| `voice_styles/F4.json` | 291,808 |
| `voice_styles/F5.json` | 291,479 |
| `voice_styles/M1.json` | 291,748 |
| `voice_styles/M2.json` | 292,055 |
| `voice_styles/M3.json` | 290,198 |
| `voice_styles/M4.json` | 291,522 |
| `voice_styles/M5.json` | 291,469 |

Core bundle: **398,361,202 bytes**. Voices are separate Settings downloads. All URLs are pinned to the revision above.

The public release contains preset embeddings and synthesis models, but no reference-audio style encoder. The app therefore offers preset TTS only. The external Voice Builder flow and custom JSON imports have been removed because voice cloning must run entirely inside the app on-device.

Sources: [model card](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/README.md), [official Java inference](https://github.com/supertone-inc/supertonic/blob/main/java/Helper.java).

Model: OpenRAIL-M. Runtime implementation adapted from Supertone’s MIT sample code; see `app/src/main/assets/supertonic-NOTICE.txt`.

## App usage

1. Download **Supertonic 3 (ONNX)** under Text-to-Speech models.
2. Select it as the TTS model in Settings. Preset voices and language selection appear there; this CPU-only engine does not expose the Kokoro GPU selector.
3. Download and select a preset voice in Settings. Supertonic and Kokoro use the same shared voice-download dialog: 56 dp rows, labelled filled Download buttons, red outlined Delete buttons, progress indicators and the empty-state download hint.
4. The selected voice applies to Chat, Agent, Writing Aid, Scam Analysis, VibeVoice readout, and Text-to-Speech. Translator continues using system TTS.

Previously stored custom voice selections fall back to M1. The app no longer imports, lists, or synthesizes custom Supertonic voices. Existing user files are left untouched.

## Validation

- Android `:app:compileDebugKotlin` and `:app:processDebugResources` passed.
- All 24 Android strings resource sets contain all four remaining translated strings; resource compilation passed.
- Eight focused `SupertonicTest` tests passed: flat/nested preset tensors, malformed dimensions, truncated and nonnumeric tensors, preprocessing, and bounded multilingual chunking without broken surrogate pairs.
- The existing `GeniexPromptTest.kt` does not compile because it references the missing `prepareVlmUserText`. Focused tests were run with a temporary Gradle init script setting `compileDebugUnitTestKotlin` sources to `**/SupertonicTest.kt`; the unrelated test source was left untouched.
- The actual Kotlin engine synthesized non-silent, finite audio from the pinned assets with M1 in English (100,092 samples), Japanese (66,677), and French (81,831), at 44,100 Hz. Cancellation passed. This was a macOS JVM smoke test using desktop ONNX Runtime 1.24.2 (1.24.1's desktop JAR lacks macOS natives); the Android app retains ONNX Runtime Android 1.24.1.
- Android device playback and visual UI testing have not been performed.

## Removal verification

After removing the external cloning UI, JSON import handling, custom-voice resolution, and cloning strings from all locales, Android Kotlin compilation and resource processing passed. See [on-device cloning research](on-device-voice-cloning.md) for replacement candidates.
