# LLM-Vibe-Hub: Copilot Instructions

## Project Overview
**LLM-Vibe-Hub** is a privacy-first Android app providing on-device LLM chat, image generation, writing aids, transcription, translation, and scam detection. All inference runs locally with GPU/NPU acceleration.

## Architecture Essentials

### Multi-Runtime Inference System
The app routes models through three parallel inference backends in [UnifiedInferenceService.kt](app/src/main/java/com/llmhub/llmhub/inference/UnifiedInferenceService.kt):

1. **MediaPipe LiteRT** (`MediaPipeInferenceService`) — `.task`/`.litertlm` models (legacy)
2. **ONNX Runtime** (`OnnxInferenceService`) — `.onnx` models with NNAPI GPU acceleration; handles Ministral-3 vision encoder and Ministral/Mistral chat templates with `[INST].../[INST]` formatting
3. **Nexa SDK** (`NexaInferenceService`) — `.gguf` models; includes vision (VLM) support via `VlmWrapper`; requires image downscaling to 300px max for performance

**Selection logic**: Model format in [LLMModel.kt](app/src/main/java/com/llmhub/llmhub/data/LLMModel.kt) (`modelFormat` field) determines which service loads the model. **Important**: Nexa's `libonnxruntime.so` can conflict with ONNX Runtime's; clean builds required if UnsatisfiedLinkError occurs.

### Image Generation (Stable Diffusion)
Native `libstable_diffusion_core.so` runs as a subprocess (not JNI) started by `SDBackendService`. It spins an HTTP server on `127.0.0.1:8081`, accepts prompts/config, returns PNG data. Supports:
- **QNN (NPU)** backend: `libQnnHtp.so` + version-specific libs from `app/src/main/assets/qnnlibs/`
- **MNN (CPU)** backend: Model files in `app/src/main/assets/cvtbase/` (clip, unet, vae_decoder, tokenizer)

Extraction of QNN libs to `filesDir/runtime_libs/` is handled automatically at service startup.

## Key Code Patterns

### Chat Templates
- **Ministral/Mistral** (ONNX): Use `[INST] user message [/INST]` — must NOT include "user:" label or model sees "messageassisstant" misparsed
- **LFM/Granite** (ONNX): Use ChatML: `<|im_start|>user\nmessage<|im_end|>\n<|im_start|>assistant\n`
- **Nexa (GGUF)**: SDK's `applyChatTemplate()` handles automatically; see [NexaInferenceService.kt#L800+](app/src/main/java/com/llmhub/llmhub/inference/NexaInferenceService.kt#L800)

### Web Search Integration
`SearchIntentDetector` (in [WebSearchService.kt](app/src/main/java/com/llmhub/llmhub/websearch/WebSearchService.kt)) detects queries needing web context; `DuckDuckGoSearchService` fetches results via HTML parsing (handles multiple DOM formats). Results injected into prompt before inference. **Note**: Reasoning models (LFM-2.5 Thinking) buffer output to detect `<think>` tags for improved thinking token handling.

### Vision (Multimodal)
- **Nexa VLM**: Downscales images to 300×300, saves as JPEG (70% quality), paths passed via config; SDK handles encoding
- **Ministral-3 ONNX**: Custom vision encoder (`vision_encoder_q4.onnx`) + image token insertion; requires `config.json` for `text_config.image_token_index`

### GPU/NPU Acceleration
ONNX uses Android NNAPI (`OrtSession.SessionOptions.addNnapi()`):
- Try NNAPI-only first (FP16, CPU disabled) for max device acceleration
- Fall back to NNAPI+CPU if ops unsupported
- Ministral embedding (`embed_tokens_fp16.onnx`) loaded separately if decoder expects `inputs_embeds`

## Build & Development

### Prerequisites
- Android SDK 36 (target), min SDK 27; AGP 8.13.1; Kotlin 2.2.20
- For native image gen: asset pack `:qnn_pack` (built from `qnn_pack/build.gradle.kts`)
- Gradle wrapper: `./gradlew` (Linux/Mac) or `gradlew.bat` (Windows)

### Common Tasks
```bash
# Debug build
./gradlew assembleDebug
./gradlew installDebug

# Full app + asset packs
./gradlew bundleDebug

# Tests
./gradlew test

# Lint
./gradlew lint
```

### Model Download Workflow
- UI: [ModelDownloadScreen.kt](app/src/main/java/com/llmhub/llmhub/screens/ModelDownloadScreen.kt) handles model selection & HuggingFace token auth
- Service: [ModelDownloadService.kt](app/src/main/java/com/llmhub/llmhub/services/ModelDownloadService.kt) streams to `context.filesDir/models/<model_name>/`
- Additional files (e.g., ONNX `.data_00000` shards) listed in `additionalFiles` of [ModelData.kt](app/src/main/java/com/llmhub/llmhub/data/ModelData.kt); always download together

### Sensitive Config
- **HuggingFace Token**: Add to `local.properties` (not committed):
  ```properties
  HF_TOKEN=hf_xxxxx
  ```
  Read at build time as `BuildConfig.HF_TOKEN`

## Database & Persistence
[Room](gradle/libs.versions.toml) (v2.8.4) stores chat history, user settings, model metadata. DAOs in `app/src/main/java/com/llmhub/llmhub/data/`. Migrations auto-applied; model.isDownloaded boolean updated after successful downloads.

## Jetpack Compose UI
Material 3 (2024.09.00). Screens in `app/src/main/java/com/llmhub/llmhub/screens/`:
- [HomeScreen.kt](app/src/main/java/com/llmhub/llmhub/screens/HomeScreen.kt) — feature card grid
- [ChatScreen.kt](app/src/main/java/com/llmhub/llmhub/screens/ChatScreen.kt) — multi-turn with RAG & web search
- [ImageGeneratorScreen.kt](app/src/main/java/com/llmhub/llmhub/screens/ImageGeneratorScreen.kt) — prompt → image + swipeable variations gallery
- [FeatureScreens.kt](app/src/main/java/com/llmhub/llmhub/screens/FeatureScreens.kt) — writing aid, translator, transcriber, scam detector

ViewModels use `@HiltViewModel` + Hilt dependency injection.

## Important Gotchas
1. **ONNX Runtime conflicts**: Nexa SDK bundles its own `libonnxruntime.so`. Clean build + uninstall old APK to resolve.
2. **Ministral vision config**: Must parse `config.json` to extract image token index; vision encoder must exist before loading main model.
3. **16KB page alignment**: NDK set to `debugSymbolLevel = "FULL"` for Android 15+ Play Store compliance.
4. **R8 minification disabled**: Release builds disable R8 to preserve ONNX/Nexa JNI class definitions.
5. **Native libs asset pack**: QNN libraries auto-extracted; ensure `:qnn_pack` is built and included in app's asset pack declaration.

## File Organization Summary
```
app/src/main/
  java/com/llmhub/llmhub/
    screens/           — Compose UI screens
    inference/         — UnifiedInferenceService + three backends
    services/          — ModelDownloadService, SDBackendService, web search
    data/              — LLMModel, ModelData, database DAOs
    utils/             — FileUtils, device info helpers
  res/                 — strings (16 languages), drawables, layouts
  assets/
    cvtbase/           — SD CPU model files (clip, unet, vae_decoder, tokenizer)
    qnnlibs/           — QNN libraries (auto-extracted at runtime)

qnn_pack/              — Asset pack for QNN native libs + SD base models
```

## Testing & Debugging
- **Logcat tags**: Search `OnnxInferenceService`, `NexaInferenceService`, `SDBackendService` for backend logs
- **Token usage**: Models report token counts in generation; check [ModelData.kt](app/src/main/java/com/llmhub/llmhub/data/ModelData.kt) for `contextWindowSize`
- **Crash after vision load**: Check vision encoder file exists + ONNX session init wasn't skipped
- **Image gen subprocess errors**: Monitor `127.0.0.1:8081` logs via `adb logcat | grep libstable_diffusion_core`
