# Engines and integrations: research notes

Facts gathered (October 2026) before building Agent, Design, Studio and voice. Verified against the projects' own repos,
release assets and the Hugging Face API unless marked otherwise.

## Speech to text: whisper.cpp (MIT)

- The latest tag (v1.9.5) has no assets; binaries sit on the matching nightly tag (b5454, same commit).
- Windows x64: `whisper-bin-x64.zip` (CPU), `whisper-blas-bin-x64.zip`, `whisper-bin-win-cuda-12.4.0-x64.zip` (685 MB).
  Windows Arm64: `whisper-bin-win-cpu-arm64.zip`, `whisper-bin-win-opencl-adreno-arm64.zip`.
  Linux: `whisper-bin-ubuntu-x64.tar.gz` (CPU). macOS: only an xcframework (we use the iOS app's whisper.spm package).
- `whisper-server`: `POST /inference` (multipart `file`, `language`, `response_format=json`, `temperature`, …),
  `POST /load`, `GET /health`. `--inference-path /v1/audio/transcriptions` exposes it at the OpenAI path.
- Models: `ggerganov/whisper.cpp` → `ggml-large-v3-turbo-q5_0.bin` (574 MB), `ggml-base.en-q5_1.bin` (60 MB),
  `ggml-small-q5_1.bin` (190 MB).

## Text to speech: Supertonic 3 (code MIT, weights OpenRAIL-M)

- The project is archived (`supertone-oss-archive/supertonic`); weights in `supertone-oss-archive/supertonic-3`
  (revision `aafc6e32416a594460b32413efc49d7fe4ce6d46`).
- ONNX: `onnx/duration_predictor.onnx`, `text_encoder.onnx`, `vector_estimator.onnx` (256 MB), `vocoder.onnx`,
  plus `onnx/tts.json`, `onnx/unicode_indexer.json`, and voices `voice_styles/{F1..F5,M1..M5}.json`.
- 44.1 kHz. 31 languages incl. en, es, pt, fr, de, it, ru (text wrapped as `<en>…</en>`).
- Official C# example (`Microsoft.ML.OnnxRuntime`) and Swift example (onnxruntime-swift-package-manager).

## Music: Magenta RealTime 2 (code Apache-2.0, weights CC-BY-4.0, not gated)

- `google/magenta-realtime-2`: MLX exports (`models/mrt2_small/*.mlxfn`), portable safetensors checkpoints,
  SpectroStream codec and MusicCoCa (TFLite).
- The official C++ engine is macOS only (MLX/Metal). The Python package `magenta-rt` (JAX) runs on CUDA only on Linux and
  its docs call NVIDIA "offline (non-real-time)".
- Linux with an NVIDIA GPU plays it live through a JAX sidecar (`MagentaService`, `shared/magenta/mrt_live.py`). Native
  Windows has no real-time path (MLX's CUDA backend reached 3.6 of the 25 steps/s live audio needs), so Windows offers
  Stable Audio 3 clips only and says so in Studio › Music and Settings › Studio.
- macOS reuses the iOS app's MLX driver (`ios/LLMHub/LocalPackages/magenta-runtime`).

## Images and video: ComfyUI (GPL-3.0, run as a separate process)

- Install without user Python: Comfy-Org `ComfyUI-Standalone-Environments` (python-build-standalone + torch) or the
  Windows portable 7z (`ComfyUI_windows_portable_nvidia.7z`, about 2 GB). No GPU support on Snapdragon.
- API: `POST /prompt {prompt, client_id}`, `GET /history/{id}`, `GET /view?filename=&subfolder=&type=output`,
  `POST /upload/image`, `POST /interrupt`, `GET /system_stats`, `WS /ws?clientId=` (`progress`, `executing`,
  `executed`, `execution_error`).
- Models (no Chinese makers): Flux.1 schnell fp8 (`Comfy-Org/flux1-schnell`, Apache-2.0), SDXL base (OpenRAIL++),
  LTX-Video 2B 0.9.5 (Lightricks, OpenRAIL-M; 0.9.6+ uses a revenue-limited licence), upscaler 4x-ClearRealityV1
  (Apache-2.0). Real-ESRGAN is excluded (Tencent). Flux fp8 does not fit a 32 GB Mac well (MPS cannot use fp8).

## Agent: Claude Code headless

- Install: `irm https://claude.ai/install.ps1 | iex` (Windows), `curl -fsSL https://claude.ai/install.sh | bash`.
- Drive: `claude -p --input-format stream-json --output-format stream-json --verbose --include-partial-messages
  --session-id <uuid>`; events `system/init`, `stream_event`, `assistant` (tool_use), `user` (tool_result), `result`.
- Local models: `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>`, `ANTHROPIC_AUTH_TOKEN=<key>`,
  `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL=<local id>`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`,
  `CLAUDE_CODE_ENABLE_TODO_TOOLS=1`.
- Anthropic does not support routing Claude Code to non-Claude models, and "Claude Code" may not appear in a third-party
  product name. The app treats it as an engine the user installs.

## Design: open-design (Apache-2.0)

- `nexu-io/open-design`: the model writes plain HTML in one `<artifact type="text/html">…</artifact>` block, shown in a
  sandboxed preview. Design systems are `DESIGN.md` + `tokens.css`; skills are `SKILL.md` + `assets/template.html`.
- Reusable: `apps/daemon/src/prompts/` (`core-slim.ts`), `design-systems/`, `design-templates/`, `craft/`.
