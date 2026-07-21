# LLM Hub 🤖

**LLM Hub** is an open-source mobile app for on-device LLM chat, image generation, and video generation, available for both **Android** and **iOS**. It's optimized for mobile usage (CPU/GPU/NPU acceleration) and supports multiple model formats so you can run powerful models locally and privately.

🌟 **Featured:** Featured by **RunAnywhere** on their website — read the [LLM Hub iOS Case Study](https://www.runanywhere.ai/blog/llm-hub-ios-case-study)!

## Download

<table>
  <tr>
    <td valign="center">
      <a href="https://play.google.com/store/apps/details?id=com.llmhub.llmhub"><img src="https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png" height="75"></a>
    </td>
    <td valign="center">
      <a href="https://apps.apple.com/au/app/llm-hub/id6762511820"><img src="https://developer.apple.com/app-store/marketing/guidelines/images/badge-download-on-the-app-store.svg" height="50"></a>
    </td>
  </tr>
</table>

**💻 Desktop Versions Coming:** Windows and macOS native apps are planned, bringing desktop-class experiences with advanced cursor integration and Claude Code–like capabilities.


## 📸 Demo & Screenshots

| Vibe Coder Demo on iOS | Image Generation on Android |
| :-: | :-: |
| <img src="vibecode_demo.gif" width="300" style="border-radius:8px;" /> | <img src="android_image_generation_demo.gif" width="300" style="border-radius:8px;" /> |
| Vibe Coder using Gemma 4 model on iPhone (HTML preview) | Stable Diffusion image generation on Android |

## 🚀 Features

### 🛠️ AI Tools Suite
| Tool | Description |
|------|-------------|
| **💬 Chat** | Multi-turn conversations with RAG memory, web search, TTS auto-readout, and multimodal input |
| **🤖 creAItor** | **[NEW]** Design custom AI personas with specialized system prompts (PCTF) in seconds |
| **💻 Vibe Coder** | **[NEW]** Explain your app idea and watch it be built in real-time with live HTML/JS preview |
| **✍️ Writing Aid** | Summarize, expand, rewrite, improve grammar, or generate code from descriptions |
| **🎨 Image Generator** | Create images from text prompts using Stable Diffusion 1.5 with swipeable gallery |
| **🔍 Image Upscale** | **[NEW]** Upscale images up to 4× using AI super-resolution models (RealESRGAN, UltraSharp) with NPU acceleration |
| **🎥 Video Generator** | **[NEW]** Generate videos from text prompts or images using Stable Video Diffusion on iOS |
| **🌍 Translator** | Translate text, images (OCR), and audio across 50+ languages - offline |
| **🎙️ Transcriber** | Convert speech to text with on-device processing using Whisper models |
| **🛡️ Scam Detector** | Analyze messages and images for phishing with risk assessment |
| **🗣️ VibeVoice** | **[NEW]** Hands-free AI voice chat |

### 🎙️ Supported ASR Models (Android)
The offline **Transcriber** feature on Android supports on-device Whisper models for high-accuracy local speech-to-text.

Android also includes Kokoro TTS models for local text-to-speech.

### 🔐 Privacy First
- **100% on-device processing** - no internet required for inference
- **Zero data collection** - conversations never leave your device
- **No accounts, no tracking** - completely private
- **Open-source** - fully transparent

### ⚡ Advanced Capabilities
- GPU/NPU acceleration for fast performance
- Text-to-Speech with auto-readout
- RAG with global memory for enhanced responses
- Import custom models (.task, .litertlm, qnn,.mnn, .gguf)
- Direct downloads from HuggingFace
- 17 language interfaces

## 🚧 Upcoming Features & Roadmap

### Desktop Expansion (Windows & macOS)
- **Advanced Vibe Coder** — Enhanced code generation with desktop IDE integration, cursor control, and Claude Code–like experiences
- **OpenClaw-like Agent** — Autonomous tool-using agent framework for complex multi-step tasks
- **Advanced Media Generation** — Professional-grade image/video synthesis with ComfyUI/Draw Things feature parity and extended controls

Quick Start
1. Download from **Google Play** or the **App Store**, or build from source
2. Open Settings → Download Models → Download or Import a model
3. Select a model and start chatting or generating images


Technology
- **Android**: Kotlin + Jetpack Compose (Material 3), [GenieX SDK](https://github.com/qualcomm/GenieX) (for LLM and VLM inference), [WhisperKit](https://github.com/argmaxinc/WhisperKitAndroid) (for ASR)
- **iOS**: Swift + SwiftUI, [Run Anywhere SDK](https://github.com/RunanywhereAI/runanywhere-sdks), [Draw Things (MediaGenerationKit)](https://drawthings.ai/), Apple Foundation Model, [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for on-device ASR
- **LLM & ASR Runtime**: MediaPipe, LiteRT, GenieX SDK (GGUF on Android), WhisperKit (ASR with TFLite + QNN NPU on Android), Llama.cpp (via [Run Anywhere SDK](https://github.com/RunanywhereAI/runanywhere-sdks) on iOS), whisper.cpp on iOS for transcription
- **Image & Video Gen**: [Draw Things (MediaGenerationKit)](https://drawthings.ai/) (iOS), Qualcomm QNN (Android)


Acknowledgments
- [GenieX SDK](https://github.com/qualcomm/GenieX) — GGUF model inference support (credit shown in-app About) ⚡
- [WhisperKit](https://github.com/argmaxinc/WhisperKitAndroid) — On-device ASR with TFLite + NPU acceleration
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — On-device ASR/transcription support on iOS
- [Run Anywhere SDK](https://github.com/RunanywhereAI/runanywhere-sdks) — iOS model runtime and LLM execution framework 🚀
- [Draw Things](https://drawthings.ai/) — iOS image and video generation engine (MediaGenerationKit) 🎨
- **Google, OpenAI, Meta, Microsoft, IBM, LiquidAI, Mistral, Primsm ML, HuggingFace** — model and tooling contributions

Development Setup

### Android local development (Android Studio + Gradle)
```bash
git clone https://github.com/timmyy123/LLM-Hub.git
cd LLM-Hub/android
./gradlew assembleDebug
./gradlew installDebug
```

### Android-only local configuration

#### Setting up Hugging Face Token for Development
To use private or gated models, add your HuggingFace token to `android/local.properties` (do NOT commit this file):
```properties
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
Save and sync Gradle in Android Studio; the app will read `BuildConfig.HF_TOKEN` at build time.

#### Dev Premium Flag
To skip ads and unlock all premium features locally without a real IAP purchase, add this to `android/local.properties`:
```properties
DEBUG_PREMIUM=true
```
Set it back to `false` before making a production build.

#### Model License Acceptance
Some models on HuggingFace (especially from Google and Meta) require explicit license acceptance before downloading. When building the app locally:

1. Ensure you have a valid HuggingFace read token in `local.properties` (see above)
2. **For each model you want to download:**
   - Visit the model's HuggingFace page (e.g., https://huggingface.co/google/gemma-3n-E2B-it-litert-lm)
   - Click the "Access repository" or license acceptance button
   - Grant consent to the model's license terms
   - Try downloading the model in the app again

**Note:** This is only required for local development builds. The Play Store version uses different authentication and does not require manual license acceptance for each model.

### iOS local development (macOS + Xcode)

#### Prerequisites
- macOS with Xcode installed (use a version that matches your iOS device version)
- An Apple ID signed into Xcode (free Personal Team works for local device testing)
- iPhone with Developer Mode enabled if you run on real hardware

#### Build and run on iPhone
1. Clone the repo and open the iOS project:
```bash
git clone https://github.com/timmyy123/LLM-Hub.git
cd LLM-Hub
open ios/LLMHub/LLMHub.xcodeproj
```
2. In Xcode, select target **LLMHub** → **Signing & Capabilities**:
   - Set your **Team**
   - Set a unique **Bundle Identifier** (for example: `com.yourname.llmhub`)
   - Keep **Automatically manage signing** enabled
3. Select your iPhone as the run destination and press **Run**.

#### If you use Xcode beta
If your phone is on a newer iOS build and requires Xcode beta support, switch CLI tools:
```bash
sudo xcode-select -s /Applications/Xcode-beta.app/Contents/Developer
xcodebuild -version
```

#### iOS TTS voice troubleshooting

**TTS falls back to a robotic default voice after changing Siri voices**

iOS can get into a state where the system TTS voice cache doesn't update after a Siri voice change, causing the app to fall back to the old default voice. To fix:

1. Go to **Settings → Accessibility → Spoken Content → Voices**
2. Tap your language and select any voice (even a different one)
3. Switch back to your preferred Siri voice or whichever voice you want

This forces iOS to refresh the voice selection and the app will use the correct voice immediately.

#### Useful iOS dev troubleshooting
- If signing fails, re-check Team + Bundle Identifier in target settings.
- If build cache acts stale, clean DerivedData:
```bash
rm -rf ~/Library/Developer/Xcode/DerivedData/LLMHub-*
```
- Build logs: **Report Navigator** (`Cmd+9`)
- Runtime logs: **Debug Console** (`Cmd+Shift+Y`)

Contributing
- Fork → branch → PR. See CONTRIBUTING.md (or open an issue/discussion if unsure).

License
- Source code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
- You are free to use, study, and build on this project for non-commercial purposes.
- **Commercial use — including distributing the app, charging for it, or monetizing it with ads or IAP — is not permitted without explicit written permission from the author.**
- Contact timmy@llm-hub.app for commercial licensing enquiries.

Support
- Email: timmy@llm-hub.app
- Issues & Discussions: GitHub

Notes
- This README is intentionally concise — consult `ModelData.kt` for exact model variants, sizes, and format details.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=timmyy123/LLM-Hub&type=date&legend=top-left)](https://www.star-history.com/#timmyy123/LLM-Hub&type=date&legend=top-left)





