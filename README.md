# LLM Hub 🤖

**LLM Hub** is an open-source Android app for on-device LLM chat and image generation. It's optimized for mobile usage (CPU/GPU/NPU acceleration) and supports multiple model formats so you can run powerful models locally and privately.

## 📸 Screenshots

<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start;">
   <figure style="margin:0;flex:0 1 300px;max-width:300px;text-align:center">
      <img src="assets\screenshots\Screenshot_20260214_201455.png" alt="AI Models" style="width:300px;height:auto;border-radius:8px;display:block;" />
   </figure>
   <figure style="margin:0;flex:0 1 300px;max-width:300px;text-align:center">
      <img src="assets/screenshots/Screenshot_20251007_042114_LLM%20Hub.jpg" alt="AI Features" style="width:300px;height:auto;border-radius:8px;display:block;" />
   </figure>
   <figure style="margin:0;flex:0 1 300px;max-width:300px;text-align:center">
      <img src="assets/screenshots/Screenshot_20251007_042146_LLM%20Hub.jpg" alt="Chat Interface" style="width:300px;height:auto;border-radius:8px;display:block;" />
   </figure>
</div>

## 🚀 Features

### 🛠️ AI Tools Suite
| Tool | Description |
|------|-------------|
| **💬 Chat** | Multi-turn conversations with RAG memory, web search, TTS auto-readout, and multimodal input |
| **🤖 creAItor** | **[NEW]** Design custom AI personas with specialized system prompts (PCTF) in seconds |
| **💻 Vibe Coder** | **[NEW]** Explain your app idea and watch it be built in real-time with live HTML/JS preview |
| **✍️ Writing Aid** | Summarize, expand, rewrite, improve grammar, or generate code from descriptions |
| **🎨 Image Generator** | Create images from text prompts using Stable Diffusion 1.5 with swipeable gallery |
| **🌍 Translator** | Translate text, images (OCR), and audio across 50+ languages - offline |
| **🎙️ Transcriber** | Convert speech to text with on-device processing |
| **🛡️ Scam Detector** | Analyze messages and images for phishing with risk assessment |

### ✨ Vibes & Creators
- **Vibes**: A full on-device coding environment. The LLM writes HTML/JS/CSS based on your requirements, and you can preview/run the app instantly in a secure sandbox.
- **creAItor**: Powerful persona generation to create anything from characters with fun personalities to system architects. Just describe a creAItor ("respond like a pirate" or "respond with a markdown spec for a code agent to generate a full-stack system"), and the on-device LLM generates a complex system prompt (PCTF format) that you can use in chat.

###  Kid Mode
Activate this in Settings under Kid Mode. Set a PIN, and the mode will remain in effect until you unlock it with the same PIN.
- **Safe Exploration**: Families can introduce children to private local AI with confidence.
- **Model-Level Guardrails**: Automatically activates strict safety protocols at the model inference layer across **all** tools (Chat, Writing Aid, Image Gen, etc.).
- **Peace of Mind**: Ensures all generated content is age-appropriate without compromising the privacy benefits of on-device processing.

### �🔐 Privacy First
- **100% on-device processing** - no internet required for inference
- **Zero data collection** - conversations never leave your device
- **No accounts, no tracking** - completely private
- **Open-source** - fully transparent

### ⚡ Advanced Capabilities
- GPU/NPU acceleration for fast performance
- Text-to-Speech with auto-readout
- RAG with global memory for enhanced responses
- Import custom models (.task, .litertlm, .mnn, .gguf)
- Direct downloads from HuggingFace
- 16 language interfaces

Quick Start
1. Download from Google Play or build from source
2. Open Settings → Download Models → Download or Import a model
3. Select a model and start chatting or generating images

Supported Model Families (summary)
- Gemma (LiteRT Task)
- Llama (Task + GGUF variants)
- Phi (LiteRT LM)
- LiquidAI LFM (LFM 2.5 1.2B + LFM VL 1.6B vision-enabled)
- Ministral / Mistral family (GGUF / ONNX)
- IBM Granite (GGUF)

Model Formats
- Task / LiteRT (.task): MediaPipe/LiteRT optimized models (GPU/NPU capable)
- LiteRT LM (.litertlm): LiteRT language models
- GGUF (.gguf): Quantized models — CPU inference powered by Nexa SDK; some vision-capable GGUF models require an additional `mmproj` vision project file
- ONNX (.onnx): Cross-platform model runtime

GGUF Compatibility Notes
- Not all Android devices can load GGUF models in this app.
- GGUF loading/runtime depends on Nexa SDK native libraries and device/ABI support; on unsupported devices, GGUF model loading can fail even if the model file is valid.
- In this app, the GGUF NPU option is intentionally shown only for Snapdragon 8 Gen 4-class devices.

Importing models
- Settings → Download Models → Import Model → choose `.task`, `.litertlm`, `.mnn`, `.gguf`, or `.onnx`
- The full model list and download links live in `app/src/.../data/ModelData.kt` (do not exhaustively list variants in the README)

Technology
- Kotlin + Jetpack Compose (Material 3)
- LLM Runtime: MediaPipe, LiteRT, Nexa SDK
- Image Gen: MNN / Qualcomm QNN
- Quantization: INT4/INT8

Acknowledgments
- **Nexa SDK** — GGUF model inference support (credit shown in-app About) ⚡
- **Google, Meta, Microsoft, IBM, LiquidAI, Mistral, HuggingFace** — model and tooling contributions

Development Setup

### Building from source
```bash
git clone https://github.com/timmyy123/LLM-Hub.git
cd LLM-Hub
./gradlew assembleDebug
./gradlew installDebug
```

### Setting up Hugging Face Token for Development
To use private or gated models, add your HuggingFace token to `local.properties` (do NOT commit this file):
```properties
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
Save and sync Gradle in Android Studio; the app will read `BuildConfig.HF_TOKEN` at build time.

### Model License Acceptance
Some models on HuggingFace (especially from Google and Meta) require explicit license acceptance before downloading. When building the app locally:

1. Ensure you have a valid HuggingFace read token in `local.properties` (see above)
2. **For each model you want to download:**
   - Visit the model's HuggingFace page (e.g., https://huggingface.co/google/gemma-3n-E2B-it-litert-lm)
   - Click the "Access repository" or license acceptance button
   - Grant consent to the model's license terms
   - Try downloading the model in the app again

**Note:** This is only required for local development builds. The Play Store version uses different authentication and does not require manual license acceptance for each model.

Contributing
- Fork → branch → PR. See CONTRIBUTING.md (or open an issue/discussion if unsure).

License
- MIT (see LICENSE)

Support
- Email: timmyboy0623@gmail.com
- Issues & Discussions: GitHub

Notes
- This README is intentionally concise — consult `ModelData.kt` for exact model variants, sizes, and format details.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=timmyy123/LLM-Hub&type=date&legend=top-left)](https://www.star-history.com/#timmyy123/LLM-Hub&type=date&legend=top-left)

---
*If you want, I can also add a short “Release notes / changelog” section and a quick performance guide for device profiles.*

