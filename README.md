# LLM Hub 🤖

[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://android.com)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-3.0-blue.svg)](https://github.com/timmyy123/LLM-Hub/releases)

**LLM Hub** is an open-source Android application that brings the power of Large Language Models and AI image generation directly to your mobile device. Experience AI conversations, create images, translate languages, and more - all running locally on your phone for maximum privacy and offline accessibility.

## 📸 Screenshots

<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start;">
   <figure style="margin:0;flex:0 1 300px;max-width:300px;text-align:center">
      <img src="assets/screenshots/Screenshot_20251007_041833_LLM%20Hub.jpg" alt="AI Models" style="width:300px;height:auto;border-radius:8px;display:block;" />
   </figure>
   <figure style="margin:0;flex:0 1 300px;max-width:300px;text-align:center">
      <img src="assets/screenshots/Screenshot_20251007_042114_LLM%20Hub.jpg" alt="AI Features" style="width:300px;height:auto;border-radius:8px;display:block;" />
   </figure>
   <figure style="margin:0;flex:0 1 300px;max-width:300px;text-align:center">
      <img src="assets/screenshots/Screenshot_20251007_042146_LLM%20Hub.jpg" alt="Chat Interface" style="width:300px;height:auto;border-radius:8px;display:block;" />
   </figure>
</div>

## Download

[![Get it on Google Play](https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png)](https://play.google.com/store/apps/details?id=com.llmhub.llmhub)

---

## 🚀 Features

### 🛠️ Six AI Tools
| Tool | Description |
|------|-------------|
| **💬 Chat** | Multi-turn conversations with RAG memory, web search, TTS auto-readout, and multimodal input (text, images, audio) |
| **✍️ Writing Aid** | Summarize, expand, rewrite, improve grammar, or generate code from descriptions |
| **🎨 Image Generator** | Create images from text prompts using Stable Diffusion 1.5 with swipeable gallery for variations |
| **🌍 Translator** | Translate text, images (OCR), and audio across 50+ languages - works offline |
| **🎙️ Transcriber** | Convert speech to text with on-device processing |
| **🛡️ Scam Detector** | Analyze messages and images for phishing with risk assessment |

### 🔐 Privacy First
- **100% on-device processing** - no internet required for inference
- **Zero data collection** - conversations never leave your device
- **No accounts, no tracking** - completely private
- **Open-source** - fully transparent

### ⚡ Advanced Capabilities
- GPU/NPU acceleration for fast performance
- Text-to-Speech with auto-readout
- RAG with global memory for enhanced responses
- Import custom models (.task, .litertlm, .mnn)
- Direct downloads from HuggingFace
- 16 language interfaces

## 📱 Supported Models

### Text & Multimodal Models
| Model | Type | Context |
|-------|------|---------|
| **Gemma-3 1B** (Google) | Text | 2k-4k |
| **Gemma-3n E2B/E4B** (Google) | Text + Vision + Audio | 4k |
| **Llama-3.2 1B/3B** (Meta) | Text | 1.2k |
| **Phi-4 Mini** (Microsoft) | Text | 4k |

### Image Generation Models
| Model | Backend |
|-------|---------|
| **Absolute Reality SD1.5** | MNN (CPU) / QNN (NPU) |

### Embedding Models (RAG)
- **Gecko-110M** - Compact embeddings (64D-1024D)
- **EmbeddingGemma-300M** - High-quality text embeddings

## 🧠 RAG & Memory System

- **On-device RAG & Embeddings**: Retrieval-augmented generation runs locally using embedding models
- **Global Memory**: Upload documents to a shared memory store for RAG lookups across conversations
- **Chunking & Persistence**: Documents are split into chunks with embeddings stored in Room database
- **Privacy**: All embeddings and searches happen locally - no external endpoints

## 🔎 Web Search

- **DuckDuckGo integration** for optional web augmentation
- **URL content extraction** for fetching page content
- **Privacy-focused**: Uses public DuckDuckGo endpoints, no API keys required

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Kotlin |
| UI | Jetpack Compose + Material 3 |
| LLM Runtime | MediaPipe & LiteRT |
| Image Gen | MNN / QNN (Stable Diffusion) |
| Quantization | INT4/INT8 |
| GPU | LiteRT XNNPACK delegate |

## 📋 Requirements

- **Android 8.0+** (API 26)
- **RAM**: 2GB minimum, 6GB+ recommended
- **Storage**: 1-5GB depending on models
- **Internet**: Only for model downloads

## 🚀 Getting Started

### Installation

1. Download from [Google Play](https://play.google.com/store/apps/details?id=com.llmhub.llmhub) or [Releases](https://github.com/timmyy123/LLM-Hub/releases)
2. Open the app and go to **Settings → Download Models**
3. Download your preferred models
4. Start using AI tools offline!

### Building from Source

```bash
git clone https://github.com/timmyy123/LLM-Hub.git
cd LLM-Hub
./gradlew assembleDebug
./gradlew installDebug
```

### Importing Custom Models

- **Supported formats**: `.task`, `.litertlm`, `.mnn`
- Go to **Settings → Download Models → Import Model**
- Select your model file from device storage

## 🔧 Model Selection Guide

| RAM | Recommended Models |
|-----|-------------------|
| 2GB | Gemma-3 1B INT4 |
| 4GB | Gemma-3 1B INT8, Llama-3.2 1B |
| 6GB+ | Gemma-3n, Llama-3.2 3B, SD Image Gen |
| 8GB+ | Phi-4 Mini with GPU |


## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install Android Studio
# Open project in Android Studio
# Sync Gradle files
# Run on device/emulator
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google** for Gemma models and MediaPipe framework
- **Meta** for Llama models
- **Microsoft** for Phi models
- **Alibaba** for MNN framework (Stable Diffusion)
- **HuggingFace** for model hosting and community

## 📞 Support

- **Email**: [timmyboy0623@gmail.com](mailto:timmyboy0623@gmail.com)
- **Issues**: [GitHub Issues](https://github.com/timmyy123/LLM-Hub/issues)
- **Discussions**: [GitHub Discussions](https://github.com/timmyy123/LLM-Hub/discussions)


**Made with ❤️ by Timmy**

*Bringing AI to your pocket, privately and securely.*

## Setting up Hugging Face Token for Development

To use private or gated models, you need to provide your Hugging Face (HF) access token. This project is set up to securely load your token from your local machine using `local.properties` (never commit your token to source control).

### Steps:

1. **Open or create `local.properties` in your project root.**
   - This file is usually already present and is ignored by git by default.

2. **Add your Hugging Face token:**
   ```properties
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   Replace `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` with your actual token from https://huggingface.co/settings/tokens

3. **Sync Gradle:**
   - In Android Studio, click "Sync Project with Gradle Files" after saving `local.properties`.

4. **How it works:**
   - The build system injects your token into the app at build time as `BuildConfig.HF_TOKEN`.
   - The app uses this token for authenticated model downloads.

**Note:**
- Never commit your `local.properties` file or your token to version control.
- If you change your token, update `local.properties` and re-sync Gradle.

