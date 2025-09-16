# LLM Hub 🤖

[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://android.com)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📸 Screenshots

<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start;">
   <figure style="margin:0;flex:0 1 300px;max-width:300px;text-align:center">
      <img src="assets/screenshots/Screenshot_20250916_155747_LLM%20Hub.jpg" alt="Chat screen" style="width:300px;height:auto;border-radius:8px;display:block;" />
      <figcaption style="font-size:12px;margin-top:6px;color:var(--color-fg-muted,#6b7280)">Chat screen</figcaption>
   </figure>

   <figure style="margin:0;flex:0 1 300px;max-width:300px;text-align:center">
      <img src="assets/screenshots/Screenshot_20250916_155727_LLM%20Hub.jpg" alt="Model list" style="width:300px;height:auto;border-radius:8px;display:block;" />
      <figcaption style="font-size:12px;margin-top:6px;color:var(--color-fg-muted,#6b7280)">Model list</figcaption>
   </figure>
</div>

**LLM Hub** is an open-source Android application that brings the power of Large Language Models (LLMs) directly to your mobile device. Experience AI conversations with state-of-the-art models like Gemma, Llama, and Phi - all running locally on your phone for maximum privacy and offline accessibility.

## 🚀 Features

- **🤖 Multiple LLM Models**: Support for Gemma-3, Llama-3.2, Phi-4, and Gemma-3n
- **📱 On-Device Processing**: Complete privacy - no internet required for inference
- **🖼️ Vision Support**: Multimodal models that understand text, images, and audio input
- **🎙️ Audio Input**: Voice recording support for Gemma-3n models with speech recognition
- **⚡ GPU Acceleration**: Optimized performance on supported devices (8GB+ RAM)
- **💾 Offline Usage**: Chat without internet connection after model download
- **🔒 Privacy First**: Your conversations never leave your device
- **🎨 Modern UI**: Clean, intuitive Material Design interface
- **📥 Direct Downloads**: Download models directly from HuggingFace

##  Download

[![Get it on Google Play](https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png)](https://play.google.com/store/apps/details?id=com.llmhub.llmhub)

*Available on Google Play Store for easy installation and automatic updates*

## 📱 Supported Models

### Text Models
- **Gemma-3 1B Series** (Google)
   - INT4 quantization - 2k context
   - INT8 quantization - 1.2k context
   - INT8 quantization - 2k context
   - INT8 quantization - 4k context

- **Llama-3.2 Series** (Meta)
   - 1B model - 1.2k context
   - 3B model - 1.2k context

- **Phi-4 Mini** (Microsoft)
   - INT8 quantization - 1.2k context

### Multimodal Models (Vision + Audio + Text)
- **Gemma-3n E2B** (Gemma-3n family) - Supports text, images, and audio input
- **Gemma-3n E4B** (Gemma-3n family) - Supports text, images, and audio input

**Memory & RAG (Global Context)**

- **On-device RAG & Embeddings:** The app performs retrieval-augmented generation (RAG) locally on the device. Embeddings and semantic search are implemented using the app's RAG manager and embedding models (see `RagServiceManager`, `MemoryProcessor`, and the compact Gecko embedding entry in `ModelData.kt`).
- **Global Memory (import-only):** Users can upload or paste documents into a single global memory store. This is a global context used for RAG lookups — it is not a per-conversation conversational memory. The global memory is managed via the Room database (`memoryDao`) and exposed in the Settings and Memory screens.
- **Chunking & Persistence:** Uploaded documents are split into chunks; chunk embeddings are computed and persisted. On startup the app restores persisted chunk embeddings from the database and repopulates the in-memory RAG index.
- **RAG Flow in Chat:** The chat pipeline queries the RAG index (both per-chat documents and optional global memory) to build a RAG context that is inserted into the prompt (the code assembles a "USER MEMORY FACTS" block before the assistant prompt). See `ChatViewModel` for the exact integration points where embeddings are generated (`generateEmbedding`) and searched (`searchRelevantContext`, `searchGlobalContext`).
- **Controls & Settings:** Embeddings and RAG can be enabled/disabled in Settings, and the user can choose the embedding model used for semantic search (the UI exposes embedding model selection via the settings and `ThemeViewModel`).
- **Local-only:** All embeddings, RAG searches and document chunk storage happen locally (Room DB + in-memory index). No external endpoints are used for RAG or memory lookups.


## 🛠️ Technology Stack

- **Language**: Kotlin
- **UI Framework**: Jetpack Compose
- **AI Runtime**: MediaPipe & LiteRT (formerly TensorFlow Lite)
- **Model Optimization**: INT4/INT8 quantization
- **GPU Acceleration**: LiteRT XNNPACK delegate
- **Model Source**: HuggingFace & Google repositories

## 📋 Requirements

- **Android 8.0** (API level 26) or higher
- **RAM**: 
  - Minimum 2GB for small models
  - 6GB+ recommended for better performance
- **Storage**: 1GB - 5GB depending on selected models
- **Internet**: Required only for model downloads

## 🚀 Getting Started

### Installation

1. **Download APK**: Get the latest release from [Releases](https://github.com/timmyy123/LLM-Hub/releases)
2. **Install**: Enable "Unknown Sources" and install the APK
3. **Download Models**: Use the in-app model downloader to get your desired models

### Building from Source

```bash
# Clone the repository
git clone https://github.com/timmyy123/LLM-Hub.git

# Navigate to project directory
cd LLM-Hub

# Build the project
./gradlew assembleDebug

# Install on device
./gradlew installDebug
```

### Usage

1. **Launch the app**
2. **Go to Settings → Download Models**
3. **Select and download your preferred model**
4. **Start chatting** once the model is downloaded
5. **For vision models**: Tap the image icon to upload photos

## 📖 How It Works

LLM Hub uses Google's MediaPipe framework with LiteRT to run quantized AI models directly on your Android device. The app:

1. **Downloads** pre-optimized `.task` files from HuggingFace
2. **Loads** models into MediaPipe's LLM Inference API
3. **Processes** your input locally using CPU or GPU
4. **Generates** responses without sending data to external servers

## 🔧 Configuration

### GPU Acceleration
- **Gemma-3 1B models**: recommend at least 4GB RAM for GPU acceleration
- **Gemma-3n models**: recommend at least 8GB RAM for GPU acceleration
- **Llama & Phi models**: CPU only (compatibility issues)

### Model Selection
Choose models based on your device capabilities:
- **2GB RAM**: Gemma-3 1B INT4
- **4GB RAM**: Gemma-3 1B INT8, Llama-3.2 1B
- **6GB+ RAM**: Gemma-3n, Llama-3.2 3B
- **8GB+ RAM**: Phi-4 Mini, GPU acceleration

## 🔎 Web Search

- **Built-in web search:** LLM Hub includes an on-device web search integration used for document lookups and optional augmentation of model responses. The implementation is a DuckDuckGo-based service (`WebSearchService` / `DuckDuckGoSearchService`) bundled with the app.
- **How it works:** The search service first attempts content-aware searches: it detects if a query contains a URL and will fetch page content directly. For general queries it:
   - tries DuckDuckGo Instant Answer API (JSON) for short answers and definitions,
   - falls back to DuckDuckGo HTML search scraping when needed,
   - performs optional content extraction: fetches result pages and extracts text snippets to return richer snippets to the app.
- **Privacy & limits:** Searches use public DuckDuckGo endpoints (no API key required). The app performs HTTP requests from the device; network access is required for web search and content fetching. The web search implementation includes timeouts and result limits to avoid excessive requests.
- **Usage in app:** Search results are returned as title/snippet/url tuples and can be used by the chat UI or RAG/document upload flows to provide external context or to fetch page content when users paste a URL.


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
- **HuggingFace** for model hosting and community
- **Android Community** for development tools and libraries

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
 - Never commit your `local.properties` file or your token to version control.
 - If you change your token, update `local.properties` and re-sync Gradle.


