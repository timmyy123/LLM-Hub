# LLM Hub 🤖

[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](https://android.com)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-blue.svg)](https://kotlinlang.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LLM Hub** is an open-source Android application that brings the power of Large Language Models (LLMs) directly to your mobile device. Experience AI conversations with state-of-the-art models like Gemma, Llama, and Phi - all running locally on your phone for maximum privacy and offline accessibility.

## 🚀 Features

- **🤖 Multiple LLM Models**: Support for Gemma-3, Llama-3.2, Phi-4, and Gemma-3n
- **📱 On-Device Processing**: Complete privacy - no internet required for inference
- **🖼️ Vision Support**: Multimodal models that understand both text and images
- **⚡ GPU Acceleration**: Optimized performance on supported devices (8GB+ RAM)
- **💾 Offline Usage**: Chat without internet connection after model download
- **🔒 Privacy First**: Your conversations never leave your device
- **🎨 Modern UI**: Clean, intuitive Material Design interface
- **📥 Direct Downloads**: Download models directly from HuggingFace

## 📱 Supported Models

### Text Models
- **Gemma-3 1B Series** (Google)
  - INT4 quantization (529MB) - 2k context
  - INT8 quantization (1005MB) - 1.2k context
  - INT8 quantization (1024MB) - 2k context
  - INT8 quantization (1005MB) - 4k context

- **Llama-3.2 Series** (Meta)
  - 1B model (1.20GB) - 1.2k context
  - 3B model (3.08GB) - 1.2k context

- **Phi-4 Mini** (Microsoft)
  - INT8 quantization (3.67GB) - 1.2k context

### Multimodal Models (Vision + Text)
- **Gemma-3n E2B** (2.92GB) - 4k context
- **Gemma-3n E4B** (4.10GB) - 4k context

## 🛠️ Technology Stack

- **Language**: Kotlin
- **UI Framework**: Jetpack Compose
- **AI Runtime**: MediaPipe & LiteRT (formerly TensorFlow Lite)
- **Model Optimization**: INT4/INT8 quantization
- **GPU Acceleration**: LiteRT XNNPACK delegate
- **Model Source**: HuggingFace & Google repositories

## 📋 Requirements

- **Android 7.0** (API level 24) or higher
- **RAM**: 
  - Minimum 2GB for small models
  - 4GB+ recommended for better performance
  - 8GB+ required for GPU acceleration on vision models
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
- **Gemma-3 1B models**: GPU supported on all devices
- **Gemma-3n models**: GPU requires >8GB RAM
- **Llama & Phi models**: CPU only (compatibility issues)

### Model Selection
Choose models based on your device capabilities:
- **2GB RAM**: Gemma-3 1B INT4
- **4GB RAM**: Gemma-3 1B INT8, Llama-3.2 1B
- **6GB+ RAM**: Gemma-3n, Llama-3.2 3B
- **8GB+ RAM**: Phi-4 Mini, GPU acceleration

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

## 🔄 Version History

- **v1.0.0**: Initial release with Gemma, Llama, Phi, and Gemma-3n support

---

**Made with ❤️ by the LLM Hub Team**

*Bringing AI to your pocket, privately and securely.*