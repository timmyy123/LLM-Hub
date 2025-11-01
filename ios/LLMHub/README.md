# LLM Hub - iOS Version

This is the native iOS version of LLM Hub, a mobile application that brings Large Language Models (LLMs) directly to your iOS device using on-device inference.

## Overview

LLM Hub for iOS is built with:
- **SwiftUI** for the user interface
- **MediaPipe Tasks GenAI** for LLM inference
- **Native iOS frameworks** for optimal performance

## Features

- 🤖 Multiple LLM model support (Gemma, Llama, Phi)
- 📱 On-device processing for complete privacy
- 🖼️ Vision support for multimodal models
- 🎙️ Audio input support
- ⚡ GPU acceleration on supported devices
- 💾 Offline usage after model download

## Requirements

- iOS 15.0 or later
- Xcode 15.0 or later
- Swift 5.9 or later
- 2GB+ RAM (4GB+ recommended)

## Installation

### Using CocoaPods

1. Install CocoaPods if you haven't already:
   ```bash
   sudo gem install cocoapods
   ```

2. Navigate to the iOS project directory:
   ```bash
   cd ios/LLMHub
   ```

3. Install dependencies:
   ```bash
   pod install
   ```

4. Open the workspace:
   ```bash
   open LLMHub.xcworkspace
   ```

### Using Swift Package Manager (Alternative)

The project is configured to use Swift Package Manager. Simply open the project in Xcode and it will automatically fetch dependencies.

## Building from Source

1. Clone the repository (if not already cloned)
2. Navigate to the iOS project:
   ```bash
   cd ios/LLMHub
   ```
3. Open in Xcode:
   ```bash
   open LLMHub.xcodeproj
   ```
4. Select your target device or simulator
5. Click Run (⌘R)

## Project Structure

```
ios/LLMHub/
├── LLMHub/
│   ├── LLMHubApp.swift          # App entry point
│   ├── ContentView.swift        # Main navigation
│   ├── Models/
│   │   ├── LLMModel.swift       # Model data structures
│   │   └── ChatMessage.swift    # Chat message model
│   ├── Services/
│   │   └── InferenceService.swift # MediaPipe inference wrapper
│   ├── Views/
│   │   ├── ChatView.swift       # Chat interface
│   │   ├── ModelSelectorView.swift # Model selection & download
│   │   └── SettingsView.swift   # Settings & configuration
│   ├── ViewModels/
│   │   └── ChatViewModel.swift  # Chat logic
│   └── Assets.xcassets/         # App icons and colors
├── LLMHub.xcodeproj/            # Xcode project file
└── Package.swift                # Swift Package Manager manifest
```

## MediaPipe Integration

This app uses MediaPipe Tasks GenAI framework for LLM inference. Follow the official documentation for detailed integration:

📚 [MediaPipe LLM Inference for iOS](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios)

### Key Integration Points

The `InferenceService.swift` file contains the MediaPipe integration. To complete the integration:

1. Add MediaPipe Tasks GenAI via CocoaPods:
   ```ruby
   pod 'MediaPipeTasksGenAI'
   ```

2. Import in InferenceService:
   ```swift
   import MediaPipeTasksGenAI
   ```

3. Initialize LLMInference:
   ```swift
   let options = LlmInferenceOptions()
   options.modelPath = modelPath
   options.maxTokens = 512
   options.temperature = 0.8
   let llmInference = try LlmInference(options: options)
   ```

4. Generate responses:
   ```swift
   let response = try await llmInference.generateResponse(inputText: prompt)
   ```

## Usage

1. **Download Models**: Navigate to the Models tab and download your preferred LLM model
2. **Load Model**: Tap "Load" on a downloaded model to initialize it
3. **Start Chatting**: Go to the Chat tab and start conversing with the AI
4. **Configure Settings**: Adjust generation parameters in the Settings tab

## Supported Models

The iOS version supports the same models as the Android version:

### Text Models
- Gemma-3 1B (INT4/INT8 variants)
- Llama-3.2 (1B/3B)
- Phi-4 Mini

### Multimodal Models
- Gemma-3n E2B/E4B (text + vision + audio)

## Configuration

Generation parameters can be configured in Settings:
- **Max Tokens**: 128-2048 tokens
- **Temperature**: 0.0-2.0
- **Top K**: 1-100
- **Top P**: 0.0-1.0

## Privacy

All inference happens on-device. Your conversations and data never leave your device unless you explicitly enable cloud features (not implemented in this version).

## Development Notes

### Current Implementation Status

This is the initial iOS implementation with the following structure in place:
- ✅ SwiftUI interface
- ✅ Model management system
- ✅ Chat interface
- ✅ Settings screen
- ⚠️ MediaPipe integration (placeholder - requires actual MediaPipe framework)

### TODO

- [ ] Complete MediaPipe Tasks GenAI integration
- [ ] Add actual model download progress tracking
- [ ] Implement vision and audio input support
- [ ] Add database persistence for chat history
- [ ] Implement RAG (Retrieval-Augmented Generation)
- [ ] Add web search integration
- [ ] Implement Text-to-Speech
- [ ] Add additional AI tools (Translator, Writing Aid, etc.)

## Contributing

We welcome contributions! If you'd like to improve the iOS version:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly on iOS devices
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Acknowledgments

- **Google** for MediaPipe framework and Gemma models
- **Meta** for Llama models
- **Microsoft** for Phi models
- **Apple** for SwiftUI and iOS development tools

## Support

For issues and questions:
- GitHub Issues: [timmyy123/LLM-Hub/issues](https://github.com/timmyy123/LLM-Hub/issues)
- Email: timmyboy0623@gmail.com

---

**Made with ❤️ for iOS by the LLM Hub team**

*Bringing AI to your pocket, privately and securely.*
