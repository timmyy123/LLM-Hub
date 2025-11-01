# MediaPipe Integration Guide for LLM Hub iOS

This guide provides step-by-step instructions for integrating MediaPipe Tasks GenAI into the LLM Hub iOS application.

## Overview

MediaPipe Tasks GenAI provides on-device LLM inference capabilities for iOS. This document explains how to complete the integration that has been prepared in the `InferenceService.swift` file.

## Official Documentation

📚 [MediaPipe LLM Inference for iOS](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios)

## Prerequisites

- Xcode 15.0 or later
- iOS 15.0 or later
- CocoaPods installed
- Downloaded LLM model in `.task` format

## Installation Steps

### Option 1: Using CocoaPods (Recommended)

1. Navigate to the iOS project directory:
   ```bash
   cd ios/LLMHub
   ```

2. Install dependencies:
   ```bash
   pod install
   ```

3. Open the workspace (not the project):
   ```bash
   open LLMHub.xcworkspace
   ```

### Option 2: Using Swift Package Manager

1. Open `LLMHub.xcodeproj` in Xcode
2. Go to File > Add Packages
3. Add MediaPipe package (when available via SPM)

## Code Integration

### 1. Import MediaPipe Framework

In `Services/InferenceService.swift`, add the import at the top:

```swift
import MediaPipeTasksGenAI
```

### 2. Update InferenceService Implementation

Replace the placeholder LLMInferenceProtocol with actual MediaPipe types:

```swift
class InferenceService: ObservableObject {
    @Published var currentModel: LLMModel?
    @Published var isModelLoaded = false
    @Published var isGenerating = false
    @Published var errorMessage: String?
    
    private var llmInference: LlmInference?
    private var cancellables = Set<AnyCancellable>()
    
    // Load model implementation
    func loadModel(_ model: LLMModel) async throws {
        guard let modelPath = model.localFilePath, 
              FileManager.default.fileExists(atPath: modelPath.path) else {
            throw InferenceError.modelNotFound
        }
        
        await MainActor.run {
            self.isModelLoaded = false
            self.currentModel = model
        }
        
        // Configure MediaPipe options
        let options = LlmInferenceOptions()
        options.modelPath = modelPath.path
        options.maxTokens = model.contextWindowSize
        options.topK = 40
        options.temperature = 0.8
        
        // Enable GPU if supported
        if model.supportsGpu {
            // GPU delegate configuration
            let baseOptions = BaseOptions()
            baseOptions.delegateOptions = GPUOptions()
            options.baseOptions = baseOptions
        }
        
        // Initialize LLM Inference
        llmInference = try LlmInference(options: options)
        
        await MainActor.run {
            self.isModelLoaded = true
        }
    }
    
    // Generate response implementation
    func generateResponse(prompt: String) async throws -> String {
        guard isModelLoaded, let inference = llmInference else {
            throw InferenceError.modelNotLoaded
        }
        
        await MainActor.run {
            self.isGenerating = true
        }
        
        defer {
            Task { @MainActor in
                self.isGenerating = false
            }
        }
        
        // Generate response using MediaPipe
        let response = try await inference.generateResponse(inputText: prompt)
        return response
    }
    
    // Stream response implementation
    func generateResponseStream(prompt: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard isModelLoaded, let inference = llmInference else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }
                
                await MainActor.run {
                    self.isGenerating = true
                }
                
                // Use MediaPipe's streaming API
                inference.generateResponseAsync(
                    inputText: prompt,
                    progress: { partialResponse, error in
                        if let error = error {
                            continuation.finish(throwing: error)
                            return
                        }
                        if let text = partialResponse {
                            continuation.yield(text)
                        }
                    },
                    completion: { finalResponse, error in
                        Task { @MainActor in
                            self.isGenerating = false
                        }
                        if let error = error {
                            continuation.finish(throwing: error)
                        } else {
                            continuation.finish()
                        }
                    }
                )
            }
        }
    }
}
```

### 3. Vision Support (Optional)

For models that support vision (multimodal models):

```swift
import MediaPipeTasksVision

// In InferenceService, add method for image-based inference
func generateResponseWithImage(prompt: String, image: UIImage) async throws -> String {
    guard isModelLoaded, let inference = llmInference else {
        throw InferenceError.modelNotLoaded
    }
    
    // Convert UIImage to MediaPipe Image
    guard let cgImage = image.cgImage else {
        throw InferenceError.invalidImage
    }
    
    let mpImage = MPImage(image: image)
    
    // Generate response with image
    let response = try await inference.generateResponse(
        inputText: prompt,
        image: mpImage
    )
    
    return response
}
```

### 4. Error Handling

Add proper error handling for MediaPipe operations:

```swift
enum InferenceError: LocalizedError {
    case modelNotFound
    case modelNotLoaded
    case invalidURL
    case invalidPath
    case invalidImage
    case inferenceFailure(String)
    case mediaPipeError(Error)
    
    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Model file not found. Please download the model first."
        case .modelNotLoaded:
            return "No model is currently loaded."
        case .invalidURL:
            return "Invalid model URL."
        case .invalidPath:
            return "Invalid file path."
        case .invalidImage:
            return "Invalid image format."
        case .inferenceFailure(let message):
            return "Inference failed: \(message)"
        case .mediaPipeError(let error):
            return "MediaPipe error: \(error.localizedDescription)"
        }
    }
}
```

## Testing

### 1. Download a Model

You can download a test model manually:

```bash
# Download Gemma 1B INT4 model
curl -L "https://huggingface.co/google/gemma-3-1b-int4-2k/resolve/main/gemma-3-1b-int4-2k.task" \
  -o ~/Downloads/gemma-3-1b-int4-2k.task
```

### 2. Add Model to App

In the iOS Simulator or device:
1. Use Files app to copy the `.task` file
2. Or implement file sharing in Info.plist:
   ```xml
   <key>UIFileSharingEnabled</key>
   <true/>
   <key>LSSupportsOpeningDocumentsInPlace</key>
   <true/>
   ```

### 3. Test Inference

1. Launch the app
2. Go to Models tab
3. Load the downloaded model
4. Go to Chat tab
5. Send a test message

## Performance Optimization

### GPU Acceleration

For devices with sufficient memory (6GB+):

```swift
let options = LlmInferenceOptions()
// ... other settings ...

// Enable GPU delegate
let baseOptions = BaseOptions()
baseOptions.delegateOptions = GPUOptions()
options.baseOptions = baseOptions
```

### Memory Management

```swift
// Unload model when not in use
func unloadModel() {
    llmInference = nil
    currentModel = nil
    isModelLoaded = false
}

// Monitor memory usage
func checkMemoryUsage() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_,
                     task_flavor_t(MACH_TASK_BASIC_INFO),
                     $0,
                     &count)
        }
    }
    
    if kerr == KERN_SUCCESS {
        return info.resident_size
    }
    return 0
}
```

## Troubleshooting

### Common Issues

1. **Model not found error**
   - Verify the model file exists in the correct location
   - Check file permissions

2. **Out of memory error**
   - Try a smaller quantized model (INT4 instead of INT8)
   - Close other apps
   - Test on a device with more RAM

3. **Slow inference**
   - Enable GPU acceleration if available
   - Reduce maxTokens parameter
   - Use a smaller model

4. **Build errors**
   - Clean build folder (Cmd+Shift+K)
   - Delete derived data
   - Run `pod install` again

## Advanced Features

### Streaming Responses

The current implementation includes streaming support. To use it:

```swift
for try await chunk in inferenceService.generateResponseStream(prompt: prompt) {
    // Process each chunk as it arrives
    updateUI(with: chunk)
}
```

### Custom Generation Parameters

Adjust parameters per request:

```swift
let options = LlmInferenceOptions()
options.temperature = 0.7  // Creativity (0.0-2.0)
options.topK = 40          // Top-K sampling
options.topP = 0.95        // Nucleus sampling
options.maxTokens = 512    // Maximum response length
```

## Resources

- [MediaPipe LLM Inference iOS Guide](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios)
- [MediaPipe Tasks API Reference](https://developers.google.com/mediapipe/api/solutions/swift/mediapipe_tasks_genai)
- [Model Download Hub](https://huggingface.co/models?library=mediapipe)
- [LLM Hub GitHub Repository](https://github.com/timmyy123/LLM-Hub)

## Support

For issues specific to MediaPipe integration:
- Check [MediaPipe Issues](https://github.com/google/mediapipe/issues)
- Review [MediaPipe Discussions](https://github.com/google/mediapipe/discussions)

For LLM Hub iOS app issues:
- [GitHub Issues](https://github.com/timmyy123/LLM-Hub/issues)
- Email: timmyboy0623@gmail.com

---

Last updated: November 2025
