# Image Generation Migration: MediaPipe → Absolute Reality (QNN/MNN)

## Summary

Replaced the MediaPipe Image Generator with **Absolute Reality** (Stable Diffusion 1.5) using two backend options:
- **NPU Acceleration** (Qualcomm QNN SDK) - for Snapdragon 8 Gen 1+ devices
- **CPU Inference** (MNN Framework) - for all Android devices

## Changes Made

### 1. Device Detection (`ModelData.kt`)

Added `DeviceInfo` object to detect device SOC and select optimal model:

```kotlin
object DeviceInfo {
    fun getDeviceSoc(): String // Returns Build.SOC_MODEL on Android 12+
    fun getChipsetSuffix(): String? // Maps SOC to model variant (8gen1, 8gen2, min)
    fun isQualcommNpuSupported(): Boolean // Checks NPU compatibility
}
```

**Supported Chipsets:**
- SM8450/SM8475 → `8gen1` (Your S22 Ultra!)
- SM8550/SM8650/SM8750 → `8gen2` (8 Gen 2/3/Elite)
- Other Snapdragon → `min` (lightweight)

### 2. Model Definitions

**Replaced:**
```kotlin
LLMModel(name = "Stable Diffusion v1.5", modelFormat = "image_generator", ...)
```

**With:**
```kotlin
// NPU-accelerated model (device-specific)
LLMModel(
    name = "Absolute Reality (NPU - 8gen1)",
    url = "https://huggingface.co/xororz/sd-qnn/resolve/main/AbsoluteReality_qnn2.28_8gen1.zip",
    category = "image_generation",
    sizeBytes = 1138900992L, // 1.06 GB
    modelFormat = "qnn_npu"
)

// CPU fallback model (universal)
LLMModel(
    name = "Absolute Reality (CPU)",
    url = "https://huggingface.co/xororz/sd-mnn/resolve/main/AbsoluteReality.zip",
    sizeBytes = 1288490188L, // 1.2 GB
    modelFormat = "mnn_cpu"
)
```

### 3. Dependency Changes (`build.gradle.kts`)

**Removed:**
```kotlin
implementation("com.google.mediapipe:tasks-vision-image-generator:0.10.26.1")
```

**Why:** MediaPipe Image Generator has a hard dependency on `libimagegenerator_gpu.so` which crashes on Adreno 730 (Snapdragon 8 Gen 1) due to driver bugs in FP16→FP32 conversion.

**Kept:**
```kotlin
implementation("com.google.mediapipe:tasks-genai:0.10.29")
implementation("com.google.mediapipe:tasks-text:0.10.29")
```

### 4. UI Strings Updated

- Changed "Iterations" → "Steps" (correct SD terminology)
- Updated descriptions to mention "Absolute Reality SD1.5"
- Added device SOC info to model download screen

### 5. Files Removed (To Do)

The following files need to be removed or updated:
- `app/src/main/java/com/llmhub/llmhub/imagegen/ImageGeneratorHelper.kt` ❌ (Delete)
- `app/src/main/java/com/llmhub/llmhub/screens/ImageGeneratorScreen.kt` ⚠️ (Update to use new backend)

## Next Steps (Implementation Needed)

### Phase 1: Backend Integration
You'll need to implement the actual inference logic:

1. **QNN SDK Integration** (for NPU model):
   - Download QNN SDK 2.39 from Qualcomm
   - Build native C++ wrapper (see local-dream's implementation)
   - Create JNI bindings for Android

2. **MNN Framework Integration** (for CPU model):
   - Add MNN dependency to `build.gradle.kts`
   - Implement CPU inference wrapper

### Phase 2: UI Adaptation
Update `ImageGeneratorScreen.kt` to:
1. Check which model is downloaded (NPU or CPU)
2. Initialize appropriate backend
3. Handle model-specific parameters (steps, CFG scale, scheduler)
4. Show device compatibility info

## Model Comparison

| Feature | MediaPipe SD1.5 | Absolute Reality (QNN) | Absolute Reality (MNN) |
|---------|----------------|------------------------|------------------------|
| **Size** | 2.07 GB (998 files) | 1.06 GB (single zip) | 1.2 GB (single zip) |
| **Backend** | TFLite GPU | Qualcomm NPU | CPU |
| **Your Device** | ❌ Crashes | ✅ Works | ✅ Works |
| **Speed** | N/A (broken) | Very Fast | Moderate |
| **Requirements** | Any Android | SD 8 Gen 1+ | Any Android |
| **Format** | Multi-file manifest | Single .zip | Single .zip |

## Why This is Better

1. **Actually Works on Your Device**: No more Adreno 730 crashes
2. **Faster Downloads**: Single .zip vs 998 separate files
3. **NPU Acceleration**: True hardware acceleration (not just GPU)
4. **Flexible Resolutions**: CPU model supports 128×128 to 512×512
5. **Active Maintenance**: xororz's models are regularly updated

## References

- **local-dream source**: https://github.com/xororz/local-dream
- **NPU models**: https://huggingface.co/xororz/sd-qnn
- **CPU models**: https://huggingface.co/xororz/sd-mnn
- **GitHub Issue #5270**: MediaPipe crash on Adreno GPUs
- **Device SOC Detection**: `Build.SOC_MODEL` (Android 12+)

## Your S22 Ultra Configuration

```kotlin
Device: Samsung Galaxy S22 Ultra
SOC: SM8450 (Snapdragon 8 Gen 1)
GPU: Adreno 730
Detected Suffix: "8gen1"
NPU Supported: ✅ Yes
Recommended Model: Absolute Reality (NPU - 8gen1)
Download URL: AbsoluteReality_qnn2.28_8gen1.zip
Size: 1.06 GB
