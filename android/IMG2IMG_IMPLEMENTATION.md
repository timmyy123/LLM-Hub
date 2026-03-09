# Image-to-Image (img2img) Implementation

## Overview

Added image-to-image functionality to the Image Generator feature, inspired by the [local-dream](https://github.com/xororz/local-dream) repository. This allows users to transform existing images using text prompts.

## Changes Made

### 1. StableDiffusionHelper.kt
**Location**: `app/src/main/java/com/llmhub/llmhub/imagegen/StableDiffusionHelper.kt`

**Changes**:
- Added `inputImage: Bitmap?` parameter to `generateImage()` function
- Added `denoiseStrength: Float = 0.7f` parameter (controls how much the input image is modified)
- Automatically encodes input image to base64 and includes it in the JSON request
- Resizes input image to match target dimensions before encoding
- Updated logging to show when img2img mode is active

**Key Code**:
```kotlin
// Add img2img support
if (inputImage != null) {
    put("denoise_strength", denoiseStrength.toDouble())
    
    // Encode input image to base64
    val resizedImage = Bitmap.createScaledBitmap(inputImage, width, height, true)
    val outputStream = java.io.ByteArrayOutputStream()
    resizedImage.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
    val imageBytes = outputStream.toByteArray()
    val imageBase64 = Base64.encodeToString(imageBytes, Base64.NO_WRAP)
    put("image", imageBase64)
}
```

### 2. ImageGeneratorHelper.kt
**Location**: `app/src/main/java/com/llmhub/llmhub/imagegen/ImageGeneratorHelper.kt`

**Changes**:
- Added `inputImage: Bitmap?` parameter to `generateImage()` function
- Added `denoiseStrength: Float = 0.7f` parameter
- Passes these parameters through to `StableDiffusionHelper`
- Maintains API compatibility with existing UI code

### 3. ImageGeneratorScreen.kt
**Location**: `app/src/main/java/com/llmhub/llmhub/screens/ImageGeneratorScreen.kt`

**New Imports**:
```kotlin
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
```

**New State Variables**:
```kotlin
// Img2img state
var useImg2img by remember { mutableStateOf(prefs.getBoolean("use_img2img", false)) }
var denoiseStrength by remember { mutableStateOf(prefs.getFloat("denoise_strength", 0.7f)) }
var inputImageUri by remember { mutableStateOf<Uri?>(null) }
var inputImageBitmap by remember { mutableStateOf<Bitmap?>(null) }
```

**Image Picker Launcher**:
```kotlin
// Image picker launcher
val imagePickerLauncher = rememberLauncherForActivityResult(
    contract = ActivityResultContracts.GetContent()
) { uri: Uri? ->
    inputImageUri = uri
    uri?.let {
        try {
            inputImageBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = android.graphics.ImageDecoder.createSource(context.contentResolver, it)
                android.graphics.ImageDecoder.decodeBitmap(source)
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(context.contentResolver, it)
            }
        } catch (e: Exception) {
            errorMessage = "Failed to load image: ${e.message}"
            inputImageUri = null
            inputImageBitmap = null
        }
    }
}
```

**New UI Components**:
1. **Toggle Switch** - Enable/disable img2img mode
2. **Image Picker Button** - Select input image from gallery
3. **Image Preview** - Shows selected input image as thumbnail
4. **Denoise Strength Slider** - Control how much the input image is modified (0.1 - 1.0)

**Updated Generation Calls**:
All `imageGeneratorHelper.generateImage()` calls now include:
```kotlin
inputImage = if (useImg2img) inputImageBitmap else null,
denoiseStrength = denoiseStrength
```

### 4. SDBackendService.kt
**Location**: `app/src/main/java/com/llmhub/llmhub/service/SDBackendService.kt`

**Changes**:
- Reads `use_img2img` preference from SharedPreferences
- Conditionally includes VAE encoder in command line arguments
- Supports both QNN (`.bin`) and MNN (`.mnn`) VAE encoder models
- Logs warnings if img2img is enabled but VAE encoder file is missing

**For QNN models**:
```kotlin
if (useImg2img) {
    val vaeEncoderFile = File(actualDir, "vae_encoder.bin")
    if (vaeEncoderFile.exists()) {
        command.add("--vae_encoder")
        command.add(vaeEncoderFile.absolutePath)
        Log.i(TAG, "img2img enabled: VAE encoder included")
    }
}
```

**For MNN models**:
```kotlin
if (useImg2img) {
    val vaeEncoderFile = File(actualDir, "vae_encoder.mnn")
    if (vaeEncoderFile.exists()) {
        command.add("--vae_encoder")
        command.add(vaeEncoderFile.absolutePath)
        Log.i(TAG, "img2img enabled: VAE encoder included")
    }
}
```

## How It Works

1. **User enables img2img** in the settings card
2. **User selects an input image** using the image picker
3. **User adjusts denoise strength** (0.1 = subtle changes, 1.0 = major transformation)
4. **User generates** with a text prompt describing desired changes
5. **Backend receives**:
   - Text prompt
   - Negative prompt
   - Input image (base64 encoded)
   - Denoise strength
   - Other parameters (steps, CFG, seed, etc.)
6. **Backend processes**:
   - VAE Encoder converts input image to latent space
   - Adds noise based on denoise strength
   - UNet denoises according to prompt
   - VAE Decoder converts back to image
7. **Result** combines input image structure with prompt-guided modifications

## Denoise Strength Explained

- **0.1 - 0.3**: Minimal changes, mostly color/lighting adjustments
- **0.4 - 0.6**: Moderate changes, preserves main structure
- **0.7 - 0.8**: Significant changes, good balance (default)
- **0.9 - 1.0**: Major transformation, mostly respects prompt over input

## Model Requirements

For img2img to work, models must include:
- `vae_encoder.bin` (for QNN/NPU models)
- `vae_encoder.mnn` (for MNN/CPU models)

If these files are missing, the backend will start without img2img support and log a warning.

## UI/UX Features

- Settings are automatically saved to SharedPreferences
- Input image is shown as a preview thumbnail
- Can change input image without toggling img2img off
- Denoise strength has helpful tooltip
- Works in both portrait and landscape layouts
- Compatible with variation generation (generates multiple outputs from same input)

## Based On

This implementation is inspired by the [local-dream](https://github.com/xororz/local-dream) repository's img2img functionality, which uses:
- VAE Encoder for converting images to latent space
- Denoise strength parameter for controlling transformation amount
- Base64 image encoding for HTTP transport
- Scheduler noise addition for img2img workflow

## Testing

To test img2img:
1. Load a Stable Diffusion model with VAE encoder
2. Enable "Image-to-Image" toggle in settings
3. Click "Select Input Image" and choose a photo
4. Adjust denoise strength (try 0.7 first)
5. Enter a prompt describing desired changes
6. Generate and compare variations

Example prompts:
- "oil painting style" (artistic transformation)
- "cyberpunk cityscape at night" (scenery transformation)
- "watercolor sketch" (medium transformation)
- "professional photography, dramatic lighting" (enhancement)
