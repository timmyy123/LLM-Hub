# Building Native Backend for Stable Diffusion

This guide explains how to integrate the native C++ inference engine from local-dream into LLM-Hub.

## Overview

The current implementation provides a complete Kotlin integration layer (HTTP client, service management, model download) but requires a native C++ backend to perform actual inference. This backend:

- Loads Stable Diffusion model files (QNN `.bin` or MNN `.mnn`)
- Runs inference on Qualcomm NPU or CPU
- Serves an HTTP API on localhost:8081
- Returns generated images as Base64-encoded PNGs

## Prerequisites

### Required Tools
1. **Android NDK** (r25c or later)
   ```bash
   # Install via Android Studio SDK Manager
   # Or download from: https://developer.android.com/ndk/downloads
   ```

2. **CMake** (3.18+)
   ```bash
   # Bundled with Android Studio
   # Or install via SDK Manager
   ```

3. **Qualcomm QNN SDK** (2.39)
   ```bash
   # Download from: https://developer.qualcomm.com/software/qualcomm-ai-stack
   # Requires Qualcomm developer account
   ```

4. **Git** (for cloning repos with submodules)

### Optional Tools
- **ccache**: Speed up rebuilds
- **ninja**: Faster build system

## Method 1: Build from local-dream Source

### Step 1: Clone Repository
```bash
cd /path/to/workspace
git clone --recursive https://github.com/xororz/local-dream.git
cd local-dream
```

**Important**: Use `--recursive` to fetch all submodules (MNN, OpenCV, etc.)

### Step 2: Setup Environment Variables
```bash
# Set paths (adjust for your system)
export ANDROID_NDK=/path/to/android-ndk-r25c
export QNN_SDK_ROOT=/path/to/qnn-sdk-2.39.0

# Add NDK to PATH
export PATH=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH
```

### Step 3: Configure Build
```bash
mkdir build && cd build

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
  -DBUILD_NPU=ON \
  -DBUILD_CPU=ON \
  -DCMAKE_BUILD_TYPE=Release
```

**Build Options:**
- `BUILD_NPU=ON`: Include QNN backend for Qualcomm NPU
- `BUILD_CPU=ON`: Include MNN backend for CPU fallback
- `ANDROID_PLATFORM`: Minimum Android API level (28 = Android 9)

### Step 4: Build Libraries
```bash
cmake --build . -j$(nproc)
```

This produces:
- `libstable_diffusion_core.so`: Main inference library
- `libqnn_backend.so`: QNN runtime (if BUILD_NPU=ON)
- `libmnn_backend.so`: MNN runtime (if BUILD_CPU=ON)

### Step 5: Copy to Android Project
```bash
# Create jniLibs directory structure
mkdir -p /path/to/LLM-Hub/app/src/main/jniLibs/arm64-v8a

# Copy built libraries
cp build/libstable_diffusion_core.so \
   /path/to/LLM-Hub/app/src/main/jniLibs/arm64-v8a/

cp build/libqnn_backend.so \
   /path/to/LLM-Hub/app/src/main/jniLibs/arm64-v8a/

cp build/libmnn_backend.so \
   /path/to/LLM-Hub/app/src/main/jniLibs/arm64-v8a/
```

## Method 2: Use Pre-built Binaries

If building from source is not feasible, you can use pre-built binaries:

### Step 1: Download Releases
```bash
# Check for releases
wget https://github.com/xororz/local-dream/releases/download/v1.0/android-libs.zip
unzip android-libs.zip
```

### Step 2: Verify Architecture
```bash
file libstable_diffusion_core.so
# Should output: ELF 64-bit LSB shared object, ARM aarch64
```

### Step 3: Copy to Project
```bash
mkdir -p app/src/main/jniLibs/arm64-v8a
cp android-libs/*.so app/src/main/jniLibs/arm64-v8a/
```

## JNI Integration

### Step 1: Create JNI Wrapper Class

Create `app/src/main/java/com/llmhub/llmhub/jni/SDBackendJNI.kt`:

```kotlin
package com.llmhub.llmhub.jni

/**
 * JNI interface to native Stable Diffusion backend
 */
object SDBackendJNI {
    init {
        System.loadLibrary("stable_diffusion_core")
    }
    
    /**
     * Start HTTP server on specified port
     * @param modelPath Path to model directory (e.g., /data/data/com.llmhub/files/sd_models)
     * @param modelType "qnn" or "mnn"
     * @param port Port number (default 8081)
     * @return true if server started successfully
     */
    external fun startServer(modelPath: String, modelType: String, port: Int): Boolean
    
    /**
     * Stop HTTP server
     */
    external fun stopServer()
    
    /**
     * Check if server is running
     */
    external fun isRunning(): Boolean
}
```

### Step 2: Implement Native Methods

Create `app/src/main/cpp/sd_backend_jni.cpp`:

```cpp
#include <jni.h>
#include <string>
#include <android/log.h>
#include "stable_diffusion/server.h"

#define LOG_TAG "SDBackendJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static sd::Server* g_server = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_llmhub_llmhub_jni_SDBackendJNI_startServer(
    JNIEnv* env,
    jobject /* this */,
    jstring modelPath,
    jstring modelType,
    jint port
) {
    const char* path_cstr = env->GetStringUTFChars(modelPath, nullptr);
    const char* type_cstr = env->GetStringUTFChars(modelType, nullptr);
    
    std::string path(path_cstr);
    std::string type(type_cstr);
    
    env->ReleaseStringUTFChars(modelPath, path_cstr);
    env->ReleaseStringUTFChars(modelType, type_cstr);
    
    LOGI("Starting SD server: path=%s, type=%s, port=%d", path.c_str(), type.c_str(), port);
    
    try {
        sd::ServerConfig config;
        config.model_path = path;
        config.model_type = (type == "qnn") ? sd::ModelType::QNN : sd::ModelType::MNN;
        config.port = port;
        config.host = "127.0.0.1";
        
        g_server = new sd::Server(config);
        bool started = g_server->start();
        
        if (started) {
            LOGI("SD server started successfully");
        } else {
            LOGE("Failed to start SD server");
            delete g_server;
            g_server = nullptr;
        }
        
        return started;
    } catch (const std::exception& e) {
        LOGE("Exception starting server: %s", e.what());
        return false;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_jni_SDBackendJNI_stopServer(
    JNIEnv* env,
    jobject /* this */
) {
    if (g_server) {
        LOGI("Stopping SD server");
        g_server->stop();
        delete g_server;
        g_server = nullptr;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_llmhub_llmhub_jni_SDBackendJNI_isRunning(
    JNIEnv* env,
    jobject /* this */
) {
    return (g_server != nullptr && g_server->is_running());
}
```

### Step 3: Update SDBackendService

Modify `SDBackendService.kt` to use JNI:

```kotlin
import com.llmhub.llmhub.jni.SDBackendJNI

class SDBackendService : Service() {
    // ... existing code ...
    
    private fun startBackend() {
        try {
            val modelDir = File(filesDir, "sd_models")
            val modelType = detectModelType(modelDir)
            
            Log.i(TAG, "Starting backend: modelDir=${modelDir.absolutePath}, type=$modelType")
            
            val started = SDBackendJNI.startServer(
                modelPath = modelDir.absolutePath,
                modelType = modelType,
                port = 8081
            )
            
            if (started) {
                Log.i(TAG, "Backend started successfully")
            } else {
                Log.e(TAG, "Failed to start backend")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error starting backend: ${e.message}", e)
        }
    }
    
    private fun stopBackend() {
        try {
            SDBackendJNI.stopServer()
            Log.i(TAG, "Backend stopped")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping backend: ${e.message}", e)
        }
    }
}
```

### Step 4: Configure CMake

Create `app/src/main/cpp/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18.1)
project("sd_backend_jni")

# Find pre-built stable_diffusion_core library
add_library(stable_diffusion_core SHARED IMPORTED)
set_target_properties(stable_diffusion_core PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libstable_diffusion_core.so
)

# Build JNI wrapper
add_library(sd_backend_jni SHARED
    sd_backend_jni.cpp
)

target_link_libraries(sd_backend_jni
    stable_diffusion_core
    android
    log
)
```

### Step 5: Link in build.gradle.kts

Add to `app/build.gradle.kts`:

```kotlin
android {
    // ... existing config ...
    
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    
    defaultConfig {
        // ... existing config ...
        
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-frtti", "-fexceptions")
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_PLATFORM=android-28"
                )
            }
        }
    }
}
```

## Testing

### 1. Verify Library Loading
```kotlin
try {
    System.loadLibrary("stable_diffusion_core")
    Log.i("NativeTest", "Library loaded successfully")
} catch (e: UnsatisfiedLinkError) {
    Log.e("NativeTest", "Failed to load library: ${e.message}")
}
```

### 2. Test Server Startup
```kotlin
val started = SDBackendJNI.startServer(
    modelPath = "/data/data/com.llmhub.llmhub/files/sd_models",
    modelType = "qnn",
    port = 8081
)
Log.i("Test", "Server started: $started")
```

### 3. Test HTTP Endpoint
```bash
adb shell am start-foreground-service \
  -n com.llmhub.llmhub/.service.SDBackendService \
  -a START

# Wait a few seconds for startup

adb forward tcp:8081 tcp:8081
curl http://localhost:8081/health
# Should return: {"status": "ok"}
```

### 4. Test Image Generation
```bash
curl -X POST http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat wearing a hat",
    "steps": 20,
    "seed": 42
  }' > response.json

# Extract base64 image
cat response.json | jq -r '.image' | base64 -d > generated.png
```

## Troubleshooting

### Library Not Found
```
Error: dlopen failed: library "libstable_diffusion_core.so" not found
```

**Solution**: Verify library is in correct path:
```bash
ls app/src/main/jniLibs/arm64-v8a/
# Should show: libstable_diffusion_core.so
```

### Symbol Not Found
```
Error: dlopen failed: cannot locate symbol "some_function"
```

**Solution**: Rebuild native library with correct API level and STL

### Port Already in Use
```
Error: bind failed: Address already in use
```

**Solution**: Kill existing process or use different port:
```bash
adb shell "ps -A | grep sd_backend"
adb shell "kill <PID>"
```

### Model Load Failure
```
Error: Failed to load model from /data/data/.../sd_models
```

**Solution**: Verify model files exist and have correct permissions:
```bash
adb shell "ls -la /data/data/com.llmhub.llmhub/files/sd_models/"
# Should show: unet.bin, clip.bin, vae_decoder.bin, etc.
```

## Performance Optimization

### 1. Enable ARM NEON
```cmake
target_compile_options(sd_backend_jni PRIVATE
    -march=armv8-a+fp+simd
)
```

### 2. Use LTO (Link-Time Optimization)
```cmake
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
```

### 3. Profile with Android Studio
- Run app with Profiler attached
- Check CPU usage during generation
- Monitor memory allocations

## Alternative: Use Existing APK

If native integration proves too complex, you can:

1. Download **local-dream APK** from releases
2. Extract native libraries:
   ```bash
   unzip local-dream.apk -d extracted
   cp extracted/lib/arm64-v8a/*.so app/src/main/jniLibs/arm64-v8a/
   ```
3. Follow JNI integration steps above

## References

- **local-dream GitHub**: https://github.com/xororz/local-dream
- **Android NDK Guide**: https://developer.android.com/ndk/guides
- **JNI Tips**: https://developer.android.com/training/articles/perf-jni
- **QNN SDK Docs**: https://developer.qualcomm.com/qnn
- **MNN Documentation**: https://www.yuque.com/mnn/en

---

**Note**: This is a complex integration. Consider allocating 1-2 days for initial setup and testing.
