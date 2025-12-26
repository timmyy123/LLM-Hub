# Simplified Native Backend Setup

## Problem: Full Build Too Complex

Building from source requires:
- Qualcomm QNN SDK 2.39 (requires Qualcomm developer account)
- Android NDK r25c+
- Rust 1.84.0
- Ninja build system
- Linux/WSL environment

**This is complex and time-consuming.** Here's a simpler approach:

## â­ Recommended: Extract from Local-Dream APK

### Step 1: Download Local-Dream APK

Download the latest APK from one of these sources:
1. **GitHub Releases**: https://github.com/xororz/local-dream/releases
2. **Google Play** (if you have an Android device)

### Step 2: Extract Native Libraries

**On Windows (PowerShell):**
```powershell
# Download APK (manual download from browser if needed)
# Place it in Downloads folder as local-dream.apk

cd $env:USERPROFILE\Downloads

# APKs are just ZIP files - extract them
Expand-Archive -Path "local-dream.apk" -DestinationPath "local-dream-extracted" -Force

# Copy native libraries to your project
$sourceLibs = "local-dream-extracted\lib\arm64-v8a"
$targetLibs = "LLM-Hub\app\src\main\jniLibs\arm64-v8a"

if (!(Test-Path $targetLibs)) {
    New-Item -ItemType Directory -Path $targetLibs -Force
}

# Copy all .so files
Copy-Item "$sourceLibs\*.so" -Destination $targetLibs -Force

Write-Host "Native libraries copied successfully!"
Get-ChildItem $targetLibs
```

**Expected files:**
- `libstable_diffusion_core.so` - Main inference engine
- Various dependency libraries

### Step 3: Copy QNN Runtime Libraries

Local-dream also includes QNN runtime libraries in the APK assets:

```powershell
$sourceQnn = "local-dream-extracted\assets\qnnlibs"
$targetQnn = "LLM-Hub\app\src\main\assets\qnnlibs"

if (!(Test-Path $targetQnn)) {
    New-Item -ItemType Directory -Path $targetQnn -Force
}

Copy-Item "$sourceQnn\*" -Destination $targetQnn -Recurse -Force

Write-Host "QNN libraries copied successfully!"
```

**Expected files:**
- `libQnnHtp.so` - Main QNN library
- `libQnnSystem.so` - QNN system library
- `libQnnHtpV*.so` - Version-specific stub libraries
- Hexagon DSP libraries

## Alternative: Request Pre-built Binaries

If you can't access the APK, you can:

1. **Contact the local-dream developer** on their Telegram group: https://t.me/local_dream
2. **Ask for pre-built binaries** specifically for integration
3. They may provide a separate package with just the `.so` files

## Next Steps After Getting Libraries

Once you have the native libraries:

1. **Verify library structure:**
   ```powershell
   Get-ChildItem app\src\main\jniLibs\arm64-v8a\
   Get-ChildItem app\src\main\assets\qnnlibs\
   ```

2. **Create JNI wrapper** (see [NATIVE_BACKEND_GUIDE.md](NATIVE_BACKEND_GUIDE.md) - JNI Integration section)

3. **Update SDBackendService** to load and call native functions

4. **Test on device:**
   ```powershell
   # Build and install
   cd LLM-Hub
   .\gradlew installDebug
   
   # Check if libraries load
   adb logcat | Select-String "SDBackend"
   ```

## Why This Approach Works

- **No build tools needed** - Libraries already compiled
- **Same binaries** - Exactly what local-dream uses in production
- **Proven compatibility** - Known to work on target devices
- **Fast setup** - Minutes instead of hours/days

## Important Notes

### Library Dependencies

`libstable_diffusion_core.so` depends on:
- Android system libraries (automatically available)
- C++ standard library (include `c++_shared`)
- QNN libraries (must be in assets/qnnlibs/)

Make sure to configure your `build.gradle.kts` to extract assets:

```kotlin
android {
    // ... existing config ...
    
    defaultConfig {
        // ... existing config ...
        
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }
    
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }
}
```

### Load QNN Libraries at Runtime

The QNN libraries in assets need to be extracted to a writable location at runtime. Add to SDBackendService:

```kotlin
private fun extractQnnLibraries() {
    val qnnDir = File(filesDir, "qnnlibs")
    if (!qnnDir.exists()) {
        qnnDir.mkdirs()
        
        // Extract all QNN libraries from assets
        assets.list("qnnlibs")?.forEach { libName ->
            val outputFile = File(qnnDir, libName)
            assets.open("qnnlibs/$libName").use { input ->
                outputFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "Extracted: $libName")
        }
    }
    
    // Add to library search path
    System.setProperty("java.library.path", 
        "${System.getProperty("java.library.path")}:${qnnDir.absolutePath}")
}
```

## Troubleshooting

### "Library not found"
- Verify files are in `app/src/main/jniLibs/arm64-v8a/`
- Check build output: `ls app/build/intermediates/merged_native_libs/*/out/lib/arm64-v8a/`
- Clean and rebuild: `.\gradlew clean assembleDebug`

### "Symbol not found"  
- Missing dependency library
- Check with: `adb shell "readelf -d /data/app/.../lib/arm64/libstable_diffusion_core.so"`
- Ensure QNN libraries are extracted and accessible

### "QNN initialization failed"
- QNN libraries not in correct path
- Device doesn't support NPU (check chipset compatibility)
- Try fallback to MNN/CPU mode

## Minimal Integration Test

Create a simple test to verify library loading:

```kotlin
// In SDBackendService.kt or test activity
fun testLibraryLoad(): Boolean {
    return try {
        // Extract QNN libs first
        extractQnnLibraries()
        
        // Try to load main library
        System.loadLibrary("stable_diffusion_core")
        Log.i("LibTest", "✅ Native library loaded successfully")
        true
    } catch (e: UnsatisfiedLinkError) {
        Log.e("LibTest", "❌ Failed to load library: ${e.message}")
        false
    }
}
```

## Summary

**Instead of building from source:**
1. ✅ Download local-dream APK
2. ✅ Extract `.so` files to `jniLibs/arm64-v8a/`
3. ✅ Extract QNN libs to `assets/qnnlibs/`
4. ✅ Implement JNI wrapper (see NATIVE_BACKEND_GUIDE.md)
5. ✅ Test library loading

**Total time: ~30 minutes** vs. days of setup for building from source.

---

**Next**: See [NATIVE_BACKEND_GUIDE.md](NATIVE_BACKEND_GUIDE.md) JNI Integration section for implementing the Kotlin↔C++ interface.
