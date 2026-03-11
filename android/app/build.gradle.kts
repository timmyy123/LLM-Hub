import java.util.Properties
import java.io.FileInputStream
import java.util.zip.ZipFile
import java.util.zip.ZipEntry

// Load local.properties at the top-level so it's available everywhere
val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localProperties.load(FileInputStream(localPropertiesFile))
}

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.llmhub.llmhub"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.llmhub.llmhub"
        minSdk = 27
        targetSdk = 36
        versionCode = 81
        versionName = "3.6.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        val hfToken: String = localProperties.getProperty("HF_TOKEN", "")
        buildConfigField("String", "HF_TOKEN", "\"$hfToken\"")
        
        // Enable 16KB page size support for Android 15+ compatibility
        // Required for Google Play Store submission starting Nov 1st, 2025
        ndk {
            // Only package arm64-v8a — excludes x86/x86_64/armeabi-v7a slices from all dependency AARs (~150 MB saved).
            abiFilters += setOf("arm64-v8a")
            // This helps with alignment but ultimate fix requires library maintainers
            // to rebuild native libraries with 16KB alignment
            debugSymbolLevel = "FULL"
        }
    }
    
    // Specify supported locales to ensure proper resource loading
    // Note: Indonesian uses both "id" (modern) and "in" (legacy) for maximum compatibility
    androidResources {
        localeFilters += listOf("en", "es", "pt", "de", "fr", "ru", "it", "tr", "pl", "ar", "ja", "id", "in", "ko", "fa", "he", "iw", "uk", "zh")
    }

    // Configure asset packs for install-time delivery
    // nexa_npu_pack delivers assets/npu/htp-files-v81/ and htp-files-v85/ (~230 MB)
    // keeping the base module well under Play Store's 200 MB limit
    assetPacks += mutableSetOf(":qnn_pack", ":sd_pack", ":nexa_npu_pack")

    buildTypes {
        release {
            // Disable R8 minification to prevent stripping ONNX/Nexa JNI classes
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // Strip native debug symbols for release builds to reduce native library sizes.
            ndk {
                debugSymbolLevel = "NONE"
            }
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlin {
        compilerOptions {
            jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11
        }
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.0"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            // Exclude common duplicate license and metadata files from dependencies (e.g., Flexmark)
            excludes += "META-INF/LICENSE-LGPL-3.txt"
            excludes += "META-INF/LICENSE-LGPL-3.0.txt"
            excludes += "META-INF/LICENSE*"
            excludes += "META-INF/DEPENDENCIES"
            excludes += "META-INF/DEPENDENCIES.txt"
            excludes += "META-INF/NOTICE"
            excludes += "META-INF/NOTICE.txt"
            excludes += "META-INF/LICENSE.txt"
            // Avoid duplicate Kotlin module descriptors
            excludes += "META-INF/*.kotlin_module"
            // Exclude duplicate protobuf .proto files
            excludes += "google/protobuf/*.proto"
        }
        // Configure JNI libraries packaging
        // useLegacyPackaging = true is REQUIRED because SDBackendService uses ProcessBuilder
        // to execute libstable_diffusion_core.so as a standalone process, which needs the
        // library extracted to disk (not compressed in APK)
        jniLibs {
            useLegacyPackaging = true
            // Note: strip native debug symbols via release.ndk.debugSymbolLevel = "NONE"
            // Some AGP APIs around keepDebugSymbols vary by AGP version; avoid using
            // keepDebugSymbols to preserve compatibility across AGP releases.

            // Pick only the architecture we need to reduce size and alignment issues
            // Prevent duplicate .so files from different MediaPipe tasks modules
            pickFirsts += setOf("**/libmediapipe_tasks_text_jni.so")
            // Safety fallback — pickFirsts ensures Microsoft's ORT always wins over Nexa's bundled copy.
            pickFirsts += setOf("**/libonnxruntime.so")
            // Exclude DeepSeek OCR library to avoid 16KB page alignment issues
            excludes += setOf("**/libdeepseek-ocr.so")
            // Exclude Nexa SDK's bundled stable-diffusion — app uses its own libstable_diffusion_core.so subprocess
            excludes += setOf("**/libstable-diffusion.so")
        }
    }
    
    // Prevent Play Store from removing unused language resources when generating app bundles.
    // This ensures all supported languages packaged in `resourceConfigurations` remain
    // available at runtime for per-app locale switching (AppCompat per-app locales).
    bundle {
        language {
            // Keep all languages in the base APK rather than splitting them into configuration-specific
            // APKs. When enabled, Play may remove some language resources from the installed split
            // APK which prevents runtime calls to update the app locale from finding translations.
            enableSplit = false
        }
    }

    // Removed externalNativeBuild - now using MediaPipe instead of native llama.cpp
}

// The Image Generator requires full protobuf-java, not the lite version
configurations.all {
    resolutionStrategy {
        force("com.google.protobuf:protobuf-java:3.25.1")
        // Force Microsoft's ONNX Runtime version to win over any version Nexa SDK pulls in
        force("com.microsoft.onnxruntime:onnxruntime-android:1.24.1")
    }
    // Exclude protobuf-javalite from all dependencies to prevent duplicate classes
    exclude(group = "com.google.protobuf", module = "protobuf-javalite")
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    
    // Material Icons Extended
    implementation(libs.androidx.material.icons.extended)
    
    // Navigation
    implementation(libs.androidx.navigation.compose)
    
    // ViewModel
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    
    // Room Database
    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)
    
    // Networking
    implementation(libs.retrofit)
    implementation(libs.retrofit.gson)
    implementation(libs.okhttp)
    implementation(libs.okhttp.logging)
    
    // Coroutines
    implementation(libs.kotlinx.coroutines.android)
    
    // LocalBroadcastManager for service communication
    implementation("androidx.localbroadcastmanager:localbroadcastmanager:1.1.0")
    
    // DataStore for settings
    implementation("androidx.datastore:datastore-preferences:1.0.0")

    // Security (EncryptedSharedPreferences)
    // Use 1.1.0-alpha06 for MasterKey API support (replaces deprecated MasterKeys)
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    // AppCompat for per-app locale APIs
    implementation("androidx.appcompat:appcompat:1.7.0")
    
    // Permissions
    implementation(libs.accompanist.permissions)
    
    // Image Loading
    implementation(libs.coil.compose)
    
    // JSON
    implementation(libs.gson)
    
    // Document and text file parsing
    implementation("org.apache.commons:commons-csv:1.10.0")
    
    // PDF text extraction - using iText7 Community for Android compatibility
    implementation("com.itextpdf:itext7-core:7.2.5")
    
    // Ktor for networking
    implementation("io.ktor:ktor-client-android:2.3.6")
    implementation("io.ktor:ktor-client-content-negotiation:2.3.6")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.6")
    
    // OkHttp for SD backend communication
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // MediaPipe Tasks (updated to latest as of Oct 2025)
    // NOTE: Version 0.10.29 has slower initial load for multimodal models due to eager
    // vision/audio component initialization. Disable vision/audio when not needed for faster loading.
    // tasks-genai latest: 0.10.29; tasks-text latest: 0.10.29
    implementation("com.google.mediapipe:tasks-genai:0.10.32")
    implementation("com.google.mediapipe:tasks-vision:0.10.32")
    implementation("com.google.mediapipe:tasks-text:0.10.32")
    
    // Protobuf - required for MediaPipe
    implementation("com.google.protobuf:protobuf-java:3.25.1")
    // Provide a no-op SLF4J binder so R8 finds org.slf4j.impl.StaticLoggerBinder
    implementation("org.slf4j:slf4j-nop:2.0.9")
    
    // AI Edge RAG SDK for proper Gecko embedding support
    implementation("com.google.ai.edge.localagents:localagents-rag:0.3.0")
    // Note: MediaPipe tasks-genai 0.10.22 is required for RAG SDK, but using 0.10.27 should be compatible
    
    // Compose Markdown - temporarily removed due to version conflicts
    implementation("com.github.jeziellago:compose-markdown:0.3.0")

    // Markdown parser for extracting code blocks
    implementation("com.vladsch.flexmark:flexmark-all:0.64.8")

    // ONNX Runtime for Android - supports ONNX model inference
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.1")

    // Nexa SDK for GGUF model support
    // Nexa bundles libonnxruntime.so directly in its AAR (6.5MB) which conflicts with
    // Microsoft's ORT JNI bridge. The task below strips it from the cached AAR so only
    // Microsoft's version ends up in the APK.
    implementation("ai.nexa:core:0.0.24") {
        exclude(group = "com.microsoft.onnxruntime")
    }

    // Play Core for asset pack access at runtime
    implementation("com.google.android.play:asset-delivery:2.2.2")
    implementation("com.google.android.play:asset-delivery-ktx:2.2.2")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}

// ── Extract npu HTP assets from Nexa AAR into nexa_npu_pack ──────────────────
// Nexa 0.0.24 bundles assets/npu/htp-files-v81/ (~67 MB) and htp-files-v85/
// (~80 MB) inside its AAR. We extract them here (where Nexa is already a
// resolved dependency) into the nexa_npu_pack asset pack source directory so
// Play Asset Delivery can serve them at install time. This keeps the base
// module well under Play Store's 200 MB compressed APK split limit.
//
// All assets/npu/ content is stripped from the base module via the hook below.
// For APK sideloads, NPU falls back to NNAPI GPU / CPU automatically.

val nexaAarConfig by configurations.creating {
    isCanBeConsumed = false
    isCanBeResolved = true
}
dependencies { nexaAarConfig("ai.nexa:core:0.0.24@aar") }

val npuPackAssetsDir = rootProject.file("nexa_npu_pack/src/main/assets/npu")

val extractNexaNpuAssets by tasks.registering {
    description = "Extracts npu/htp-files-v81/ and htp-files-v85/ from Nexa AAR into nexa_npu_pack"
    group = "build setup"
    inputs.files(nexaAarConfig)
    outputs.dir(npuPackAssetsDir)
    outputs.upToDateWhen {
        npuPackAssetsDir.resolve("htp-files-v81").exists() &&
        npuPackAssetsDir.resolve("htp-files-v85").exists()
    }
    doLast {
        val aar = nexaAarConfig.singleFile
        npuPackAssetsDir.deleteRecursively()
        npuPackAssetsDir.mkdirs()
        var extracted = 0
        ZipFile(aar).use { zip ->
            zip.entries().toList().asSequence()
                .filter { !it.isDirectory && it.name.startsWith("assets/npu/htp-files-v") }
                .forEach { entry ->
                    // entry.name: "assets/npu/htp-files-v81/libFoo.so"
                    // place as:   nexa_npu_pack/src/main/assets/npu/htp-files-v81/libFoo.so
                    val rel = entry.name.removePrefix("assets/npu/")
                    val target = npuPackAssetsDir.resolve(rel)
                    target.parentFile.mkdirs()
                    zip.getInputStream(entry).use { src ->
                        target.outputStream().use { dst -> src.copyTo(dst) }
                    }
                    extracted++
                }
        }
        logger.lifecycle("extractNexaNpu: extracted $extracted files → ${npuPackAssetsDir.absolutePath}")
    }
}

// Detect at configuration time whether this is an AAB bundle build or an APK build.
// gradle.startParameter.taskNames contains the tasks requested (e.g. "bundleRelease" vs "assembleRelease").
// For APK builds: npu/ must stay in base (no asset pack delivery mechanism).
// For AAB builds: npu/ is delivered via nexa_npu_pack, so strip it from base.
val isBundleBuild = gradle.startParameter.taskNames.any { it.contains("bundle", ignoreCase = true) }

// Run extraction + wire dependency only during AAB bundle builds
if (isBundleBuild) {
    tasks.configureEach {
        val n = name
        if ((n.startsWith("merge") && n.contains("Assets", ignoreCase = true)) ||
            (n.startsWith("assetPack") && n.contains("PreBundleTask", ignoreCase = true))
        ) {
            dependsOn(extractNexaNpuAssets)
        }
    }
}

// ── Strip ALL assets/npu/ from base module (AAB builds only) ─────────────────
// npu HTP runtime libs are delivered via :nexa_npu_pack (install-time Play Asset
// Delivery) for AAB. For APK sideloads, npu stays in base so NPU works normally.
if (isBundleBuild) {
    tasks.configureEach {
        if (name.startsWith("merge") && name.contains("Assets", ignoreCase = true)) {
            doLast {
                // Delete everything under npu/ from the base module's merged assets
                outputs.files.asFileTree.matching { include("npu/**") }
                    .filter { it.isFile }
                    .forEach { f ->
                        logger.lifecycle("stripNpu: removed ${f.parentFile.name}/${f.name} from base module")
                        f.delete()
                    }
                // Remove empty npu dirs (deepest first)
                outputs.files.asFileTree.matching { include("npu/**") }
                    .filter { it.isDirectory }
                    .sortedByDescending { it.absolutePath.length }
                    .forEach { it.delete() }
                // Remove root npu/ dir if empty
                outputs.files.asFileTree.matching { include("npu") }
                    .filter { it.isDirectory && (it.listFiles()?.isEmpty() == true) }
                    .forEach { it.delete() }
            }
        }
    }
}

// ── Conditionally exclude qnnlibs from base module during AAB builds ──────────
// When building an AAB, qnnlibs are delivered via asset pack (qnn_pack).
// When building an APK, qnnlibs must be in app/src/main/assets for functionality.
// These tasks automatically hide qnnlibs during bundle builds and restore after.

val qnnlibsDir = project.file("src/main/assets/qnnlibs")
val qnnlibsHiddenDir = project.file("src/main/assets/.qnnlibs_hidden")
val cvtbaseDir = project.file("src/main/assets/cvtbase")
val cvtbaseHiddenDir = project.file("src/main/assets/.cvtbase_hidden")

tasks.register("hideAssetsForBundle") {
    doLast {
        if (qnnlibsDir.exists() && !qnnlibsHiddenDir.exists()) {
            logger.lifecycle("Hiding qnnlibs from base module for AAB build (will use asset pack)")
            qnnlibsDir.renameTo(qnnlibsHiddenDir)
        }
        if (cvtbaseDir.exists() && !cvtbaseHiddenDir.exists()) {
            logger.lifecycle("Hiding cvtbase from base module for AAB build (will use sd_pack)")
            cvtbaseDir.renameTo(cvtbaseHiddenDir)
        }
    }
}

tasks.register("restoreAssetsAfterBundle") {
    doLast {
        if (qnnlibsHiddenDir.exists()) {
            logger.lifecycle("Restoring qnnlibs to app/src/main/assets")
            qnnlibsHiddenDir.renameTo(qnnlibsDir)
        }
        if (cvtbaseHiddenDir.exists()) {
            logger.lifecycle("Restoring cvtbase to app/src/main/assets")
            cvtbaseHiddenDir.renameTo(cvtbaseDir)
        }
    }
}

tasks.register("ensureAssetsForApk") {
    doLast {
        if (qnnlibsHiddenDir.exists() && !qnnlibsDir.exists()) {
            logger.lifecycle("Restoring qnnlibs for APK build")
            qnnlibsHiddenDir.renameTo(qnnlibsDir)
        }
        if (cvtbaseHiddenDir.exists() && !cvtbaseDir.exists()) {
            logger.lifecycle("Restoring cvtbase for APK build")
            cvtbaseHiddenDir.renameTo(cvtbaseDir)
        }
    }
}

// Hook into bundle tasks to hide base-module assets that are delivered via asset packs in AAB
tasks.configureEach {
    if (name.startsWith("bundle") && name.contains("Release", ignoreCase = true)) {
        dependsOn("hideAssetsForBundle")
        finalizedBy("restoreAssetsAfterBundle")
    }
    // Hook into assemble tasks to ensure assets are present for APK builds
    if (name.startsWith("assemble") && name.contains("Release", ignoreCase = true)) {
        dependsOn("ensureAssetsForApk")
    }
}
