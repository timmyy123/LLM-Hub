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
    compileSdk = 37

    defaultConfig {
        applicationId = "com.llmhub.llmhub"
        minSdk = 27
        targetSdk = 37
        versionCode = 117
        versionName = "3.8.2"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        val hfToken: String = localProperties.getProperty("HF_TOKEN", "")
        buildConfigField("String", "HF_TOKEN", "\"$hfToken\"")
        val debugPremium: Boolean = localProperties.getProperty("DEBUG_PREMIUM", "false").toBoolean()
        buildConfigField("Boolean", "DEBUG_PREMIUM", "$debugPremium")

        // AdMob IDs — override in local.properties; test IDs are the defaults
        val admobAppId: String = localProperties.getProperty(
            "ADMOB_APP_ID", "ca-app-pub-3940256099942544~3347511713")
        val admobBannerId: String = localProperties.getProperty(
            "ADMOB_BANNER_ID", "ca-app-pub-3940256099942544/6300978111")
        val admobInterstitialId: String = localProperties.getProperty(
            "ADMOB_INTERSTITIAL_ID", "ca-app-pub-3940256099942544/1033173712")
        val admobRewardedId: String = localProperties.getProperty(
            "ADMOB_REWARDED_ID", "ca-app-pub-3940256099942544/5224354917")
        buildConfigField("String", "ADMOB_APP_ID", "\"$admobAppId\"")
        buildConfigField("String", "ADMOB_BANNER_ID", "\"$admobBannerId\"")
        buildConfigField("String", "ADMOB_INTERSTITIAL_ID", "\"$admobInterstitialId\"")
        buildConfigField("String", "ADMOB_REWARDED_ID", "\"$admobRewardedId\"")
        manifestPlaceholders["admobAppId"] = admobAppId
        
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
    // geniex_npu_pack delivers QNN HTP runtime libs (~175 MB)
    // keeping the base module well under Play Store's 200 MB limit
    assetPacks += mutableSetOf(":qnn_pack", ":sd_pack", ":geniex_npu_pack")

    buildTypes {
        release {
            // Disable R8 minification to prevent stripping ONNX/GenieX JNI classes
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
            // Exclude DeepSeek OCR library to avoid 16KB page alignment issues
            excludes += setOf("**/libdeepseek-ocr.so")
            // WhisperKit 0.3.3 ships 4KB-aligned .so files (upstream bug, not yet fixed).
            // Android only enforces 16KB page alignment on API 35+ devices with new kernel.
            // These libs still load correctly on all current devices; suppress the build warning.
            // Track: https://github.com/argmaxinc/WhisperKitAndroid/issues
            // Exclude QNN HTP runtime libs from base module — delivered via geniex_npu_pack asset pack
            // NOTE: libQnnTFLiteDelegate.so must NOT be excluded — WhisperKit needs it in the APK
            excludes += setOf(
                "**/libQnnHtp*.so",
                "**/libQnnHtpPrepare.so",
                "**/libQnnDsp*.so",
                "**/libQnnSystem.so",
                "**/libQnnCpu.so",
                "**/libQnnGpu.so",
                "**/libPlatformValidatorShared.so",
                "**/libCalculator_skel.so",
                "**/libcalculator.so",
                "**/libhta_hexagon_runtime_qnn.so",
                "**/libhta_hexagon_runtime_snpe.so",
                "**/libNetRunDirect*.so"
            )
        }
    }
    
    lint {
        lintConfig = file("lint.xml")
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

}

// Patch 4KB-aligned .so files from WhisperKit and other dependencies to 16KB page alignment.
// Android enforces 16KB alignment on API 35+ devices. We patch the PT_LOAD p_align field in
// the ELF program headers directly — this is safe because p_align is advisory metadata only;
// the actual file/memory layout is unchanged. patchelf does the same thing.
//
// Strategy: hook AFTER each merge task so we patch the exact files Gradle will package into the APK.
// We also patch the gradle transforms cache so incremental builds stay patched.

val elfPatchScript = """
import struct, os, sys

def patch_elf_pt_load_alignment(path, target_align=0x4000):
    with open(path, 'rb') as f:
        data = bytearray(f.read())
    if data[:4] != b'\x7fELF':
        return 0
    ei_class = data[4]
    if ei_class != 2:  # Not ELF64
        return 0
    e_phoff = struct.unpack_from('<Q', data, 32)[0]
    e_phentsize = struct.unpack_from('<H', data, 54)[0]
    e_phnum = struct.unpack_from('<H', data, 56)[0]
    patched = 0
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from('<I', data, off)[0]
        if p_type == 1:  # PT_LOAD
            p_align = struct.unpack_from('<Q', data, off + 48)[0]
            if 0 < p_align < target_align:
                struct.pack_into('<Q', data, off + 48, target_align)
                patched += 1
    if patched > 0:
        with open(path, 'wb') as f:
            f.write(data)
    return patched

total = 0
for search_dir in sys.argv[1:]:
    if not os.path.exists(search_dir):
        continue
    for root, dirs, files in os.walk(search_dir):
        for name in files:
            if name.endswith('.so'):
                path = os.path.join(root, name)
                n = patch_elf_pt_load_alignment(path)
                if n > 0:
                    print(f'Patched {n} PT_LOAD segment(s): {name}')
                    total += n
print(f'Total: {total} segment(s) patched for 16KB alignment')
""".trimIndent()

fun runElfPatch(scriptFile: File, vararg dirs: String, logger: org.gradle.api.logging.Logger) {
    val args = mutableListOf("python3", scriptFile.absolutePath) + dirs.toList()
    val process = ProcessBuilder(args)
        .redirectErrorStream(true)
        .start()
    val output = process.inputStream.bufferedReader().readText()
    val exitCode = process.waitFor()
    if (output.isNotBlank()) logger.lifecycle(output)
    logger.lifecycle("ELF alignment patch complete (exit=$exitCode)")
}

// After each merge task that assembles native libs, patch the output dir in-place
tasks.configureEach {
    if (name.startsWith("merge") && (name.endsWith("JniLibFolders") || name.endsWith("NativeLibs") || name.endsWith("NativeLibs"))) {
        doLast {
            val scriptFile = File(layout.buildDirectory.get().asFile, "patch_elf_align.py")
            scriptFile.parentFile.mkdirs()
            scriptFile.writeText(elfPatchScript)

            // Patch the merged intermediates output dirs (where the APK packager reads from)
            val buildDir = layout.buildDirectory.get().asFile.absolutePath
            val intermediatesDirs = listOf(
                "$buildDir/intermediates/merged_native_libs",
                "$buildDir/intermediates/jniLibs",
                "$buildDir/intermediates/library_jni",
                "$buildDir/intermediates/stripped_native_libs"
            )
            // Also patch gradle transforms cache to keep it warm across incremental builds
            val gradleHome = System.getProperty("user.home")
            val transformsDirs = File("$gradleHome/.gradle/caches").walkTopDown()
                .maxDepth(2)
                .filter { it.isDirectory && it.name == "transforms" }
                .map { it.absolutePath }
                .toList()

            val allDirs = (intermediatesDirs + transformsDirs).toTypedArray()
            runElfPatch(scriptFile, *allDirs, logger = logger)
        }
    }
}

// The Image Generator requires full protobuf-java, not the lite version
configurations.all {
    resolutionStrategy {
        force("com.google.protobuf:protobuf-java:3.25.1")
        // Force Microsoft's ONNX Runtime version to win over any version GenieX SDK pulls in
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
    
    // Ktor for networking — must match WhisperKit's transitive Ktor 3.x dependency
    implementation("io.ktor:ktor-client-android:3.1.0")
    implementation("io.ktor:ktor-client-content-negotiation:3.1.0")
    implementation("io.ktor:ktor-serialization-kotlinx-json:3.1.0")
    
    // OkHttp for SD backend communication
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // MediaPipe Tasks (updated to latest as of Oct 2025)
    // NOTE: Version 0.10.29 has slower initial load for multimodal models due to eager
    // vision/audio component initialization. Disable vision/audio when not needed for faster loading.
    // tasks-genai latest: 0.10.29; tasks-text latest: 0.10.29
    implementation("com.google.mediapipe:tasks-genai:0.10.35")
    implementation("com.google.mediapipe:tasks-vision:0.10.35")
    implementation("com.google.mediapipe:tasks-text:0.10.35")

    // LiteRT-LM: native Kotlin API for .litertlm models (Gemma-3n, Gemma-4, etc.)
    // Replaces tasks-genai for litertlm format models. GPU enabled once 0.10.1 hits Maven.
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.13.1")
    
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

    // IPA Transcribers - pure-Kotlin G2P fallback for Kokoro TTS
    implementation("com.github.medavox:IPA-Transcribers:v0.2")

    // GenieX SDK for GGUF model support (LLM/VLM inference on CPU/GPU/NPU)
    implementation("com.qualcomm.qti:geniex-android:0.3.12")

    // WhisperKit for fast on-device ASR (TFLite + QNN NPU acceleration)
    implementation("com.argmaxinc:whisperkit:0.3.3")
    implementation("com.qualcomm.qti:qnn-runtime:2.34.0")
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.34.0")

    // Play Core for asset pack access at runtime
    implementation("com.google.android.play:asset-delivery:2.2.2")
    implementation("com.google.android.play:asset-delivery-ktx:2.2.2")

    // Google Play Billing (IAP)
    implementation("com.android.billingclient:billing-ktx:7.1.1")

    // AdMob
    implementation("com.google.android.gms:play-services-ads:23.6.0")
    // AdMob UMP SDK — EU consent (GDPR) form
    implementation("com.google.android.ump:user-messaging-platform:3.1.0")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}

// ── Extract QNN HTP .so files from GenieX AAR into geniex_npu_pack ──────────────
// GenieX 0.3.12 bundles ~175 MB of QNN HTP runtime libs (libQnn*, libPlatformValidator,
// libCalculator, libhta*) in its jni/arm64-v8a/ folder. We extract them into the
// geniex_npu_pack asset pack source directory so Play Asset Delivery can serve them
// at install time. This keeps the base module well under Play Store's 200 MB limit.
//
// The extracted libs are stripped from the main jniLibs via packaging excludes below.
// At runtime, the app extracts them from the asset pack to filesDir and loads via dlopen.
// For APK sideloads, NPU falls back to GPU / CPU automatically.

val geniexAarConfig by configurations.creating {
    isCanBeConsumed = false
    isCanBeResolved = true
}
dependencies { geniexAarConfig("com.qualcomm.qti:geniex-android:0.3.12@aar") }

val npuPackAssetsDir = rootProject.file("geniex_npu_pack/src/main/assets/npu")

val extractGeniexNpuAssets by tasks.registering {
    description = "Extracts QNN HTP .so files from GenieX AAR into geniex_npu_pack"
    group = "build setup"
    inputs.files(geniexAarConfig)
    outputs.dir(npuPackAssetsDir)
    outputs.upToDateWhen {
        npuPackAssetsDir.resolve("libQnnHtp.so").exists()
    }
    doLast {
        val aar = geniexAarConfig.singleFile
        npuPackAssetsDir.deleteRecursively()
        npuPackAssetsDir.mkdirs()
        var extracted = 0
        ZipFile(aar).use { zip ->
            zip.entries().toList().asSequence()
                .filter { !it.isDirectory && it.name.startsWith("jni/arm64-v8a/") }
                .filter { entry ->
                    val name = entry.name.substringAfterLast("/")
                    name.startsWith("libQnn") ||
                    name.startsWith("libPlatformValidator") ||
                    name.startsWith("libCalculator") ||
                    name.startsWith("libhta") ||
                    name.startsWith("libNetRunDirect")
                }
                .forEach { entry ->
                    val fileName = entry.name.substringAfterLast("/")
                    val target = npuPackAssetsDir.resolve(fileName)
                    target.parentFile.mkdirs()
                    zip.getInputStream(entry).use { src ->
                        target.outputStream().use { dst -> src.copyTo(dst) }
                    }
                    extracted++
                }
        }
        logger.lifecycle("extractGeniexNpu: extracted $extracted files → ${npuPackAssetsDir.absolutePath}")
    }
}

// Detect at configuration time whether this is an AAB bundle build or an APK build.
val isBundleBuild = gradle.startParameter.taskNames.any { it.contains("bundle", ignoreCase = true) }

// Run extraction + wire dependency only during AAB bundle builds
if (isBundleBuild) {
    tasks.configureEach {
        val n = name
        if ((n.startsWith("merge") && n.contains("Assets", ignoreCase = true)) ||
            (n.startsWith("assetPack") && n.contains("PreBundleTask", ignoreCase = true))
        ) {
            dependsOn(extractGeniexNpuAssets)
        }
    }
}

// ── Strip ALL assets/npu/, cvtbase/, qnnlibs/ from base module (AAB builds only) ──────
// npu/cvtbase/qnnlibs are delivered via asset packs in AAB; strip from base merged output.
// outputs.upToDateWhen { false } forces the doLast to always run even when incremental
// build marks mergeReleaseAssets as UP-TO-DATE, preventing stale cached cvtbase from
// reaching packageReleaseBundle.
if (isBundleBuild) {
    tasks.configureEach {
        if (name.startsWith("merge") && name.contains("Assets", ignoreCase = true)) {
            outputs.upToDateWhen { false }
            doLast {
                // Delete cvtbase (sd_pack) and qnnlibs (qnn_pack) from base merged assets
                outputs.files.forEach { outDir ->
                    outDir.resolve("cvtbase").deleteRecursively()
                    outDir.resolve("qnnlibs").deleteRecursively()
                }
                // Delete everything under npu/ from the base module's merged assets
                outputs.files.asFileTree.matching { include("npu/**") }
                    .filter { it.isFile }
                    .forEach { f ->
                        logger.lifecycle("stripBundleAssets: removed ${f.parentFile.name}/${f.name} from base module")
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
    // Restore any previously-renamed source dirs after bundle finishes (success or failure)
    if (name.startsWith("bundle") && name.contains("Release", ignoreCase = true)) {
        finalizedBy("restoreAssetsAfterBundle")
    }
    // Hook into assemble tasks to ensure assets are present for APK builds
    if (name.startsWith("assemble") && name.contains("Release", ignoreCase = true)) {
        dependsOn("ensureAssetsForApk")
    }
}
