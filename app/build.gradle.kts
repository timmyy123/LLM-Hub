import java.util.Properties
import java.io.FileInputStream

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
    compileSdk = 35

    defaultConfig {
        applicationId = "com.llmhub.llmhub"
        minSdk = 26
        targetSdk = 35
        versionCode = 45
        versionName = "3.2.6"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        val hfToken: String = localProperties.getProperty("HF_TOKEN", "")
        buildConfigField("String", "HF_TOKEN", "\"$hfToken\"")
        
        // Specify supported locales to ensure proper resource loading
        // Note: Indonesian uses both "id" (modern) and "in" (legacy) for maximum compatibility
        resourceConfigurations += listOf("en", "es", "pt", "de", "fr", "ru", "it", "tr", "pl", "ar", "ja", "id", "in", "ko", "fa", "he", "iw", "uk")
        
        // NDK ABI filters and native library configuration for 16KB alignment
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
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
        }
        // Configure JNI libraries packaging for 16 KB page size compatibility
        jniLibs {
            useLegacyPackaging = false
            // Pick only the architecture we need to reduce size and alignment issues
            pickFirsts += setOf("**/libmediapipe_tasks_text_jni.so")
        }
    }
    
    // NDK configuration for 16 KB page size compatibility
    ndkVersion = "26.1.10909125"
    
    // Add build features for better native library handling
    buildFeatures {
        compose = true
        buildConfig = true
        // Disable unused features to reduce build complexity
        aidl = false
        renderScript = false
        shaders = false
    }
    
    // Removed externalNativeBuild - now using MediaPipe instead of native llama.cpp
}

// Prevent Play Store from removing unused language resources when generating app bundles.
// This ensures all supported languages packaged in `resourceConfigurations` remain
// available at runtime for per-app locale switching (AppCompat per-app locales).
android.bundle {
    language {
        // Keep all languages in the base APK rather than splitting them into configuration-specific
        // APKs. When enabled, Play may remove some language resources from the installed split
        // APK which prevents runtime calls to update the app locale from finding translations.
        enableSplit = false
    }
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

    // MediaPipe Tasks (updated to latest as of Oct 2025)
    // NOTE: Version 0.10.29 has slower initial load for multimodal models due to eager
    // vision/audio component initialization. Disable vision/audio when not needed for faster loading.
    // tasks-genai latest: 0.10.29; tasks-vision latest: 0.10.29; tasks-text latest: 0.10.29
    implementation("com.google.mediapipe:tasks-genai:0.10.29")
    implementation("com.google.mediapipe:tasks-vision:0.10.29")
    implementation("com.google.mediapipe:tasks-text:0.10.29")
    
    // AI Edge RAG SDK for proper Gecko embedding support
    implementation("com.google.ai.edge.localagents:localagents-rag:0.3.0")
    // Note: MediaPipe tasks-genai 0.10.22 is required for RAG SDK, but using 0.10.27 should be compatible
    
    // Compose Markdown - temporarily removed due to version conflicts
    implementation("com.github.jeziellago:compose-markdown:0.3.0")

    // Markdown parser for extracting code blocks
    implementation("com.vladsch.flexmark:flexmark-all:0.64.8")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}