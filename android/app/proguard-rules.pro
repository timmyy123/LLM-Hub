# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# ── Stacktrace & De-obfuscation ───────────────────────────────────────
-keepattributes SourceFile,LineNumberTable,*Annotation*,Signature,InnerClasses,EnclosingMethod
-renamesourcefileattribute SourceFile

# Suppress missing protobuf and SLF4J binder warnings during R8
-dontwarn com.google.protobuf.Internal$ProtoNonnullApi
-dontwarn com.google.protobuf.ProtoField
-dontwarn com.google.protobuf.ProtoPresenceBits
-dontwarn com.google.protobuf.ProtoPresenceCheckedField
-dontwarn org.slf4j.impl.StaticLoggerBinder
-dontwarn com.google.protobuf.Internal$ProtoMethodMayReturnNull
-dontwarn org.slf4j.**

# ── General JNI / Native Methods ──────────────────────────────────────
-keepclasseswithmembernames class * {
    native <methods>;
}

# ── Native llama.cpp JNI Engine ───────────────────────────────────────
-keep class com.llmhub.llmhub.inference.LlamaCppInferenceService { *; }
-keepclassmembers class com.llmhub.llmhub.inference.LlamaCppInferenceService { *; }

# ── MediaPipe Tasks (GenAI, Vision, Text) ─────────────────────────────
-keep class com.google.mediapipe.** { *; }
-keepclassmembers class com.google.mediapipe.** { *; }
-dontwarn com.google.mediapipe.**

# ── LiteRT-LM & LiteRT Compiled Models (Gemma-3n, SoundGen) ───────────
-keep class com.google.ai.edge.litertlm.** { *; }
-keepclassmembers class com.google.ai.edge.litertlm.** { *; }
-keep class com.google.ai.edge.litert.** { *; }
-keepclassmembers class com.google.ai.edge.litert.** { *; }
-dontwarn com.google.ai.edge.**

# ── TensorFlow Lite Runtime ───────────────────────────────────────────
-keep class org.tensorflow.lite.** { *; }
-keepclassmembers class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# ── AI Edge RAG SDK & Protobuf ────────────────────────────────────────
-keep class com.google.ai.edge.localagents.rag.** { *; }
-keepclassmembers class com.google.ai.edge.localagents.rag.** { *; }
-keep class com.google.protobuf.** { *; }
-dontwarn com.google.protobuf.**

# ── ONNX Runtime JNI Protection ───────────────────────────────────────
# ONNX Runtime uses JNI extensively - preserve all classes/methods
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }
-keep class ai.onnxruntime.OrtSession { *; }
-keep class ai.onnxruntime.OrtSession$* { *; }
-keep class ai.onnxruntime.OrtEnvironment { *; }
-keep class ai.onnxruntime.OrtSessionOptions { *; }
-keep class ai.onnxruntime.OnnxTensor { *; }
-keep class ai.onnxruntime.TensorInfo { *; }
-keep class ai.onnxruntime.OnnxValue { *; }
-keep class ai.onnxruntime.OrtException { *; }
-keep class ai.onnxruntime.OrtProvider { *; }
-keep class ai.onnxruntime.OrtProvider$* { *; }
-keepclassmembers enum ai.onnxruntime.** {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# ── GenieX SDK Protection ────────────────────────────────────────────
# GenieX SDK uses native libraries, JNI, and reflection on ChatMessage role/content
-keep class com.geniex.sdk.** { *; }
-keepclassmembers class com.geniex.sdk.** { *; }
-dontwarn com.geniex.sdk.**

# ── WhisperKit (ASR) & Qualcomm QNN NPU ─────────────────────────────
# WhisperKitService uses reflection on WhisperKitImpl, loadModels, and isModelLoaded
-keep class com.argmaxinc.whisperkit.** { *; }
-keepclassmembers class com.argmaxinc.whisperkit.** { *; }
-dontwarn com.argmaxinc.whisperkit.**
-keep class com.qualcomm.qti.** { *; }
-keepclassmembers class com.qualcomm.qti.** { *; }
-dontwarn com.qualcomm.qti.**

# ── Stable Diffusion Local Backend ───────────────────────────────────
-keep class com.llmhub.llmhub.service.SDBackendService { *; }
-keep class com.llmhub.llmhub.imagegen.** { *; }

# ── IPA Transcribers (Kokoro TTS G2P) ──────────────────────────────────
# Suppress missing JavaFX desktop GUI classes bundled in the desktop library
-dontwarn javafx.**
-dontwarn com.github.medavox.ipa_transcribers.Gui**
-keep class com.github.medavox.ipa_transcribers.** { *; }
-keepclassmembers class com.github.medavox.ipa_transcribers.** { *; }

# ── Gson Serialized Data Models ───────────────────────────────────────
# LLMModel and model specifications are serialized/deserialized with Gson
-keepclassmembers class com.llmhub.llmhub.data.** {
    <fields>;
}
-keep class com.llmhub.llmhub.data.LLMModel { *; }
-keep class com.llmhub.llmhub.data.ModelRequirements { *; }
-keep class com.google.gson.** { *; }
-dontwarn com.google.gson.**

# ── Room Database ────────────────────────────────────────────────────
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class * { *; }
-keep @androidx.room.Dao interface * { *; }
-keep class androidx.room.paging.** { *; }

# ── Networking, Parsers & Async ───────────────────────────────────────
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn retrofit2.**
-dontwarn io.ktor.**
-dontwarn kotlinx.coroutines.**
-dontwarn com.vladsch.flexmark.**
-dontwarn com.itextpdf.**
-dontwarn org.apache.commons.**
-dontwarn org.osmdroid.**

