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

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

# Suppress missing protobuf and SLF4J binder warnings during R8
-dontwarn com.google.protobuf.Internal$ProtoNonnullApi
-dontwarn com.google.protobuf.ProtoField
-dontwarn com.google.protobuf.ProtoPresenceBits
-dontwarn com.google.protobuf.ProtoPresenceCheckedField
-dontwarn org.slf4j.impl.StaticLoggerBinder
-dontwarn com.google.protobuf.Internal$ProtoMethodMayReturnNull

# Keep Protobuf generated message classes and fields used by RAG SDK
-keep class com.google.ai.edge.localagents.rag.** { *; }
-keepclassmembers class com.google.ai.edge.localagents.rag.** { *; }

# ── ONNX Runtime JNI Protection ──────────────────────────────────────
# ONNX Runtime uses JNI extensively - must preserve all classes/methods
# that are called from native code to prevent "java_class == null" crashes
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep OrtSession and related classes used via JNI reflection
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

# Keep enums used by ONNX Runtime
-keepclassmembers enum ai.onnxruntime.** {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# ── Nexa SDK Protection ─────────────────────────────────────────────
# Nexa SDK uses native libraries and JNI - preserve all classes
-keep class ai.nexa.** { *; }
-keepclassmembers class ai.nexa.** { *; }
-keep class ai.nexa.core.** { *; }
-keepclassmembers class ai.nexa.core.** { *; }
