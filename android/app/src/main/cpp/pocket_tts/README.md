# Vendored PocketTTS.cpp Android runtime

Pinned sources and local patches are documented in [android/docs/pocket-tts.md](../../../../../docs/pocket-tts.md). This directory contains PocketTTS.cpp at `e801e7d6c2692121a39e80ae525cb5265174a495`, SentencePiece v0.2.1, and dr_libs at `dfe8377631000664666519fdb83da193fd8037f4`.

`pocket_jni.cpp` is the app-owned JNI adapter. `CMakeLists.txt` builds it against the app’s pinned ONNX Android AAR. No model weights or precompiled native binaries are committed here. All third-party licenses are retained beside the sources and included in the app’s `pocket-tts-NOTICE.txt` asset. Upstream code has local bounded-audio, validation, exception-handling and embedded-build patches; SentencePiece excludes desktop CLI tools on Android.
