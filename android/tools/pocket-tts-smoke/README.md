# PocketTTS.cpp Android JNI smoke test

This harness exercises the actual JNI symbols used by the Kotlin engine without installing or replacing the app. It requires an ARM64 Android device, the pinned model bundle in `android/docs/pocket-tts.md`, and a clear English `reference.wav` of 3–30 seconds. Test audio must be yours to use.

1. Build `llmhub_pocket_tts` through Android CMake/Gradle. The ONNX AAR is extracted to `app/build/generated/pocket-onnx`.
2. Compile `PocketTtsEngine.java` with `javac --release 11 -d <classes-dir>`, then run Android SDK `d8 --min-api 27 --output <dex-dir> <class-files>` on all generated `.class` files.
3. Create a temporary test directory under `/data/local/tmp` on the device. Copy `classes.dex`, `reference.wav`, `libllmhub_pocket_tts.so`, and the extracted ARM64 `libonnxruntime.so` there. Put the seven model files under `models/` and create `voices/`.
4. Run from that directory using adb shell:

```sh
LD_LIBRARY_PATH="$PWD" CLASSPATH="$PWD/classes.dex" app_process /system/bin com.llmhub.llmhub.ui.components.PocketTtsEngine "$PWD"
```

To also test coexistence with the app’s Java ONNX environment, include the ONNX Android AAR’s `classes.jar` as an additional d8 input and copy its ARM64 `libonnxruntime4j_jni.so` to the test directory. The harness detects this optional runtime.

The test validates reference encoding, shared-session lifetime, finite/non-silent synthesis, cancellation before and during synthesis, and Java exceptions for missing, too-short and silent references. It creates invalid reference fixtures and cached voice states in that temporary directory. Remove the test directory when finished. It does not test the Compose UI or speaker playback.

## Audio import regression harness

`PocketVoiceImportSmoke.java` runs the actual importer from the built debug APK. It creates AAC/M4A fixtures on the device and verifies content detection, mono PCM16 conversion, truncation to 30 seconds, and short-reference cleanup. It does not install the APK or change app data.

1. Build `:app:assembleDebug`. Compile the harness with `javac --release 11 -cp <sdk>/platforms/android-37.0/android.jar -d <classes-dir> PocketVoiceImportSmoke.java`. Run `d8 --min-api 27 --lib <sdk>/platforms/android-37.0/android.jar --output <dex-dir> <class-files>` on all generated classes.
2. Copy `classes.dex` and `app/build/outputs/apk/debug/app-debug.apk` to a fresh directory under `/data/local/tmp`.
3. Run in that directory:

```sh
CLASSPATH="$PWD/classes.dex:$PWD/app-debug.apk" app_process /system/bin PocketVoiceImportSmoke "$PWD"
```

For an additional native encoder check, add `models/` and both native libraries as described above and set `LD_LIBRARY_PATH="$PWD"`. It encodes the converted synthetic M4A tone; optionally provide a permitted English speech recording as `speech.m4a` to use instead. Remove the temporary device directory afterward.
