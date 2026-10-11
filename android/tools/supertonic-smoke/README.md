# Supertonic FP16 Android smoke test

`SupertonicVariantSmoke.java` runs the actual engine and model manifest from a debug APK without installing it or changing app data. It requires an ARM64 Android device.

1. Build `:app:assembleDebug`.
2. Compile the harness with `javac --release 11 -cp <sdk>/platforms/android-37.0/android.jar -d <classes-dir> SupertonicVariantSmoke.java`. Run `d8 --min-api 27 --lib <sdk>/platforms/android-37.0/android.jar --output <dex-dir> <class-files>` on all generated classes, including the anonymous context class.
3. Create a fresh temporary device directory under `/data/local/tmp`. Copy `classes.dex`, `app-debug.apk`, and the APK's ARM64 `libonnxruntime.so` and `libonnxruntime4j_jni.so` there.
4. Download the six pinned FP16 core files and `voice_styles/M1.json` listed in [the model documentation](../../docs/supertonic-3.md), and copy them into that same directory using their base filenames.
5. Run from that directory:

```sh
LD_LIBRARY_PATH="$PWD" CLASSPATH="$PWD/classes.dex:$PWD/app-debug.apk" app_process /system/bin SupertonicVariantSmoke "$PWD"
```

The harness verifies separate FP32/FP16 directories, acceptance of complete bundles, rejection of missing/truncated weights, finite non-silent English/Japanese/French synthesis at 44,100 Hz, and cancellation. It creates sparse manifest fixtures under `test-files/`, not app storage. It does not test Compose screens or speaker playback. Remove the temporary directory afterward.
