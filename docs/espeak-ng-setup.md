# espeak-ng Integration for Kokoro TTS

The Kokoro pipeline uses an espeak-ng-based grapheme-to-phoneme converter
when available, falling back to a small bundled English dictionary
(~150 words) otherwise. espeak-ng gives you:

- ~95%+ vocabulary coverage for English (plus dozens of other languages)
- Real letter-to-sound rules — proper handling of unfamiliar words and proper nouns
- Stress placement that Kokoro uses to shape prosody

espeak-ng is **GPLv3-licensed**. Bundling it into your app means the app's
distribution must comply with GPLv3 (source disclosure for GPL components,
LGPL-compatible app license, etc.). If your app's license forbids that, stay
on `DictionaryG2P` or implement an MIT-friendly alternative (e.g. a CMUdict
+ ARPAbet→IPA pipeline).

## What's already wired in this repo

| Surface                    | File                                                                                  | Status |
|----------------------------|---------------------------------------------------------------------------------------|--------|
| G2P interface              | `mimobot/speech/kokoro/G2P.kt` and `MimoBot/Kokoro/G2P.swift`                         | ✓ done |
| Dictionary fallback        | `DictionaryG2P` in both platforms                                                     | ✓ done |
| Android JNI bridge (C++)   | `android/app/src/main/cpp/espeak_jni.cpp`, `CMakeLists.txt`                            | ✓ done |
| Android Kotlin wrapper     | `mimobot/speech/kokoro/EspeakG2P.kt` (loadLibrary + asset staging)                    | ✓ done |
| Android Gradle integration | `externalNativeBuild` block in `android/app/build.gradle.kts`                          | ✓ done |
| iOS Swift wrapper          | `MimoBot/Kokoro/EspeakG2P.swift` (stub — `tryLoad()` returns nil until you finish below) | ⚠ stub |
| Build scripts              | `scripts/build_espeak_android.sh`, `scripts/build_espeak_ios.sh`                       | ✓ done |

`KokoroTts` / `KokoroTTS` already calls `G2P.best(...)` /
`G2PFactory.best()`, so the moment the per-platform setup below succeeds the
pipeline starts using espeak-ng with **no other code changes**.

## Android setup

Requirements on the build machine:
- Android NDK r25+ (`$ANDROID_NDK_HOME` or `$ANDROID_NDK_ROOT` set)
- cmake, make, git
- ~250 MB disk

```bash
# default builds arm64-v8a only (matches the Gradle abiFilter):
./scripts/build_espeak_android.sh

# add other ABIs if you re-enable them in build.gradle.kts:
ABIS="arm64-v8a armeabi-v7a x86_64" ./scripts/build_espeak_android.sh
```

What it does:
1. Clones espeak-ng `1.52.0` into `build/espeak-android/espeak-ng`
2. Builds a host copy first (needed to generate espeak's data files)
3. Cross-compiles `libespeak-ng.so` for each ABI →
   `android/app/src/main/jniLibs/<abi>/libespeak-ng.so`
4. Stages `speak_lib.h` →
   `android/app/src/main/cpp/espeak-ng/include/speak_lib.h`
5. Stages voice/phoneme tables →
   `android/app/src/main/assets/espeak-ng-data/`

Then a normal Gradle build:
- Compiles `src/main/cpp/espeak_jni.cpp` into `libespeak-jni.so` per ABI
  (the `externalNativeBuild` block in `app/build.gradle.kts` handles this)
- Packages both `.so` files plus the `espeak-ng-data` assets into the APK
- `EspeakG2P.tryLoad(context)` returns a non-null instance at runtime
- The Mimo Bot test screen should now show "espeak-ng" as the active G2P

If anything goes wrong, the wrapper falls back to `DictionaryG2P` and logs
a warning under the `MimoEspeakG2P` tag.

## iOS setup (manual — see warnings below)

Requirements:
- macOS with Xcode 15+
- cmake, git

```bash
./scripts/build_espeak_ios.sh
```

This produces:
- `ios/vendor/EspeakNG.xcframework/` (device + simulator slices, static libs)
- `ios/vendor/EspeakNG/share/espeak-ng-data/` (voice data)

The script stops there. You then need to do the **manual second half** —
SwiftPM doesn't allow optional binary targets, so we can't make the
XCFramework reference conditional from `Package.swift`:

1. **Make the XCFramework available to Swift.** Two options:
   - **Recommended:** add the XCFramework as a dependency of the host iOS
     app target (drag `EspeakNG.xcframework` into the Xcode project,
     check "Embed & Sign"). This keeps `Package.swift` clean.
   - **Alternative:** add a binary target to a separate SwiftPM package
     that depends on the XCFramework, then add that package as a dep of
     `LLMHub`. Requires that everyone who builds the project has the
     XCFramework checked in or fetched first.

2. **Bundle the data files.** Drag
   `ios/vendor/EspeakNG/share/espeak-ng-data/` into the Xcode project as
   a Folder Reference (blue folder). Confirm it lands at
   `Bundle.main.bundlePath/espeak-ng-data/`.

3. **Wire up the Swift wrapper.** Replace the body of
   `EspeakG2P.swift::tryLoad()` with:

   ```swift
   import EspeakNG  // module from the XCFramework

   static func tryLoad() -> EspeakG2P? {
       let dataPath = Bundle.main.bundlePath + "/espeak-ng-data"
       let rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, dataPath, 0)
       guard rate >= 0 else { return nil }
       return EspeakG2P()
   }

   func phonemize(_ text: String, language: String) -> String {
       espeak_SetVoiceByName(language)
       var input: UnsafeRawPointer? = UnsafeRawPointer(strdup(text))
       defer { free(UnsafeMutableRawPointer(mutating: input)) }
       var out = ""
       while input != nil {
           guard let cstr = espeak_TextToPhonemes(&input, /*charsAuto*/ 1, /*ipa*/ 2) else { break }
           out += String(cString: cstr)
       }
       return out
   }
   ```

4. **Verify.** Open the test screen. The G2P label under "Voice" should
   read "espeak-ng" instead of "dictionary". Try a multisyllabic word that
   isn't in the bundled dictionary (e.g. "philosophy", "parallelogram") —
   Kokoro should pronounce it cleanly.

## Verifying which G2P is active

Both platforms surface the active G2P engine on the Mimo Bot test screen
under the **Voice** card when Kokoro is selected. You should see one of:

- "G2P: dictionary (~150 words)" — espeak-ng failed to load (or is missing)
- "G2P: espeak-ng" — neural-quality phonemization is in use

If you expected espeak-ng but see the dictionary fallback, check:
- Android: `adb logcat | grep MimoEspeakG2P` — look for load failures
- iOS: console for "EspeakG2P" or `espeak_Initialize` warnings

## Troubleshooting

**Android: `UnsatisfiedLinkError: dlopen failed: library "libespeak-ng.so" not found`**
- The `.so` wasn't packaged. Confirm `android/app/src/main/jniLibs/<abi>/libespeak-ng.so` exists after running the build script and rerun `./gradlew :app:clean :app:assembleDebug`.

**Android: `Failed to stage espeak-ng-data: ...`**
- `android/app/src/main/assets/espeak-ng-data/` is empty or missing. Re-run the build script — Stage 3 stages the data dir.

**iOS: linker errors mentioning `_espeak_Initialize`**
- The XCFramework wasn't linked into the app target. In Xcode, select the
  app target → General → "Frameworks, Libraries, and Embedded Content" →
  add `EspeakNG.xcframework` with "Embed & Sign".

**iOS: `espeak_Initialize` returns -1**
- Data files aren't at `Bundle.main.bundlePath/espeak-ng-data/`. Verify the
  folder reference was added to the app's "Copy Bundle Resources" build phase.
