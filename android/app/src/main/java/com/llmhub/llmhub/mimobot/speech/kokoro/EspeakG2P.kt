package com.llmhub.llmhub.mimobot.speech.kokoro

import android.content.Context
import android.util.Log
import java.io.File

/**
 * espeak-ng-backed [G2P]. Produces high-quality multilingual IPA via the
 * `libespeak-ng.so` runtime + a JNI bridge (`libespeak-jni.so`).
 *
 * The native libs are NOT bundled by default. Run
 * `scripts/build_espeak_android.sh` once to:
 *   1. Cross-compile espeak-ng for each enabled ABI →
 *      `android/app/src/main/jniLibs/<abi>/libespeak-ng.so`
 *   2. Copy headers to `android/app/src/main/cpp/espeak-ng/include/` so the
 *      CMakeLists in src/main/cpp/ links the JNI bridge.
 *   3. Stage espeak-ng-data under `android/app/src/main/assets/espeak-ng-data/`
 *      so it ships in the APK.
 *
 * If any of those are missing at runtime, [tryLoad] returns null and the
 * pipeline falls back to [DictionaryG2P].
 *
 * On first use we copy the assets/espeak-ng-data tree to the app's filesDir
 * (espeak's C API needs a real filesystem path, not an asset URI) and call
 * `espeak_Initialize` with that path.
 */
class EspeakG2P private constructor() : G2P {

    override val displayName: String = "espeak-ng"

    override fun phonemize(text: String, language: String): String {
        if (!nativeAvailable) return ""
        // Voice changes are cheap; do them per call so the caller can switch
        // languages mid-session.
        nativeSetVoice(language)
        return nativePhonemize(text)
    }

    fun close() {
        if (nativeAvailable) nativeShutdown()
    }

    companion object {
        private const val TAG = "MimoEspeakG2P"
        private const val ASSET_DIR = "espeak-ng-data"
        private const val INSTALL_MARKER = ".installed"

        @Volatile private var nativeAvailable: Boolean = false

        /**
         * Best-effort load. Returns a working instance, or null if the native
         * libs / data are missing. Safe to call repeatedly.
         */
        fun tryLoad(context: Context): EspeakG2P? {
            // Idempotent: a second call after a successful first one returns
            // the same logical singleton (espeak-ng has process-global state).
            if (nativeAvailable) return EspeakG2P()

            return try {
                System.loadLibrary("espeak-ng")
                System.loadLibrary("espeak-jni")
                val dataDir = stageDataDir(context) ?: return null
                val rc = nativeInit(dataDir.absolutePath)
                if (rc != 0) {
                    Log.w(TAG, "nativeInit failed rc=$rc — falling back")
                    return null
                }
                nativeAvailable = true
                Log.i(TAG, "espeak-ng ready (data=${dataDir.absolutePath})")
                EspeakG2P()
            } catch (t: UnsatisfiedLinkError) {
                Log.i(TAG, "espeak native libs not present — falling back to DictionaryG2P")
                null
            } catch (t: Throwable) {
                Log.w(TAG, "espeak load failed: ${t.message}")
                null
            }
        }

        /**
         * espeak-ng's C API wants a real filesystem path. We ship its data
         * tree as Android assets and copy on first use into filesDir.
         * The marker file under the destination directory means we skip the
         * copy on subsequent launches (data files don't change between
         * versions of the app unless the user updates them deliberately).
         */
        private fun stageDataDir(context: Context): File? {
            val dest = File(context.filesDir, ASSET_DIR)
            val marker = File(dest, INSTALL_MARKER)
            if (marker.exists()) return dest
            try {
                val assets = context.assets
                if (!hasAsset(assets, ASSET_DIR)) {
                    Log.i(TAG, "espeak-ng-data assets not present — skipping")
                    return null
                }
                copyAssetTree(context, ASSET_DIR, dest)
                marker.createNewFile()
                Log.i(TAG, "Staged espeak-ng-data → ${dest.absolutePath}")
                return dest
            } catch (t: Throwable) {
                Log.w(TAG, "Failed to stage espeak-ng-data: ${t.message}")
                return null
            }
        }

        private fun hasAsset(assets: android.content.res.AssetManager, name: String): Boolean {
            return try { assets.list(name)?.isNotEmpty() == true } catch (_: Throwable) { false }
        }

        private fun copyAssetTree(context: Context, src: String, dst: File) {
            val assets = context.assets
            val children = assets.list(src) ?: return
            if (children.isEmpty()) {
                // It's a file — copy it.
                dst.parentFile?.mkdirs()
                assets.open(src).use { input ->
                    dst.outputStream().use { input.copyTo(it) }
                }
                return
            }
            dst.mkdirs()
            for (child in children) {
                copyAssetTree(context, "$src/$child", File(dst, child))
            }
        }

        // ── Native bindings ─────────────────────────────────────────────────
        @JvmStatic private external fun nativeInit(dataPath: String): Int
        @JvmStatic private external fun nativeSetVoice(voice: String): Int
        @JvmStatic private external fun nativePhonemize(text: String): String
        @JvmStatic private external fun nativeShutdown()
    }
}
