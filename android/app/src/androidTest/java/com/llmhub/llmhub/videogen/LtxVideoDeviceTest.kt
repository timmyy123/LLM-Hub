package com.llmhub.llmhub.videogen

import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.ModelData
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * End-to-end LTX-Video run for remote device farms: downloads the model from its pinned Hugging Face
 * links, loads it on the device's default backend and generates one clip.
 *
 * Arguments (am instrument -e): `model` (substring of the model name, default "Q4_0"),
 * `backend` (NPU/GPU/CPU, default: first supported backend), `frames` (default 25), `skip_download`.
 * Results (MP4, result.json) go to the app's external files dir under ltxv_device_test/.
 */
@RunWith(AndroidJUnit4::class)
class LtxVideoDeviceTest {
    private val args = InstrumentationRegistry.getArguments()
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun generateClip() {
        val out = File(context.getExternalFilesDir(null), "ltxv_device_test").apply { mkdirs() }
        val result = JSONObject()
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("soc", if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE)
            .put("android", Build.VERSION.RELEASE)
        fun save() = File(out, "result.json").writeText(result.toString(2))
        try {
            val pick = args.getString("model") ?: "Q4_0"
            val model = ModelData.videoGenerationModels.first { pick in it.name }
            val backend = args.getString("backend")?.let { VideoGenBackend.valueOf(it) }
                ?: VideoGenEngine.supportedBackends(model).first()
            result.put("model", model.name).put("backend", backend.name)
            save()

            val dir = VideoGenEngine.modelDir(context, model).apply { mkdirs() }
            val t0 = System.currentTimeMillis()
            // -e skip_download true: use files already pushed to the model dir.
            if (args.getString("skip_download") != "true") {
                for (url in listOf(model.url) + model.additionalFiles) {
                    download(url, File(dir, url.substringAfterLast('/').substringBefore('?')))
                }
            }
            result.put("download_s", (System.currentTimeMillis() - t0) / 1000.0)
            save()

            SdCppNative.ensureLoaded()
            result.put("devices", SdCppNative.nativeListDevices().trim())
            val engine = VideoGenEngine(context)
            val t1 = System.currentTimeMillis()
            runBlocking { engine.load(model, backend) {} }.getOrThrow()
            result.put("load_s", (System.currentTimeMillis() - t1) / 1000.0)
            save()

            val (w, h, f) = VideoGenEngine.fixedShape(model) ?: Triple(512, 320, args.getString("frames")?.toInt() ?: 25)
            val request = VideoGenRequest(
                prompt = "A red vintage sports car drives along a winding coastal road at sunset, waves crashing " +
                    "against the rocks below, golden light on the paint, the camera tracks alongside the car, cinematic.",
                width = w, height = h, numFrames = f, steps = 8, seed = 42L
            )
            val clip = runBlocking { engine.generate(request) { Log.i(TAG, "progress $it") } }.getOrThrow()
            clip.videoFile.copyTo(File(out, "clip_${w}x${h}x$f.mp4"), overwrite = true)
            result.put("generate_s", clip.elapsedMs / 1000.0).put("frames", clip.frameCount).put("ok", true)
            engine.unload()
        } catch (t: Throwable) {
            Log.e(TAG, "device test failed", t)
            result.put("ok", false).put("error", "${t.javaClass.simpleName}: ${t.message}")
            throw t
        } finally {
            save()
            Log.i(TAG, "result: $result")
        }
    }

    // Resumable download with retries (device-farm networks drop long transfers); skips complete files.
    private fun download(url: String, target: File) {
        var attempt = 0
        while (true) {
            try {
                downloadOnce(url, target)
                return
            } catch (e: java.io.IOException) {
                if (++attempt >= 6) throw e
                Log.w(TAG, "download of ${target.name} interrupted (${e.message}), resuming (attempt $attempt)")
                Thread.sleep(5_000)
            }
        }
    }

    private fun downloadOnce(url: String, target: File) {
        var have = if (target.exists()) target.length() else 0L
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = true
            connectTimeout = 30_000
            readTimeout = 120_000
            if (have > 0) setRequestProperty("Range", "bytes=$have-")
        }
        when (conn.responseCode) {
            416 -> return
            200 -> have = 0L
        }
        conn.inputStream.use { input ->
            java.io.FileOutputStream(target, have > 0).use { input.copyTo(it, 1 shl 20) }
        }
        Log.i(TAG, "downloaded ${target.name}: ${target.length()} bytes")
    }

    private companion object {
        const val TAG = "LtxVideoDeviceTest"
    }
}
