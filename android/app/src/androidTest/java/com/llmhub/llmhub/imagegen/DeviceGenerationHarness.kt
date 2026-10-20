package com.llmhub.llmhub.imagegen

import android.graphics.Bitmap
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.core.content.FileProvider
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.videogen.SdCppNative
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import java.io.File
import java.security.MessageDigest

/**
 * Fixture import, native rendering and result capture for opt-in physical-device checkpoint tests.
 * Fixtures are staged under files/<suite>-test-fixtures; PNGs and results.json go to files/<suite>-test-results.
 */
internal class DeviceGenerationHarness(private val suite: String, private val prompt: String, private val size: Int = 512) : AutoCloseable {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val fixtures = File(context.filesDir, "$suite-test-fixtures")
    private val results = File(context.filesDir, "$suite-test-results").apply { mkdirs() }
    private val records = JSONArray()
    private val loads = JSONArray()
    private val store = ImageExtrasStore(context)
    private val imported = mutableListOf<ImageExtra>()
    private var handle = 0L
    var options = ImageGenerationOptions()

    init {
        // Optional ggml-vulkan switches for driver diagnosis, e.g. -e vkEnv GGML_VK_DISABLE_DOT2=1,GGML_VK_DISABLE_COOPMAT=1.
        // ggml reads them when the Vulkan device is first created, so each value needs its own instrumentation process.
        InstrumentationRegistry.getArguments().getString("vkEnv")?.split(',')?.filter { '=' in it }?.forEach {
            android.system.Os.setenv(it.substringBefore('='), it.substringAfter('='), true)
            Log.i("DeviceGenerationTest", "vkEnv $it")
        }
    }

    /** Missing fixtures fail rather than silently skip. */
    fun import(name: String, kind: String): ImageExtra {
        val file = File(fixtures, name)
        assertTrue("Missing $suite fixture: $name", file.isFile)
        return store.importFile(FileProvider.getUriForFile(context, context.packageName + ".fileprovider", file), kind)
            .also { imported += it }
    }

    fun load(model: ImageExtra, value: ImageGenerationOptions) {
        if (handle != 0L) { SdCppImageNative.close(handle); handle = 0 }
        SdCppNative.ensureLoaded()
        options = value
        val started = SystemClock.elapsedRealtime()
        handle = SdCppImageNative.create(model.path, value.json().toString(), SdCppImageRuntime.usesGpu(true))
        assertTrue("$suite pipeline must load", handle != 0L)
        loads.put(JSONObject().apply {
            put("assets", JSONArray(value.assets.map { "${it.kind}:${it.name}" }))
            put("gpu", SdCppImageRuntime.usesGpu(true))
            put("milliseconds", SystemClock.elapsedRealtime() - started)
        })
    }

    fun render(label: String, strength: Float = 1f, input: ByteArray? = null, steps: Int = 2): ByteArray {
        val started = SystemClock.elapsedRealtime()
        val output = SdCppImageNative.generate(handle, prompt, options.json().toString(), size, size, steps, 12345, strength, input)
        assertNotNull(label, output)
        val rgb = output!![1]
        assertEquals(label, size * size * 3, rgb.size)
        assertTrue("$label must have image variation", rgb.distinct().size > 32)
        val pixels = IntArray(size * size) { i -> (0xff shl 24) or
            ((rgb[i * 3].toInt() and 255) shl 16) or ((rgb[i * 3 + 1].toInt() and 255) shl 8) or
            (rgb[i * 3 + 2].toInt() and 255) }
        val bitmap = Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
        File(results, "$label.png").outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bitmap.recycle()
        val digest = MessageDigest.getInstance("SHA-256").digest(rgb).joinToString("") { "%02x".format(it) }
        records.put(JSONObject().apply {
            put("label", label); put("sha256", digest); put("steps", steps)
            put("milliseconds", SystemClock.elapsedRealtime() - started)
            put("pssKb", Debug.getPss()); put("options", options.json())
        })
        report()
        Log.i("DeviceGenerationTest", "PASS $suite $label sha256=$digest")
        return rgb
    }

    /** Mean absolute per-channel difference; GPU backends may drift slightly across model reloads. */
    fun distance(a: ByteArray, b: ByteArray): Double {
        assertEquals(a.size, b.size)
        var total = 0L
        for (i in a.indices) total += kotlin.math.abs((a[i].toInt() and 255) - (b[i].toInt() and 255))
        return total.toDouble() / a.size
    }

    /** An identical component reloaded must stay much closer to the reference than a changed one. */
    fun assertReloadMatches(message: String, reference: ByteArray, reloaded: ByteArray, changed: ByteArray) {
        val same = distance(reference, reloaded)
        val different = distance(reference, changed)
        Log.i("DeviceGenerationTest", "$message: reload distance $same, changed distance $different")
        assertTrue("$message (reload $same vs changed $different)", same < 0.25 * different)
    }

    private fun report() {
        File(results, "results.json").writeText(JSONObject().apply {
            put("manufacturer", Build.MANUFACTURER); put("model", Build.MODEL)
            put("soc", Build.SOC_MODEL); put("android", Build.VERSION.RELEASE)
            put("loads", loads); put("renders", records)
        }.toString(2))
    }

    override fun close() {
        if (handle != 0L) SdCppImageNative.close(handle)
        imported.forEach(store::remove)
        report()
    }
}
