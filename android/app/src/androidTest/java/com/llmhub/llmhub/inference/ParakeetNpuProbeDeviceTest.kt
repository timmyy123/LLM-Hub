package com.llmhub.llmhub.inference

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.imagegen.LiteRtSdPipeline
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.util.Random
import kotlin.math.abs
import kotlin.math.sin

/**
 * Physical-device probe of a chip's ahead-of-time compiled Parakeet NPU build, for chips we cannot
 * hold in hand (Device Farm, Remote Test Lab). It needs no audio: it feeds the encoder and decoder
 * known tensors and reports whether their outputs depend on the inputs, which is what separates a
 * working NPU build from one that loads and then returns the same thing for everything.
 *
 * Downloads the build for this device's SoC and logs one `PROBE {...}` line per step to logcat.
 * `-e stateless 1` probes the stateless build instead; `-e soc <Build.SOC_MODEL>` overrides the chip.
 * `-e file NAME` probes a build already placed at files/models/NAME (one we compiled ourselves);
 * with `-e zipUrl64 BASE64` as well, a missing NAME is first fetched from that URL (a zip holding NAME).
 */
class ParakeetNpuProbeDeviceTest {
    private val context: Context = InstrumentationRegistry.getInstrumentation().targetContext
    private val arguments = InstrumentationRegistry.getArguments()

    @Test fun encoderAndDecoderFollowTheirInputs() {
        val file = modelFile()
        val name = file.name

        val environment = environment()
        log(JSONObject().put("step", "environment").put("accelerators", environment.getAvailableAccelerators().toString()))
        var started = SystemClock.elapsedRealtime()
        val model = try {
            CompiledModel.create(file.absolutePath, CompiledModel.Options(setOf(Accelerator.NPU, Accelerator.CPU)), environment)
        } catch (t: Throwable) {
            log(JSONObject().put("step", "load").put("error", t.toString()))
            throw t
        }
        log(JSONObject().put("step", "load").put("ms", SystemClock.elapsedRealtime() - started))
        try {
            for (signature in listOf("encode", "decode", "decode_1")) {
                fun shapes(prefix: String, type: (String) -> com.google.ai.edge.litert.TensorType) = JSONArray(
                    (0..5).mapNotNull { i -> runCatching { type("$prefix$i").layout?.dimensions?.toList().toString() }.getOrNull() }
                )
                runCatching {
                    log(JSONObject().put("step", "signature").put("name", signature)
                        .put("inputs", shapes("args_") { model.getInputTensorType(it, signature) })
                        .put("outputs", shapes("output_") { model.getOutputTensorType(it, signature) }))
                }.onFailure { log(JSONObject().put("step", "signature").put("name", signature).put("error", it.toString())) }
            }

            // Encoder: silence, a tone-like pattern and noise must not encode to the same thing.
            val encodeIn = model.createInputBuffers("encode")
            val encodeOut = model.createOutputBuffers("encode")
            val size = model.getInputTensorType("args_0", "encode").layout!!.dimensions.fold(1, Int::times)
            val random = Random(7)
            val inputs = linkedMapOf(
                "zeros" to FloatArray(size),
                "pattern" to FloatArray(size) { sin(it * 0.37f) },
                "noise" to FloatArray(size) { random.nextGaussian().toFloat() }
            )
            val encodings = linkedMapOf<String, FloatArray>()
            for ((label, input) in inputs) {
                encodeIn[0].writeFloat(input)
                started = SystemClock.elapsedRealtime()
                model.run(encodeIn, encodeOut, "encode")
                val ms = SystemClock.elapsedRealtime() - started
                val output = encodeOut[0].readFloat()
                encodings[label] = output
                log(JSONObject().put("step", "encode").put("input", label).put("ms", ms).put("summary", summary(output)))
            }
            val encoderFollows = !encodings.getValue("zeros").contentEquals(encodings.getValue("pattern")) &&
                !encodings.getValue("pattern").contentEquals(encodings.getValue("noise"))

            // Decoder: change one input at a time and see whether the logits move.
            val decodeIn = model.createInputBuffers("decode")
            val decodeOut = model.createOutputBuffers("decode")
            fun elements(index: Int) = model.getInputTensorType("args_$index", "decode").layout!!.dimensions.fold(1, Int::times)
            fun decode(label: String, encoded: FloatArray, token: Int, state: Float): FloatArray {
                decodeIn[0].writeFloat(encoded)
                decodeIn[1].writeInt(IntArray(elements(1)).also { it[0] = token })
                for (i in 2 until decodeIn.size) decodeIn[i].writeFloat(FloatArray(elements(i)) { state })
                val begun = SystemClock.elapsedRealtime()
                model.run(decodeIn, decodeOut, "decode")
                val logits = decodeOut[0].readFloat()
                log(JSONObject().put("step", "decode").put("case", label).put("ms", SystemClock.elapsedRealtime() - begun)
                    .put("logits", summary(logits)).put("outputs", decodeOut.size))
                return logits
            }
            val noiseEncoding = FloatArray(elements(0)) { (random.nextGaussian() * 0.05).toFloat() }
            val base = decode("base", encodings.getValue("pattern"), 8192, 0f)
            val otherEncoding = decode("other encoding", noiseEncoding, 8192, 0f)
            val otherToken = decode("other token", encodings.getValue("pattern"), 100, 0f)
            val otherState = if (decodeIn.size > 2) decode("other state", encodings.getValue("pattern"), 8192, 0.5f) else null
            log(JSONObject().put("step", "verdict").put("build", name)
                .put("encoderFollowsInput", encoderFollows)
                .put("decoderFollowsEncoding", !base.contentEquals(otherEncoding))
                .put("decoderFollowsToken", !base.contentEquals(otherToken))
                .put("decoderFollowsState", otherState?.let { !base.contentEquals(it) }))
        } finally {
            model.close()
            environment.close()
        }
    }

    /** The app's own engine on a build placed with `-e file`: a spoken passage has to come back as text. */
    @Test fun customBuildTranscribes() = kotlinx.coroutines.runBlocking<Unit> {
        assumeTrue("Needs -e file NAME", arguments.getString("file") != null)
        val file = modelFile()
        val speech = SpeechModelsDeviceTest()
        val audio = speech.speech(com.llmhub.llmhub.data.LiteRtTtsModel.KOKORO_NAME, SpeechModelsDeviceTest.PASSAGE)
        var started = SystemClock.elapsedRealtime()
        val engine = ParakeetNpuEngine.create(context, file)
        val report = JSONObject().put("step", "transcribe").put("build", file.name).put("soc", Build.SOC_MODEL)
            .put("loaded", engine != null).put("loadMs", SystemClock.elapsedRealtime() - started)
        try {
            if (engine != null) {
                started = SystemClock.elapsedRealtime()
                val transcript = engine.transcribe(audio).orEmpty()
                report.put("transcribeMs", SystemClock.elapsedRealtime() - started).put("audioSeconds", audio.size / 32000.0)
                    .put("transcript", transcript)
            }
        } finally {
            log(report)
            engine?.close()
        }
        org.junit.Assert.assertTrue("The engine rejected ${file.name}", engine != null)
        org.junit.Assert.assertTrue("Nothing came back: $report", report.optString("transcript").contains("quick brown fox", ignoreCase = true))
    }

    /** The build under test: `-e file NAME` in files/models, otherwise Google's build for this chip, downloaded. */
    private fun modelFile(): File {
        val models = File(context.filesDir, "models").apply { mkdirs() }
        val soc = arguments.getString("soc") ?: Build.SOC_MODEL
        val custom = arguments.getString("file")
        log(JSONObject().put("step", "device").put("model", Build.MODEL).put("soc", Build.SOC_MODEL)
            .put("manufacturer", Build.SOC_MANUFACTURER).put("android", Build.VERSION.RELEASE).put("file", custom))
        if (custom != null) {
            val file = File(models, custom)
            val zipUrl = arguments.getString("zipUrl64")?.let { String(android.util.Base64.decode(it, android.util.Base64.URL_SAFE)) }
            if (!file.isFile && zipUrl != null) fetchZipped(zipUrl, file)
            // -e url URL -e sha256 HEX: a plain download, for a build that is already hosted.
            val url = arguments.getString("url")
            if (!file.isFile && url != null) OnDeviceFixtures.download(file, url, arguments.getString("sha256")!!)
            org.junit.Assert.assertTrue("Missing ${file.path}", file.isFile)
            return file
        }
        val tag = TAG_BY_SOC.entries.firstOrNull { soc.uppercase().replace(Regex("[^A-Z0-9]"), "").startsWith(it.key) }?.value
        assumeTrue("No Parakeet NPU build for $soc", tag != null)
        val stateless = arguments.getString("stateless") == "1"
        val name = "parakeet_tdt_0.6b_v3_5s_f32_${if (stateless) "" else "stateful_"}$tag.tflite"
        val sha256 = (if (stateless) STATELESS_SHA256 else STATEFUL_SHA256)[tag]
        assumeTrue("No checksum recorded for $name", sha256 != null)
        val file = File(models, name)
        OnDeviceFixtures.download(file, "$REPO/$name", sha256!!)
        return file
    }

    /** Downloads a zip to the cache, resuming after a dropped connection, and extracts [target]'s entry. */
    private fun fetchZipped(url: String, target: File) {
        val zip = File(context.cacheDir, target.name + ".zip")
        var failure: Exception? = null
        for (attempt in 1..6) {
            try {
                val connection = java.net.URL(url).openConnection() as java.net.HttpURLConnection
                if (zip.length() > 0) connection.setRequestProperty("Range", "bytes=${zip.length()}-")
                val code = connection.responseCode
                if (code == 416) break // already complete
                check(code in 200..299) { "HTTP $code" }
                connection.inputStream.use { input -> java.io.FileOutputStream(zip, code == 206).use { input.copyTo(it, 1 shl 20) } }
                failure = null
                break
            } catch (e: java.io.IOException) {
                failure = e
                log(JSONObject().put("step", "download").put("try", attempt).put("bytes", zip.length()).put("error", e.toString()))
                Thread.sleep(5_000)
            }
        }
        failure?.let { throw it }
        java.util.zip.ZipFile(zip).use { archive ->
            val entry = archive.getEntry(target.name) ?: error("No ${target.name} in the zip")
            archive.getInputStream(entry).use { input -> target.outputStream().use { input.copyTo(it, 1 shl 20) } }
        }
        zip.delete()
        log(JSONObject().put("step", "download").put("file", target.name).put("bytes", target.length()))
    }

    /** The same NPU environment the app's engine uses: the SoC vendor's dispatch library in its own folder. */
    private fun environment(): Environment {
        val dir = if (DeviceInfo.isSnapdragon()) qualcommDir() else LiteRtSdPipeline.vendorLibraryDir(context)
        log(JSONObject().put("step", "libraries").put("dir", dir?.path).put("files", JSONArray(dir?.list()?.sorted().orEmpty())))
        return dir?.let { Environment.create(mapOf(Environment.Option.DispatchLibraryDir to it.path)) }
            ?: Environment.create(BuiltinNpuAcceleratorProvider(context))
    }

    private fun qualcommDir(): File? {
        val dispatch = File(context.applicationInfo.nativeLibraryDir, "libLiteRtDispatch_Qualcomm.so")
        val names = context.assets.list("qnnlibs_litert").orEmpty()
        if (!dispatch.isFile || names.isEmpty()) return null
        val dir = File(context.filesDir, "litert_npu_qnn_250").apply { mkdirs() }
        if (File(dir, dispatch.name).length() != dispatch.length()) dispatch.copyTo(File(dir, dispatch.name), overwrite = true)
        for (name in names) {
            val target = File(dir, name)
            if (target.length() > 0L) continue
            context.assets.open("qnnlibs_litert/$name").use { input -> target.outputStream().use { input.copyTo(it) } }
            target.setReadable(true, false)
            target.setExecutable(true, false)
        }
        return dir
    }

    private fun summary(values: FloatArray): JSONObject {
        var sum = 0.0
        var absolute = 0.0
        var nan = 0
        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        for (value in values) {
            if (value.isNaN()) { nan++; continue }
            sum += value
            absolute += abs(value)
            if (value < min) min = value
            if (value > max) max = value
        }
        return JSONObject().put("n", values.size).put("nan", nan).put("min", min.toDouble()).put("max", max.toDouble())
            .put("sum", sum).put("absSum", absolute)
    }

    private fun log(report: JSONObject) = Log.i(TAG, "PROBE $report")

    private companion object {
        const val TAG = "ParakeetNpuProbe"
        const val REPO = "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/resolve/3ce205aa928f443f423214195cfe83ebe4802826"
        /** Every chip Google publishes a build for, including the ones the app does not offer. */
        val TAG_BY_SOC = mapOf(
            "SM8450" to "Qualcomm_SM8450", "SM8550" to "Qualcomm_SM8550", "SM8650" to "Qualcomm_SM8650",
            "SM8750" to "Qualcomm_SM8750", "SM8850" to "Qualcomm_SM8850",
            "TENSORG5" to "Google_Tensor_G5", "TENSORG6" to "Google_Tensor_G6",
            "MT6983" to "MediaTek_MT6983", "MT6985" to "MediaTek_MT6985", "MT6989" to "MediaTek_MT6989",
            "MT6991" to "MediaTek_MT6991", "MT6993" to "MediaTek_MT6993",
        )
        val STATEFUL_SHA256 = mapOf(
            "Qualcomm_SM8450" to "b3843c6749a7aebe56178238300d9d28672d16fcf058a0b877af40f2f4b51137",
            "Qualcomm_SM8550" to "f8847f3541d47110009adb866bbfcc0dcd3cb5a37cf8170d9e72b47e77719750",
            "Qualcomm_SM8650" to "85b9d54be7f86a56e268a6baf0c9680f969ec98d6014759fe921ec192c4d40f1",
            "Qualcomm_SM8750" to "1dd2f826fb669b6de3910975573021531f65411a80b8dbdafc7a70b4d33f8233",
            "Qualcomm_SM8850" to "e9edabe48ffd853ddebac6c83804752edf0e1e567b26954aae0d3e964f72a486",
            "Google_Tensor_G5" to "5d720a785a90991eec74a898049c4a8e86f78c6fe5b6bba02aea0e59a2c10184",
            "Google_Tensor_G6" to "815710739575332c188d484bc77106c9abeb953dc0cf572548aa9c7400611475",
            "MediaTek_MT6983" to "fb4008d9b43cda15f972ef4bec547f222fc32eeaa53a76c5ad829946c7fd0d35",
            "MediaTek_MT6985" to "f169c3b5d1618414b931d6a10995240e65ce924f585efe1b63c3a07c19547651",
            "MediaTek_MT6989" to "f7cb23f849d51409f0c4af9619ee060b27dd5cf756972afe6be16377bed2dd3a",
            "MediaTek_MT6991" to "a16ba15392666d1ba73218abee3654e6ce1eaf1a9f7568e38ecfd25df84aca22",
            "MediaTek_MT6993" to "2848b34272940b10808f68479607fb243e92a0958c06e2e0553b76cd35343830",
        )
        val STATELESS_SHA256 = mapOf(
            "Qualcomm_SM8450" to "61a91535fa114c2bb986cdb6b6e3e653f7feef1aac0002848ee9c1034c4edda1",
            "Qualcomm_SM8650" to "82a9551a059f808644718d75dc663f81ac81c5aa8c4516c5977bf41abdef1c85",
            "Qualcomm_SM8750" to "9e2023bd7f28d42a7fe66a2b828d74c2fd3fe557fdb6541b7c7b8a4bff06aee4",
            "Qualcomm_SM8850" to "e45e9a0d527356f3f2a580385e050c4d63a02cd39c86c1244dd744e573fd2dde",
            "Google_Tensor_G5" to "a2b49b05cad12ae1e2bf3efacc2befd6005d2b3dc59b4bd42b6de4c24e7f480b",
            "Google_Tensor_G6" to "a9d353a5c4272228e231ee7094048e8e9b14201cf064565168f57e69805b9592",
            "MediaTek_MT6983" to "39d490ccb0f4eb82250c03e6f547ebd8672e71b9a4bd25825825de9be113cc9d",
            "MediaTek_MT6985" to "a20b69b576bab35537e8318da8b40d4d31ba6d8020f71daabc0bbe844e06df55",
            "MediaTek_MT6989" to "db7a2fae12a56f21dcf61e2edfb804d36488b03cf882d12a1875c310cf83cf35",
            "MediaTek_MT6991" to "25b622b575fa69597d45405b69fc50510214b7f367465766988c08bcf88f8770",
            "MediaTek_MT6993" to "1cd591e61692396309320ddd358fbf3ee66d1bd1477d3a750cbb59f0ff0ccb44",
        )
    }
}
