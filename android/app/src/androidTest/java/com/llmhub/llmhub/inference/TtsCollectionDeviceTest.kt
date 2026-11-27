package com.llmhub.llmhub.inference

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.llmhub.llmhub.data.AsrModels
import com.llmhub.llmhub.data.LiteRtTtsModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import com.llmhub.llmhub.ui.components.LiteRtTtsEngine
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin

/**
 * The non-Chinese text-to-speech models in litert-community's Audio collection that the app does not
 * ship (Phonon 31M, Matcha-TTS), with Kokoro for reference. Every voice reads [SpeechModelsDeviceTest.SENTENCE]
 * and [SpeechModelsDeviceTest.PASSAGE]; Moonshine Tiny transcribes the audio back, so `recall` is the
 * share of words a recognizer understood. Results go to logcat as `RESULT {...}`.
 */
class TtsCollectionDeviceTest {
    private val context: Context = InstrumentationRegistry.getInstrumentation().targetContext
    private val models = File(context.filesDir, "models").apply { mkdirs() }
    private val texts = listOf("sentence" to SpeechModelsDeviceTest.SENTENCE, "passage" to SpeechModelsDeviceTest.PASSAGE)

    @Test fun phononCpu() = mediaPipe(phonon(), "phonon", PHONON_VOICES, gpu = false)

    @Test fun phononGpu() = mediaPipe(phonon(), "phonon", PHONON_VOICES, gpu = true)

    @Test fun kokoroReference() = mediaPipe(LiteRtTtsModel.file(context, LiteRtTtsModel.KOKORO_NAME), "kokoro", listOf("af_heart"), gpu = true)

    /**
     * Kokoro on the Pixel 10 (Tensor G5) garbled "Please open the settings page and download a new voice model
     * before tomorrow morning." on the GPU. Every sentence below is synthesized on its own, the failing one first on
     * a fresh engine, with each Kokoro bundle and delegate; Moonshine transcribes each one back. `-e kokoroSetups`
     * picks setups by name (comma separated), default all.
     */
    @Test fun kokoroDiagnosis() = runBlocking<Unit> {
        val base = "https://huggingface.co/litert-community/Kokoro-82M/resolve/81e43a025bccabcbbd487f08bc3ebd65b5fc7295"
        val bundles = mapOf(
            "gpu_fp16" to ("kokoro_int8_gpu_fp16.litertlm" to "1ef178b127ec48c737daf17eabb4cea5f6b3ffbc510aced270fe5c267bbeac52"),
            "fp16" to ("kokoro_int8_fp16.litertlm" to "e7b931ee5d4b7104012a838c9eee240d19ea69cb3ca269041a3274f83dbb7fa1"),
            "gpu" to ("kokoro_int8_gpu.litertlm" to "c7219534a814099166ba382b55ccf90b67fecad3f630179e1e65f91a3a27adda"),
        )
        val setups = listOf("gpu_fp16@GPU", "gpu_fp16@CPU", "fp16@GPU", "fp16@CPU", "gpu@GPU")
        val wanted = InstrumentationRegistry.getArguments().getString("kokoroSetups")?.split(",")?.toSet()
        val sentences = listOf(
            "Please open the settings page and download a new voice model before tomorrow morning.",
            "The quick brown fox jumps over the lazy dog.",
            "Speech recognition on this tablet now runs on the neural processor.",
            "Hello.",
            "Our team met on Tuesday to plan the new library building near the river.",
            "The architect showed drawings of a glass roof, wooden shelves and a quiet reading garden.",
            "Several parents asked whether the children's section would open before the summer holidays.",
            "After the meeting we walked across the bridge and had coffee at the small bakery on the corner, then went home.",
            "Please open the settings page and download a new voice model before tomorrow morning.",
        )
        val asr = recognizer()
        val cacheDir = File(context.cacheDir, "kokoro_diagnosis").apply { mkdirs() }
        try {
            for (setup in setups) {
                if (wanted != null && setup !in wanted) continue
                val (bundle, delegateName) = setup.split("@")
                val (fileName, sha) = bundles.getValue(bundle)
                val file = File(models, if (bundle == "gpu_fp16") LiteRtTtsModel.file(context, LiteRtTtsModel.KOKORO_NAME).name else "test_$fileName")
                OnDeviceFixtures.download(file, "$base/$fileName", sha)
                for (bucket in listOf(0)) { // forcing buckets 1-4 garbles speech on every chip (Tab S9+ too)
                    val report = JSONObject().put("kokoroSetup", setup).put("targetBucket", bucket).put("soc", Build.SOC_MODEL)
                    val tts = runCatching {
                        com.google.mediapipe.tasks.audio.texttospeech.TextToSpeech.create(
                            com.google.mediapipe.tasks.audio.texttospeech.TextToSpeechOptions(
                                baseOptions = com.google.mediapipe.tasks.core.BaseOptions.builder().setModelAssetPath(file.absolutePath)
                                    .setDelegate(com.google.mediapipe.tasks.core.Delegate.valueOf(delegateName)).build(),
                                defaultVoiceId = "af_heart", defaultLocale = "en-US", speakingRate = 1f,
                                targetBucket = bucket, cacheDirectory = cacheDir.absolutePath))
                    }
                    if (tts.isFailure) { Log.i(TAG, "RESULT " + report.put("loadError", tts.exceptionOrNull().toString())); continue }
                    val engine = tts.getOrThrow()
                    val runs = JSONArray()
                    try {
                        for (sentence in sentences) {
                            val started = SystemClock.elapsedRealtime()
                            val result = runCatching { engine.synthesize(sentence, com.google.mediapipe.tasks.audio.texttospeech.SynthesisOptions("af_heart", "en-US")) }
                            val ms = SystemClock.elapsedRealtime() - started
                            val samples = result.getOrNull()?.samples ?: FloatArray(0)
                            var sum = 0.0; var nonFinite = 0; var clipped = 0
                            for (v in samples) { if (!v.isFinite()) nonFinite++ else { sum += v * v; if (abs(v) >= 0.999f) clipped++ } }
                            // Loudness per 250 ms, to see where the audio goes wrong.
                            val envelope = samples.toList().chunked(6000).map { c -> Math.round(Math.sqrt(c.sumOf { (it * it).toDouble() } / c.size) * 1000) }
                            val heard = if (samples.isEmpty() || nonFinite > 0) "" else asr.transcribe(pcm16k(samples, 24000)).orEmpty()
                            runs.put(JSONObject().put("text", sentence.take(40)).put("chars", sentence.length).put("ms", ms)
                                .put("error", result.exceptionOrNull()?.toString() ?: JSONObject.NULL)
                                .put("seconds", samples.size / 24000.0).put("peak", samples.filter { it.isFinite() }.maxOfOrNull { abs(it) } ?: 0f)
                                .put("rms", if (samples.isEmpty()) 0.0 else Math.sqrt(sum / samples.size)).put("nonFinite", nonFinite).put("clipped", clipped)
                                .put("envelope", JSONArray(envelope)).put("recall", recall(sentence, heard)).put("heard", heard))
                        }
                    } finally {
                        engine.close()
                        Log.i(TAG, "RESULT " + report.put("runs", runs))
                    }
                }
                if (file.name.startsWith("test_")) file.delete()
            }
        } finally {
            asr.release()
        }
    }

    @Test fun matchaCpu() = matcha(gpu = false)

    /** Text encoder and vocoder on the GPU, decoder on the CPU, as the model card runs it. */
    @Test fun matchaGpu() = matcha(gpu = true)

    private fun phonon(): File = File(models, "phonon_int8.litertlm").also {
        OnDeviceFixtures.download(it, "https://huggingface.co/litert-community/phonon-31m-en/resolve/ee374c33f0f6f8ed1d9f761e21cdb97df6accfe8/phonon_int8.litertlm",
            "b33a9a0d65acec36762db38a2bff6d899f4aab34b196bb56d89fab5665eabebe")
    }

    private fun mediaPipe(file: File, label: String, voices: List<String>, gpu: Boolean) = runBlocking<Unit> {
        System.gc()
        val pssBefore = Debug.getPss()
        var started = SystemClock.elapsedRealtime()
        val engine = LiteRtTtsEngine.create(context, file, gpu, voices.first(), "en-US", 1f)
        val report = JSONObject().put("tts", label).put("soc", Build.SOC_MODEL).put("requestedGpu", gpu).put("usesGpu", engine.usesGpu)
            .put("loadMs", SystemClock.elapsedRealtime() - started).put("pssAfterLoadMb", (Debug.getPss() - pssBefore) / 1024)
        val runs = JSONArray()
        val asr = recognizer()
        try {
            for (voice in voices) for ((name, text) in texts) {
                val chunks = mutableListOf<FloatArray>()
                var firstChunkMs = -1L
                started = SystemClock.elapsedRealtime()
                val failure = runCatching {
                    engine.synthesize(LiteRtTtsEngine.sanitize(text), voice, "en-US") {
                        if (firstChunkMs < 0) firstChunkMs = SystemClock.elapsedRealtime() - started
                        chunks += it
                    }
                }.exceptionOrNull()
                val ms = SystemClock.elapsedRealtime() - started
                val samples = concat(chunks)
                runs.put(judge(asr, JSONObject().put("voice", voice).put("text", name).put("firstAudioMs", firstChunkMs).put("ms", ms)
                    .put("error", failure?.toString() ?: JSONObject.NULL), text, samples, LiteRtTtsEngine.SAMPLE_RATE))
            }
            report.put("runs", runs).put("peakPssMb", (Debug.getPss() - pssBefore) / 1024)
        } finally {
            engine.close()
            asr.release()
            Log.i(TAG, "RESULT $report")
        }
    }

    /** Matcha-TTS on LiteRT CompiledModel, following the model card: dictionary G2P, text encoder, 10 Euler steps, HiFi-GAN. */
    private fun matcha(gpu: Boolean) = runBlocking<Unit> {
        val base = "https://huggingface.co/litert-community/Matcha-TTS/resolve/9b0cf415f5ffe240f7206048dc440179fdc8f555"
        val dir = File(models, "test_matcha").apply { mkdirs() }
        for ((name, sha) in MATCHA_FILES) OnDeviceFixtures.download(File(dir, name), "$base/$name", sha)
        val config = JSONObject(File(dir, "config.json").readText())
        val symbols = config.getJSONArray("symbols").let { a -> (0 until a.length()).associateBy { a.getString(it) } }
        val dictionary = HashMap<String, String>(300_000)
        java.util.zip.GZIPInputStream(File(dir, "g2p_dict.txt.gz").inputStream()).bufferedReader().forEachLine { line ->
            val tab = line.indexOf('\t')
            if (tab > 0) dictionary[line.substring(0, tab)] = line.substring(tab + 1)
        }
        val embedding = ByteBuffer.wrap(File(dir, "emb.bin").readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            .let { b -> FloatArray(b.remaining()).also { b.get(it) } }
        val melMean = config.getDouble("mel_mean").toFloat()
        val melStd = config.getDouble("mel_std").toFloat()
        val hop = config.getInt("hop")
        val sampleRate = config.getInt("sample_rate")

        System.gc()
        val pssBefore = Debug.getPss()
        var started = SystemClock.elapsedRealtime()
        fun load(name: String, accelerator: Accelerator) = CompiledModel.create(File(dir, name).absolutePath, CompiledModel.Options(accelerator))
        val textEncoder = load("matcha_textenc_fp16.tflite", if (gpu) Accelerator.GPU else Accelerator.CPU)
        val decoder = load("matcha_decoder_fp16.tflite", Accelerator.CPU)
        val vocoder = load("matcha_vocoder_fp16.tflite", if (gpu) Accelerator.GPU else Accelerator.CPU)
        val report = JSONObject().put("tts", "matcha").put("soc", Build.SOC_MODEL).put("gpu", gpu)
            .put("loadMs", SystemClock.elapsedRealtime() - started).put("pssAfterLoadMb", (Debug.getPss() - pssBefore) / 1024)
        val teIn = textEncoder.createInputBuffers(); val teOut = textEncoder.createOutputBuffers()
        val decIn = decoder.createInputBuffers(); val decOut = decoder.createOutputBuffers()
        val vocIn = vocoder.createInputBuffers(); val vocOut = vocoder.createOutputBuffers()
        val random = java.util.Random(7)

        /** One sentence (at most 256 interspersed phonemes, 512 mel frames, about 5.9 s) to 22.05 kHz audio. */
        fun speak(sentence: String, missing: MutableList<String>): FloatArray {
            val ipa = StringBuilder()
            for (token in Regex("[A-Za-z']+|[,.!?;:]").findAll(sentence).map { it.value }) {
                if (token[0].isLetter() || token[0] == '\'') {
                    val phonemes = dictionary[token.lowercase()]
                    if (phonemes == null) { missing += token; continue }
                    if (ipa.isNotEmpty() && ipa.last() != ' ') ipa.append(' ')
                    ipa.append(phonemes)
                } else {
                    ipa.append(token)
                }
            }
            val ids = ipa.mapNotNull { symbols[it.toString()] }.take(MAX_TEXT / 2 - 1)
            val length = 2 * ids.size + 1
            val emb = FloatArray(MAX_TEXT * CHANNELS)
            val tokenIds = IntArray(MAX_TEXT).also { arr -> ids.forEachIndexed { i, id -> arr[2 * i + 1] = id } }
            for (t in 0 until MAX_TEXT) System.arraycopy(embedding, tokenIds[t] * CHANNELS, emb, t * CHANNELS, CHANNELS)
            val textMask = FloatArray(MAX_TEXT) { if (it < length) 1f else 0f }
            teIn[0].writeFloat(emb); teIn[1].writeFloat(textMask)
            textEncoder.run(teIn, teOut)
            val (mu, logw) = teOut.map { it.readFloat() }.sortedByDescending { it.size }
            // Durations, then the length regulator: each phoneme's mu column repeated for its frames.
            val cumulative = FloatArray(MAX_TEXT)
            var total = 0f
            for (t in 0 until MAX_TEXT) { total += ceil(exp(logw[t]) * textMask[t]) * LENGTH_SCALE; cumulative[t] = total }
            val frames = total.toInt().coerceIn(1, MAX_MEL)
            val muY = FloatArray(FEATS * MAX_MEL)
            var phoneme = 0
            for (f in 0 until frames) {
                while (phoneme < MAX_TEXT - 1 && cumulative[phoneme] <= f) phoneme++
                for (c in 0 until FEATS) muY[c * MAX_MEL + f] = mu[c * MAX_TEXT + phoneme]
            }
            val melMask = FloatArray(MAX_MEL) { if (it < frames) 1f else 0f }
            val x = FloatArray(FEATS * MAX_MEL)
            for (c in 0 until FEATS) for (f in 0 until frames) x[c * MAX_MEL + f] = (random.nextGaussian() * TEMPERATURE).toFloat()
            for (k in 0 until STEPS) {
                val t = k.toFloat() / STEPS
                val timeEmbedding = FloatArray(160) { i ->
                    val e = 1000f * t * exp(-(i % 80) * ln(10000f) / 79f)
                    if (i < 80) sin(e) else cos(e)
                }
                decIn[0].writeFloat(x); decIn[1].writeFloat(muY); decIn[2].writeFloat(timeEmbedding); decIn[3].writeFloat(melMask)
                decoder.run(decIn, decOut)
                val v = decOut[0].readFloat()
                for (i in x.indices) x[i] += v[i] / STEPS
            }
            val mel = FloatArray(FEATS * MAX_MEL)
            for (c in 0 until FEATS) for (f in 0 until frames) mel[c * MAX_MEL + f] = x[c * MAX_MEL + f] * melStd + melMean
            vocIn[0].writeFloat(mel)
            vocoder.run(vocIn, vocOut)
            return vocOut[0].readFloat().copyOf(frames * hop).also { w -> for (i in w.indices) w[i] = w[i].coerceIn(-1f, 1f) }
        }

        val runs = JSONArray()
        val asr = recognizer()
        try {
            for ((name, text) in texts) {
                val missing = mutableListOf<String>()
                started = SystemClock.elapsedRealtime()
                val parts = text.split(Regex("(?<=[.!?])\\s+")).map { speak(it, missing) }
                val ms = SystemClock.elapsedRealtime() - started
                runs.put(judge(asr, JSONObject().put("voice", "ljspeech").put("text", name).put("ms", ms)
                    .put("notInDictionary", JSONArray(missing)), text, concat(parts), sampleRate))
            }
            report.put("runs", runs).put("peakPssMb", (Debug.getPss() - pssBefore) / 1024)
        } finally {
            textEncoder.close(); decoder.close(); vocoder.close()
            asr.release()
            dir.deleteRecursively()
            Log.i(TAG, "RESULT $report")
        }
    }

    /** Moonshine Tiny on the CPU, the recognizer the intelligibility check uses. */
    private suspend fun recognizer(): SpeechRecognitionService {
        val model = ModelData.models.first { it.name == AsrModels.MOONSHINE_NAME }
        OnDeviceFixtures.download(File(models, model.localFileName()), model.url, SpeechModelsDeviceTest.MOONSHINE_SHA256)
        return SpeechRecognitionService(context).also { check(it.loadModel(model, WhisperBackend.CPU)) { "Moonshine did not load" } }
    }

    private suspend fun judge(asr: SpeechRecognitionService, run: JSONObject, text: String, samples: FloatArray, rate: Int): JSONObject {
        val seconds = samples.size / rate.toDouble()
        val transcript = if (samples.isEmpty()) "" else asr.transcribe(pcm16k(samples, rate)).orEmpty()
        return run.put("audioSeconds", seconds).put("realTimeFactor", if (seconds > 0) run.getLong("ms") / 1000.0 / seconds else JSONObject.NULL)
            .put("peak", samples.maxOfOrNull { abs(it) } ?: 0f).put("recall", recall(text, transcript)).put("heard", transcript)
    }

    private fun concat(parts: List<FloatArray>) = FloatArray(parts.sumOf { it.size }).also { out ->
        var o = 0
        for (p in parts) { p.copyInto(out, o); o += p.size }
    }

    private fun pcm16k(samples: FloatArray, rate: Int): ByteArray {
        val count = (samples.size * 16000L / rate).toInt()
        val pcm = ByteBuffer.allocate(count * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until count) {
            val position = i * (rate / 16000.0)
            val index = position.toInt().coerceAtMost(samples.lastIndex)
            val next = samples[minOf(index + 1, samples.lastIndex)]
            val value = samples[index] + (next - samples[index]) * (position - index).toFloat()
            pcm.putShort((value.coerceIn(-1f, 1f) * 32767).toInt().toShort())
        }
        return pcm.array()
    }

    private fun words(text: String) = text.lowercase().split(Regex("[^\\p{L}\\p{N}']+")).filter { it.isNotEmpty() }

    private fun recall(expected: String, transcript: String): Double {
        val heard = words(transcript).groupingBy { it }.eachCount().toMutableMap()
        val wanted = words(expected)
        return wanted.count { word -> (heard[word] ?: 0) > 0 && heard.merge(word, -1, Int::plus) != null }.toDouble() / wanted.size
    }

    private companion object {
        const val TAG = "TtsCollectionDeviceTest"
        val PHONON_VOICES = listOf("freya", "claire", "lukas", "sofia", "lucas")
        const val MAX_TEXT = 256
        const val MAX_MEL = 512
        const val CHANNELS = 192
        const val FEATS = 80
        const val LENGTH_SCALE = 0.95f
        const val STEPS = 10
        const val TEMPERATURE = 0.667
        val MATCHA_FILES = listOf(
            "config.json" to "7363e9e4dda1613aebff7005f2e8c0c76d9b0a1cac31de0f3483bef3089c6906",
            "emb.bin" to "18be34cda4afa1f59b9fb0b80d03f048bd9dd650e29846e4c6467e0b614c1ee7",
            "g2p_dict.txt.gz" to "5b3493a8cd4d20b72c7b91415afaf3f32335ebd81f349698e1cedc898c59f979",
            "matcha_textenc_fp16.tflite" to "adf2642699b09a9f0cc302a64396554ac698401d93cffad4462792d6df8589a5",
            "matcha_decoder_fp16.tflite" to "2c2e70235f0daaa9d82366416aad69b4f6e916f080a2da8b4b3b759f7cfd3069",
            "matcha_vocoder_fp16.tflite" to "7d0b4d975eb1558d02c0fa781b329f5fc650cfee6bffcf311de4b190d62020ac",
        )
    }
}
