package com.llmhub.llmhub.inference

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.llmhub.llmhub.data.AsrModels
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.LiteRtTtsModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.imagegen.OnDeviceFixtures
import com.llmhub.llmhub.ui.components.LiteRtTtsEngine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Physical-device round trip for the LiteRT speech models: a LiteRT-LM voice reads a passage and
 * each recognizer has to transcribe it. Every test downloads the model files it needs into the
 * app's models folder and leaves them there; results go to logcat as `RESULT {...}`.
 */
class SpeechModelsDeviceTest {
    private val context: Context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun kokoroSpeaks() = runBlocking { assertTrue(speech(LiteRtTtsModel.KOKORO_NAME, PASSAGE).size > 16000 * 2 * 5) }

    @Test fun pocketSpeaks() = runBlocking { assertTrue(speech(LiteRtTtsModel.POCKET_NAME, SENTENCE).size > 16000 * 2) }

    @Test fun parakeetGpuHearsThePassage() = hears(model(AsrModels.PARAKEET_NAME, PARAKEET_SHA256), WhisperBackend.GPU, PASSAGE)

    @Test fun parakeetCpuHearsThePassage() = hears(model(AsrModels.PARAKEET_NAME, PARAKEET_SHA256), WhisperBackend.CPU, PASSAGE)

    @Test fun parakeetNpuHearsThePassage() = hears(npuModel(), WhisperBackend.NPU, PASSAGE)

    /** A clip shorter than one 5-second window, so most of the window is padding. */
    @Test fun parakeetNpuHearsAShortSentence() = hears(npuModel(), WhisperBackend.NPU, SENTENCE)

    @Test fun parakeetNpuHearsPocketTts() = hears(npuModel(), WhisperBackend.NPU, SENTENCE, LiteRtTtsModel.POCKET_NAME)

    /**
     * Microphone-like audio: quiet, noisy, padded with silence, very short. The NPU engine (our own
     * front end and decoder) should hear what Google's LiteRT-LM pipeline hears in the same samples.
     */
    @Test fun parakeetNpuMatchesLiteRtLmOnRoughAudio() = runBlocking<Unit> {
        fun floats(pcm: ByteArray): FloatArray {
            val shorts = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            return FloatArray(shorts.remaining()) { shorts.get(it) / 32768f }
        }
        fun pcm(samples: FloatArray): ByteArray {
            val out = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (v in samples) out.putShort((v.coerceIn(-1f, 1f) * 32767).toInt().toShort())
            return out.array()
        }
        val passage = floats(speech(LiteRtTtsModel.KOKORO_NAME, PASSAGE))
        val sentence = floats(speech(LiteRtTtsModel.KOKORO_NAME, SENTENCE))
        val word = floats(speech(LiteRtTtsModel.KOKORO_NAME, "Who is your king?"))
        val random = java.util.Random(11)
        fun noise(n: Int, level: Float) = FloatArray(n) { (random.nextGaussian() * level).toFloat() }
        fun mix(a: FloatArray, gain: Float, level: Float) = FloatArray(a.size) { a[it] * gain + (random.nextGaussian() * level).toFloat() }
        val cases = linkedMapOf(
            "clean passage" to passage,
            "quiet passage (5%)" to mix(passage, 0.05f, 0f),
            "quiet + room noise" to mix(passage, 0.05f, 0.004f),
            "loud noise (SNR ~6 dB)" to mix(passage, 0.3f, 0.03f),
            "1.5 s noise, sentence, 1.2 s noise" to noise(24000, 0.003f) + mix(sentence, 0.1f, 0.003f) + noise(19200, 0.003f),
            "short question + 1.2 s silence" to mix(word, 0.1f, 0.002f) + noise(19200, 0.002f),
            "0.6 s noise, short question, 1.2 s noise" to noise(9600, 0.004f) + mix(word, 0.06f, 0.004f) + noise(19200, 0.004f),
            "14.12 s: passage then noise" to (mix(passage, 0.1f, 0.003f) + noise(40000, 0.003f)).copyOf(225920),
            "only room noise, 2.6 s" to noise(42240, 0.004f),
            "DC offset + quiet passage" to FloatArray(passage.size) { passage[it] * 0.05f + 0.02f },
        )
        val reference = SpeechRecognitionService(context)
        val npu = SpeechRecognitionService(context)
        try {
            assertTrue(reference.loadModel(model(AsrModels.PARAKEET_NAME, PARAKEET_SHA256), WhisperBackend.CPU))
            assertTrue(npu.loadModel(npuModel(), WhisperBackend.NPU))
            for ((label, samples) in cases) {
                val bytes = pcm(samples)
                val expected = reference.transcribe(bytes)
                val actual = npu.transcribe(bytes)
                Log.i(TAG, "RESULT rough " + JSONObject().put("case", label).put("seconds", samples.size / 16000.0)
                    .put("litertlm", expected ?: JSONObject.NULL).put("npu", actual ?: JSONObject.NULL))
            }
        } finally {
            reference.release()
            npu.release()
        }
    }

    @Test fun mlKitBasicHearsThePassage() = hears(ModelData.models.first { it.name == AsrModels.MLKIT_BASIC_NAME }, WhisperBackend.CPU, PASSAGE)

    @Test fun readsAloudWithKokoro() = readsAloud(LiteRtTtsModel.KOKORO_NAME)

    @Test fun readsAloudWithPocket() = readsAloud(LiteRtTtsModel.POCKET_NAME)

    /** Every packaged Kokoro voice must produce audio in an engine that was loaded with another voice. */
    @Test fun everyKokoroVoiceSpeaks() = runBlocking {
        speech(LiteRtTtsModel.KOKORO_NAME, SENTENCE)
        val samples = mapOf(
            'a' to "Hello, this is a test.", 'b' to "Hello, this is a test.", 'e' to "Hola, esto es una prueba.",
            'f' to "Bonjour, ceci est un test.", 'h' to "नमस्ते, यह एक परीक्षण है।", 'i' to "Ciao, questa è una prova.",
            'j' to "こんにちは、これはテストです。", 'p' to "Olá, isto é um teste.", 'z' to "你好，这是一个测试。"
        )
        val engine = LiteRtTtsEngine.create(context, LiteRtTtsModel.file(context, LiteRtTtsModel.KOKORO_NAME), true, "af_heart", null, 1f)
        val silent = mutableListOf<String>()
        try {
            for ((voice, _) in LiteRtTtsModel.voices(LiteRtTtsModel.KOKORO_NAME)) {
                var count = 0
                var peak = 0f
                val failure = runCatching {
                    engine.synthesize(samples.getValue(voice[0]), voice, null) { chunk ->
                        count += chunk.size
                        peak = maxOf(peak, chunk.maxOf { kotlin.math.abs(it) })
                    }
                }.exceptionOrNull()
                Log.i(TAG, "RESULT voice $voice samples=$count peak=$peak ${failure ?: ""}")
                if (failure != null || count < 4800 || peak < 0.05f) silent += voice
            }
        } finally {
            engine.close()
        }
        assertTrue("Voices without audio: $silent", silent.isEmpty())
    }

    /** A non-English Kokoro voice, picked the way Settings does it. */
    @Test fun readsAloudWithKokoroInSpanish() =
        readsAloud(LiteRtTtsModel.KOKORO_NAME, "ef_dora", "Hola, me llamo Dora y hoy hace un día muy bonito en Madrid.")

    /**
     * What the Text to Speech screen does: the service is already loaded when Speak is pressed, the
     * voice is changed in Settings while it stays loaded, and Speak is pressed again.
     */
    @Test fun speaksAgainAfterAVoiceChange() = runBlocking {
        speech(LiteRtTtsModel.KOKORO_NAME, SENTENCE)
        val preferences = com.llmhub.llmhub.data.ThemePreferences(context)
        val previous = preferences.selectedTtsModel.first()
        val previousVoice = preferences.selectedTtsVoice.first()
        preferences.setSelectedTtsModel(LiteRtTtsModel.KOKORO_NAME)
        preferences.setSelectedTtsVoice("af_heart")
        val service = com.llmhub.llmhub.ui.components.TtsService(context)
        try {
            kotlinx.coroutines.delay(8_000) // model load
            val failures = mutableListOf<String>()
            for ((voice, text) in listOf("af_heart" to SENTENCE, "ef_dora" to "Hola, me llamo Dora y hoy hace un día muy bonito.",
                "ef_dora" to "Esta es la segunda frase en español.", "af_heart" to SENTENCE)) {
                preferences.setSelectedTtsVoice(voice)
                kotlinx.coroutines.delay(500)
                val started = SystemClock.elapsedRealtime()
                service.speak(text)
                val began = kotlinx.coroutines.withTimeoutOrNull(20_000) { service.isSpeaking.first { it } }
                val ended = kotlinx.coroutines.withTimeoutOrNull(30_000) { service.isSpeaking.first { !it } }
                val spokeMs = SystemClock.elapsedRealtime() - started
                Log.i(TAG, "RESULT speakAgain voice=$voice began=${began != null} ended=${ended != null} ms=$spokeMs error=${service.error.value}")
                if (began == null || ended == null || spokeMs < 1500 || service.error.value != null) failures += "$voice ($spokeMs ms)"
            }
            assertTrue("Silent or failed: $failures", failures.isEmpty())
        } finally {
            service.shutdown()
            preferences.setSelectedTtsModel(previous)
            preferences.setSelectedTtsVoice(previousVoice)
        }
    }

    @Test fun readsHindiAloud() = readsAloud(LiteRtTtsModel.KOKORO_NAME, "hf_alpha", "नमस्ते, मेरा नाम अल्फा है और आज मौसम बहुत अच्छा है।")

    @Test fun readsEnglishAloudWithHindiVoice() = readsAloud(LiteRtTtsModel.KOKORO_NAME, "hf_alpha", SENTENCE)

    @Test fun readsAloudWithSpanishMaleVoice() =
        readsAloud(LiteRtTtsModel.KOKORO_NAME, "em_alex", "Hola, me llamo Alex y hoy hace un día muy bonito en Madrid.")

    /** A Spanish voice asked to read English, as happens when the voice is changed but the text is not. */
    @Test fun readsEnglishAloudWithSpanishMaleVoice() = readsAloud(LiteRtTtsModel.KOKORO_NAME, "em_alex", SENTENCE)

    /** The app's own read-aloud path: TtsService with the model picked in Settings, through to playback. */
    private fun readsAloud(voiceModel: String, voice: String? = null, text: String = SENTENCE) = runBlocking {
        speech(voiceModel, SENTENCE) // makes sure the model file is there
        val preferences = com.llmhub.llmhub.data.ThemePreferences(context)
        val previous = preferences.selectedTtsModel.first()
        val previousVoice = preferences.selectedTtsVoice.first()
        preferences.setSelectedTtsModel(voiceModel)
        if (voice != null) preferences.setSelectedTtsVoice(voice)
        val service = com.llmhub.llmhub.ui.components.TtsService(context)
        try {
            val started = SystemClock.elapsedRealtime()
            service.speak(text)
            val began = kotlinx.coroutines.withTimeoutOrNull(60_000) { service.isSpeaking.first { it } }
            val startMs = SystemClock.elapsedRealtime() - started
            val ended = kotlinx.coroutines.withTimeoutOrNull(60_000) { service.isSpeaking.first { !it } }
            val report = JSONObject().put("readAloud", voiceModel).put("startedSpeakingMs", startMs)
                .put("spokeForMs", SystemClock.elapsedRealtime() - started - startMs).put("error", service.error.value)
            Log.i(TAG, "RESULT $report")
            assertNull("TTS reported an error", service.error.value)
            assertNotNull("Speech never started", began)
            assertNotNull("Speech never finished", ended)
            // The sentence takes about three seconds to say.
            assertTrue("Playback ended too soon: $report", report.getLong("spokeForMs") > 1500)
        } finally {
            service.shutdown()
            preferences.setSelectedTtsModel(previous)
            if (voice != null) preferences.setSelectedTtsVoice(previousVoice)
        }
    }

    @Test fun mlKitAdvancedHearsThePassage() = hears(ModelData.models.first { it.name == AsrModels.MLKIT_ADVANCED_NAME }, WhisperBackend.CPU, PASSAGE)

    private fun hears(model: LLMModel, backend: WhisperBackend, text: String, voiceModel: String = LiteRtTtsModel.KOKORO_NAME) = runBlocking {
        val audio = speech(voiceModel, text)
        val report = JSONObject().put("model", model.name).put("backend", backend.name).put("soc", Build.SOC_MODEL)
            .put("audioSeconds", audio.size / 32000.0)
        val service = SpeechRecognitionService(context)
        try {
            if (AsrModels.isMlKit(model)) {
                report.put("mlKit", installMlKit(model))
                // -e mlKitFeedSpeed N overrides how fast the recording is fed to ML Kit.
                InstrumentationRegistry.getArguments().getString("mlKitFeedSpeed")?.toFloatOrNull()?.let { MlKitAsrEngine.feedSpeed = it }
                report.put("feedSpeed", MlKitAsrEngine.feedSpeed.toDouble())
            }
            var started = SystemClock.elapsedRealtime()
            val loaded = service.loadModel(model, backend)
            report.put("loaded", loaded).put("loadMs", SystemClock.elapsedRealtime() - started)
            if (AsrModels.isMlKit(model)) assumeTrue("ML Kit recognizer is not available on this device", loaded)
            assertTrue("${model.name} must load", loaded)
            started = SystemClock.elapsedRealtime()
            val transcript = service.transcribe(audio).orEmpty()
            val recall = recall(text, transcript)
            report.put("transcribeMs", SystemClock.elapsedRealtime() - started).put("recall", recall).put("transcript", transcript)
            Log.i(TAG, "RESULT $report")
            assertTrue("Only ${(recall * 100).toInt()}% of the words came back: $transcript", recall >= 0.85)
            // Stitching windows must not repeat words.
            assertTrue("Transcript is too long: $transcript", words(transcript).size <= words(text).size + 2)
        } finally {
            Log.i(TAG, "REPORT $report")
            service.release()
        }
    }

    /** Reports the recognizer's feature status and, when the system offers the model, installs it. */
    private suspend fun installMlKit(model: LLMModel): String {
        val engine = MlKitAsrEngine(AsrModels.isMlKitAdvanced(model))
        try {
            val status = engine.status()
            if (status != com.google.mlkit.genai.common.FeatureStatus.DOWNLOADABLE &&
                status != com.google.mlkit.genai.common.FeatureStatus.DOWNLOADING) return "status $status"
            var last = ""
            kotlinx.coroutines.withTimeoutOrNull(300_000) {
                engine.download()?.collect {
                    last = it.javaClass.simpleName +
                        ((it as? com.google.mlkit.genai.common.DownloadStatus.DownloadFailed)?.e?.let { e -> " ($e)" } ?: "")
                }
            }
            return "status $status, download $last, now ${engine.status()}"
        } finally {
            engine.close()
        }
    }

    /** Device labs drop DNS now and then in the middle of a run: a failed download resumes on the next try. */
    private fun download(target: File, url: String, sha256: String) {
        var failure: Exception? = null
        repeat(4) { attempt ->
            try {
                return OnDeviceFixtures.download(target, url, sha256)
            } catch (e: java.io.IOException) {
                failure = e
                Log.w(TAG, "Download of ${target.name} failed (try ${attempt + 1}): $e")
                Thread.sleep(10_000)
            }
        }
        throw failure!!
    }

    private fun model(name: String, sha256: String): LLMModel {
        val model = ModelData.models.first { it.name == name }
        download(File(File(context.filesDir, "models").apply { mkdirs() }, model.localFileName()), model.url, sha256)
        return model
    }

    private fun npuModel(): LLMModel {
        val model = ModelData.models.firstOrNull { AsrModels.isParakeetNpu(it) }
        assumeTrue("No Parakeet NPU build for ${Build.SOC_MODEL}", model != null)
        val sha256 = NPU_SHA256[model!!.url.substringAfterLast("stateful_").removeSuffix(".tflite")]
        assumeTrue("No checksum recorded for ${model.name}", sha256 != null)
        return model(model.name, sha256!!)
    }

    /** [text] read by [voiceModel], as 16 kHz mono PCM16. Cached, so the recognizers hear the same audio. */
    internal suspend fun speech(voiceModel: String, text: String): ByteArray {
        val cache = File(context.cacheDir, "speech_test/${voiceModel.hashCode()}_${text.hashCode()}.pcm")
        if (cache.length() > 0) return cache.readBytes()
        val catalog = ModelData.ttsModels.first { it.name == voiceModel }
        val file = LiteRtTtsModel.file(context, voiceModel).also { it.parentFile?.mkdirs() }
        download(file, catalog.url, TTS_SHA256.getValue(voiceModel))
        assertTrue(LiteRtTtsModel.isComplete(context, voiceModel))

        val voice = LiteRtTtsModel.defaultVoice(voiceModel)
        val locale = LiteRtTtsModel.synthesisLocale(voiceModel, voice)
        var started = SystemClock.elapsedRealtime()
        val engine = LiteRtTtsEngine.create(context, file, LiteRtTtsModel.supportsGpu(voiceModel), voice, locale, 1f)
        val loadMs = SystemClock.elapsedRealtime() - started
        val chunks = mutableListOf<FloatArray>()
        try {
            started = SystemClock.elapsedRealtime()
            for (sentence in text.split(Regex("(?<=[.!?])\\s+"))) {
                engine.synthesize(LiteRtTtsEngine.sanitize(sentence), voice, locale) { chunks += it }
            }
        } finally {
            engine.close()
        }
        val samples = FloatArray(chunks.sumOf { it.size })
        var offset = 0
        for (chunk in chunks) { chunk.copyInto(samples, offset); offset += chunk.size }
        Log.i(TAG, "RESULT " + JSONObject().put("voiceModel", voiceModel).put("gpu", engine.usesGpu).put("loadMs", loadMs)
            .put("synthesisMs", SystemClock.elapsedRealtime() - started)
            .put("audioSeconds", samples.size / LiteRtTtsEngine.SAMPLE_RATE.toDouble())
            .put("peak", samples.maxOfOrNull { kotlin.math.abs(it) } ?: 0f))
        assertTrue("$voiceModel produced no audio", samples.any { kotlin.math.abs(it) > 0.05f })

        // 24 kHz to 16 kHz by linear interpolation.
        val count = (samples.size * 16000L / LiteRtTtsEngine.SAMPLE_RATE).toInt()
        val pcm = ByteBuffer.allocate(count * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until count) {
            val position = i * (LiteRtTtsEngine.SAMPLE_RATE / 16000.0)
            val index = position.toInt()
            val next = samples[minOf(index + 1, samples.lastIndex)]
            val value = samples[index] + (next - samples[index]) * (position - index).toFloat()
            pcm.putShort((value.coerceIn(-1f, 1f) * 32767).toInt().toShort())
        }
        cache.parentFile?.mkdirs()
        cache.writeBytes(pcm.array())
        return pcm.array()
    }

    private fun words(text: String) = text.lowercase().split(Regex("[^\\p{L}\\p{N}']+")).filter { it.isNotEmpty() }

    /** Share of the words of [expected] that [transcript] contains. */
    private fun recall(expected: String, transcript: String): Double {
        val heard = words(transcript).groupingBy { it }.eachCount().toMutableMap()
        val wanted = words(expected)
        val found = wanted.count { word -> (heard[word] ?: 0) > 0 && heard.merge(word, -1, Int::plus) != null }
        return found.toDouble() / wanted.size
    }

    internal companion object {
        const val TAG = "SpeechModelsDeviceTest"
        const val SENTENCE = "The quick brown fox jumps over the lazy dog."
        const val PASSAGE = "The quick brown fox jumps over the lazy dog. " +
            "Speech recognition on this tablet now runs on the neural processor. " +
            "Please open the settings page and download a new voice model before tomorrow morning."
        const val PARAKEET_SHA256 = "66c5a74cf6c2c1d2b990901f6cece5bb53e3214dbceafa0989900a6f0eeb3504"
        /** Checksums of the builds the catalog offers, by SoC tag. */
        val NPU_SHA256 = mapOf(
            "Qualcomm_SM8550" to "f8847f3541d47110009adb866bbfcc0dcd3cb5a37cf8170d9e72b47e77719750",
            "Qualcomm_SM8650" to "85b9d54be7f86a56e268a6baf0c9680f969ec98d6014759fe921ec192c4d40f1",
            "Qualcomm_SM8750" to "b2715c0418dd7b4fcf3012615fdac8041f68a267109011d2d9dc35d834afb7c5",
            "Qualcomm_SM8850" to "42d2244882b50aa211192174f16dacc9d7cfff4ef0ccd6f50ee0bddd6bf3f4c1",
            "Google_Tensor_G5" to "5d720a785a90991eec74a898049c4a8e86f78c6fe5b6bba02aea0e59a2c10184",
            "Google_Tensor_G6" to "815710739575332c188d484bc77106c9abeb953dc0cf572548aa9c7400611475",
            "MediaTek_MT6991" to "cbd36a96b8033b5a7e28e5de4525688aa3f08696275d8b56bcb1e94d5db127ac",
            "MediaTek_MT6993" to "12e66f4b693364cb43fcce372bb7a89df63baf51a0b7c06d16a5697f54bab99f",
        )
        val TTS_SHA256 = mapOf(
            LiteRtTtsModel.KOKORO_NAME to "1ef178b127ec48c737daf17eabb4cea5f6b3ffbc510aced270fe5c267bbeac52",
            LiteRtTtsModel.POCKET_NAME to "9c5427cf3fe2d6680f8268e1c06537496be52d818ef12233a461bf193f314b3c",
        )
    }
}
