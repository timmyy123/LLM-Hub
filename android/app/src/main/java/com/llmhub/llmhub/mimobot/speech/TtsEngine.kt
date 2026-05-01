package com.llmhub.llmhub.mimobot.speech

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.llmhub.llmhub.mimobot.MimoBotIds
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.UUID

/**
 * System TTS via Android's built-in [TextToSpeech]. Free, on-device, ships on
 * every phone.
 *
 * Synthesis strategy: [TextToSpeech.synthesizeToFile] to a temp WAV, then
 * read + resample + chunk. Not streaming, but for short assistant responses
 * (a few seconds) latency is acceptable. If we need lower first-audio latency
 * later, switch to the API 24+ [UtteranceProgressListener.onAudioAvailable]
 * callback which streams PCM as the voice renders.
 */
class SystemTts(private val context: Context) : Tts {

    private var tts: TextToSpeech? = null
    private var initialized = false
    private var currentUtteranceId: String? = null

    override suspend fun init() {
        if (initialized) return
        val ready = CompletableDeferred<Boolean>()
        val engine = TextToSpeech(context.applicationContext) { status ->
            ready.complete(status == TextToSpeech.SUCCESS)
        }
        tts = engine
        val ok = ready.await()
        if (!ok) throw IllegalStateException("TextToSpeech init failed")
        // Default to English; callers can pass a language on each speak.
        engine.language = Locale.US
        initialized = true
    }

    override fun speakToPcm(text: String, language: String): Flow<ShortArray> = flow {
        if (!initialized) init()
        val engine = tts ?: throw IllegalStateException("TTS not initialized")
        engine.language = Locale.forLanguageTag(language)

        val outFile = File.createTempFile("mimo-tts-", ".wav", context.cacheDir)
        val utteranceId = "mimo-${UUID.randomUUID()}"
        currentUtteranceId = utteranceId

        val done = CompletableDeferred<Boolean>()
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(uid: String?) {}
            override fun onDone(uid: String?) {
                if (uid == utteranceId && !done.isCompleted) done.complete(true)
            }
            @Suppress("OVERRIDE_DEPRECATION")
            override fun onError(uid: String?) {
                if (uid == utteranceId && !done.isCompleted) done.complete(false)
            }
            override fun onError(uid: String?, errorCode: Int) {
                if (uid == utteranceId && !done.isCompleted) done.complete(false)
            }
        })

        val params = Bundle().apply {
            putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId)
        }
        val queueRc = engine.synthesizeToFile(text, params, outFile, utteranceId)
        if (queueRc != TextToSpeech.SUCCESS) {
            outFile.delete()
            throw IllegalStateException("synthesizeToFile queue failed: $queueRc")
        }

        val ok = done.await()
        if (!ok) {
            outFile.delete()
            return@flow
        }

        try {
            val (srcRate, pcm) = readWav(outFile)
            val resampled = if (srcRate == MimoBotIds.SAMPLE_RATE_HZ) pcm
                            else resampleLinear(pcm, srcRate, MimoBotIds.SAMPLE_RATE_HZ)
            val frame = MimoBotIds.FRAME_SAMPLES
            var i = 0
            while (i + frame <= resampled.size) {
                val chunk = resampled.copyOfRange(i, i + frame)
                emit(chunk)
                i += frame
            }
            if (i < resampled.size) {
                // Zero-pad the last partial frame so sinks downstream never see short frames.
                val tail = ShortArray(frame)
                System.arraycopy(resampled, i, tail, 0, resampled.size - i)
                emit(tail)
            }
        } finally {
            outFile.delete()
        }
    }.flowOn(Dispatchers.IO)

    override fun stop() {
        try { tts?.stop() } catch (_: Throwable) {}
        currentUtteranceId = null
    }

    override fun close() {
        stop()
        try { tts?.shutdown() } catch (_: Throwable) {}
        tts = null
        initialized = false
    }

    // --- helpers ---------------------------------------------------------

    /** Read a RIFF/WAVE file. Returns (sampleRate, monoInt16Pcm). */
    private fun readWav(file: File): Pair<Int, ShortArray> {
        RandomAccessFile(file, "r").use { raf ->
            val header = ByteArray(44)
            if (raf.read(header) != 44) throw IllegalStateException("wav too short")
            val bb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
            // RIFF ... WAVEfmt chunk at offset 12
            val channels = bb.getShort(22).toInt()
            val sampleRate = bb.getInt(24)
            val bitsPerSample = bb.getShort(34).toInt()
            if (bitsPerSample != 16) throw IllegalStateException("expected 16-bit WAV, got $bitsPerSample")

            // Find "data" chunk (not always at 36 — skip past any LIST/INFO junk).
            var pos = 36L
            raf.seek(pos)
            var dataSize = -1
            val chunkId = ByteArray(4)
            val sizeBuf = ByteArray(4)
            while (pos < raf.length()) {
                if (raf.read(chunkId) != 4) break
                if (raf.read(sizeBuf) != 4) break
                val size = ByteBuffer.wrap(sizeBuf).order(ByteOrder.LITTLE_ENDIAN).int
                if (chunkId[0] == 'd'.code.toByte() && chunkId[1] == 'a'.code.toByte() &&
                    chunkId[2] == 't'.code.toByte() && chunkId[3] == 'a'.code.toByte()) {
                    dataSize = size
                    break
                }
                raf.skipBytes(size)
                pos = raf.filePointer
            }
            if (dataSize <= 0) throw IllegalStateException("no data chunk in wav")

            val bytes = ByteArray(dataSize)
            raf.readFully(bytes)
            val framesTotal = dataSize / 2
            val shorts = ShortArray(framesTotal)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                .asShortBuffer().get(shorts)

            return if (channels == 1) {
                Pair(sampleRate, shorts)
            } else {
                // Downmix to mono by averaging channels.
                val mono = ShortArray(framesTotal / channels)
                for (i in mono.indices) {
                    var sum = 0
                    for (c in 0 until channels) sum += shorts[i * channels + c]
                    mono[i] = (sum / channels).toShort()
                }
                Pair(sampleRate, mono)
            }
        }
    }

    /** Simple linear-interpolation resampler. Good enough for TTS speech. */
    private fun resampleLinear(input: ShortArray, srcRate: Int, dstRate: Int): ShortArray {
        if (srcRate == dstRate) return input
        val ratio = srcRate.toDouble() / dstRate.toDouble()
        val outLen = (input.size / ratio).toInt()
        val out = ShortArray(outLen)
        var srcPos = 0.0
        for (i in 0 until outLen) {
            val idx = srcPos.toInt()
            val frac = srcPos - idx
            val s0 = input[idx].toInt()
            val s1 = if (idx + 1 < input.size) input[idx + 1].toInt() else s0
            out[i] = (s0 + (s1 - s0) * frac).toInt().toShort()
            srcPos += ratio
        }
        return out
    }

    companion object {
        @Suppress("unused") private const val TAG = "MimoSystemTts"
    }
}

@Deprecated("Use SystemTts", ReplaceWith("SystemTts(context)"))
typealias TtsEngine = SystemTts
