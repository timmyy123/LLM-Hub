package com.llmhub.llmhub.utils

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder

object AudioConversionUtils {
    private const val TAG = "AudioConversionUtils"
    private const val TARGET_SAMPLE_RATE = 16000
    private const val TARGET_CHANNELS = 1
    private const val TIMEOUT_US = 10_000L

    suspend fun convertUriToFloat32Wav(context: Context, uri: Uri): ByteArray? = withContext(Dispatchers.IO) {
        try {
            // Fast path: if already float32 mono 16k WAV, pass through.
            context.contentResolver.openInputStream(uri)?.use { input ->
                val raw = input.readBytes()
                if (isTargetFloat32Wav(raw)) {
                    return@withContext raw
                }
            }

            val extractor = MediaExtractor()
            extractor.setDataSource(context, uri, null)

            val trackIndex = findAudioTrack(extractor)
            if (trackIndex < 0) {
                Log.w(TAG, "No audio track found for URI: $uri")
                extractor.release()
                return@withContext null
            }

            extractor.selectTrack(trackIndex)
            val format = extractor.getTrackFormat(trackIndex)
            val mime = format.getString(android.media.MediaFormat.KEY_MIME)
            if (mime.isNullOrBlank()) {
                Log.w(TAG, "Missing MIME type for audio track")
                extractor.release()
                return@withContext null
            }

            val decoder = MediaCodec.createDecoderByType(mime)
            decoder.configure(format, null, null, 0)
            decoder.start()

            val monoBuilder = FloatBuilder()
            var sourceSampleRate = format.getIntegerOrDefault(android.media.MediaFormat.KEY_SAMPLE_RATE, TARGET_SAMPLE_RATE)
            var channelCount = format.getIntegerOrDefault(android.media.MediaFormat.KEY_CHANNEL_COUNT, TARGET_CHANNELS)
            var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT

            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false

            while (!outputDone) {
                if (!inputDone) {
                    val inIndex = decoder.dequeueInputBuffer(TIMEOUT_US)
                    if (inIndex >= 0) {
                        val inBuffer = decoder.getInputBuffer(inIndex)
                        if (inBuffer != null) {
                            val sampleSize = extractor.readSampleData(inBuffer, 0)
                            if (sampleSize < 0) {
                                decoder.queueInputBuffer(
                                    inIndex,
                                    0,
                                    0,
                                    0,
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM
                                )
                                inputDone = true
                            } else {
                                val presentationTimeUs = extractor.sampleTime
                                decoder.queueInputBuffer(
                                    inIndex,
                                    0,
                                    sampleSize,
                                    presentationTimeUs,
                                    0
                                )
                                extractor.advance()
                            }
                        }
                    }
                }

                val outIndex = decoder.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
                when {
                    outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val outFormat = decoder.outputFormat
                        sourceSampleRate = outFormat.getIntegerOrDefault(
                            android.media.MediaFormat.KEY_SAMPLE_RATE,
                            sourceSampleRate
                        )
                        channelCount = outFormat.getIntegerOrDefault(
                            android.media.MediaFormat.KEY_CHANNEL_COUNT,
                            channelCount
                        )
                        pcmEncoding = if (outFormat.containsKey(android.media.MediaFormat.KEY_PCM_ENCODING)) {
                            outFormat.getInteger(android.media.MediaFormat.KEY_PCM_ENCODING)
                        } else {
                            AudioFormat.ENCODING_PCM_16BIT
                        }
                    }

                    outIndex >= 0 -> {
                        if (bufferInfo.size > 0) {
                            val outBuffer = decoder.getOutputBuffer(outIndex)
                            if (outBuffer != null) {
                                val chunk = outBuffer.duplicate().apply {
                                    position(bufferInfo.offset)
                                    limit(bufferInfo.offset + bufferInfo.size)
                                }
                                appendMonoFloats(chunk, pcmEncoding, channelCount, monoBuilder)
                            }
                        }

                        decoder.releaseOutputBuffer(outIndex, false)

                        if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                            outputDone = true
                        }
                    }
                }
            }

            decoder.stop()
            decoder.release()
            extractor.release()

            val monoFloats = monoBuilder.toArray()
            if (monoFloats.isEmpty()) {
                Log.w(TAG, "Decoded audio is empty for URI: $uri")
                return@withContext null
            }

            val resampled = if (sourceSampleRate != TARGET_SAMPLE_RATE) {
                resampleLinear(monoFloats, sourceSampleRate, TARGET_SAMPLE_RATE)
            } else {
                monoFloats
            }

            return@withContext createFloat32Wav(resampled, TARGET_SAMPLE_RATE, TARGET_CHANNELS)
        } catch (e: Exception) {
            Log.e(TAG, "Failed converting URI to float32 WAV: ${e.message}", e)
            null
        }
    }

    private fun findAudioTrack(extractor: MediaExtractor): Int {
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val mime = format.getString(android.media.MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) return i
        }
        return -1
    }

    private fun appendMonoFloats(
        buffer: ByteBuffer,
        pcmEncoding: Int,
        channelCount: Int,
        out: FloatBuilder
    ) {
        val channels = channelCount.coerceAtLeast(1)

        when (pcmEncoding) {
            AudioFormat.ENCODING_PCM_FLOAT -> {
                val fb = buffer.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                val samples = FloatArray(fb.remaining())
                fb.get(samples)
                var i = 0
                while (i + channels - 1 < samples.size) {
                    var sum = 0f
                    for (c in 0 until channels) sum += samples[i + c]
                    out.add((sum / channels).coerceIn(-1f, 1f))
                    i += channels
                }
            }

            AudioFormat.ENCODING_PCM_8BIT -> {
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                var i = 0
                while (i + channels - 1 < bytes.size) {
                    var sum = 0f
                    for (c in 0 until channels) {
                        val v = ((bytes[i + c].toInt() and 0xFF) - 128) / 128f
                        sum += v
                    }
                    out.add((sum / channels).coerceIn(-1f, 1f))
                    i += channels
                }
            }

            else -> {
                // Default/fallback: 16-bit PCM.
                val sb = buffer.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                val samples = ShortArray(sb.remaining())
                sb.get(samples)
                var i = 0
                while (i + channels - 1 < samples.size) {
                    var sum = 0f
                    for (c in 0 until channels) {
                        sum += samples[i + c] / 32768f
                    }
                    out.add((sum / channels).coerceIn(-1f, 1f))
                    i += channels
                }
            }
        }
    }

    private fun resampleLinear(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        if (input.isEmpty() || fromRate <= 0 || toRate <= 0 || fromRate == toRate) return input

        val outLen = ((input.size.toDouble() * toRate) / fromRate).toInt().coerceAtLeast(1)
        val out = FloatArray(outLen)
        val ratio = fromRate.toDouble() / toRate

        for (i in out.indices) {
            val srcPos = i * ratio
            val i0 = srcPos.toInt().coerceIn(0, input.lastIndex)
            val i1 = (i0 + 1).coerceAtMost(input.lastIndex)
            val frac = (srcPos - i0)
            out[i] = ((1.0 - frac) * input[i0] + frac * input[i1]).toFloat()
        }

        return out
    }

    private fun createFloat32Wav(samples: FloatArray, sampleRate: Int, channels: Int): ByteArray {
        val audioDataSize = samples.size * 4
        val header = ByteArray(44)
        val hb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)

        hb.put("RIFF".toByteArray())
        hb.putInt(36 + audioDataSize)
        hb.put("WAVE".toByteArray())

        hb.put("fmt ".toByteArray())
        hb.putInt(16)
        hb.putShort(3) // IEEE float
        hb.putShort(channels.toShort())
        hb.putInt(sampleRate)
        hb.putInt(sampleRate * channels * 4)
        hb.putShort((channels * 4).toShort())
        hb.putShort(32)

        hb.put("data".toByteArray())
        hb.putInt(audioDataSize)

        val audioBytes = ByteArray(audioDataSize)
        val ab = ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN)
        for (s in samples) ab.putFloat(s.coerceIn(-1f, 1f))

        return header + audioBytes
    }

    private fun isTargetFloat32Wav(bytes: ByteArray): Boolean {
        if (bytes.size < 44) return false
        if (!(bytes[0] == 'R'.code.toByte() && bytes[1] == 'I'.code.toByte() && bytes[2] == 'F'.code.toByte() && bytes[3] == 'F'.code.toByte())) return false
        if (!(bytes[8] == 'W'.code.toByte() && bytes[9] == 'A'.code.toByte() && bytes[10] == 'V'.code.toByte() && bytes[11] == 'E'.code.toByte())) return false

        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val audioFormat = bb.getShort(20).toInt() and 0xFFFF
        val channels = bb.getShort(22).toInt() and 0xFFFF
        val sampleRate = bb.getInt(24)
        val bitsPerSample = bb.getShort(34).toInt() and 0xFFFF

        return audioFormat == 3 && channels == 1 && sampleRate == TARGET_SAMPLE_RATE && bitsPerSample == 32
    }

    fun float32WavToPcm16Wav(float32Wav: ByteArray): ByteArray {
        if (float32Wav.size < 44) return float32Wav
        val bb = ByteBuffer.wrap(float32Wav).order(ByteOrder.LITTLE_ENDIAN)
        // Check RIFF/WAVE magic
        val riff = String(float32Wav, 0, 4)
        val wave = String(float32Wav, 8, 4)
        if (riff != "RIFF" || wave != "WAVE") return float32Wav // not a WAV
        val audioFormat = bb.getShort(20).toInt() and 0xFFFF
        if (audioFormat != 3) return float32Wav // already PCM or other; pass through
        val channels = bb.getShort(22).toInt() and 0xFFFF
        val sampleRate = bb.getInt(24)
        val dataSize = bb.getInt(40)
        val numSamples = dataSize / 4
        // Convert float32 samples -> int16 samples
        val pcm16Bytes = ByteArray(numSamples * 2)
        val pcm16Buf = ByteBuffer.wrap(pcm16Bytes).order(ByteOrder.LITTLE_ENDIAN)
        bb.position(44)
        for (i in 0 until numSamples) {
            val f = bb.getFloat().coerceIn(-1f, 1f)
            pcm16Buf.putShort((f * 32767f).toInt().toShort())
        }
        // Build PCM16 WAV header
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray())
        header.putInt(36 + pcm16Bytes.size)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16)
        header.putShort(1)                              // PCM
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(sampleRate * channels * 2)        // ByteRate
        header.putShort((channels * 2).toShort())       // BlockAlign
        header.putShort(16)                             // BitsPerSample
        header.put("data".toByteArray())
        header.putInt(pcm16Bytes.size)
        return header.array() + pcm16Bytes
    }

    private fun android.media.MediaFormat.getIntegerOrDefault(key: String, defaultValue: Int): Int {
        return if (containsKey(key)) getInteger(key) else defaultValue
    }

    private class FloatBuilder(initialCapacity: Int = 16_384) {
        private var data = FloatArray(initialCapacity)
        private var size = 0

        fun add(value: Float) {
            if (size == data.size) {
                data = data.copyOf((data.size * 2).coerceAtLeast(1))
            }
            data[size] = value
            size++
        }

        fun toArray(): FloatArray = data.copyOf(size)
    }
}
