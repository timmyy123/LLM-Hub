package com.llmhub.llmhub.ui.components

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.provider.OpenableColumns
import com.llmhub.llmhub.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

/** File names and provider MIME types are labels, never evidence of audio format. */
internal object PocketVoiceImport {
    const val MAX_INPUT_BYTES = 20L * 1024 * 1024

    suspend fun import(context: Context, uri: Uri, directory: File, track: (File) -> Unit): Pair<File, String> =
        withContext(Dispatchers.IO) {
            val name = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
                if (it.moveToFirst()) it.getString(0) else null
            }?.take(80) ?: context.getString(R.string.tts_voice_setting)
            val source = File.createTempFile("pocket-reference-", ".input", context.cacheDir)
            try {
                val input = context.contentResolver.openInputStream(uri) ?: throw IOException("Cannot open reference audio")
                input.use { stream ->
                    source.outputStream().use { target ->
                        val buffer = ByteArray(16384)
                        var total = 0L
                        while (true) {
                            coroutineContext.ensureActive()
                            val read = stream.read(buffer)
                            if (read < 0) break
                            total += read
                            require(total <= MAX_INPUT_BYTES) { "Reference audio exceeds 20 MiB" }
                            target.write(buffer, 0, read)
                        }
                    }
                }
                val worker = coroutineContext
                val output = prepare(source, directory, track) { worker.ensureActive() }
                output to name.substringBeforeLast('.', name).take(80)
            } finally { source.delete() }
        }

    /** Retain native-supported audio; decode other Android-supported formats to mono WAV. */
    internal fun prepare(source: File, directory: File, track: (File) -> Unit, checkCancelled: () -> Unit = {}): File {
        checkCancelled()
        val detectedFormat = nativeFormat(source)
        val extension = detectedFormat ?: "wav"
        val output = File(directory, "${UUID.randomUUID()}.$extension")
        track(output)
        try {
            if (detectedFormat != null) source.inputStream().use { input ->
                output.outputStream().use { target ->
                    val buffer = ByteArray(16384)
                    while (true) {
                        checkCancelled()
                        val read = input.read(buffer)
                        if (read < 0) break
                        target.write(buffer, 0, read)
                    }
                }
            } else decodeToWav(source, output, checkCancelled)
            checkCancelled()
            return output
        } catch (e: Exception) { output.delete(); throw e }
    }

    internal fun nativeFormat(source: File): String? {
        val header = ByteArray(12)
        val count = source.inputStream().use { it.read(header) }
        fun tag(offset: Int, value: String) = count >= offset + value.length &&
            value.indices.all { header[offset + it] == value[it].code.toByte() }
        return when {
            (tag(0, "RIFF") || tag(0, "RF64")) && tag(8, "WAVE") -> "wav"
            tag(0, "fLaC") -> "flac"
            tag(0, "ID3") -> "mp3"
            // MPEG audio sync, with valid version and layer bits. AAC/ADTS has layer 00.
            count >= 3 && (header[0].toInt() and 255) == 255 &&
                (header[1].toInt() and 224) == 224 && (header[1].toInt() and 24) != 8 &&
                (header[1].toInt() and 6) != 0 && (header[2].toInt() and 240) !in listOf(0, 240) -> "mp3"
            else -> null
        }
    }

    private fun decodeToWav(source: File, output: File, checkCancelled: () -> Unit) {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        var started = false
        try {
            extractor.setDataSource(source.absolutePath)
            val audioTrack = (0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: throw IOException("No supported audio track in reference")
            extractor.selectTrack(audioTrack)
            val inputFormat = extractor.getTrackFormat(audioTrack)
            val mime = checkNotNull(inputFormat.getString(MediaFormat.KEY_MIME))
            var rate = 0
            var channels = 0
            var encoding = AudioFormat.ENCODING_PCM_16BIT
            var frames = 0
            var mono = FloatArray(0)
            fun format(format: MediaFormat) {
                val nextRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                val nextChannels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                val nextEncoding = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING))
                    format.getInteger(MediaFormat.KEY_PCM_ENCODING) else AudioFormat.ENCODING_PCM_16BIT
                require(nextRate in 8000..192000 && nextChannels in 1..8) { "Unsupported audio sample format" }
                require(nextEncoding == AudioFormat.ENCODING_PCM_16BIT || nextEncoding == AudioFormat.ENCODING_PCM_FLOAT) {
                    "Unsupported decoded PCM encoding: $nextEncoding"
                }
                require(frames == 0 || (nextRate == rate && nextChannels == channels && nextEncoding == encoding))
                rate = nextRate; channels = nextChannels; encoding = nextEncoding
                if (frames == 0) mono = FloatArray(rate * 30)
            }
            fun append(data: ByteBuffer) {
                data.order(ByteOrder.LITTLE_ENDIAN)
                val bytesPerSample = if (encoding == AudioFormat.ENCODING_PCM_FLOAT) 4 else 2
                require(data.remaining() % (channels * bytesPerSample) == 0) { "Incomplete decoded PCM frame" }
                while (data.remaining() >= channels * bytesPerSample && frames < mono.size) {
                    var sum = 0f
                    repeat(channels) {
                        val sample = if (encoding == AudioFormat.ENCODING_PCM_FLOAT) data.float else data.short / 32768f
                        require(sample.isFinite())
                        sum += sample
                    }
                    mono[frames++] = sum / channels
                }
            }
            if (mime == "audio/raw") {
                format(inputFormat)
                val buffer = ByteBuffer.allocate(1024 * 1024)
                while (frames < mono.size) {
                    checkCancelled()
                    buffer.clear()
                    val size = extractor.readSampleData(buffer, 0)
                    if (size < 0) break
                    buffer.position(0); buffer.limit(size)
                    append(buffer)
                    extractor.advance()
                }
            } else {
                val decoder = MediaCodec.createDecoderByType(mime)
                codec = decoder
                inputFormat.setInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
                decoder.configure(inputFormat, null, null, 0)
                decoder.start(); started = true
                val info = MediaCodec.BufferInfo()
                var inputDone = false
                var outputDone = false
                val deadline = android.os.SystemClock.elapsedRealtime() + 60000L
                while (!outputDone && (rate == 0 || frames < mono.size)) {
                    checkCancelled()
                    check(android.os.SystemClock.elapsedRealtime() < deadline) { "Reference decoding timed out" }
                    if (!inputDone) {
                        val index = decoder.dequeueInputBuffer(10000)
                        if (index >= 0) {
                            val input = checkNotNull(decoder.getInputBuffer(index))
                            input.clear()
                            val size = extractor.readSampleData(input, 0)
                            if (size < 0) {
                                decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                require(extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_ENCRYPTED == 0) { "Encrypted reference audio" }
                                decoder.queueInputBuffer(index, 0, size, extractor.sampleTime, 0)
                                extractor.advance()
                            }
                        }
                    }
                    val index = decoder.dequeueOutputBuffer(info, 10000)
                    if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) format(decoder.outputFormat)
                    else if (index >= 0) {
                        try {
                            if (info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                                if (rate == 0) format(decoder.outputFormat)
                                val data = checkNotNull(decoder.getOutputBuffer(index))
                                data.position(info.offset); data.limit(info.offset + info.size)
                                append(data)
                            }
                            outputDone = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        } finally { decoder.releaseOutputBuffer(index, false) }
                    }
                }
            }
            require(rate > 0 && frames >= rate * 3) { "Reference must contain at least 3 seconds of audio" }
            checkCancelled()
            writeWav(output, mono, frames, rate, checkCancelled)
        } finally {
            try { if (started) codec?.stop() } finally {
                try { codec?.release() } finally { extractor.release() }
            }
        }
    }

    private fun writeWav(file: File, samples: FloatArray, frames: Int, rate: Int, checkCancelled: () -> Unit) {
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + frames * 2); put("WAVEfmt ".toByteArray()); putInt(16)
            putShort(1); putShort(1); putInt(rate); putInt(rate * 2); putShort(2); putShort(16)
            put("data".toByteArray()); putInt(frames * 2)
        }
        file.outputStream().use { output ->
            output.write(header.array())
            val buffer = ByteBuffer.allocate(8192).order(ByteOrder.LITTLE_ENDIAN)
            for (frame in 0 until frames) {
                buffer.putShort((samples[frame].coerceIn(-1f, 1f) * 32767).toInt().toShort())
                if (!buffer.hasRemaining()) {
                    checkCancelled(); output.write(buffer.array()); buffer.clear()
                }
            }
            if (buffer.position() > 0) output.write(buffer.array(), 0, buffer.position())
        }
    }
}
