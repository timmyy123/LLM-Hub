package com.llmhub.llmhub.videogen

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File

/** Encodes packed RGB frames into an H.264 MP4 using the platform encoder. */
internal object Mp4Writer {
    private const val MIME = MediaFormat.MIMETYPE_VIDEO_AVC
    private const val TIMEOUT_US = 10_000L

    fun write(frames: List<ByteArray>, width: Int, height: Int, fps: Int, output: File) {
        require(frames.isNotEmpty()) { "no frames to encode" }
        val format = MediaFormat.createVideoFormat(MIME, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            // ~1 bit per pixel per frame: clips are short, so favour fidelity over file size.
            setInteger(MediaFormat.KEY_BIT_RATE, (width * height * fps * 1.0f).toInt().coerceAtLeast(2_000_000))
            setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val codec = MediaCodec.createEncoderByType(MIME)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
        output.parentFile?.mkdirs()
        val muxer = MediaMuxer(output.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var track = -1
        var muxerStarted = false
        val info = MediaCodec.BufferInfo()
        var nextFrame = 0
        var inputDone = false
        try {
            while (true) {
                if (!inputDone) {
                    val inIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (inIndex >= 0) {
                        val ptsUs = nextFrame * 1_000_000L / fps
                        if (nextFrame < frames.size) {
                            val image = codec.getInputImage(inIndex) ?: error("encoder has no input image")
                            fillYuv(image, frames[nextFrame], width, height)
                            codec.queueInputBuffer(inIndex, 0, width * height * 3 / 2, ptsUs, 0)
                            nextFrame++
                        } else {
                            codec.queueInputBuffer(inIndex, 0, 0, ptsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        }
                    }
                }
                val outIndex = codec.dequeueOutputBuffer(info, TIMEOUT_US)
                when {
                    outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        track = muxer.addTrack(codec.outputFormat)
                        muxer.start()
                        muxerStarted = true
                    }
                    outIndex >= 0 -> {
                        val buffer = codec.getOutputBuffer(outIndex)
                        if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                            info.size = 0
                        }
                        if (info.size > 0 && buffer != null && muxerStarted) {
                            buffer.position(info.offset)
                            buffer.limit(info.offset + info.size)
                            muxer.writeSampleData(track, buffer, info)
                        }
                        codec.releaseOutputBuffer(outIndex, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            break
                        }
                    }
                }
            }
        } finally {
            codec.stop()
            codec.release()
            if (muxerStarted) {
                muxer.stop()
            }
            muxer.release()
        }
    }

    fun toBitmap(rgb: ByteArray, width: Int, height: Int): Bitmap {
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val r = rgb[i * 3].toInt() and 0xFF
            val g = rgb[i * 3 + 1].toInt() and 0xFF
            val b = rgb[i * 3 + 2].toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    fun bitmapToRgb(bitmap: Bitmap): ByteArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val out = ByteArray(pixels.size * 3)
        for (i in pixels.indices) {
            val p = pixels[i]
            out[i * 3] = (p shr 16 and 0xFF).toByte()
            out[i * 3 + 1] = (p shr 8 and 0xFF).toByte()
            out[i * 3 + 2] = (p and 0xFF).toByte()
        }
        return out
    }

    // BT.601 limited-range RGB -> YUV420 written through the Image plane strides, so it works for
    // both planar and semi-planar encoder layouts.
    private fun fillYuv(image: android.media.Image, rgb: ByteArray, width: Int, height: Int) {
        val planes = image.planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        for (y in 0 until height) {
            for (x in 0 until width) {
                val i = (y * width + x) * 3
                val r = rgb[i].toInt() and 0xFF
                val g = rgb[i + 1].toInt() and 0xFF
                val b = rgb[i + 2].toInt() and 0xFF
                val luma = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                yBuf.put(y * yPlane.rowStride + x * yPlane.pixelStride, luma.coerceIn(0, 255).toByte())
                if (y % 2 == 0 && x % 2 == 0) {
                    val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                    val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                    val cy = y / 2
                    val cx = x / 2
                    uBuf.put(cy * uPlane.rowStride + cx * uPlane.pixelStride, u.coerceIn(0, 255).toByte())
                    vBuf.put(cy * vPlane.rowStride + cx * vPlane.pixelStride, v.coerceIn(0, 255).toByte())
                }
            }
        }
    }
}
