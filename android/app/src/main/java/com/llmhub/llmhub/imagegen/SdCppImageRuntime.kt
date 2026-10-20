package com.llmhub.llmhub.imagegen

import android.graphics.Bitmap
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.videogen.SdCppNative
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder

internal object SdCppImageNative {
    external fun create(checkpoint: String, options: String, gpu: Boolean): Long
    external fun generate(handle: Long, prompt: String, options: String, width: Int, height: Int,
                          steps: Int, seed: Long, strength: Float, input: ByteArray?): Array<ByteArray>?
    external fun close(handle: Long)
}

internal class SdCppImageRuntime {
    companion object {
        /**
         * Imported checkpoints run on the CPU where ggml's Vulkan backend is unusable, as video GGUF models do:
         * Adreno drivers fail its shaders, and the Tensor G5's PowerVR driver produces noise.
         */
        fun usesGpu(requested: Boolean) = requested && !DeviceInfo.isSnapdragon() && !DeviceInfo.isTensorG5()
    }
    private var handle = 0L
    fun load(path: String, options: ImageGenerationOptions, gpu: Boolean) {
        SdCppNative.ensureLoaded()
        handle = SdCppImageNative.create(path, options.json().toString(), usesGpu(gpu))
        check(handle != 0L)
    }
    fun generate(prompt: String, options: ImageGenerationOptions, width: Int, height: Int,
                 steps: Int, seed: Long, input: Bitmap?, strength: Float): Bitmap? {
        val rgb = input?.let {
            val bitmap = Bitmap.createScaledBitmap(it.copy(Bitmap.Config.ARGB_8888, false), width, height, true)
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            ByteArray(pixels.size * 3).also { data ->
                pixels.forEachIndexed { index, pixel ->
                    data[index * 3] = (pixel shr 16).toByte(); data[index * 3 + 1] = (pixel shr 8).toByte(); data[index * 3 + 2] = pixel.toByte()
                }
            }
        }
        val output = SdCppImageNative.generate(handle, options.translatedPrompt(prompt), options.json().toString(),
            width, height, steps, seed, strength, rgb) ?: return null
        val dimensions = ByteBuffer.wrap(output[0]).order(ByteOrder.LITTLE_ENDIAN)
        val w = dimensions.int; val h = dimensions.int
        val data = output[1]
        val pixels = IntArray(w * h) { index ->
            (0xff shl 24) or ((data[index * 3].toInt() and 255) shl 16) or
                ((data[index * 3 + 1].toInt() and 255) shl 8) or (data[index * 3 + 2].toInt() and 255)
        }
        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
    }
    fun close() {
        val previous = handle
        handle = 0
        if (previous != 0L) CoroutineScope(Dispatchers.IO).launch { SdCppImageNative.close(previous) }
    }
}
