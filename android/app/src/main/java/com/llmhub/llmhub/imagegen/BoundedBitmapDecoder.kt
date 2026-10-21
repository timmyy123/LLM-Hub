package com.llmhub.llmhub.imagegen

import android.graphics.Bitmap
import android.graphics.BitmapFactory

/**
 * BitmapFactory with downsampling: reads the image size first and sets
 * [BitmapFactory.Options.inSampleSize] so neither side exceeds [maxSide]. Generated images
 * (512 to 1024 px) decode at full resolution; only unexpectedly large files are reduced.
 */
internal object BoundedBitmapDecoder {
    const val MAX_SIDE = 2048

    fun decodeFile(path: String, maxSide: Int = MAX_SIDE): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        return BitmapFactory.decodeFile(path, sampled(bounds, maxSide))
    }

    fun decodeByteArray(data: ByteArray, maxSide: Int = MAX_SIDE): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(data, 0, data.size, bounds)
        return BitmapFactory.decodeByteArray(data, 0, data.size, sampled(bounds, maxSide))
    }

    private fun sampled(bounds: BitmapFactory.Options, maxSide: Int) = BitmapFactory.Options().apply {
        var sample = 1
        while (bounds.outWidth / sample > maxSide || bounds.outHeight / sample > maxSide) sample *= 2
        inSampleSize = sample
    }
}
