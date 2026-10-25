package com.llmhub.llmhub.imagegen

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build

/**
 * BitmapFactory with downsampling: reads the image size first and sets
 * [BitmapFactory.Options.inSampleSize] so neither side exceeds [maxSide]. Generated images
 * (512 to 1024 px) decode at full resolution; only unexpectedly large files are reduced.
 */
internal object BoundedBitmapDecoder {
    const val MAX_SIDE = 2048
    const val PICKED_MAX_SIDE = 4096

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

    /**
     * A picked photo as a software ARGB bitmap with neither side above [maxSide]. Full-resolution
     * camera photos (50 MP is about 200 MB) exceed the 100 MB a Canvas can draw.
     */
    fun decodeUri(context: Context, uri: Uri, maxSide: Int = PICKED_MAX_SIDE): Bitmap? {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            return ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val longest = maxOf(info.size.width, info.size.height)
                if (longest > maxSide) {
                    val scale = maxSide.toFloat() / longest
                    decoder.setTargetSize(
                        (info.size.width * scale).toInt().coerceAtLeast(1),
                        (info.size.height * scale).toInt().coerceAtLeast(1),
                    )
                }
            }
        }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        return context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, sampled(bounds, maxSide)) }
    }

    private fun sampled(bounds: BitmapFactory.Options, maxSide: Int) = BitmapFactory.Options().apply {
        var sample = 1
        while (bounds.outWidth / sample > maxSide || bounds.outHeight / sample > maxSide) sample *= 2
        inSampleSize = sample
    }
}
