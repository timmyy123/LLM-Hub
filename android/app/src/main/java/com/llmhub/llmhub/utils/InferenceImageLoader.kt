package com.llmhub.llmhub.utils

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.core.graphics.drawable.toBitmap
import coil.imageLoader
import coil.request.ImageRequest
import coil.request.SuccessResult
import coil.size.Scale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Decode local image attachments at a bounded size for vision inference. */
suspend fun loadInferenceBitmap(context: Context, uri: Uri): Bitmap? = withContext(Dispatchers.IO) {
    if (uri.scheme !in setOf("content", "file", "android.resource")) return@withContext null

    val request = ImageRequest.Builder(context)
        .data(uri)
        .size(2048, 2048)
        .scale(Scale.FIT)
        .allowHardware(false)
        .build()
    val result = context.imageLoader.execute(request) as? SuccessResult
    result?.drawable?.toBitmap()
}
