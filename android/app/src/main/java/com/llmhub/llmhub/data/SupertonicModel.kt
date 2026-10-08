package com.llmhub.llmhub.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Pinned public Supertonic 3 assets; sizes verified using curl -I -L. */
object SupertonicModel {
    const val NAME = "Supertonic 3 (ONNX)"
    const val REVISION = "3cadd1ee6394adea1bd021217a0e650ede09a323"
    const val BASE_URL = "https://huggingface.co/Supertone/supertonic-3/resolve/$REVISION"
    val coreFiles = linkedMapOf(
        "duration_predictor.onnx" to 3700147L,
        "text_encoder.onnx" to 36416150L,
        "vector_estimator.onnx" to 256534781L,
        "vocoder.onnx" to 101424195L,
        "tts.json" to 8253L,
        "unicode_indexer.json" to 277676L
    )
    val voices = linkedMapOf(
        "F1" to 292046L, "F2" to 292423L, "F3" to 290794L,
        "F4" to 291808L, "F5" to 291479L, "M1" to 291748L,
        "M2" to 292055L, "M3" to 290198L, "M4" to 291522L, "M5" to 291469L
    )
    val languages = listOf("en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi")
    fun directory(context: Context) = File(context.filesDir, "models/Supertonic_3_ONNX")
    fun isComplete(context: Context) = coreFiles.all { (name, size) ->
        File(directory(context), name).let { it.isFile && it.length() == size }
    }
    fun voiceFile(context: Context, key: String): File? {
        if (key !in voices) return null
        val file = File(directory(context), "$key.json")
        return file.takeIf { it.isFile && it.length() in 1..MAX_VOICE_BYTES.toLong() }
    }
    const val MAX_VOICE_BYTES = 2 * 1024 * 1024

    data class Voice(val ttl: FloatArray, val dp: FloatArray)
    fun parseVoice(json: String): Voice {
        require(json.toByteArray(Charsets.UTF_8).size <= MAX_VOICE_BYTES)
        val root = JSONObject(json)
        fun tensor(name: String, expected: List<Int>): FloatArray {
            val obj = root.getJSONObject(name)
            val dims = obj.getJSONArray("dims")
            require(dims.length() == expected.size && expected.indices.all { dims.getInt(it) == expected[it] })
            val values = ArrayList<Float>()
            fun flatten(array: JSONArray, depth: Int = 1) {
                require(depth <= 3)
                for (i in 0 until array.length()) {
                    val value = array.get(i)
                    if (value is JSONArray) flatten(value, depth + 1) else {
                        require(value is Number)
                        val f = value.toFloat()
                        require(f.isFinite())
                        values.add(f)
                        require(values.size <= expected.reduce(Int::times))
                    }
                }
            }
            flatten(obj.getJSONArray("data"))
            require(values.size == expected.reduce(Int::times))
            return values.toFloatArray()
        }
        return Voice(tensor("style_ttl", listOf(1, 50, 256)), tensor("style_dp", listOf(1, 8, 16)))
    }
}
