package com.llmhub.llmhub.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.util.UUID

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
    private val customKey = Regex("custom_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    fun isVoiceKey(key: String) = key in voices || customKey.matches(key)
    private fun customDirectory(context: Context) = File(context.filesDir, "supertonic-voices").apply { mkdirs() }
    fun voiceFile(context: Context, key: String): File? {
        val file = when {
            key in voices -> File(directory(context), "$key.json")
            customKey.matches(key) -> File(customDirectory(context), "$key.json")
            else -> return null
        }
        return file.takeIf { it.isFile && it.length() in 1..MAX_VOICE_BYTES.toLong() }
    }
    fun customVoices(context: Context): List<Pair<String, String>> = customDirectory(context).listFiles()
        ?.filter { it.isFile && it.extension == "json" && customKey.matches(it.nameWithoutExtension) && it.length() in 1..MAX_VOICE_BYTES.toLong() }
        ?.sortedBy { it.name }
        ?.map { it.nameWithoutExtension to voiceLabel(context, it.nameWithoutExtension) } ?: emptyList()
    fun voiceLabel(context: Context, key: String): String {
        if (!customKey.matches(key)) return key
        return File(customDirectory(context), "$key.txt").takeIf { it.isFile }?.readText()?.take(80) ?: key
    }
    /** Decode only a bounded stream, validate before publishing, and never use source filenames as paths. */
    fun importVoice(context: Context, input: InputStream, label: String): String =
        importVoice(customDirectory(context), input, label)

    internal fun importVoice(directory: File, input: InputStream, label: String): String {
        val output = java.io.ByteArrayOutputStream()
        val buffer = ByteArray(8192)
        while (output.size() <= MAX_VOICE_BYTES) {
            val read = input.read(buffer, 0, minOf(buffer.size, MAX_VOICE_BYTES + 1 - output.size()))
            if (read < 0) break
            if (read > 0) output.write(buffer, 0, read)
        }
        val bytes = output.toByteArray()
        require(bytes.size <= MAX_VOICE_BYTES)
        val json = bytes.toString(Charsets.UTF_8)
        parseVoice(json)
        val key = "custom_${UUID.randomUUID()}"
        val partial = File(directory, "$key.part")
        val target = File(directory, "$key.json")
        val name = File(directory, "$key.txt")
        try {
            partial.writeBytes(bytes)
            name.writeText(label.take(80))
            check(partial.renameTo(target))
        } catch (e: Exception) {
            partial.delete(); target.delete(); name.delete()
            throw e
        }
        return key
    }
    fun deleteVoice(context: Context, key: String) {
        voiceFile(context, key)?.delete()
        if (customKey.matches(key)) File(customDirectory(context), "$key.txt").delete()
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
