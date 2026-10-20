package com.llmhub.llmhub.imagegen

import android.util.Log
import java.io.DataInputStream
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * Fixture preparation on the phone itself, for devices without a Device Farm host (e.g. Samsung Remote Test Lab or
 * BrowserStack). Mirrors tools/imagegen_devicefarm/prepare_fixtures.py for the SD3 suite.
 */
internal object OnDeviceFixtures {
    private const val SD3_BASE =
        "https://huggingface.co/second-state/stable-diffusion-3.5-medium-GGUF/resolve/58b78c305a43ddfcffe1ab54d7022995f61667ac/"
    private val SD3 = mapOf(
        "sd3.gguf" to ("sd3.5_medium-Q4_0.gguf" to "3bb8c5e9ab0a841117089ed4ed81d885bb85161df2a766b812f829bc55b31adf"),
        "clip-l.safetensors" to ("clip_l.safetensors" to "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"),
        "clip-g.safetensors" to ("clip_g.safetensors" to "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4"),
        "t5.gguf" to ("t5xxl-Q4_0.gguf" to "987ba47c158b890c274f78fd35324419f50941e846a49789f0977e9fe9d97ab7"),
    )

    private const val NEGATED_T5_SHA256 = "102328eabfae53cd01c73510d7c68cb0fedd6c50fc80b23012ee424d9d34c964"

    fun prepareSd3(directory: File) {
        directory.mkdirs()
        for ((name, source) in SD3) download(File(directory, name), SD3_BASE + source.first, source.second)
        val altered = File(directory, "t5-negated-norm.gguf")
        if (!altered.isFile) {
            val temporary = File(directory, "t5-negated-norm.part")
            File(directory, "t5.gguf").copyTo(temporary, overwrite = true)
            negateQ4Tensor(temporary, "encoder.final_layer_norm.weight")
            // Same bytes as the host-derived Device Farm fixture.
            check(sha256(temporary) == NEGATED_T5_SHA256) { "Derived T5 does not match the host-derived fixture" }
            temporary.renameTo(altered)
        }
    }

    fun download(target: File, url: String, sha256: String) {
        if (target.isFile && sha256(target) == sha256) return
        val partial = File(target.path + ".part")
        val connection = URL(url).openConnection() as HttpURLConnection
        if (partial.length() > 0) connection.setRequestProperty("Range", "bytes=${partial.length()}-")
        connection.instanceFollowRedirects = true
        val append = connection.responseCode == 206
        check(connection.responseCode in 200..299) { "HTTP ${connection.responseCode} for $url" }
        connection.inputStream.use { input -> java.io.FileOutputStream(partial, append).use { input.copyTo(it, 1 shl 20) } }
        check(sha256(partial) == sha256) { "SHA-256 mismatch for ${target.name}" }
        partial.renameTo(target)
        Log.i("OnDeviceFixtures", "Downloaded ${target.name} (${target.length()} bytes)")
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1 shl 20)
            while (true) { val read = input.read(buffer); if (read < 0) break; digest.update(buffer, 0, read) }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    /** Flips the sign bit of every block scale of a Q4_0 tensor, negating it, as prepare_fixtures.py does. */
    private fun negateQ4Tensor(file: File, tensor: String) {
        val (type, offset, elements) = ggufTensor(file, tensor)
        check(type == 2L && elements % 32 == 0L) { "Expected a Q4_0 $tensor, found type $type" }
        RandomAccessFile(file, "rw").use { raf ->
            for (block in 0 until elements / 32) {
                val position = offset + block * 18 + 1 // High byte of the block's F16 scale.
                raf.seek(position)
                val high = raf.read()
                raf.seek(position)
                raf.write(high xor 0x80)
            }
        }
    }

    /** (type, absolute data offset, element count) of one tensor in a GGUF v3 file. */
    private fun ggufTensor(file: File, wanted: String): Triple<Long, Long, Long> {
        val input = DataInputStream(file.inputStream().buffered(1 shl 20))
        var position = 0L
        fun bytes(count: Int): ByteBuffer {
            val data = ByteArray(count); input.readFully(data); position += count
            return ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        }
        fun u32() = bytes(4).int.toLong() and 0xffffffffL
        fun u64() = bytes(8).long
        fun string() = String(bytes(u64().toInt()).array(), Charsets.UTF_8)
        val scalarSizes = mapOf(0 to 1, 1 to 1, 2 to 2, 3 to 2, 4 to 4, 5 to 4, 6 to 4, 7 to 1, 10 to 8, 11 to 8, 12 to 8)
        var alignment = 32L
        fun value(type: Int): Any? = when (type) {
            8 -> string()
            9 -> { val itemType = u32().toInt(); repeat(u64().toInt()) { value(itemType) }; null }
            4 -> u32()
            else -> { bytes(scalarSizes.getValue(type)); null }
        }
        input.use {
            check(String(bytes(4).array()) == "GGUF" && u32() == 3L) { "Expected GGUF v3" }
            val tensorCount = u64()
            val kvCount = u64()
            repeat(kvCount.toInt()) {
                val key = string()
                val v = value(u32().toInt())
                if (key == "general.alignment" && v is Long) alignment = v
            }
            var found: Triple<Long, Long, Long>? = null
            repeat(tensorCount.toInt()) {
                val name = string()
                var elements = 1L
                repeat(u32().toInt()) { elements *= u64() }
                val type = u32()
                val offset = u64()
                if (name == wanted) found = Triple(type, offset, elements)
            }
            val dataStart = position + (alignment - position % alignment) % alignment
            val (type, offset, elements) = found ?: error("Missing GGUF tensor $wanted")
            return Triple(type, dataStart + offset, elements)
        }
    }
}
