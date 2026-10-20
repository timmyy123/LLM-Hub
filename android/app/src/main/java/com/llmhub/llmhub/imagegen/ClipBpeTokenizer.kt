package com.llmhub.llmhub.imagegen

import org.json.JSONObject
import java.io.File
import java.text.Normalizer
import java.util.regex.Pattern

/**
 * CLIP byte-level BPE tokenizer for Stable Diffusion 1.x LiteRT text encoders ([LiteRtSdPipeline]).
 *
 * A port of litert_torch's Stable Diffusion example tokenizer, which produced the token IDs the LiteRT text encoders
 * were converted against: BOS, BPE tokens, EOS, truncated and padded with EOS to 77 tokens.
 */
internal class ClipBpeTokenizer(vocabFile: File, mergesFile: File) {
    private val vocab: Map<String, Int>
    private val ranks: Map<Pair<String, String>, Int>
    private val bytesTable: Array<String>
    private val cache = HashMap<String, List<String>>()
    private val bos: Int
    private val eos: Int

    init {
        val json = JSONObject(vocabFile.readText())
        vocab = json.keys().asSequence().associateWith { json.getInt(it) }
        // The first line is a version header and the file ends with a newline.
        val lines = mergesFile.readText().split("\n").drop(1).dropLast(1)
        ranks = lines.withIndex().associate { (rank, line) ->
            val parts = line.trim().split(Regex("\\s+"))
            (parts[0] to parts[1]) to rank
        }
        bos = vocab.getValue("<|startoftext|>")
        eos = vocab.getValue("<|endoftext|>")
        // Printable bytes map to themselves; control and space bytes map to code points from 256 up.
        var special = 0
        bytesTable = Array(256) { byte ->
            val type = Character.getType(byte)
            val isControlOrSpace = type == Character.CONTROL.toInt() || type == Character.FORMAT.toInt() ||
                type == Character.PRIVATE_USE.toInt() || type == Character.SURROGATE.toInt() ||
                type == Character.UNASSIGNED.toInt() || type == Character.SPACE_SEPARATOR.toInt() ||
                type == Character.LINE_SEPARATOR.toInt() || type == Character.PARAGRAPH_SEPARATOR.toInt()
            if (isControlOrSpace) String(Character.toChars(256 + special++)) else String(Character.toChars(byte))
        }
    }

    fun encode(text: String): IntArray {
        val normalized = WHITESPACE.matcher(Normalizer.normalize(text, Normalizer.Form.NFC))
            .replaceAll(" ").trim().lowercase()
        val tokens = mutableListOf(bos)
        val matcher = CHUNK.matcher(normalized)
        while (matcher.find()) {
            val chunk = matcher.group().toByteArray(Charsets.UTF_8).joinToString("") { bytesTable[it.toInt() and 0xff] }
            bpe(chunk).forEach { word -> vocab[word]?.let(tokens::add) }
        }
        tokens.add(eos)
        return IntArray(MAX_LENGTH) { index -> tokens.getOrElse(index) { eos } }
    }

    private fun bpe(chunk: String): List<String> = cache.getOrPut(chunk) {
        var words = chunk.map { it.toString() }.toMutableList()
        words[words.lastIndex] = words.last() + "</w>"
        while (words.size > 1) {
            val best = words.zipWithNext().filter { it in ranks }.minByOrNull { ranks.getValue(it) } ?: break
            val (first, second) = best
            val merged = mutableListOf<String>()
            for (word in words) {
                if (word == second && merged.isNotEmpty() && merged.last() == first) merged[merged.lastIndex] = first + second
                else merged.add(word)
            }
            words = merged
        }
        words
    }

    companion object {
        const val MAX_LENGTH = 77
        // Android's ICU regex treats \s and \p{L} as Unicode and rejects UNICODE_CHARACTER_CLASS.
        private val WHITESPACE = Pattern.compile("\\s+")
        private val CHUNK = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE
        )
    }
}
