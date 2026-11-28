package com.llmhub.llmhub.inference

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonToken
import java.io.File
import java.io.Reader
import java.text.Normalizer
import java.util.PriorityQueue
import java.util.concurrent.ConcurrentHashMap
import java.util.regex.Pattern

/**
 * HuggingFace `tokenizer.json` BPE tokenizer for the ONNX chat engine, written to give the same ids as the
 * Python/Rust `tokenizers` library for the files the catalog ships (LFM-2.5 and Ministral-3).
 *
 * Pipeline, as in `tokenizers`: added tokens are cut out of the text first (leftmost-longest, non-normalized
 * ones on the raw text, normalized ones after the normalizer), the rest goes through the normalizer, the
 * pre-tokenizer (Split regex / ByteLevel / Metaspace) and the BPE model with ranked merges. Decoding runs the
 * file's decoder chain (ByteLevel, ByteFallback, Metaspace, Fuse, Strip, Replace).
 *
 * [encode] never adds BOS/EOS itself: the chat prompt already carries them as literal special tokens.
 * Pure JVM code (no android.*) so it runs in local unit tests.
 */
class OnnxTokenizer(
    tokenizerJson: Reader,
    tokenizerConfigJson: String? = null,
    generationConfigJson: String? = null,
) {
    /** Reads `tokenizer.json` plus `tokenizer_config.json` / `generation_config.json` next to it, if present. */
    constructor(tokenizerFile: File) : this(
        tokenizerFile.bufferedReader(Charsets.UTF_8),
        siblingText(tokenizerFile, "tokenizer_config.json"),
        siblingText(tokenizerFile, "generation_config.json"),
    )

    private class AddedToken(
        val id: Int,
        val content: String,
        val special: Boolean,
        val normalized: Boolean,
        val lstrip: Boolean,
        val rstrip: Boolean,
        val singleWord: Boolean,
    )

    private val vocab = HashMap<String, Int>(1 shl 17)
    private var idToToken: Array<String?> = emptyArray()
    private var merges = LongLongMap(16)
    private var byteFallback = false
    private var ignoreMerges = false
    private var fuseUnk = false
    private var unkToken: String? = null
    private var continuingSubwordPrefix: String? = null
    private var endOfWordSuffix: String? = null

    private val addedById = HashMap<Int, AddedToken>()
    private val addedByContent = HashMap<String, AddedToken>()
    private var rawAddedMatcher: AddedTokenMatcher? = null
    private var normalizedAddedMatcher: AddedTokenMatcher? = null

    private var normalizer: ((String) -> String)? = null
    private var preTokenizers: List<PreTokenizer> = emptyList()
    private var decoders: List<TokenDecoder>? = null

    private val wordCache = ConcurrentHashMap<String, IntArray>()

    private val eosTokenId: Int
    private val eosTokenIds: Set<Long>

    /** BOS text from tokenizer_config.json (e.g. `<|startoftext|>` for LFM, `<s>` for Mistral), if any. */
    val bosToken: String?

    /** EOS text from tokenizer_config.json, if any. */
    val eosToken: String?

    init {
        var modelType: String? = null
        var mergePairsRaw: ArrayList<String>? = null
        var addedJson: JsonElement? = null
        var normalizerJson: JsonElement? = null
        var preTokenizerJson: JsonElement? = null
        var decoderJson: JsonElement? = null

        JsonReader(tokenizerJson).use { r ->
            r.beginObject()
            while (r.hasNext()) {
                when (r.nextName()) {
                    "model" -> {
                        r.beginObject()
                        while (r.hasNext()) {
                            when (r.nextName()) {
                                "type" -> modelType = r.nullableString()
                                "vocab" -> {
                                    r.beginObject()
                                    while (r.hasNext()) {
                                        val k = r.nextName()
                                        vocab[k] = r.nextInt()
                                    }
                                    r.endObject()
                                }
                                "merges" -> {
                                    // "a b" strings (old format) or ["a", "b"] pairs (new format), stored flat a,b,a,b...
                                    val flat = ArrayList<String>(1 shl 18)
                                    r.beginArray()
                                    while (r.hasNext()) {
                                        if (r.peek() == JsonToken.BEGIN_ARRAY) {
                                            r.beginArray()
                                            flat.add(r.nextString())
                                            flat.add(r.nextString())
                                            while (r.hasNext()) r.skipValue()
                                            r.endArray()
                                        } else {
                                            val s = r.nextString()
                                            val sp = s.indexOf(' ')
                                            require(sp > 0) { "Bad merge entry: $s" }
                                            flat.add(s.substring(0, sp))
                                            flat.add(s.substring(sp + 1))
                                        }
                                    }
                                    r.endArray()
                                    mergePairsRaw = flat
                                }
                                "byte_fallback" -> byteFallback = r.nullableBoolean() ?: false
                                "ignore_merges" -> ignoreMerges = r.nullableBoolean() ?: false
                                "fuse_unk" -> fuseUnk = r.nullableBoolean() ?: false
                                "unk_token" -> unkToken = r.nullableString()
                                "continuing_subword_prefix" -> continuingSubwordPrefix = r.nullableString()
                                "end_of_word_suffix" -> endOfWordSuffix = r.nullableString()
                                else -> r.skipValue()
                            }
                        }
                        r.endObject()
                    }
                    "added_tokens" -> addedJson = JsonParser.parseReader(r)
                    "normalizer" -> normalizerJson = JsonParser.parseReader(r)
                    "pre_tokenizer" -> preTokenizerJson = JsonParser.parseReader(r)
                    "decoder" -> decoderJson = JsonParser.parseReader(r)
                    else -> r.skipValue()
                }
            }
            r.endObject()
        }

        require(modelType == null || modelType == "BPE") { "Unsupported tokenizer model type: $modelType" }
        require(vocab.isNotEmpty()) { "tokenizer.json has no model.vocab" }

        // Added tokens (special and non-special). Their ids win over the model vocab.
        addedJson?.takeIf { it.isJsonArray }?.asJsonArray?.forEach { el ->
            val o = el.asJsonObject
            val t = AddedToken(
                id = o["id"].asInt,
                content = o["content"].asString,
                special = o.bool("special", false),
                normalized = o.bool("normalized", !o.bool("special", false)),
                lstrip = o.bool("lstrip", false),
                rstrip = o.bool("rstrip", false),
                singleWord = o.bool("single_word", false),
            )
            addedById[t.id] = t
            addedByContent[t.content] = t
        }
        val rawTokens = addedById.values.filter { !it.normalized }
        val normTokens = addedById.values.filter { it.normalized }
        if (rawTokens.isNotEmpty()) rawAddedMatcher = AddedTokenMatcher(rawTokens)
        if (normTokens.isNotEmpty()) normalizedAddedMatcher = AddedTokenMatcher(normTokens)

        var maxId = vocab.values.maxOrNull() ?: 0
        addedById.keys.maxOrNull()?.let { if (it > maxId) maxId = it }
        val reverse = arrayOfNulls<String>(maxId + 1)
        for ((k, v) in vocab) if (v >= 0) reverse[v] = k
        for ((id, t) in addedById) if (id >= 0) reverse[id] = t.content
        idToToken = reverse

        // Merges: (leftId, rightId) -> (rank, mergedId)
        mergePairsRaw?.let { flat ->
            val n = flat.size / 2
            val map = LongLongMap(n)
            val prefixLen = continuingSubwordPrefix?.length ?: 0
            for (rank in 0 until n) {
                val a = flat[2 * rank]
                val b = flat[2 * rank + 1]
                val aId = vocab[a] ?: continue
                val bId = vocab[b] ?: continue
                val merged = if (prefixLen > 0 && b.startsWith(continuingSubwordPrefix!!)) a + b.substring(prefixLen) else a + b
                val newId = vocab[merged] ?: continue
                map.put(pairKey(aId, bId), (rank.toLong() shl 32) or (newId.toLong() and 0xffffffffL))
            }
            merges = map
        }

        normalizer = buildNormalizer(normalizerJson)
        preTokenizers = buildPreTokenizers(preTokenizerJson)
        decoders = buildDecoders(decoderJson)

        // BOS / EOS from tokenizer_config.json and generation_config.json
        val gson = com.google.gson.Gson()
        val config = tokenizerConfigJson?.let { runCatching { gson.fromJson(it, JsonObject::class.java) }.getOrNull() }
        fun tokenText(el: JsonElement?): String? = when {
            el == null || el.isJsonNull -> null
            el.isJsonPrimitive -> el.asString
            el.isJsonObject -> el.asJsonObject["content"]?.takeIf { it.isJsonPrimitive }?.asString
            else -> null
        }
        bosToken = tokenText(config?.get("bos_token"))
        eosToken = tokenText(config?.get("eos_token"))

        val generation = generationConfigJson?.let { runCatching { gson.fromJson(it, JsonObject::class.java) }.getOrNull() }
        val generationEos = mutableListOf<Int>()
        generation?.get("eos_token_id")?.let { el ->
            when {
                el.isJsonArray -> el.asJsonArray.forEach { e -> runCatching { generationEos += e.asInt } }
                el.isJsonPrimitive -> runCatching { generationEos += el.asInt }
            }
        }

        val configEos = eosToken?.let { tokenToId(it) }
        val primary = configEos
            ?: generationEos.firstOrNull()
            ?: listOf("<|im_end|>", "</s>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "[EOS]")
                .firstNotNullOfOrNull { tokenToId(it) }
            ?: 2
        eosTokenId = primary
        val all = LinkedHashSet<Long>()
        all += primary.toLong()
        generationEos.forEach { all += it.toLong() }
        // End-of-turn markers the model may emit instead of the configured EOS.
        listOf("<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>", "<end_of_turn>").forEach { name ->
            addedByContent[name]?.takeIf { it.special }?.let { all += it.id.toLong() }
        }
        eosTokenIds = all
    }

    // ---------------------------------------------------------------- public API

    /** Token ids for [text] (no BOS/EOS added; special tokens written in the text are matched as whole tokens). */
    fun encode(text: String): List<Long> {
        val out = ArrayList<Int>(text.length / 3 + 8)
        // 1. Cut out non-normalized added tokens on the raw text.
        val rawSegments = rawAddedMatcher?.split(text) ?: listOf(Segment(text, -1))
        var firstSegment = true
        for (seg in rawSegments) {
            if (seg.tokenId >= 0) { out += seg.tokenId; firstSegment = false; continue }
            // 2. Normalize, then cut out normalized added tokens.
            val normalized = normalizer?.invoke(seg.text) ?: seg.text
            val pieces = normalizedAddedMatcher?.split(normalized) ?: listOf(Segment(normalized, -1))
            for (p in pieces) {
                if (p.tokenId >= 0) { out += p.tokenId; firstSegment = false; continue }
                // 3. Pre-tokenize and run BPE on each word.
                var words: List<String> = listOf(p.text)
                for (pt in preTokenizers) words = pt.apply(words, firstSegment)
                for (w in words) if (w.isNotEmpty()) bpe(w, out)
                firstSegment = false
            }
        }
        return out.map { it.toLong() }
    }

    /** Text for [tokenIds]; special tokens are dropped when [skipSpecialTokens] (the `tokenizers` default for chat). */
    fun decode(tokenIds: List<Int>, skipSpecialTokens: Boolean = true): String {
        val tokens = ArrayList<String>(tokenIds.size)
        for (id in tokenIds) {
            val added = addedById[id]
            if (added != null) {
                if (skipSpecialTokens && added.special) continue
                tokens += added.content
                continue
            }
            val t = idToToken.getOrNull(id) ?: continue
            tokens += t
        }
        val chain = decoders ?: return tokens.joinToString(" ")
        var cur: List<String> = tokens
        for (d in chain) cur = d.decodeChain(cur)
        return cur.joinToString("")
    }

    fun getEosTokenId(): Long = eosTokenId.toLong()

    /** Every id that ends a reply: tokenizer_config eos, generation_config eos ids and special end-of-turn tokens. */
    fun getEosTokenIds(): Set<Long> = eosTokenIds

    fun tokenToId(token: String): Int? = addedByContent[token]?.id ?: vocab[token]

    fun idToTokenString(id: Int): String? = idToToken.getOrNull(id)

    /**
     * Control tokens that never belong in chat text: special added tokens and `<|...|>` markers
     * (LFM's tool-call markers are added as non-special). `<think>` / `</think>` stay visible.
     */
    fun isHiddenControlToken(id: Int): Boolean {
        val t = addedById[id] ?: return false
        return t.special || (t.content.startsWith("<|") && t.content.endsWith("|>"))
    }

    /**
     * Incremental detokenizer for streaming: push one generated id at a time and get only the new, complete
     * text back. Multi-byte UTF-8 characters split across tokens are held until they are complete, and the
     * previous token is kept as context so decoders that strip a leading space (Metaspace) stay correct.
     */
    inner class StreamDecoder(private val skipSpecialTokens: Boolean = true) {
        private val ids = ArrayList<Int>()
        private var prefixOffset = 0
        private var readOffset = 0

        fun push(id: Int): String {
            ids += id
            val prefixText = decode(ids.subList(prefixOffset, readOffset), skipSpecialTokens)
            val newText = decode(ids.subList(prefixOffset, ids.size), skipSpecialTokens)
            if (newText.length > prefixText.length && !newText.endsWith(REPLACEMENT) && newText.startsWith(prefixText)) {
                prefixOffset = readOffset
                readOffset = ids.size
                return newText.substring(prefixText.length)
            }
            return ""
        }

        /** Whatever is still held back at the end of generation (an incomplete character comes out as U+FFFD). */
        fun flush(): String {
            if (readOffset >= ids.size) return ""
            val prefixText = decode(ids.subList(prefixOffset, readOffset), skipSpecialTokens)
            val newText = decode(ids.subList(prefixOffset, ids.size), skipSpecialTokens)
            prefixOffset = ids.size
            readOffset = ids.size
            return if (newText.length > prefixText.length && newText.startsWith(prefixText)) newText.substring(prefixText.length) else ""
        }
    }

    fun newStreamDecoder(skipSpecialTokens: Boolean = true): StreamDecoder = StreamDecoder(skipSpecialTokens)

    // ---------------------------------------------------------------- BPE

    private class MergeCandidate(val rank: Int, val pos: Int, val newId: Int) : Comparable<MergeCandidate> {
        override fun compareTo(other: MergeCandidate): Int =
            if (rank != other.rank) rank.compareTo(other.rank) else pos.compareTo(other.pos)
    }

    private fun bpe(word: String, out: MutableList<Int>) {
        if (ignoreMerges) vocab[word]?.let { out += it; return }
        wordCache[word]?.let { cached -> cached.forEach { out += it }; return }

        // Initial symbols: one per code point (with byte fallback / unk handling as in tokenizers).
        val ids = ArrayList<Int>(word.length)
        val unkId = unkToken?.let { vocab[it] }
        var i = 0
        var lastWasUnk = false
        while (i < word.length) {
            val cp = word.codePointAt(i)
            val cpLen = Character.charCount(cp)
            val isFirst = i == 0
            val isLast = i + cpLen >= word.length
            var s = word.substring(i, i + cpLen)
            if (!isFirst && continuingSubwordPrefix != null) s = continuingSubwordPrefix + s
            if (isLast && endOfWordSuffix != null) s += endOfWordSuffix
            val id = vocab[s]
            if (id != null) {
                ids += id
                lastWasUnk = false
            } else {
                var handled = false
                if (byteFallback) {
                    val bytes = word.substring(i, i + cpLen).toByteArray(Charsets.UTF_8)
                    val byteIds = bytes.map { b -> vocab[String.format("<0x%02X>", b.toInt() and 0xff)] }
                    if (byteIds.all { it != null }) {
                        byteIds.forEach { ids += it!! }
                        handled = true
                        lastWasUnk = false
                    }
                }
                if (!handled && unkId != null) {
                    if (!(fuseUnk && lastWasUnk)) ids += unkId
                    lastWasUnk = true
                }
            }
            i += cpLen
        }

        val n = ids.size
        if (n > 1 && merges.size > 0) {
            val c = ids.toIntArray()
            val prev = IntArray(n) { it - 1 }
            val next = IntArray(n) { if (it + 1 < n) it + 1 else -1 }
            val alive = BooleanArray(n) { true }
            val queue = PriorityQueue<MergeCandidate>(n)
            for (p in 0 until n - 1) {
                val m = merges.get(pairKey(c[p], c[p + 1]))
                if (m != LongLongMap.MISSING) queue.add(MergeCandidate((m ushr 32).toInt(), p, m.toInt()))
            }
            while (true) {
                val top = queue.poll() ?: break
                if (!alive[top.pos]) continue
                val nextPos = next[top.pos]
                if (nextPos == -1) continue
                val m = merges.get(pairKey(c[top.pos], c[nextPos]))
                if (m == LongLongMap.MISSING || m.toInt() != top.newId) continue
                // Merge right symbol into the left one.
                c[top.pos] = top.newId
                alive[nextPos] = false
                val nn = next[nextPos]
                next[top.pos] = nn
                if (nn != -1) prev[nn] = top.pos
                val pp = prev[top.pos]
                if (pp >= 0) {
                    val mp = merges.get(pairKey(c[pp], c[top.pos]))
                    if (mp != LongLongMap.MISSING) queue.add(MergeCandidate((mp ushr 32).toInt(), pp, mp.toInt()))
                }
                if (nn != -1) {
                    val mn = merges.get(pairKey(c[top.pos], c[nn]))
                    if (mn != LongLongMap.MISSING) queue.add(MergeCandidate((mn ushr 32).toInt(), top.pos, mn.toInt()))
                }
            }
            ids.clear()
            for (p in 0 until n) if (alive[p]) ids += c[p]
        }

        if (word.length < 256) {
            if (wordCache.size > 50_000) wordCache.clear()
            wordCache[word] = ids.toIntArray()
        }
        out.addAll(ids)
    }

    // ---------------------------------------------------------------- added tokens

    private class Segment(val text: String, val tokenId: Int)

    /** Leftmost-longest, non-overlapping matcher for added tokens (same semantics as tokenizers' Aho-Corasick). */
    private class AddedTokenMatcher(tokens: List<AddedToken>) {
        private val byFirstChar = HashMap<Char, List<AddedToken>>()

        init {
            tokens.filter { it.content.isNotEmpty() }
                .groupBy { it.content[0] }
                .forEach { (ch, list) -> byFirstChar[ch] = list.sortedByDescending { it.content.length } }
        }

        fun split(text: String): List<Segment> {
            if (byFirstChar.isEmpty() || text.isEmpty()) return listOf(Segment(text, -1))
            val out = ArrayList<Segment>()
            var segStart = 0
            var i = 0
            while (i < text.length) {
                val candidates = byFirstChar[text[i]]
                val match = candidates?.firstOrNull { text.startsWith(it.content, i) }
                if (match == null) { i++; continue }
                var start = i
                var stop = i + match.content.length
                if (match.singleWord) {
                    val startOk = start == 0 || !isWordChar(text.codePointBefore(start))
                    val stopOk = stop == text.length || !isWordChar(text.codePointAt(stop))
                    if (!startOk || !stopOk) { i = stop; continue }
                }
                if (match.lstrip) while (start > segStart && Character.isWhitespace(text[start - 1])) start--
                if (match.rstrip) while (stop < text.length && Character.isWhitespace(text[stop])) stop++
                if (start > segStart) out += Segment(text.substring(segStart, start), -1)
                out += Segment(text.substring(start, stop), match.id)
                segStart = stop
                i = stop
            }
            if (segStart < text.length) out += Segment(text.substring(segStart), -1)
            return out
        }

        private fun isWordChar(cp: Int) = Character.isLetterOrDigit(cp) || cp == '_'.code
    }

    // ---------------------------------------------------------------- normalizers

    private fun buildNormalizer(el: JsonElement?): ((String) -> String)? {
        if (el == null || el.isJsonNull) return null
        val o = el.asJsonObject
        return when (val type = o["type"]?.asString) {
            "Sequence" -> sequence(o["normalizers"].asJsonArray.mapNotNull { buildNormalizer(it) })
            "NFC" -> unicodeForm(Normalizer.Form.NFC)
            "NFD" -> unicodeForm(Normalizer.Form.NFD)
            "NFKC" -> unicodeForm(Normalizer.Form.NFKC)
            "NFKD" -> unicodeForm(Normalizer.Form.NFKD)
            "Lowercase" -> lowercase()
            "Prepend" -> prepend(o["prepend"].asString)
            "Strip" -> strip(o.bool("strip_left", true), o.bool("strip_right", true))
            "Replace" -> replace(compilePattern(o["pattern"].asJsonObject), o["content"].asString)
            else -> throw IllegalArgumentException("Unsupported tokenizer normalizer: $type")
        }
    }

    private fun sequence(parts: List<(String) -> String>): (String) -> String = { s -> parts.fold(s) { acc, f -> f(acc) } }

    private fun unicodeForm(form: Normalizer.Form): (String) -> String = { s -> Normalizer.normalize(s, form) }

    private fun lowercase(): (String) -> String = { s -> s.lowercase() }

    private fun prepend(p: String): (String) -> String = { s -> if (s.isEmpty()) s else p + s }

    private fun strip(left: Boolean, right: Boolean): (String) -> String =
        { s -> s.let { if (left) it.trimStart() else it }.let { if (right) it.trimEnd() else it } }

    private fun replace(pattern: Pattern, content: String): (String) -> String =
        { s -> pattern.matcher(s).replaceAll(java.util.regex.Matcher.quoteReplacement(content)) }

    // ---------------------------------------------------------------- pre-tokenizers

    private fun interface PreTokenizer {
        fun apply(pieces: List<String>, firstSegment: Boolean): List<String>
    }

    private fun buildPreTokenizers(el: JsonElement?): List<PreTokenizer> {
        if (el == null || el.isJsonNull) return emptyList()
        val o = el.asJsonObject
        return when (val type = o["type"]?.asString) {
            "Sequence" -> o["pretokenizers"].asJsonArray.flatMap { buildPreTokenizers(it) }
            "Split" -> {
                val pattern = compilePattern(o["pattern"].asJsonObject)
                val behavior = o["behavior"]?.asString ?: "Isolated"
                val invert = o.bool("invert", false)
                listOf(PreTokenizer { pieces, _ -> pieces.flatMap { splitWithBehavior(it, pattern, behavior, invert) } })
            }
            "ByteLevel" -> {
                val addPrefixSpace = o.bool("add_prefix_space", true)
                val useRegex = o.bool("use_regex", true)
                listOf(PreTokenizer { pieces, _ ->
                    val split = if (useRegex) pieces.flatMap { splitWithBehavior(it, GPT2_PATTERN, "Isolated", false) } else pieces
                    split.map { p -> byteLevelEncode(if (addPrefixSpace && !p.startsWith(" ")) " $p" else p) }
                })
            }
            "Metaspace" -> {
                val replacement = o["replacement"]?.asString ?: "▁"
                val scheme = o["prepend_scheme"]?.takeIf { !it.isJsonNull }?.asString
                    ?: if (o.bool("add_prefix_space", true)) "always" else "never"
                val split = o.bool("split", true)
                val quoted = Pattern.compile(Pattern.quote(replacement))
                listOf(PreTokenizer { pieces, firstSegment ->
                    pieces.flatMapIndexed { idx, p ->
                        var s = p.replace(" ", replacement)
                        val prepend = when (scheme) {
                            "always" -> true
                            "first" -> firstSegment && idx == 0
                            else -> false
                        }
                        if (prepend && !s.startsWith(replacement)) s = replacement + s
                        if (split) splitWithBehavior(s, quoted, "MergedWithNext", false) else listOf(s)
                    }
                })
            }
            else -> throw IllegalArgumentException("Unsupported tokenizer pre_tokenizer: $type")
        }
    }

    private fun splitWithBehavior(text: String, pattern: Pattern, behavior: String, invert: Boolean): List<String> {
        if (text.isEmpty()) return emptyList()
        // (start, end, isMatch) spans covering the whole text
        val spans = ArrayList<IntArray>()
        val m = pattern.matcher(text)
        var prevEnd = 0
        while (m.find()) {
            if (m.start() == m.end()) continue
            if (m.start() > prevEnd) spans += intArrayOf(prevEnd, m.start(), 0)
            spans += intArrayOf(m.start(), m.end(), 1)
            prevEnd = m.end()
        }
        if (prevEnd < text.length) spans += intArrayOf(prevEnd, text.length, 0)
        if (invert) spans.forEach { it[2] = 1 - it[2] }

        val result: List<IntArray> = when (behavior) {
            "Isolated" -> spans
            "Removed" -> spans.filter { it[2] == 0 }
            "MergedWithPrevious" -> {
                val acc = ArrayList<IntArray>()
                var previousMatch = false
                for (s in spans) {
                    val isMatch = s[2] == 1
                    if (isMatch && !previousMatch && acc.isNotEmpty()) acc.last()[1] = s[1]
                    else acc += s.copyOf()
                    previousMatch = isMatch
                }
                acc
            }
            "MergedWithNext" -> {
                val acc = ArrayList<IntArray>()
                var previousMatch = false
                for (s in spans.asReversed()) {
                    val isMatch = s[2] == 1
                    if (isMatch && !previousMatch && acc.isNotEmpty()) acc.last()[0] = s[0]
                    else acc += s.copyOf()
                    previousMatch = isMatch
                }
                acc.reverse()
                acc
            }
            "Contiguous" -> {
                val acc = ArrayList<IntArray>()
                var previousMatch = false
                for (s in spans) {
                    val isMatch = s[2] == 1
                    if (isMatch == previousMatch && acc.isNotEmpty()) acc.last()[1] = s[1]
                    else acc += s.copyOf()
                    previousMatch = isMatch
                }
                acc
            }
            else -> throw IllegalArgumentException("Unsupported Split behavior: $behavior")
        }
        return result.filter { it[1] > it[0] }.map { text.substring(it[0], it[1]) }
    }

    // ---------------------------------------------------------------- decoders

    private fun interface TokenDecoder {
        fun decodeChain(tokens: List<String>): List<String>
    }

    private fun buildDecoders(el: JsonElement?): List<TokenDecoder>? {
        if (el == null || el.isJsonNull) return null
        val o = el.asJsonObject
        return when (val type = o["type"]?.asString) {
            "Sequence" -> o["decoders"].asJsonArray.flatMap { buildDecoders(it) ?: emptyList() }
            "ByteLevel" -> listOf(TokenDecoder { tokens ->
                val buf = java.io.ByteArrayOutputStream()
                for (t in tokens) {
                    val bytes = byteLevelDecodeOrNull(t)
                    if (bytes != null) buf.write(bytes) else buf.write(t.toByteArray(Charsets.UTF_8))
                }
                listOf(String(buf.toByteArray(), Charsets.UTF_8))
            })
            "ByteFallback" -> listOf(TokenDecoder { tokens ->
                val out = ArrayList<String>()
                val pending = java.io.ByteArrayOutputStream()
                fun flushBytes() {
                    if (pending.size() == 0) return
                    val bytes = pending.toByteArray()
                    out += strictUtf8(bytes) ?: REPLACEMENT.toString().repeat(bytes.size)
                    pending.reset()
                }
                for (t in tokens) {
                    if (t.length == 6 && t.startsWith("<0x") && t.endsWith(">")) {
                        val b = t.substring(3, 5).toIntOrNull(16)
                        if (b != null) { pending.write(b); continue }
                    }
                    flushBytes()
                    out += t
                }
                flushBytes()
                out
            })
            "Fuse" -> listOf(TokenDecoder { tokens -> listOf(tokens.joinToString("")) })
            "Strip" -> {
                val content = o["content"].asString
                val start = o["start"]?.asInt ?: 0
                val stop = o["stop"]?.asInt ?: 0
                listOf(TokenDecoder { tokens ->
                    tokens.map { t ->
                        var s = t
                        var n = 0
                        while (n < start && s.startsWith(content)) { s = s.substring(content.length); n++ }
                        n = 0
                        while (n < stop && s.endsWith(content)) { s = s.substring(0, s.length - content.length); n++ }
                        s
                    }
                })
            }
            "Metaspace" -> {
                val replacement = o["replacement"]?.asString ?: "▁"
                val scheme = o["prepend_scheme"]?.takeIf { !it.isJsonNull }?.asString
                    ?: if (o.bool("add_prefix_space", true)) "always" else "never"
                listOf(TokenDecoder { tokens ->
                    tokens.mapIndexed { i, t ->
                        if (i == 0 && scheme != "never") t.replace(replacement, "") else t.replace(replacement, " ")
                    }
                })
            }
            "Replace" -> {
                val content = o["content"].asString
                val pattern = compilePattern(o["pattern"].asJsonObject)
                listOf(TokenDecoder { tokens ->
                    tokens.map { pattern.matcher(it).replaceAll(java.util.regex.Matcher.quoteReplacement(content)) }
                })
            }
            else -> throw IllegalArgumentException("Unsupported tokenizer decoder: $type")
        }
    }

    // ---------------------------------------------------------------- helpers

    private fun pairKey(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xffffffffL)

    private fun byteLevelEncode(s: String): String {
        val bytes = s.toByteArray(Charsets.UTF_8)
        val sb = StringBuilder(bytes.size)
        for (b in bytes) sb.append(BYTE_TO_CHAR[b.toInt() and 0xff])
        return sb.toString()
    }

    /** Bytes for a ByteLevel token, or null when one of its chars is not in the byte map (added tokens etc.). */
    private fun byteLevelDecodeOrNull(t: String): ByteArray? {
        val out = ByteArray(t.length)
        for (i in t.indices) {
            val c = t[i].code
            val b = if (c < CHAR_TO_BYTE.size) CHAR_TO_BYTE[c] else -1
            if (b < 0) return null
            out[i] = b.toByte()
        }
        return out
    }

    /** Minimal open-addressing Long -> Long hash map (keys are non-negative), for ~270k merges without boxing. */
    private class LongLongMap(expected: Int) {
        private val capacity: Int
        private val keys: LongArray
        private val values: LongArray
        var size = 0
            private set

        init {
            var cap = 16
            while (cap < expected * 2) cap = cap shl 1
            capacity = cap
            keys = LongArray(cap) { EMPTY }
            values = LongArray(cap)
        }

        private fun slot(key: Long): Int {
            var h = key * -0x61c8864680b583ebL
            h = h xor (h ushr 29)
            return (h.toInt()) and (capacity - 1)
        }

        fun put(key: Long, value: Long) {
            var i = slot(key)
            while (true) {
                val k = keys[i]
                if (k == EMPTY) { keys[i] = key; values[i] = value; size++; return }
                if (k == key) { values[i] = value; return }
                i = (i + 1) and (capacity - 1)
            }
        }

        fun get(key: Long): Long {
            var i = slot(key)
            while (true) {
                val k = keys[i]
                if (k == EMPTY) return MISSING
                if (k == key) return values[i]
                i = (i + 1) and (capacity - 1)
            }
        }

        companion object {
            const val EMPTY = -1L
            const val MISSING = Long.MIN_VALUE
        }
    }

    companion object {
        private const val REPLACEMENT = '�'

        private fun siblingText(file: File, name: String): String? =
            File(file.parentFile, name).takeIf { it.isFile }?.readText(Charsets.UTF_8)

        /** GPT-2 bytes_to_unicode table. */
        private val BYTE_TO_CHAR: CharArray
        private val CHAR_TO_BYTE: IntArray

        init {
            val bs = ArrayList<Int>()
            bs += '!'.code..'~'.code
            bs += 0xA1..0xAC
            bs += 0xAE..0xFF
            val cs = ArrayList<Int>(bs)
            var n = 0
            for (b in 0..255) {
                if (b !in bs) { bs += b; cs += 256 + n; n++ }
            }
            BYTE_TO_CHAR = CharArray(256)
            CHAR_TO_BYTE = IntArray(256 + n) { -1 }
            for (k in bs.indices) {
                BYTE_TO_CHAR[bs[k]] = cs[k].toChar()
                CHAR_TO_BYTE[cs[k]] = bs[k]
            }
        }

        /**
         * Oniguruma's `\s` (what `tokenizers` uses) is Unicode white space; java.util.regex's `\s` is ASCII-only
         * on the JVM. Spell the set out so the JVM and Android's ICU engine agree with the Rust tokenizer.
         */
        private const val WS_CHARS = "\\t\\n\\x0B\\f\\r \\x{85}\\x{A0}\\x{1680}\\x{2000}-\\x{200A}\\x{2028}\\x{2029}\\x{202F}\\x{205F}\\x{3000}"

        internal fun translateRegex(p: String): String {
            val sb = StringBuilder(p.length + 64)
            var depth = 0
            var i = 0
            while (i < p.length) {
                val c = p[i]
                if (c == '\\' && i + 1 < p.length) {
                    when (val n = p[i + 1]) {
                        's' -> sb.append(if (depth > 0) WS_CHARS else "[$WS_CHARS]")
                        'S' -> sb.append("[^$WS_CHARS]")
                        else -> sb.append(c).append(n)
                    }
                    i += 2
                    continue
                }
                if (c == '[') depth++ else if (c == ']' && depth > 0) depth--
                sb.append(c)
                i++
            }
            return sb.toString()
        }

        /** Oniguruma's (?i) folds Unicode case (e.g. U+017F matches `s`); UNICODE_CASE does the same here. */
        private fun compileRegex(p: String): Pattern = Pattern.compile(translateRegex(p), Pattern.UNICODE_CASE)

        private fun compilePattern(o: JsonObject): Pattern = when {
            o.has("Regex") -> compileRegex(o["Regex"].asString)
            o.has("String") -> Pattern.compile(Pattern.quote(o["String"].asString))
            else -> throw IllegalArgumentException("Unsupported pattern: $o")
        }

        private val GPT2_PATTERN: Pattern =
            compileRegex("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")

        private fun strictUtf8(bytes: ByteArray): String? = try {
            Charsets.UTF_8.newDecoder()
                .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
                .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
                .decode(java.nio.ByteBuffer.wrap(bytes)).toString()
        } catch (_: java.nio.charset.CharacterCodingException) {
            null
        }

        private fun JsonReader.nullableString(): String? =
            if (peek() == JsonToken.NULL) { nextNull(); null } else nextString()

        private fun JsonReader.nullableBoolean(): Boolean? =
            if (peek() == JsonToken.NULL) { nextNull(); null } else nextBoolean()

        private fun JsonObject.bool(name: String, default: Boolean): Boolean =
            get(name)?.takeIf { it.isJsonPrimitive }?.asBoolean ?: default
    }
}

/**
 * Builds the exact chat prompt each ONNX model's `chat_template` renders, from the app's role-labelled
 * transcript (`system: ...\n\nuser: ...\nassistant: ...\n\nuser: ...\nassistant:`).
 */
internal object OnnxChatPrompt {
    /** Marks where the image tokens go in a vision prompt; never tokenized (the prompt is split on it). */
    const val IMAGE_PLACEHOLDER = ""

    fun isMistral(modelName: String): Boolean =
        modelName.contains("Ministral", ignoreCase = true) || modelName.contains("Mistral", ignoreCase = true)

    private fun splitTurns(prompt: String): Pair<String, List<VlmPromptTurn>> {
        val turns = parseVlmPromptTurns(prompt)
        val system = turns.filter { it.role == "system" }.joinToString("\n\n") { it.text }
        val conversation = turns.filter { it.role != "system" }
        if (conversation.isEmpty()) {
            // Only a system text (or nothing): send it as the user's message rather than an empty chat.
            val text = system.ifEmpty { prompt.trim() }
            return "" to listOf(VlmPromptTurn("user", text))
        }
        return system to conversation
    }

    /**
     * LFM2 / LFM2.5 template (tokenizer_config.json chat_template):
     * `<|startoftext|>[<|im_start|>system\n{sys}<|im_end|>\n]` + `<|im_start|>{role}\n{content}<|im_end|>\n`
     * per message + `<|im_start|>assistant\n`. Earlier assistant replies drop their `<think>` part.
     * [noThink] prefixes the last user message with `/no_think` (the app's thinking-off switch).
     */
    fun chatMl(prompt: String, bosToken: String = "<|startoftext|>", noThink: Boolean = false): String {
        val (system, turns) = splitTurns(prompt)
        val sb = StringBuilder(bosToken)
        if (system.isNotEmpty()) sb.append("<|im_start|>system\n").append(system).append("<|im_end|>\n")
        val lastAssistant = turns.indexOfLast { it.role == "assistant" }
        val lastUser = turns.indexOfLast { it.role == "user" }
        turns.forEachIndexed { i, t ->
            var content = t.text
            if (t.role == "assistant" && i != lastAssistant && content.contains("</think>")) {
                content = content.substringAfterLast("</think>").trim()
            }
            if (noThink && i == lastUser) content = "/no_think $content"
            sb.append("<|im_start|>").append(t.role).append('\n').append(content).append("<|im_end|>\n")
        }
        sb.append("<|im_start|>assistant\n")
        return sb.toString()
    }

    /**
     * Ministral-3 template (chat_template.jinja): `<s>[SYSTEM_PROMPT]{sys}[/SYSTEM_PROMPT]` then
     * `[INST]{user}[/INST]{assistant}</s>` per exchange, consecutive same-role messages joined by a blank line.
     * With [withImage] the image placeholder opens the last user message (the template puts images first).
     * The template's long default system prompt (used when there is none) is not added.
     */
    fun mistral(prompt: String, bosToken: String = "<s>", eosToken: String = "</s>", withImage: Boolean = false): String {
        val (system, rawTurns) = splitTurns(prompt)
        val turns = ArrayList<VlmPromptTurn>()
        for (t in rawTurns) {
            val last = turns.lastOrNull()
            if (last != null && last.role == t.role) turns[turns.lastIndex] = VlmPromptTurn(t.role, last.text + "\n\n" + t.text)
            else turns += t
        }
        val sb = StringBuilder(bosToken)
        if (system.isNotEmpty()) sb.append("[SYSTEM_PROMPT]").append(system).append("[/SYSTEM_PROMPT]")
        val lastUser = turns.indexOfLast { it.role == "user" }
        turns.forEachIndexed { i, t ->
            if (t.role == "user") {
                sb.append("[INST]")
                if (withImage && i == lastUser) sb.append(IMAGE_PLACEHOLDER)
                sb.append(t.text).append("[/INST]")
            } else {
                sb.append(t.text)
                // A trailing assistant turn is a reply being continued, so it stays open.
                if (i != turns.lastIndex) sb.append(eosToken)
            }
        }
        return sb.toString()
    }
}
