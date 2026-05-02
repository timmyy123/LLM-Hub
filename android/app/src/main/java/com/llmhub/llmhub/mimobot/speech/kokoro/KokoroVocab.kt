package com.llmhub.llmhub.mimobot.speech.kokoro

/**
 * Canonical Kokoro phoneme vocabulary.
 *
 * Token IDs must match the export at
 *   https://huggingface.co/onnx-community/Kokoro-82M-ONNX
 * which itself follows hexgrad/kokoro/__init__.py: SYMBOLS = [PAD] + PUNCT + LETTERS + LETTERS_IPA.
 *
 * Each entry is a single grapheme (one IPA "character" — some are multi-codepoint
 * like the i-with-breve `ɪ̆`). When tokenising we walk the input as Unicode
 * grapheme clusters, not raw chars, to keep these aligned.
 */
object KokoroVocab {

    const val PAD = "$"

    // The ordered list of symbols. Index = token ID.
    val SYMBOLS: List<String> = buildList {
        // 0 — pad
        add(PAD)
        // 1..16 — punctuation (16 entries)
        addAll(listOf(";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "\"", "«", "»", "“", "”", " "))
        // 17..68 — uppercase + lowercase ASCII letters (52)
        addAll("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".map { it.toString() })
        // 69+ — IPA letters / diacritics (109 entries)
        // NOTE: order MUST match hexgrad/kokoro/__init__.py LETTERS_IPA exactly.
        addAll(listOf(
            "ɑ","ɐ","ɒ","æ","b","ɓ","ʙ","β","c","ɕ","ç","ɗ","ɖ","ð","ʤ","ə",
            "ɘ","ɚ","ɛ","ɜ","ɝ","e","ɞ","ɟ","ʄ","ɡ","ɠ","ɢ","ʛ","h","ɦ","ɧ",
            "ħ","ɥ","ʜ","i","ɪ","ɨ","ɪ̆","j","ʝ","ɟ","ʄ","k","ɭ","l","ɫ","ɮ",
            "ɬ","ʟ","m","ɱ","n","ɳ","ɲ","ŋ","ɴ","o","ɵ","ø","œ","ɶ","ɔ","ɸ",
            "p","ɸ","q","ʁ","ɽ","ɾ","r","ɹ","ɻ","ʀ","ʀ̥","ɽ","s","ʂ","ɕ","ʃ",
            "ʈ","t","ʈ","ʇ","θ","u","ʉ","ʊ","ʋ","v","ⱱ","ʌ","w","ʍ","x","χ",
            "y","ʏ","ʎ","z","ʑ","ʐ","ʒ","ʔ","ʕ","ʡ","ʢ"," ","ǀ","ǁ","ǂ","ǃ",
            "ˈ","ˌ","ː","ˑ","ʼ","ʴ","ʰ","ʱ","ʲ","ʷ","ˠ","ˤ","˞","↓","↑","→",
            "↗","↘","'","̩","'","ᵻ"
        ))
    }

    private val ID_BY_SYMBOL: Map<String, Int> = SYMBOLS.withIndex().associate { (i, s) -> s to i }

    fun idOf(symbol: String): Int? = ID_BY_SYMBOL[symbol]

    /** Tokenise an IPA phoneme string as Unicode grapheme clusters. */
    fun tokenise(ipa: String): IntArray {
        if (ipa.isEmpty()) return IntArray(0)
        val out = ArrayList<Int>(ipa.length)
        // Walk by code-point, then merge combining marks into the previous cluster.
        var i = 0
        while (i < ipa.length) {
            val cp = ipa.codePointAt(i)
            val baseLen = Character.charCount(cp)
            // Greedily attach combining marks (e.g. "̩" U+0329, "̥" U+0325).
            var j = i + baseLen
            while (j < ipa.length) {
                val cp2 = ipa.codePointAt(j)
                if (Character.getType(cp2) != Character.NON_SPACING_MARK.toInt()) break
                j += Character.charCount(cp2)
            }
            val cluster = ipa.substring(i, j)
            ID_BY_SYMBOL[cluster]?.let(out::add)
                ?: ID_BY_SYMBOL[ipa.substring(i, i + baseLen)]?.let(out::add)
            i = j
        }
        return out.toIntArray()
    }
}
