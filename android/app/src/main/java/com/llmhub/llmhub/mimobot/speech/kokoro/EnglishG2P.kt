package com.llmhub.llmhub.mimobot.speech.kokoro

/**
 * Tiny English grapheme-to-phoneme converter.
 *
 * **This is a starter, not a finished G2P.** It ships a small hand-curated
 * dictionary covering the vocabulary a voice companion is most likely to need
 * (greetings, pronouns, numbers, common verbs/nouns/adjectives) so the Kokoro
 * pipeline produces real, intelligible audio for those words. Out-of-vocabulary
 * words fall back to a deterministic letter-by-letter spelling — usable but
 * sounds like a robot literally spelling things.
 *
 * Replace this with a proper espeak-ng or misaki-style phonemizer for
 * production. The expected drop-in shape is the same: text in → space-joined
 * IPA out, using only symbols present in [KokoroVocab.SYMBOLS].
 *
 * TODO(g2p):
 *   1. Compile espeak-ng for arm64-v8a + armeabi-v7a → libespeak-ng.so.
 *   2. Bundle espeak-ng-data (~7 MB) under src/main/assets/espeak-data.
 *   3. Replace [phonemize] with an espeak_TextToPhonemes JNI call.
 *   4. Match the exact phonemizer that the chosen Kokoro ONNX export was
 *      trained against (kokoro-onnx uses espeak-ng with `--ipa=3` flags).
 */
object EnglishG2P {

    /**
     * Convert a text utterance to a phoneme string ready for [KokoroVocab.tokenise].
     */
    fun phonemize(text: String): String {
        val tokens = text.lowercase().split(Regex("[^a-z']+")).filter { it.isNotEmpty() }
        val sb = StringBuilder()
        for ((i, w) in tokens.withIndex()) {
            if (i > 0) sb.append(' ')
            sb.append(DICT[w] ?: spell(w))
        }
        return sb.toString()
    }

    /** Last-resort fallback: speak the word letter by letter. */
    private fun spell(word: String): String {
        val sb = StringBuilder()
        for ((i, c) in word.withIndex()) {
            if (i > 0) sb.append(' ')
            sb.append(LETTER[c] ?: "")
        }
        return sb.toString()
    }

    /** ASCII letter → English letter name in IPA. */
    private val LETTER: Map<Char, String> = mapOf(
        'a' to "eɪ", 'b' to "bi", 'c' to "si", 'd' to "di", 'e' to "i",
        'f' to "ɛf", 'g' to "ʤi", 'h' to "eɪʧ", 'i' to "aɪ", 'j' to "ʤeɪ",
        'k' to "keɪ", 'l' to "ɛl", 'm' to "ɛm", 'n' to "ɛn", 'o' to "oʊ",
        'p' to "pi", 'q' to "kju", 'r' to "ɑɹ", 's' to "ɛs", 't' to "ti",
        'u' to "ju", 'v' to "vi", 'w' to "ˈdʌbəlju", 'x' to "ɛks", 'y' to "waɪ",
        'z' to "zi"
    )

    /**
     * Hand-curated starter dictionary. Stress markers are omitted for
     * simplicity — Kokoro handles bare phonemes OK. Diphthongs are written as
     * two adjacent vowel symbols (e.g. "oʊ", "aɪ") since Kokoro tokenises per
     * symbol.
     */
    private val DICT: Map<String, String> = mapOf(
        // greetings
        "hello" to "həloʊ", "hi" to "haɪ", "hey" to "heɪ",
        "bye" to "baɪ", "goodbye" to "ɡʊdbaɪ",
        // affirmation / negation
        "yes" to "jɛs", "yeah" to "jɛə", "yep" to "jɛp",
        "no" to "noʊ", "nope" to "noʊp", "not" to "nɑt",
        "maybe" to "meɪbi", "ok" to "oʊkeɪ", "okay" to "oʊkeɪ",
        "sure" to "ʃʊɹ", "alright" to "ɔlɹaɪt",
        // pronouns
        "i" to "aɪ", "me" to "mi", "my" to "maɪ", "mine" to "maɪn",
        "you" to "ju", "your" to "jɔɹ", "yours" to "jɔɹz",
        "we" to "wi", "us" to "ʌs", "our" to "aʊɹ", "ours" to "aʊɹz",
        "he" to "hi", "him" to "hɪm", "his" to "hɪz",
        "she" to "ʃi", "her" to "hɚ", "hers" to "hɚz",
        "it" to "ɪt", "its" to "ɪts",
        "they" to "ðeɪ", "them" to "ðɛm", "their" to "ðɛɹ",
        // copula / aux
        "is" to "ɪz", "am" to "æm", "are" to "ɑɹ",
        "was" to "wʌz", "were" to "wɚ", "be" to "bi", "been" to "bɪn",
        "have" to "hæv", "has" to "hæz", "had" to "hæd",
        "do" to "du", "does" to "dʌz", "did" to "dɪd",
        "will" to "wɪl", "would" to "wʊd", "can" to "kæn", "could" to "kʊd",
        "should" to "ʃʊd", "shall" to "ʃæl", "may" to "meɪ", "might" to "maɪt",
        // conjunctions / preps
        "and" to "ænd", "or" to "ɔɹ", "but" to "bʌt", "if" to "ɪf",
        "then" to "ðɛn", "than" to "ðæn", "so" to "soʊ", "because" to "bɪkʌz",
        "the" to "ðə", "a" to "ə", "an" to "ən", "of" to "əv",
        "in" to "ɪn", "on" to "ɔn", "at" to "æt", "to" to "tu",
        "from" to "fɹʌm", "for" to "fɔɹ", "with" to "wɪð",
        "by" to "baɪ", "about" to "əbaʊt", "as" to "æz",
        // questions
        "what" to "wʌt", "who" to "hu", "where" to "wɛɹ", "when" to "wɛn",
        "why" to "waɪ", "how" to "haʊ", "which" to "wɪʧ",
        // demonstratives / location
        "this" to "ðɪs", "that" to "ðæt", "these" to "ðiz", "those" to "ðoʊz",
        "here" to "hɪɹ", "there" to "ðɛɹ", "now" to "naʊ",
        // numbers
        "zero" to "zɪɹoʊ", "one" to "wʌn", "two" to "tu", "three" to "θɹi",
        "four" to "fɔɹ", "five" to "faɪv", "six" to "sɪks", "seven" to "sɛvən",
        "eight" to "eɪt", "nine" to "naɪn", "ten" to "tɛn",
        // courtesy
        "please" to "pliz", "thanks" to "θæŋks", "thank" to "θæŋk",
        "sorry" to "sɑɹi", "excuse" to "ɪkskjuz",
        // common verbs
        "go" to "ɡoʊ", "going" to "ɡoʊɪŋ", "gone" to "ɡɔn",
        "come" to "kʌm", "coming" to "kʌmɪŋ",
        "see" to "si", "saw" to "sɔ", "seen" to "sin",
        "say" to "seɪ", "said" to "sɛd", "tell" to "tɛl", "told" to "toʊld",
        "think" to "θɪŋk", "thought" to "θɔt",
        "know" to "noʊ", "knew" to "nu",
        "like" to "laɪk", "want" to "wɑnt", "wanted" to "wɑntɪd",
        "need" to "nid", "needs" to "nidz",
        "help" to "hɛlp", "make" to "meɪk", "made" to "meɪd",
        "take" to "teɪk", "took" to "tʊk", "taken" to "teɪkən",
        "give" to "ɡɪv", "gave" to "ɡeɪv", "got" to "ɡɑt", "get" to "ɡɛt",
        "find" to "faɪnd", "found" to "faʊnd",
        "look" to "lʊk", "ask" to "æsk", "play" to "pleɪ",
        "work" to "wɝk", "open" to "oʊpən", "close" to "kloʊz",
        "start" to "stɑɹt", "stop" to "stɑp",
        "is" to "ɪz", "do" to "du",
        // common nouns
        "thing" to "θɪŋ", "things" to "θɪŋz", "time" to "taɪm",
        "day" to "deɪ", "night" to "naɪt", "today" to "tədeɪ",
        "tomorrow" to "təmɑɹoʊ", "yesterday" to "jɛstɚdeɪ",
        "name" to "neɪm", "person" to "pɝsən", "people" to "pipəl",
        "thing" to "θɪŋ", "way" to "weɪ", "world" to "wɝld",
        "home" to "hoʊm", "house" to "haʊs",
        "music" to "mjuzɪk", "song" to "sɔŋ", "light" to "laɪt",
        "lights" to "laɪts", "volume" to "vɑljəm", "battery" to "bætəɹi",
        "robot" to "ɹoʊbɑt", "bot" to "bɑt", "voice" to "vɔɪs",
        "model" to "mɑdəl", "phone" to "foʊn",
        // common adjectives
        "good" to "ɡʊd", "bad" to "bæd", "great" to "ɡɹeɪt", "nice" to "naɪs",
        "hot" to "hɑt", "cold" to "koʊld", "warm" to "wɔɹm",
        "big" to "bɪɡ", "small" to "smɔl", "little" to "lɪtəl",
        "old" to "oʊld", "new" to "nu", "young" to "jʌŋ",
        "happy" to "hæpi", "sad" to "sæd", "fine" to "faɪn",
        "right" to "ɹaɪt", "wrong" to "ɹɔŋ", "real" to "ɹiəl",
        "slow" to "sloʊ", "fast" to "fæst", "loud" to "laʊd", "quiet" to "kwaɪət",
        // adverbs / common
        "very" to "vɛɹi", "really" to "ɹiəli", "too" to "tu",
        "also" to "ɔlsoʊ", "just" to "ʤʌst", "only" to "oʊnli",
        "more" to "mɔɹ", "less" to "lɛs", "most" to "moʊst",
        "all" to "ɔl", "any" to "ɛni", "some" to "sʌm", "none" to "nʌn",
        "always" to "ɔlweɪz", "never" to "nɛvɚ", "again" to "əɡɛn",
        // contractions
        "i'm" to "aɪm", "you're" to "jʊɹ", "he's" to "hiz", "she's" to "ʃiz",
        "it's" to "ɪts", "we're" to "wɪɹ", "they're" to "ðɛɹ",
        "don't" to "doʊnt", "doesn't" to "dʌzənt", "didn't" to "dɪdənt",
        "can't" to "kænt", "won't" to "woʊnt", "isn't" to "ɪzənt",
        "aren't" to "ɑɹənt", "wasn't" to "wʌzənt",
        "i'll" to "aɪl", "you'll" to "jul", "we'll" to "wil",
        "i've" to "aɪv", "you've" to "juv", "we've" to "wiv",
        // commands a voice bot hears a lot
        "turn" to "tɝn", "set" to "sɛt", "show" to "ʃoʊ",
        "tell" to "tɛl", "play" to "pleɪ", "pause" to "pɔz",
        "next" to "nɛkst", "back" to "bæk", "off" to "ɔf", "on" to "ɔn"
    )
}
