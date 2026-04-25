import Foundation

/// Tiny English grapheme-to-phoneme converter.
///
/// **Starter, not finished.** Hand-curated dictionary covers what a voice
/// companion is most likely to say (greetings, pronouns, numbers, common
/// verbs / nouns / adjectives). Out-of-vocabulary words fall back to
/// letter-by-letter spelling — usable but robotic.
///
/// Replace with espeak-ng or misaki for production. Drop-in shape is the
/// same: text → space-joined IPA using only `KokoroVocab.symbols`.
///
/// TODO(g2p):
///   1. Add an espeak-ng XCFramework (or a Swift port of misaki).
///   2. Replace `phonemize` with the upstream G2P call.
///   3. Match the exact phonemizer the chosen Kokoro ONNX export was
///      trained against (kokoro-onnx → espeak-ng with `--ipa=3`).
enum EnglishG2P {

    static func phonemize(_ text: String) -> String {
        let words = text.lowercased()
            .components(separatedBy: CharacterSet.letters.inverted.subtracting(["'"]))
            .filter { !$0.isEmpty }
        var parts: [String] = []
        for w in words {
            parts.append(dict[w] ?? spell(w))
        }
        return parts.joined(separator: " ")
    }

    private static func spell(_ word: String) -> String {
        word.compactMap { letter[$0] }.joined(separator: " ")
    }

    private static let letter: [Character: String] = [
        "a": "eɪ", "b": "bi", "c": "si", "d": "di", "e": "i",
        "f": "ɛf", "g": "ʤi", "h": "eɪʧ", "i": "aɪ", "j": "ʤeɪ",
        "k": "keɪ", "l": "ɛl", "m": "ɛm", "n": "ɛn", "o": "oʊ",
        "p": "pi", "q": "kju", "r": "ɑɹ", "s": "ɛs", "t": "ti",
        "u": "ju", "v": "vi", "w": "ˈdʌbəlju", "x": "ɛks", "y": "waɪ",
        "z": "zi"
    ]

    /// Hand-curated dictionary. Stress markers omitted; diphthongs split into
    /// adjacent vowel symbols since Kokoro tokenises per IPA char.
    private static let dict: [String: String] = [
        // greetings
        "hello": "həloʊ", "hi": "haɪ", "hey": "heɪ",
        "bye": "baɪ", "goodbye": "ɡʊdbaɪ",
        // affirm / deny
        "yes": "jɛs", "yeah": "jɛə", "yep": "jɛp",
        "no": "noʊ", "nope": "noʊp", "not": "nɑt",
        "maybe": "meɪbi", "ok": "oʊkeɪ", "okay": "oʊkeɪ",
        "sure": "ʃʊɹ", "alright": "ɔlɹaɪt",
        // pronouns
        "i": "aɪ", "me": "mi", "my": "maɪ", "mine": "maɪn",
        "you": "ju", "your": "jɔɹ", "yours": "jɔɹz",
        "we": "wi", "us": "ʌs", "our": "aʊɹ", "ours": "aʊɹz",
        "he": "hi", "him": "hɪm", "his": "hɪz",
        "she": "ʃi", "her": "hɚ", "hers": "hɚz",
        "it": "ɪt", "its": "ɪts",
        "they": "ðeɪ", "them": "ðɛm", "their": "ðɛɹ",
        // copula / aux
        "is": "ɪz", "am": "æm", "are": "ɑɹ",
        "was": "wʌz", "were": "wɚ", "be": "bi", "been": "bɪn",
        "have": "hæv", "has": "hæz", "had": "hæd",
        "do": "du", "does": "dʌz", "did": "dɪd",
        "will": "wɪl", "would": "wʊd", "can": "kæn", "could": "kʊd",
        "should": "ʃʊd", "shall": "ʃæl", "may": "meɪ", "might": "maɪt",
        // conjunctions / preps
        "and": "ænd", "or": "ɔɹ", "but": "bʌt", "if": "ɪf",
        "then": "ðɛn", "than": "ðæn", "so": "soʊ", "because": "bɪkʌz",
        "the": "ðə", "a": "ə", "an": "ən", "of": "əv",
        "in": "ɪn", "on": "ɔn", "at": "æt", "to": "tu",
        "from": "fɹʌm", "for": "fɔɹ", "with": "wɪð",
        "by": "baɪ", "about": "əbaʊt", "as": "æz",
        // questions
        "what": "wʌt", "who": "hu", "where": "wɛɹ", "when": "wɛn",
        "why": "waɪ", "how": "haʊ", "which": "wɪʧ",
        // demonstratives / location
        "this": "ðɪs", "that": "ðæt", "these": "ðiz", "those": "ðoʊz",
        "here": "hɪɹ", "there": "ðɛɹ", "now": "naʊ",
        // numbers
        "zero": "zɪɹoʊ", "one": "wʌn", "two": "tu", "three": "θɹi",
        "four": "fɔɹ", "five": "faɪv", "six": "sɪks", "seven": "sɛvən",
        "eight": "eɪt", "nine": "naɪn", "ten": "tɛn",
        // courtesy
        "please": "pliz", "thanks": "θæŋks", "thank": "θæŋk",
        "sorry": "sɑɹi", "excuse": "ɪkskjuz",
        // common verbs
        "go": "ɡoʊ", "going": "ɡoʊɪŋ", "gone": "ɡɔn",
        "come": "kʌm", "coming": "kʌmɪŋ",
        "see": "si", "saw": "sɔ", "seen": "sin",
        "say": "seɪ", "said": "sɛd", "tell": "tɛl", "told": "toʊld",
        "think": "θɪŋk", "thought": "θɔt",
        "know": "noʊ", "knew": "nu",
        "like": "laɪk", "want": "wɑnt", "wanted": "wɑntɪd",
        "need": "nid", "needs": "nidz",
        "help": "hɛlp", "make": "meɪk", "made": "meɪd",
        "take": "teɪk", "took": "tʊk", "taken": "teɪkən",
        "give": "ɡɪv", "gave": "ɡeɪv", "got": "ɡɑt", "get": "ɡɛt",
        "find": "faɪnd", "found": "faʊnd",
        "look": "lʊk", "ask": "æsk", "play": "pleɪ",
        "work": "wɝk", "open": "oʊpən", "close": "kloʊz",
        "start": "stɑɹt", "stop": "stɑp",
        // common nouns
        "thing": "θɪŋ", "things": "θɪŋz", "time": "taɪm",
        "day": "deɪ", "night": "naɪt", "today": "tədeɪ",
        "tomorrow": "təmɑɹoʊ", "yesterday": "jɛstɚdeɪ",
        "name": "neɪm", "person": "pɝsən", "people": "pipəl",
        "way": "weɪ", "world": "wɝld",
        "home": "hoʊm", "house": "haʊs",
        "music": "mjuzɪk", "song": "sɔŋ", "light": "laɪt",
        "lights": "laɪts", "volume": "vɑljəm", "battery": "bætəɹi",
        "robot": "ɹoʊbɑt", "bot": "bɑt", "voice": "vɔɪs",
        "model": "mɑdəl", "phone": "foʊn",
        // common adjectives
        "good": "ɡʊd", "bad": "bæd", "great": "ɡɹeɪt", "nice": "naɪs",
        "hot": "hɑt", "cold": "koʊld", "warm": "wɔɹm",
        "big": "bɪɡ", "small": "smɔl", "little": "lɪtəl",
        "old": "oʊld", "new": "nu", "young": "jʌŋ",
        "happy": "hæpi", "sad": "sæd", "fine": "faɪn",
        "right": "ɹaɪt", "wrong": "ɹɔŋ", "real": "ɹiəl",
        "slow": "sloʊ", "fast": "fæst", "loud": "laʊd", "quiet": "kwaɪət",
        // adverbs / common
        "very": "vɛɹi", "really": "ɹiəli", "too": "tu",
        "also": "ɔlsoʊ", "just": "ʤʌst", "only": "oʊnli",
        "more": "mɔɹ", "less": "lɛs", "most": "moʊst",
        "all": "ɔl", "any": "ɛni", "some": "sʌm", "none": "nʌn",
        "always": "ɔlweɪz", "never": "nɛvɚ", "again": "əɡɛn",
        // contractions
        "i'm": "aɪm", "you're": "jʊɹ", "he's": "hiz", "she's": "ʃiz",
        "it's": "ɪts", "we're": "wɪɹ", "they're": "ðɛɹ",
        "don't": "doʊnt", "doesn't": "dʌzənt", "didn't": "dɪdənt",
        "can't": "kænt", "won't": "woʊnt", "isn't": "ɪzənt",
        "aren't": "ɑɹənt", "wasn't": "wʌzənt",
        "i'll": "aɪl", "you'll": "jul", "we'll": "wil",
        "i've": "aɪv", "you've": "juv", "we've": "wiv",
        // commands a voice bot hears a lot
        "turn": "tɝn", "set": "sɛt", "show": "ʃoʊ",
        "pause": "pɔz",
        "next": "nɛkst", "back": "bæk", "off": "ɔf"
    ]
}
