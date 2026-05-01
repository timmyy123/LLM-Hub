import Foundation

/// Canonical Kokoro phoneme vocabulary. Mirrors
/// `hexgrad/kokoro/__init__.py: SYMBOLS = [PAD] + PUNCT + LETTERS + LETTERS_IPA`
/// — exact order is required, token IDs are positional.
enum KokoroVocab {

    static let pad = "$"

    static let symbols: [String] = {
        var s: [String] = []
        s.append(pad)
        // 16 punctuation
        s.append(contentsOf: [";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "\"", "«", "»", "“", "”", " "])
        // 52 ASCII letters
        s.append(contentsOf: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".map { String($0) })
        // 109 IPA — must match LETTERS_IPA in hexgrad/kokoro
        s.append(contentsOf: [
            "ɑ","ɐ","ɒ","æ","b","ɓ","ʙ","β","c","ɕ","ç","ɗ","ɖ","ð","ʤ","ə",
            "ɘ","ɚ","ɛ","ɜ","ɝ","e","ɞ","ɟ","ʄ","ɡ","ɠ","ɢ","ʛ","h","ɦ","ɧ",
            "ħ","ɥ","ʜ","i","ɪ","ɨ","ɪ̆","j","ʝ","ɟ","ʄ","k","ɭ","l","ɫ","ɮ",
            "ɬ","ʟ","m","ɱ","n","ɳ","ɲ","ŋ","ɴ","o","ɵ","ø","œ","ɶ","ɔ","ɸ",
            "p","ɸ","q","ʁ","ɽ","ɾ","r","ɹ","ɻ","ʀ","ʀ̥","ɽ","s","ʂ","ɕ","ʃ",
            "ʈ","t","ʈ","ʇ","θ","u","ʉ","ʊ","ʋ","v","ⱱ","ʌ","w","ʍ","x","χ",
            "y","ʏ","ʎ","z","ʑ","ʐ","ʒ","ʔ","ʕ","ʡ","ʢ"," ","ǀ","ǁ","ǂ","ǃ",
            "ˈ","ˌ","ː","ˑ","ʼ","ʴ","ʰ","ʱ","ʲ","ʷ","ˠ","ˤ","˞","↓","↑","→",
            "↗","↘","'","̩","'","ᵻ"
        ])
        return s
    }()

    static let idBySymbol: [String: Int] = {
        var m: [String: Int] = [:]
        for (i, s) in symbols.enumerated() { m[s] = i }
        return m
    }()

    /// Tokenise an IPA phoneme string as Unicode grapheme clusters so that
    /// e.g. "ɪ̆" stays a single token.
    static func tokenize(_ ipa: String) -> [Int] {
        var out: [Int] = []
        for cluster in ipa.unicodeScalars.grouped() {
            let s = String(String.UnicodeScalarView(cluster))
            if let id = idBySymbol[s] {
                out.append(id)
            } else if let first = cluster.first, let id = idBySymbol[String(first)] {
                out.append(id)
            }
        }
        return out
    }
}

/// Group code points into clusters: a base character followed by any combining
/// marks. Mirrors the Android implementation that merges NON_SPACING_MARK
/// (combining) code points onto the preceding base.
private extension String.UnicodeScalarView {
    func grouped() -> [[Unicode.Scalar]] {
        var clusters: [[Unicode.Scalar]] = []
        for scalar in self {
            let cat = Unicode.Scalar(scalar.value)?.properties.generalCategory
            let isCombining = cat == .nonspacingMark || cat == .spacingMark || cat == .enclosingMark
            if isCombining, !clusters.isEmpty {
                clusters[clusters.count - 1].append(scalar)
            } else {
                clusters.append([scalar])
            }
        }
        return clusters
    }
}
