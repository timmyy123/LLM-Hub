import Foundation

private func normalize(_ text: String) -> String {
  let decomposed = text.precomposedStringWithCompatibilityMapping
    .decomposedStringWithCanonicalMapping
  var lowercase = ""
  for scalar in decomposed.unicodeScalars {
    // Rust tokenizers removes all Unicode Mark categories, including spacing marks.
    if [.nonspacingMark, .spacingMark, .enclosingMark].contains(scalar.properties.generalCategory) {
      continue
    }
    // tokenizers lowercases scalars independently (no contextual final sigma).
    lowercase += String(scalar).lowercased()
  }
  var collapsed = ""
  var previousSpace = false
  for scalar in lowercase.unicodeScalars {
    if [9, 10, 13, 32].contains(scalar.value) {
      if !previousSpace { collapsed += " " }
      previousSpace = true
    } else {
      collapsed.unicodeScalars.append(scalar)
      previousSpace = false
    }
  }
  if collapsed == " " { collapsed = "\u{e000}" }
  let trimmed = collapsed.unicodeScalars.drop(while: { $0.properties.isWhitespace })
    .reversed().drop(while: { $0.properties.isWhitespace }).reversed()
  return String(String.UnicodeScalarView(trimmed)).replacingOccurrences(of: "\u{e000}", with: " ")
}

/// Decode one ByteLevel token at a time, normalize it, and assign compressed IDs
/// in vocabulary order. Dictionary keys use UTF-8 bytes rather than Swift's
/// canonically equivalent String equality, matching the training implementation.
public func DeepSeek4_1EngramTokenMap(
  vocabularyMap: [String: Int32], specialTokens: [String: Int32], vocabularySize: Int = 129_280
) -> [Int32] {
  precondition(vocabularySize > 0)
  var tokens = [String?](repeating: nil, count: vocabularySize)
  for (token, id) in vocabularyMap {
    precondition(
      id >= 0 && Int(id) < vocabularySize && tokens[Int(id)] == nil,
      "The Engram vocabulary has invalid or duplicate token IDs.")
    tokens[Int(id)] = token
  }
  var addedIDs = Set<Int32>()
  for (token, id) in specialTokens {
    guard id >= 0, Int(id) < vocabularySize, addedIDs.insert(id).inserted else {
      fatalError("The Engram special tokens have invalid or duplicate token IDs.")
    }
    tokens[Int(id)] = token
  }
  let direct = Set(Array(33...126) + Array(161...172) + Array(174...255))
  var byteMap = [Unicode.Scalar: UInt8]()
  var extra = 0
  for byte in 0...255 {
    if direct.contains(byte) {
      byteMap[Unicode.Scalar(byte)!] = UInt8(byte)
    } else {
      byteMap[Unicode.Scalar(256 + extra)!] = UInt8(byte)
      extra += 1
    }
  }
  var seen = [Data: Int32]()
  var result = [Int32]()
  result.reserveCapacity(vocabularySize)
  for token in tokens {
    guard let token else {
      fatalError("The Engram vocabulary is incomplete.")
    }
    let bytes = token.unicodeScalars.compactMap { byteMap[$0] }
    // ByteLevel preserves an entire token containing a character outside its
    // byte alphabet, such as the checkpoint's added special tokens.
    let text =
      bytes.count == token.unicodeScalars.count ? String(decoding: bytes, as: UTF8.self) : token
    let key: Data
    if text.unicodeScalars.contains("\u{fffd}") {
      // Test the scalar, not a Character: a replacement can share a grapheme
      // with a following accent. Partial UTF-8 tokens retain their raw spelling.
      key = Data(token.utf8)
    } else {
      let normalized = normalize(text)
      key = Data((normalized.isEmpty ? text : normalized).utf8)
    }
    let id = seen[key] ?? Int32(seen.count)
    seen[key] = id
    result.append(id)
  }
  return result
}
