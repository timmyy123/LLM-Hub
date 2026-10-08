import Foundation

public struct TextualInversionAttentionCLIPTokenizer {
  // Will use an attention CLIP tokenizer underneath.
  private var multiLineAttention: MultiLineAttentionCLIPTokenizer
  private var textualInversionMapping: [String: Int]
  public var textualInversions: [String] {
    didSet {
      textualInversionMapping.removeAll(keepingCapacity: true)
      for (i, value) in textualInversions.enumerated() {
        textualInversionMapping[value] = i
      }
    }
  }
  public var endToken: Int32 { multiLineAttention.endToken }
  public var unknownToken: Int32 { multiLineAttention.unknownToken }
  public init(vocabulary: Data, merges: Data, textualInversions: [String]) {
    multiLineAttention = MultiLineAttentionCLIPTokenizer(vocabulary: vocabulary, merges: merges)
    self.textualInversions = textualInversions
    var textualInversionMapping = [String: Int]()
    for (i, keyword) in textualInversions.enumerated() {
      textualInversionMapping[keyword] = i
    }
    self.textualInversionMapping = textualInversionMapping
  }
}

extension TextualInversionAttentionCLIPTokenizer: Tokenizer & TextualInversionPoweredTokenizer {
  public func textualInversion(for token: Int32) -> String? {
    let token = Int(token) - vocabulary.count
    return token <= 0 ? nil : textualInversions[token - 1]
  }
  public func isTextualInversion(_ token: Int32) -> Bool {
    return Int(token) >= vocabulary.count
  }
  var vocabulary: [String: Int32] { multiLineAttention.vocabulary }
  public func tokenize(
    text: String, truncation: Bool, maxLength: Int, paddingToken: Int32?, addSpecialTokens: Bool
  ) -> (
    [String], [Int32], [Float], [String?], [Int]
  ) {
    guard !text.isEmpty else {
      return multiLineAttention.tokenize(
        text: text, truncation: truncation, maxLength: maxLength, paddingToken: paddingToken,
        addSpecialTokens: addSpecialTokens)
    }
    var i = text.startIndex
    var tiStart = text.endIndex
    var lastIndex = i
    var textReplacements = [String]()
    var textInversionIndex = [Int?]()
    var characters = [String]()
    while i < text.endIndex {
      let c = text[i]
      switch c {
      case ">":
        guard tiStart != text.endIndex else { break }
        // Check if it contains a textual inversion embedding (or empty). If it does, record location and replace with the textual inversion.
        let textualInversion = String(text[text.index(after: tiStart)..<i].lowercased())
        let index = textualInversionMapping[textualInversion]
        guard textualInversion.isEmpty || index != nil || textualInversion == "|textualinversion|"
        else {
          break
        }
        textInversionIndex.append(index)
        textReplacements.append(String(text[tiStart...i].lowercased()))
        // Check if we have these textual inversions. If not, just do whatever it does. Otherwise, replace it with <|textualinversion|> special token.
        characters.append(String(text[lastIndex..<tiStart]))
        characters.append("<|textualinversion|>")
        lastIndex = text.index(after: i)
      case "<":
        tiStart = i
      default:
        break
      }
      if c.isWhitespace {
        tiStart = text.endIndex  // If it contains white-space, we don't recognize this textual inversion.
      }
      i = text.index(after: i)
    }
    if lastIndex != text.endIndex {
      characters.append(String(text[lastIndex..<text.endIndex]))
    }
    let newText = characters.joined()
    var (strs, ids, weights, canonicals, lengthOfEach) = multiLineAttention.tokenize(
      text: newText, truncation: truncation, maxLength: maxLength, paddingToken: paddingToken,
      addSpecialTokens: addSpecialTokens)
    let tiToken = Int32(vocabulary.count)
    let oldIds = ids
    for (i, id) in oldIds.enumerated() {
      if id == tiToken {
        strs[i] = textReplacements.removeFirst()
        if let index = textInversionIndex.removeFirst() {
          ids[i] = Int32(index + 1) + tiToken
        }
      }
    }
    return (strs, ids, weights, canonicals, lengthOfEach)
  }
}
