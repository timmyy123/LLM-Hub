import Foundation

public struct MultiLineAttentionCLIPTokenizer {
  // Will use a CLIP tokenizer underneath.
  private var attention: AttentionCLIPTokenizer
  public var endToken: Int32 { attention.endToken }
  public var unknownToken: Int32 { attention.unknownToken }
  public init(vocabulary: Data, merges: Data) {
    attention = AttentionCLIPTokenizer(vocabulary: vocabulary, merges: merges)
  }
}

extension MultiLineAttentionCLIPTokenizer: Tokenizer {
  var vocabulary: [String: Int32] { attention.vocabulary }
  public func tokenize(
    text: String, truncation: Bool, maxLength: Int, paddingToken: Int32?, addSpecialTokens: Bool
  ) -> (
    [String], [Int32], [Float], [String?], [Int]
  ) {
    guard !text.isEmpty else {
      var (strs, ids, weights, canonicals, _) = attention.tokenize(
        text: "", truncation: false, maxLength: 0, paddingToken: paddingToken,
        addSpecialTokens: addSpecialTokens)
      // We have to get correct lengths of each.
      let lengthOfEach = [ids.count - 2]
      if truncation {
        if ids.count > maxLength {
          let isLastPaddingToken = ids[maxLength - 1] == paddingToken
          let k = ids.count - maxLength + 1
          strs.removeLast(k)
          ids.removeLast(k)
          weights.removeLast(k)
          canonicals.removeLast(k)
          if !isLastPaddingToken {  // If the last is not padding, appending a endToken
            strs.append("")
            ids.append(endToken)
            weights.append(1)
            canonicals.append("")
          }
        }
      }
      if ids.count < maxLength {
        let paddingToken = paddingToken ?? endToken
        let k = maxLength - ids.count
        for _ in 0..<k {
          strs.append("")
          ids.append(paddingToken)
          weights.append(1)
          canonicals.append("")
        }
      }
      return (strs, ids, weights, canonicals, lengthOfEach)
    }
    let lines = text.split(separator: "\n")
    var allStrs = [String]()
    var allIds = [Int32]()
    var allWeights = [Float]()
    var allCanonicals = [String?]()
    var lengthOfEach = [Int]()
    for (i, line) in lines.enumerated() {
      var (strs, ids, weights, canonicals, _) = attention.tokenize(
        text: String(line), truncation: false, maxLength: 0, paddingToken: paddingToken,
        addSpecialTokens: addSpecialTokens)
      lengthOfEach.append(ids.count - 2)  // Remove start / end regardless.
      if i > 0 {
        strs.removeFirst()
        ids.removeFirst()
        weights.removeFirst()
        canonicals.removeFirst()
      }
      if i < lines.count - 1 {
        strs.removeLast()
        ids.removeLast()
        weights.removeLast()
        canonicals.removeLast()
      }
      allStrs.append(contentsOf: strs)
      allIds.append(contentsOf: ids)
      allWeights.append(contentsOf: weights)
      allCanonicals.append(contentsOf: canonicals)
    }
    if truncation {
      if allIds.count > maxLength {
        let isLastPaddingToken = allIds[maxLength - 1] == paddingToken
        let k = allIds.count - maxLength + 1
        allStrs.removeLast(k)
        allIds.removeLast(k)
        allWeights.removeLast(k)
        allCanonicals.removeLast(k)
        if !isLastPaddingToken {  // If the last is not padding, appending a endToken
          allStrs.append("")
          allIds.append(endToken)
          allWeights.append(1)
          allCanonicals.append("")
        }
      }
    }
    if allIds.count < maxLength {
      let paddingToken = paddingToken ?? endToken
      let k = maxLength - allIds.count
      for _ in 0..<k {
        allStrs.append("")
        allIds.append(paddingToken)
        allWeights.append(1)
        allCanonicals.append("")
      }
    }
    var lengthSum = 1
    for (i, each) in lengthOfEach.enumerated() {
      lengthSum += each
      guard lengthSum + 1 >= allIds.count else { continue }
      lengthOfEach[i] -= lengthSum + 1 - allIds.count
      if i < lengthOfEach.count - 1 {
        lengthOfEach.removeLast(lengthOfEach.count - 1 - i)
      }
      break
    }
    return (allStrs, allIds, allWeights, allCanonicals, lengthOfEach)
  }
}
