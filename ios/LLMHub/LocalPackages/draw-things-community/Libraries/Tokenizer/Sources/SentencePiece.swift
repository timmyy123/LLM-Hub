import Foundation
import SentencePiece

public struct SentencePieceTokenizer: Tokenizer {
  public var startToken: Int32?
  public var endToken: Int32 { internalEndToken ?? 0 }
  public var unknownToken: Int32 { 0 }
  private let sentencePiece: SentencePiece
  private let tokenShift: Int32
  private let internalEndToken: Int32?
  public init(data: Data, startToken: Int32?, endToken: Int32?, tokenShift: Int32) {
    vocabulary = [:]
    sentencePiece = SentencePiece(data: data)
    self.startToken = startToken
    self.internalEndToken = endToken
    self.tokenShift = tokenShift
  }

  public func tokenize(
    text: String, truncation: Bool, maxLength: Int, paddingToken: Int32?, addSpecialTokens: Bool
  ) -> (
    [String], [Int32], [Float], [String?], [Int]
  ) {
    let result = sentencePiece.encode(text)
    var strs = [String]()
    var ids = [Int32]()
    var canonicals = [String?]()
    if let startToken = startToken {
      strs.append("")
      ids.append(startToken)
      canonicals.append("<|startoftext|>")
    }
    let addedTokens = (startToken != nil ? 1 : 0) + (internalEndToken != nil ? 1 : 0)
    if truncation {
      for (i, spt) in result.enumerated() {
        guard i < maxLength + addedTokens else { break }
        strs.append(spt.surface)
        ids.append(spt.id + tokenShift)
        canonicals.append(spt.piece)
      }
    } else {
      for spt in result {
        strs.append(spt.surface)
        ids.append(spt.id + tokenShift)
        canonicals.append(spt.piece)
      }
    }
    if let endToken = internalEndToken {
      strs.append("")
      ids.append(endToken)
      canonicals.append("<|endoftext|>")
    }
    let lengthOfTokens = ids.count - addedTokens
    let paddingToken = paddingToken ?? tokenShift
    if ids.count < maxLength {
      for _ in ids.count..<maxLength {
        strs.append("")
        ids.append(paddingToken)
        canonicals.append("")
      }
    }
    return (strs, ids, [Float](repeating: 1, count: ids.count), canonicals, [lengthOfTokens])
  }

  public var vocabulary: [String: Int32]
}

extension SentencePieceTokenizer: TextualInversionPoweredTokenizer {
  public func textualInversion(for token: Int32) -> String? {
    return nil
  }
  public func isTextualInversion(_ token: Int32) -> Bool {
    return false
  }
  public var textualInversions: [String] { [] }
}
