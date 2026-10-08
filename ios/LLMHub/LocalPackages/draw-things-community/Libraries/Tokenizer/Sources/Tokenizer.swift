public protocol Tokenizer {
  var endToken: Int32 { get }
  var unknownToken: Int32 { get }
  func tokenize(
    text: String, truncation: Bool, maxLength: Int, paddingToken: Int32?, addSpecialTokens: Bool
  ) -> (
    [String], [Int32], [Float], [String?], [Int]
  )
}

extension Tokenizer {
  public func tokenize(text: String, truncation: Bool, maxLength: Int) -> (
    //  return: (strs, tokens, weights, canonicals, lengthsOfEach)
    [String], [Int32], [Float], [String?], [Int]
  ) {
    tokenize(
      text: text, truncation: truncation, maxLength: maxLength, paddingToken: nil,
      addSpecialTokens: true)
  }
}

public protocol TextualInversionPoweredTokenizer {
  func textualInversion(for token: Int32) -> String?
  func isTextualInversion(_ token: Int32) -> Bool
  var textualInversions: [String] { get }
}

extension Tokenizer {
  public func automaticLineBreak(_ prompt: String, maxLength: Int = 77) -> String {
    let (_, tokens, _, _, _) = tokenize(
      text: prompt, truncation: false, maxLength: maxLength, paddingToken: endToken,
      addSpecialTokens: true)
    if tokens.count > maxLength {
      var indexOfWord = -1
      // -1 means no word, 0 means first word, to align the real index in the array which is 0 base
      var indexOfWords = [Int]()
      for i in 0..<tokens.count {
        if tokens[i] == 267 {
          indexOfWord += 1
        }
        if i > 0 && i % (maxLength - 2) == 0 {
          // every time when current token index fills current chunk to 75, add the last valid prompt, we will use it to insert \n in the next step
          indexOfWords.append(indexOfWord)
        }
      }

      let promptArray = prompt.split(separator: ",").map { String($0) }
      var result = ""
      var results = [String]()
      indexOfWord = -1
      if !indexOfWords.isEmpty {
        indexOfWord = indexOfWords.removeFirst()
      }

      for (i, prompt) in promptArray.enumerated() {
        if !result.isEmpty {
          result += ","
        }
        result += prompt
        if indexOfWord > -1 && i == indexOfWord {
          results.append(AttentionCLIPTokenizer.format(result))
          result = ""
          if !indexOfWords.isEmpty {
            indexOfWord = indexOfWords.removeFirst()
          } else {
            indexOfWord = -1
          }
        }
      }
      if !result.isEmpty {
        results.append(AttentionCLIPTokenizer.format(result))
      }
      return results.joined(separator: "\n")
    }
    return prompt
  }
}
