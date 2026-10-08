import Foundation

public struct AttentionCLIPTokenizer {
  // Will use a CLIP tokenizer underneath.
  private var CLIP: CLIPTokenizer
  public var endToken: Int32 { CLIP.endToken }
  public var unknownToken: Int32 { CLIP.unknownToken }
  public init(vocabulary: Data, merges: Data) {
    CLIP = CLIPTokenizer(vocabulary: vocabulary, merges: merges)
  }
}

extension AttentionCLIPTokenizer: Tokenizer {
  var vocabulary: [String: Int32] { CLIP.vocabulary }
  public func tokenize(
    text: String, truncation: Bool, maxLength: Int, paddingToken: Int32?, addSpecialTokens: Bool
  ) -> (
    [String], [Int32], [Float], [String?], [Int]
  ) {
    guard !text.isEmpty else {
      return CLIP.tokenize(
        text: text, truncation: truncation, maxLength: maxLength, paddingToken: paddingToken,
        addSpecialTokens: addSpecialTokens)
    }
    // Should support same syntax:
    // (abc) - increases attention to abc by a mupltiplier of 1.1
    // (abc:3.12) - increases attention to abc by a multiplier of 3.2
    // [abc] - decreases attention to abc by a multiplier of 1.1 (1 / 1.1)
    // Some examples to look out:
    // 'normal text'
    // '(unbalanced'
    // '(literal]'
    // '(unneccessary)(parens)'
    // 'a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).'
    var i = text.endIndex
    var characters = [String]()
    var characterWeights = [Float]()
    let roundBracketScale: Float = 1.1
    let squareBracketScale: Float = 1.0 / 1.1
    var scales = [Float(1)]
    while i > text.startIndex {
      i = text.index(before: i)
      let c = text[i]
      switch c {
      case ")":
        // Look ahead to see if there is a manually specified scale. Ignoring white space.
        var fractional = ""
        var integral = ""
        var j = i
        var customScale = false
        while j > text.startIndex {
          j = text.index(before: j)
          let c = text[j]
          guard !c.isWhitespace else { continue }
          guard c != ":" && c.isASCII && (c.isNumber || c == ".") else {
            if c == ":" {
              customScale = true
              i = j  // Swallow all the way to ":". This also marks we will continue to process.
            }
            break
          }
          // Compute the number. Assuming it is integral, until encountered a .
          if c == "." && fractional.isEmpty {
            fractional = integral
            integral = ""
          } else {
            integral = String(c) + integral
          }
        }
        if customScale, !fractional.isEmpty || !integral.isEmpty,
          let scale = Float(integral + "." + fractional)
        {
          scales.append(scales.last! * scale)
        } else {
          scales.append(scales.last! * roundBracketScale)
        }
      case "]":
        scales.append(scales.last! * squareBracketScale)
      case "(", "[":  // I cannot think of anyway which one closes matters, ignoring it.
        if scales.count > 1 {
          scales.removeLast()
        }
      default:
        characters.append(String(c))
        characterWeights.append(scales.last!)
      }
    }
    characterWeights.reverse()
    let text = characters.reversed().joined().lowercased()
    let (strs, ids, _, canonicals, _) = CLIP.tokenize(
      text: text, truncation: truncation, maxLength: maxLength, paddingToken: paddingToken,
      addSpecialTokens: addSpecialTokens)
    // Looking back each strs in the text to find its character weights. Note that this is quadratic, but we have 77 as length.
    var tokenWeights = [Float](repeating: 1, count: ids.count)
    var tokenLength = ids.count
    for i in (0..<ids.count).reversed() {
      if ids[i] != CLIP.endToken {
        tokenLength = min(i + 2, ids.count)
        break
      }
    }
    for index in 0..<tokenLength {
      // Remove </w> if there is.
      var str = strs[index]
      if str.hasSuffix("</w>") {
        str = String(str.prefix(upTo: str.index(str.endIndex, offsetBy: -4)))
      }
      // Now we have raw text, count how many raw text up to the index we have in the preceding tokens.
      var precedingOccurrences = 0
      for i in 0..<index {
        precedingOccurrences += strs[i].components(separatedBy: str).count - 1
      }
      // Now, find the start position.
      var startIndex = text.startIndex
      var occurrences = 0
      var strRange: Range<String.Index>? = nil
      while let range = text.range(of: str, range: startIndex..<text.endIndex) {
        occurrences += 1
        startIndex = range.upperBound
        if occurrences > precedingOccurrences {
          // This is the range.
          strRange = range
          break
        }
      }
      if let range = strRange {  // All characters in this range should have the same weights. Just use the first.
        let i = text.distance(from: text.startIndex, to: range.lowerBound)
        tokenWeights[index] = characterWeights[i]
      }
    }
    return (strs, ids, tokenWeights, canonicals, [ids.count - 2])
  }
}

extension AttentionCLIPTokenizer {
  public static func format(_ text: String) -> String {
    // First, do some trivial fix, remove leading / trailing spaces, remove more spaces than needed.
    let firstStageFix = text.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: " ")
      .joined(separator: " ")
    // Second, go through enclosing characters and make sure they have proper spaces between.
    // These enclosing characters are [{("'<
    var secondStageFix = ""
    var i = firstStageFix.startIndex
    var doubleQuotes = 0
    var singleQuotes = 0
    var backQuotes = 0
    while i < firstStageFix.endIndex {
      let c = firstStageFix[i]
      switch c {
      case "(", "[", "{", "<":
        let next = firstStageFix.index(after: i)
        if next < firstStageFix.endIndex && firstStageFix[next].isWhitespace {
          // Move the whitespace out of (.
          secondStageFix += " "
          i = next  // Move to space.
        }
        secondStageFix += String(c)
      case ")", "]", "}", ">":
        // These may have special means ( ) and ] definitely have), we don't add space if there is none.
        if secondStageFix.last?.isWhitespace ?? false {
          secondStageFix.removeLast()
          secondStageFix += String(c) + " "
        } else {
          secondStageFix += String(c)
        }
      case ",", ";", ".":
        if secondStageFix.last?.isWhitespace ?? false {
          secondStageFix.removeLast()
        }
        secondStageFix += String(c)
        // Add additional space, it is fine, we will remove it as the last stage.
        secondStageFix += " "
      case "\"":
        doubleQuotes += 1
        if doubleQuotes % 2 == 1 {
          let next = firstStageFix.index(after: i)
          if next < firstStageFix.endIndex && firstStageFix[next].isWhitespace {
            i = next
          }
          // We add extra whitespace regardless, it is meaningless.
          secondStageFix += " " + String(c)
        } else {
          if secondStageFix.last?.isWhitespace ?? false {
            secondStageFix.removeLast()
          }
          secondStageFix += String(c) + " "
        }
      case "'":
        // Don't count if it is one of these:
        let restOfFirstStageFix = firstStageFix[i...]
        guard
          !(restOfFirstStageFix.hasPrefix("'s") || restOfFirstStageFix.hasPrefix("'t")
            || restOfFirstStageFix.hasPrefix("'m") || restOfFirstStageFix.hasPrefix("'d")
            || restOfFirstStageFix.hasPrefix("'re") || restOfFirstStageFix.hasPrefix("'ve")
            || restOfFirstStageFix.hasPrefix("'ll"))
        else {
          secondStageFix += String(c)
          break
        }
        singleQuotes += 1
        if singleQuotes % 2 == 1 {
          let next = firstStageFix.index(after: i)
          if next < firstStageFix.endIndex && firstStageFix[next].isWhitespace {
            i = next  // Skip that whitespace.
          }
          // We add extra whitespace regardless, it is meaningless.
          secondStageFix += " " + String(c)
        } else {
          if secondStageFix.last?.isWhitespace ?? false {
            secondStageFix.removeLast()
          }
          secondStageFix += String(c) + " "
        }
      case "`":
        backQuotes += 1
        if backQuotes % 2 == 1 {
          let next = firstStageFix.index(after: i)
          if next < firstStageFix.endIndex && firstStageFix[next].isWhitespace {
            i = next
          }
          // We add extra whitespace regardless, it is meaningless.
          secondStageFix += " "
          secondStageFix += String(c)
        } else {
          if secondStageFix.last?.isWhitespace ?? false {
            secondStageFix.removeLast()
          }
          secondStageFix += String(c) + " "
        }
      case ":":
        // This requires special treatment, if it is not used with attention (namely, ":number)"), we insert a space after. Otherwise, we keep it compact.
        // This look up borrows from above tokenizer.
        var fractional = ""
        var integral = ""
        var customScale = false
        var j = firstStageFix.index(after: i)
        let iOld = i
        while j < firstStageFix.endIndex {
          let c = text[j]
          guard !c.isWhitespace else {
            j = firstStageFix.index(after: j)
            continue
          }
          guard c != ")" && c.isASCII && (c.isNumber || c == ".") else {
            if c == ")" {
              customScale = true
              i = j  // Swallow all the way to ":". This also marks we will continue to process.
            }
            break
          }
          // Compute the number. Assuming it is integral, until encountered a .
          if c == "." && integral.isEmpty {
            integral = fractional
            fractional = ""
          } else {
            fractional = String(c) + fractional
          }
          j = firstStageFix.index(after: j)
        }
        if customScale, !fractional.isEmpty || !integral.isEmpty,
          let _ = Float(integral + "." + fractional)
        {
          if integral.isEmpty {
            integral = "0"
          }
          if secondStageFix.last?.isWhitespace ?? false {
            secondStageFix.removeLast()
            // Notice the extra space at the end.
            secondStageFix +=
              String(c) + integral + (fractional.isEmpty ? "" : "." + fractional) + ") "
          } else {
            secondStageFix +=
              String(c) + integral + (fractional.isEmpty ? "" : "." + fractional) + ")"
          }
        } else {
          if secondStageFix.last?.isWhitespace ?? false {
            secondStageFix.removeLast()
          }
          secondStageFix += String(c) + " "
          i = iOld
        }
      default:
        secondStageFix += String(c)
      }
      if i < firstStageFix.endIndex {
        i = firstStageFix.index(after: i)
      }
    }
    // Fix additional spaces introduced when we do second stage fixes.
    let lastStageFix = secondStageFix.split(separator: " ").joined(separator: " ")
    return lastStageFix
  }
}
