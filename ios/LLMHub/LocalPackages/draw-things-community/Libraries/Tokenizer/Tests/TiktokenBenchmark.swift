import Foundation
import QuartzCore

// Compiled with the working sources and generated reference sources by benchmark_tiktoken.py.
@main
struct TiktokenBenchmark {
  struct Sample: Decodable {
    let name: String
    let text: String
  }

  static func main() throws {
    setbuf(stdout, nil)
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    let resources = root.appendingPathComponent("Libraries/BinaryResources/Resources")
    let corpus = try JSONDecoder().decode(
      [Sample].self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])))
    for (name, stem, specialTokens, unknown, start, end, pretokenizer) in [
      (
        "DeepSeek4.1 unsafe", "deepseek4", DeepSeekModel.specialTokens,
        "<｜▁pad▁｜>", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
        TiktokenTokenizer.Pretokenizer.joyAI
      ),
      (
        "DeepSeek4.1 safe", "deepseek4",
        ["<｜begin▁of▁sentence｜>": Int32(0), "<｜end▁of▁sentence｜>": 1, "<｜▁pad▁｜>": 2],
        "<｜▁pad▁｜>", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", .joyAI
      ),
      (
        "Qwen3.5 unsafe", "qwen3.5", QwenModel.specialTokens,
        "<|endoftext|>", "<|endoftext|>", "<|im_end|>", .llama3
      ),
      (
        "Qwen3.5 safe", "qwen3.5", ["<|endoftext|>": Int32(248_044)],
        "<|endoftext|>", "<|endoftext|>", "<|endoftext|>", .llama3
      ),
    ] {
      let vocabulary = try Data(contentsOf: resources.appendingPathComponent("vocab_\(stem).json"))
      let merges = try Data(contentsOf: resources.appendingPathComponent("merges_\(stem).txt"))
      let original = ReferenceTiktokenTokenizer(
        vocabulary: vocabulary, merges: merges, specialTokens: specialTokens,
        unknownToken: unknown, startToken: start, endToken: end,
        pretokenizer: pretokenizer == .joyAI ? .joyAI : .llama3)
      let updated = TiktokenTokenizer(
        vocabulary: vocabulary, merges: merges, specialTokens: specialTokens,
        unknownToken: unknown, startToken: start, endToken: end, pretokenizer: pretokenizer)
      let samples =
        corpus + [
          Sample(
            name: "all special tokens",
            text: specialTokens.keys.sorted().joined(separator: " text\n")),
          Sample(name: "adjacent special tokens", text: specialTokens.keys.sorted().joined()),
          Sample(
            name: "partial special tokens",
            text: specialTokens.keys.sorted().map {
              String($0.dropLast())
            }.joined(separator: " text\n")),
          Sample(
            name: "Unicode and controls",
            text: String(
              repeating:
                "English 中文 日本語 한국어 العربية हिन्दी e\u{301} é 👩🏽‍💻 🇺🇸\r\n\t\u{0}\u{200b} punctuation... <think>x</think>\n",
              count: 200)),
          Sample(name: "empty", text: ""),
          Sample(name: "built-in DeepSeek4.1 prompt", text: systemPromptForDeepSeek4_1),
          Sample(name: "built-in Qwen3.5 prompt", text: systemPromptForQwen3_5),
        ]
      _ = original.tokenize(text: "Warm up <think>hello</think>\n")
      _ = updated.tokenize(text: "Warm up <think>hello</think>\n")
      var referenceSeconds = 0.0
      var updatedSeconds = 0.0
      var byteCount = 0
      var tokenCount = 0
      print("\(name): \(samples.count) samples; \(specialTokens.count) special tokens")
      for (index, sample) in samples.enumerated() {
        let expected: ([Int32], [String])
        let actual: ([Int32], [String])
        // Alternate execution order to reduce warm-cache/order bias.
        let start = CACurrentMediaTime()
        if index % 2 == 0 {
          expected = original.tokenize(text: sample.text, addSpecialTokens: true)
          let middle = CACurrentMediaTime()
          actual = updated.tokenize(text: sample.text, addSpecialTokens: true)
          referenceSeconds += middle - start
          updatedSeconds += CACurrentMediaTime() - middle
        } else {
          actual = updated.tokenize(text: sample.text, addSpecialTokens: false)
          let middle = CACurrentMediaTime()
          expected = original.tokenize(text: sample.text, addSpecialTokens: false)
          updatedSeconds += middle - start
          referenceSeconds += CACurrentMediaTime() - middle
        }
        precondition(expected.0 == actual.0, "Token ID mismatch: \(name), \(sample.name)")
        precondition(
          expected.1.count == actual.1.count, "Token piece count: \(name), \(sample.name)")
        for (a, b) in zip(expected.1, actual.1) {
          precondition(a.utf8.elementsEqual(b.utf8), "Token byte mismatch: \(name), \(sample.name)")
        }
        byteCount += sample.text.utf8.count
        tokenCount += actual.0.count
        if index % 25 == 0 || index + 1 == samples.count {
          print("Verified \(index + 1)/\(samples.count): \(byteCount) bytes, \(tokenCount) tokens")
        }
      }
      print(
        String(
          format: "TOKENIZER %@: %d bytes, %d tokens; reference %.3f s, updated %.3f s, %.2fx",
          name, byteCount, tokenCount, referenceSeconds, updatedSeconds,
          referenceSeconds / updatedSeconds))
      if name == "DeepSeek4.1 unsafe" {
        // Measure the special-token regex separately on the actual built-in prompt.
        let alternatives = specialTokens.keys.map {
          NSRegularExpression.escapedPattern(for: $0)
        }.joined(separator: "|")
        let originalRegex = try Regex("(\(alternatives))")
        let updatedRegex = try Regex(
          TiktokenTokenizer.specialTokensPattern(Array(specialTokens.keys)))
        var originalRegexSeconds = 0.0
        var updatedRegexSeconds = 0.0
        for _ in 0..<5 {
          let start = CACurrentMediaTime()
          let expected = systemPromptForDeepSeek4_1.matches(of: originalRegex).map(\.range)
          let middle = CACurrentMediaTime()
          let actual = systemPromptForDeepSeek4_1.matches(of: updatedRegex).map(\.range)
          let end = CACurrentMediaTime()
          precondition(expected == actual, "Regex match ranges changed")
          originalRegexSeconds += middle - start
          updatedRegexSeconds += end - middle
        }
        print(
          String(
            format: "REGEX mean of 5: reference %.3f ms, updated %.3f ms",
            originalRegexSeconds * 200, updatedRegexSeconds * 200))

        let safeTokens: [String: Int32] = [
          "<｜begin▁of▁sentence｜>": 0, "<｜end▁of▁sentence｜>": 1, "<｜▁pad▁｜>": 2,
        ]
        let originalSafe = ReferenceTiktokenTokenizer(
          vocabulary: vocabulary, merges: merges, specialTokens: safeTokens,
          unknownToken: unknown, startToken: "<｜begin▁of▁sentence｜>", endToken: end,
          pretokenizer: .joyAI)
        let safe = TiktokenTokenizer(
          vocabulary: vocabulary, merges: merges, specialTokens: safeTokens,
          unknownToken: unknown, startToken: "<｜begin▁of▁sentence｜>", endToken: end,
          pretokenizer: .joyAI)
        let sections: [SystemPromptSection] = [
          (systemPromptForDeepSeek4_1 + "\n", true), ("\nWorkspace: /tmp/project\n", false),
        ]
        var referencePromptSeconds = 0.0
        var updatedPromptSeconds = 0.0
        for _ in 0..<5 {
          let start = CACurrentMediaTime()
          let expected = baselineTokenizeSystemPromptSections(
            sections, unsafeTokenizer: original, safeTokenizer: originalSafe)
          let middle = CACurrentMediaTime()
          let actual = tokenizeSystemPromptSections(
            sections, unsafeTokenizer: updated, safeTokenizer: safe)
          let end = CACurrentMediaTime()
          precondition(expected == actual, "Prompt token IDs changed")
          referencePromptSeconds += middle - start
          updatedPromptSeconds += end - middle
        }
        print(
          String(
            format: "PROMPT mean of 5 (%d bytes): reference %.3f ms, updated %.3f ms",
            systemPromptForDeepSeek4_1.utf8.count, referencePromptSeconds * 200,
            updatedPromptSeconds * 200))
      }
    }
    print("PASS: all token IDs and token-piece UTF-8 bytes matched exactly.")
  }
}
