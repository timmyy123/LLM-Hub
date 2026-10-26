// Supertonic text to speech, ported to Swift from Supertone's official C# example
// (https://github.com/supertone-oss-archive/supertonic/tree/main/csharp/Helper.cs), by way of the Windows and Linux
// apps' Supertonic.cs. MIT License, Copyright (c) 2025 Supertone Inc. Changes: Swift and ONNX Runtime's Objective-C
// API, flat tensors, no console output, GPU option removed.
import Foundation
import OnnxRuntimeBindings

enum Supertonic {
    /// The 31 languages Supertonic 3 speaks, in the Android app's order (Settings › Speech and voice lists them).
    static let spoken = ["en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it",
                         "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi"]
    /// Language tags the model accepts ("na" is its language-neutral tag).
    static let languages = Set(spoken + ["na"])
    /// Input is cut into chunks of at most this many code points: Korean and Japanese pack more speech per character.
    static func maxChunk(_ lang: String) -> Int { lang == "ko" || lang == "ja" ? 120 : 300 }
    /// Imported voice styles are at most 2 MiB (SupertonicModel.MAX_VOICE_BYTES on Android).
    static let maxVoiceBytes = 2 * 1024 * 1024

    struct Config {
        let sampleRate: Int
        let baseChunkSize: Int
        let chunkCompressFactor: Int
        let latentDim: Int
    }

    /// A voice: the two style tensors from voice_styles/<name>.json.
    struct Style {
        let ttl: [Float]
        let ttlShape: [Int]
        let dp: [Float]
        let dpShape: [Int]
    }

    struct SupertonicError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    // MARK: Unicode text processor

    final class UnicodeProcessor {
        private let indexer: [Int64]

        init(unicodeIndexerPath: URL) throws {
            indexer = try JSONDecoder().decode([Int64].self, from: Data(contentsOf: unicodeIndexerPath))
        }

        private static func isEmoji(_ v: UInt32) -> Bool {
            (0x1F600...0x1F64F).contains(v) || (0x1F300...0x1F5FF).contains(v) || (0x1F680...0x1F6FF).contains(v)
                || (0x1F700...0x1F77F).contains(v) || (0x1F780...0x1F7FF).contains(v) || (0x1F800...0x1F8FF).contains(v)
                || (0x1F900...0x1F9FF).contains(v) || (0x1FA00...0x1FA6F).contains(v) || (0x1FA70...0x1FAFF).contains(v)
                || (0x2600...0x26FF).contains(v) || (0x2700...0x27BF).contains(v) || (0x1F1E6...0x1F1FF).contains(v)
        }

        private static func replace(_ text: String, _ pattern: String, _ with: String) -> String {
            text.replacingOccurrences(of: pattern, with: with, options: .regularExpression)
        }

        func preprocess(_ input: String, lang: String) throws -> String {
            // TODO (upstream): an advanced normalizer would read better.
            var text = input.decomposedStringWithCompatibilityMapping
            var scalars = String.UnicodeScalarView()
            scalars.append(contentsOf: text.unicodeScalars.filter { !Self.isEmoji($0.value) })
            text = String(scalars)

            // Dashes and symbols.
            let replacements: [(String, String)] = [
                ("–", "-"), ("‑", "-"), ("—", "-"), ("_", " "), ("\u{201C}", "\""), ("\u{201D}", "\""), ("\u{2018}", "'"), ("\u{2019}", "'"),
                ("´", "'"), ("`", "'"), ("[", " "), ("]", " "), ("|", " "), ("/", " "), ("#", " "), ("→", " "), ("←", " "),
            ]
            for (from, to) in replacements { text = text.replacingOccurrences(of: from, with: to) }
            text = Self.replace(text, #"[♥☆♡©\\]"#, "")
            for (from, to) in [("@", " at "), ("e.g.,", "for example, "), ("i.e.,", "that is, ")] {
                text = text.replacingOccurrences(of: from, with: to)
            }
            // Spacing around punctuation.
            for p in [",", ".", "!", "?", ";", ":", "'"] { text = text.replacingOccurrences(of: " " + p, with: p) }
            while text.contains("\"\"") { text = text.replacingOccurrences(of: "\"\"", with: "\"") }
            while text.contains("''") { text = text.replacingOccurrences(of: "''", with: "'") }
            while text.contains("``") { text = text.replacingOccurrences(of: "``", with: "`") }
            text = Self.replace(text, #"\s+"#, " ").trimmingCharacters(in: .whitespaces)
            // End with punctuation.
            if text.range(of: #"[.!?;:,'"“”‘’)\]}…。」』】〉》›»]$"#, options: .regularExpression) == nil {
                text += "."
            }
            guard Supertonic.languages.contains(lang) else {
                throw SupertonicError(message: "Invalid language: \(lang)")
            }
            return "<\(lang)>" + text + "</\(lang)>"
        }

        /// Text ids (UTF-16 code units through the indexer, like the C# original) and the text mask.
        func call(_ text: String, lang: String) throws -> (ids: [Int64], mask: [Float]) {
            let processed = try preprocess(text, lang: lang)
            let ids = processed.utf16.map { unit -> Int64 in
                let i = Int(unit)
                return i < indexer.count ? indexer[i] : 0
            }
            return (ids, [Float](repeating: 1, count: ids.count))
        }
    }

    // MARK: Text to speech

    final class TextToSpeech: @unchecked Sendable {
        let config: Config
        private let textProcessor: UnicodeProcessor
        private let dp: ORTSession
        private let textEncoder: ORTSession
        private let vectorEstimator: ORTSession
        private let vocoder: ORTSession
        var sampleRate: Int { config.sampleRate }

        /// `basicOptimization`: the FP16 build's graphs only keep ONNX Runtime CPU kernels at the basic level (extended
        /// fusion makes FP16 Gelu nodes the CPU provider lacks), as on Android.
        init(onnxDir: URL, basicOptimization: Bool = false) throws {
            let json = try JSONSerialization.jsonObject(with: Data(contentsOf: onnxDir.appendingPathComponent("tts.json"))) as? [String: Any] ?? [:]
            let ae = json["ae"] as? [String: Any] ?? [:]
            let ttl = json["ttl"] as? [String: Any] ?? [:]
            config = Config(sampleRate: ae["sample_rate"] as? Int ?? 44100, baseChunkSize: ae["base_chunk_size"] as? Int ?? 512,
                            chunkCompressFactor: ttl["chunk_compress_factor"] as? Int ?? 1, latentDim: ttl["latent_dim"] as? Int ?? 0)
            textProcessor = try UnicodeProcessor(unicodeIndexerPath: onnxDir.appendingPathComponent("unicode_indexer.json"))
            let env = try ORTEnv(loggingLevel: .warning)
            let options = try ORTSessionOptions()
            try options.setGraphOptimizationLevel(basicOptimization ? .basic : .all)
            try options.setIntraOpNumThreads(Int32(max(4, ProcessInfo.processInfo.activeProcessorCount / 2)))
            func load(_ name: String) throws -> ORTSession {
                try ORTSession(env: env, modelPath: onnxDir.appendingPathComponent(name).path, sessionOptions: options)
            }
            dp = try load("duration_predictor.onnx")
            textEncoder = try load("text_encoder.onnx")
            vectorEstimator = try load("vector_estimator.onnx")
            vocoder = try load("vocoder.onnx")
        }

        /// A voice style file as the Android app accepts it: at most 2 MiB of JSON with finite `style_ttl` [1, 50, 256]
        /// and `style_dp` [1, 8, 16] tensors.
        static func validateStyle(_ data: Data) throws {
            guard data.count <= Supertonic.maxVoiceBytes else { throw SupertonicError(message: "Voice file is larger than 2 MB") }
            guard let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
                throw SupertonicError(message: "Voice file is not a JSON object")
            }
            func check(_ name: String, _ expected: [Int]) throws {
                guard let tensor = json[name] as? [String: Any], let dims = (tensor["dims"] as? [NSNumber])?.map(\.intValue), dims == expected else {
                    throw SupertonicError(message: "\(name) must have dims \(expected)")
                }
                var count = 0
                func walk(_ value: Any?, _ depth: Int) throws {
                    guard depth <= 3 else { throw SupertonicError(message: "\(name) is nested too deeply") }
                    if let array = value as? [Any] {
                        for item in array { try walk(item, depth + 1) }
                    } else if let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite {
                        count += 1
                    } else {
                        throw SupertonicError(message: "\(name) holds a value that is not a finite number")
                    }
                }
                try walk(tensor["data"], 0)
                let wanted = expected.reduce(1, *)
                guard count == wanted else { throw SupertonicError(message: "\(name) has \(count) values, not \(wanted)") }
            }
            try check("style_ttl", [1, 50, 256])
            try check("style_dp", [1, 8, 16])
        }

        static func loadStyle(_ path: URL) throws -> Style {
            guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any],
                  let ttl = json["style_ttl"] as? [String: Any], let dp = json["style_dp"] as? [String: Any] else {
                throw SupertonicError(message: "Bad voice style \(path.lastPathComponent)")
            }
            func flat(_ value: Any?) -> [Float] {
                guard let array = value as? [Any] else { return (value as? NSNumber).map { [$0.floatValue] } ?? [] }
                return array.flatMap { flat($0) }
            }
            func dims(_ value: Any?) -> [Int] { (value as? [NSNumber])?.map(\.intValue) ?? [] }
            return Style(ttl: flat(ttl["data"]), ttlShape: dims(ttl["dims"]), dp: flat(dp["data"]), dpShape: dims(dp["dims"]))
        }

        private static func tensor<T>(_ values: [T], _ type: ORTTensorElementDataType, _ shape: [Int]) throws -> ORTValue {
            let data = values.withUnsafeBufferPointer { NSMutableData(bytes: $0.baseAddress, length: $0.count * MemoryLayout<T>.stride) }
            return try ORTValue(tensorData: data, elementType: type, shape: shape.map { NSNumber(value: $0) })
        }

        private static func floats(_ value: ORTValue) throws -> [Float] {
            let data = try value.tensorData() as Data
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        private static func output(_ outputs: [String: ORTValue], _ name: String) throws -> ORTValue {
            guard let value = outputs[name] else { throw SupertonicError(message: "Missing output \(name)") }
            return value
        }

        /// One chunk of text to audio samples and its duration in seconds.
        private func infer(_ text: String, lang: String, style: Style, totalStep: Int, speed: Float) throws -> (wav: [Float], duration: Float) {
            let (ids, mask) = try textProcessor.call(text, lang: lang)
            let textIds = try Self.tensor(ids, .int64, [1, ids.count])
            let textMask = try Self.tensor(mask, .float, [1, 1, mask.count])
            let styleTtl = try Self.tensor(style.ttl, .float, style.ttlShape)
            let styleDp = try Self.tensor(style.dp, .float, style.dpShape)

            // Duration predictor, then the speed factor.
            let dpOut = try dp.run(withInputs: ["text_ids": textIds, "style_dp": styleDp, "text_mask": textMask],
                                   outputNames: ["duration"], runOptions: nil)
            let duration = try Self.floats(Self.output(dpOut, "duration"))[0] / speed

            // Text encoder.
            let encOut = try textEncoder.run(withInputs: ["text_ids": textIds, "style_ttl": styleTtl, "text_mask": textMask],
                                             outputNames: ["text_emb"], runOptions: nil)
            let textEmb = try Self.output(encOut, "text_emb")

            // Noisy latent, masked to the predicted length.
            let wavLength = Int(duration * Float(config.sampleRate))
            let chunkSize = config.baseChunkSize * config.chunkCompressFactor
            let latentLength = Int((Float(wavLength) + Float(chunkSize) - 1) / Float(chunkSize))
            let latentDim = config.latentDim * config.chunkCompressFactor
            let validLength = (wavLength + chunkSize - 1) / chunkSize
            let latentMask = (0..<latentLength).map { $0 < validLength ? Float(1) : 0 }
            var xt = [Float](repeating: 0, count: latentDim * latentLength)
            for d in 0..<latentDim {
                for t in 0..<latentLength where latentMask[t] > 0 {
                    // Box-Muller transform for a normal distribution.
                    let u1 = 1 - Double.random(in: 0..<1), u2 = 1 - Double.random(in: 0..<1)
                    xt[d * latentLength + t] = Float(sqrt(-2 * log(u1)) * cos(2 * .pi * u2))
                }
            }
            let latentShape = [1, latentDim, latentLength]
            let latentMaskValue = try Self.tensor(latentMask, .float, [1, 1, latentLength])
            let total = try Self.tensor([Float(totalStep)], .float, [1])

            // Iterative denoising.
            for step in 0..<totalStep {
                let out = try vectorEstimator.run(withInputs: [
                    "noisy_latent": Self.tensor(xt, .float, latentShape), "text_emb": textEmb, "style_ttl": styleTtl, "text_mask": textMask,
                    "latent_mask": latentMaskValue, "total_step": total, "current_step": Self.tensor([Float(step)], .float, [1]),
                ], outputNames: ["denoised_latent"], runOptions: nil)
                xt = try Self.floats(Self.output(out, "denoised_latent"))
            }

            let vocOut = try vocoder.run(withInputs: ["latent": Self.tensor(xt, .float, latentShape)], outputNames: ["wav_tts"], runOptions: nil)
            return (try Self.floats(Self.output(vocOut, "wav_tts")), duration)
        }

        /// Text of any length to audio: sentence chunks joined with short silences.
        func call(_ text: String, lang: String, style: Style, totalStep: Int, speed: Float = 1.05, silence: Float = 0.3) throws -> (wav: [Float], duration: Float) {
            guard style.ttlShape.first == 1 else { throw SupertonicError(message: "Single speaker text to speech only supports single style") }
            var wav: [Float] = []
            var duration: Float = 0
            for chunk in Supertonic.chunkText(text, maxLength: Supertonic.maxChunk(lang)) {
                let (piece, seconds) = try infer(chunk, lang: lang, style: style, totalStep: totalStep, speed: speed)
                // Trim each chunk to its predicted length so silences line up.
                let keep = min(piece.count, Int(Float(sampleRate) * seconds))
                if !wav.isEmpty {
                    wav += [Float](repeating: 0, count: Int(silence * Float(sampleRate)))
                    duration += silence
                }
                wav += piece.prefix(keep)
                duration += seconds
            }
            return (wav, duration)
        }
    }

    // MARK: Chunking

    private static let sentenceSplit = try! NSRegularExpression(pattern:
        #"(?<!Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sr\.|Jr\.|Ph\.D\.|etc\.|e\.g\.|i\.e\.|vs\.|Inc\.|Ltd\.|Co\.|Corp\.|St\.|Ave\.|Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+"#)

    /// Paragraphs, then sentences, packed into chunks of at most `maxLength` code points; a sentence longer than that
    /// (or unbroken CJK text) is cut at the last space in the second half of the window, as on Android.
    static func chunkText(_ text: String, maxLength: Int = 300) -> [String] {
        packSentences(text, maxLength: maxLength).flatMap { hardSplit($0, maxLength: maxLength) }
    }

    /// SupertonicEngine.chunks from the Android app: bounded by code points, never splitting a scalar.
    static func hardSplit(_ text: String, maxLength: Int) -> [String] {
        let scalars = Array(text.unicodeScalars)
        guard scalars.count > maxLength else { return [text] }
        var result: [String] = []
        var start = 0
        while start < scalars.count {
            var end = min(scalars.count, start + maxLength)
            if end < scalars.count, let space = scalars[start ..< end].lastIndex(of: " "), space > start, space - start >= (end - start) / 2 {
                end = space
            }
            var view = String.UnicodeScalarView()
            view.append(contentsOf: scalars[start ..< end])
            let piece = String(view).trimmingCharacters(in: .whitespaces)
            if !piece.isEmpty { result.append(piece) }
            start = end
        }
        return result
    }

    private static func packSentences(_ text: String, maxLength: Int) -> [String] {
        var chunks: [String] = []
        let paragraphs = text.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: #"\n\s*\n+"#, with: "\u{0}", options: .regularExpression)
            .split(separator: "\u{0}").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        for paragraph in paragraphs {
            let ns = paragraph as NSString
            var sentences: [String] = []
            var last = 0
            for m in sentenceSplit.matches(in: paragraph, range: NSRange(location: 0, length: ns.length)) {
                sentences.append(ns.substring(with: NSRange(location: last, length: m.range.location - last)))
                last = m.range.location + m.range.length
            }
            sentences.append(ns.substring(from: last))
            var current = ""
            for sentence in sentences where !sentence.isEmpty {
                if current.unicodeScalars.count + sentence.unicodeScalars.count + 1 <= maxLength {
                    current += (current.isEmpty ? "" : " ") + sentence
                } else {
                    if !current.isEmpty { chunks.append(current.trimmingCharacters(in: .whitespaces)) }
                    current = sentence
                }
            }
            if !current.isEmpty { chunks.append(current.trimmingCharacters(in: .whitespaces)) }
        }
        return chunks.isEmpty ? [text.trimmingCharacters(in: .whitespacesAndNewlines)] : chunks
    }
}
