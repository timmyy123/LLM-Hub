import Foundation
import onnxruntime_objc

/// Kokoro-82M neural TTS via ONNX Runtime (onnxruntime-objc package).
///
/// Pipeline (per utterance):
///   text → G2P (espeak-ng or dict) → IPA → KokoroVocab.tokenize → Int64 ids
///        → pad with 0 on both ends → (1, L+2) tokens
///        → style row L from VoicePack → ORT run
///        → 24 kHz Float audio → resample to 16 kHz Int16 → 320-sample frames
///
/// Model files (download via KokoroAssets):
///   - kokoro-v1.0.fp16.onnx     (onnx-community/Kokoro-82M-ONNX export)
///   - voices/<id>.bin           (per-voice 511×256 float32 packs)
///
/// G2P defaults to `G2PFactory.best()` — espeak-ng if its XCFramework is
/// present, the bundled dictionary otherwise.
final class KokoroTTS: TTS {

    let modelURL: URL
    let voicePackURL: URL
    let voiceId: String
    let speed: Float
    let g2p: G2P
    /// Human-readable name of the active G2P engine (for UI display).
    var g2pName: String { g2p.displayName }

    private var env: ORTEnv?
    private var session: ORTSession?
    private var voicePack: VoicePack?

    private var inputTokensName: String = "input_ids"
    private var inputStyleName: String = "style"
    private var inputSpeedName: String = "speed"
    private var outputName: String = "audio"

    init(
        modelURL: URL,
        voicePackURL: URL,
        voiceId: String = "af_heart",
        speed: Float = 1.0,
        g2p: G2P = G2PFactory.best()
    ) {
        self.modelURL = modelURL
        self.voicePackURL = voicePackURL
        self.voiceId = voiceId
        self.speed = speed
        self.g2p = g2p
    }

    func load() async throws {
        if session != nil { return }
        let env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        try opts.setGraphOptimizationLevel(.all)
        // Core ML EP — silently skipped on simulators / older devices.
        do {
            let coreml = ORTCoreMLExecutionProviderOptions()
            coreml.useCPUOnly = false
            try opts.appendCoreMLExecutionProvider(with: coreml)
        } catch {
            print("ℹ️ KokoroTTS: Core ML EP unavailable, falling back to CPU: \(error)")
        }

        let session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: opts)

        // Probe input/output names.
        let inputs: Set<String> = (try? session.inputNames()).map(Set.init) ?? []
        if inputs.contains("tokens") { inputTokensName = "tokens" }
        if inputs.contains("ref_s")  { inputStyleName = "ref_s" }

        let outputs: [String] = (try? session.outputNames()) ?? []
        if let first = outputs.first { outputName = first }

        self.env = env
        self.session = session
        self.voicePack = try VoicePack.load(from: voicePackURL)
    }

    func speakToPCM(text: String, language: String = "en-US") -> AsyncStream<[Int16]> {
        AsyncStream { continuation in
            Task.detached { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    if self.session == nil { try await self.load() }
                    try self.synthesize(text: text, into: continuation)
                } catch {
                    print("⚠️ KokoroTTS synth failed: \(error)")
                }
                continuation.finish()
            }
        }
    }

    private func synthesize(text: String, into continuation: AsyncStream<[Int16]>.Continuation) throws {
        guard let session = session else { return }

        let ipa = g2p.phonemize(text, language: "en-us")
        guard !ipa.isEmpty else { return }
        let ids = KokoroVocab.tokenize(ipa)
        guard !ids.isEmpty else { return }

        // Pad with 0 on both ends.
        var padded = [Int64](repeating: 0, count: ids.count + 2)
        for (i, v) in ids.enumerated() { padded[i + 1] = Int64(v) }

        // Build tensors.
        let tokensData = NSMutableData()
        padded.withUnsafeBytes { tokensData.append($0.baseAddress!, length: $0.count) }
        let tokensValue = try ORTValue(
            tensorData: tokensData,
            elementType: .int64,
            shape: [NSNumber(value: 1), NSNumber(value: padded.count)]
        )

        let styleVec = voicePack?.style(forTokens: ids.count) ?? [Float](repeating: 0, count: 256)
        let styleData = NSMutableData()
        styleVec.withUnsafeBytes { styleData.append($0.baseAddress!, length: $0.count) }
        let styleValue = try ORTValue(
            tensorData: styleData,
            elementType: .float,
            shape: [NSNumber(value: 1), NSNumber(value: styleVec.count)]
        )

        var speedArr: [Float] = [speed]
        let speedData = NSMutableData()
        speedArr.withUnsafeBytes { speedData.append($0.baseAddress!, length: $0.count) }
        let speedValue = try ORTValue(
            tensorData: speedData,
            elementType: .float,
            shape: [NSNumber(value: 1)]
        )

        let outputs = try session.run(
            withInputs: [
                inputTokensName: tokensValue,
                inputStyleName: styleValue,
                inputSpeedName: speedValue
            ],
            outputNames: Set([outputName]),
            runOptions: nil
        )

        guard let outValue = outputs[outputName] else { return }
        let outData = try outValue.tensorData() as Data
        let floats = outData.withUnsafeBytes { raw -> [Float] in
            let bound = raw.bindMemory(to: Float.self)
            return Array(bound)
        }

        let pcm16k = Self.resampleLinear(floats, from: 24_000, to: MimoBotIds.sampleRateHz)
        let frame = MimoBotIds.frameSamples
        var i = 0
        while i + frame <= pcm16k.count {
            continuation.yield(Array(pcm16k[i..<(i + frame)]))
            i += frame
        }
        if i < pcm16k.count {
            var tail = Array(pcm16k[i...])
            while tail.count < frame { tail.append(0) }
            continuation.yield(tail)
        }
    }

    func stop() { /* one-shot synth — nothing to interrupt mid-run */ }

    static func resampleLinear(_ input: [Float], from srcRate: Int, to dstRate: Int) -> [Int16] {
        if input.isEmpty { return [] }
        if srcRate == dstRate {
            return input.map { Int16(max(-1, min(1, $0)) * 32767) }
        }
        let ratio = Double(srcRate) / Double(dstRate)
        let outLen = Int(Double(input.count) / ratio)
        var out = [Int16](repeating: 0, count: outLen)
        var srcPos = 0.0
        for i in 0..<outLen {
            let idx = Int(srcPos)
            let frac = Float(srcPos - Double(idx))
            let s0 = input[idx]
            let s1 = idx + 1 < input.count ? input[idx + 1] : s0
            let sample = max(-1, min(1, s0 + (s1 - s0) * frac))
            out[i] = Int16(sample * 32767)
            srcPos += ratio
        }
        return out
    }
}
