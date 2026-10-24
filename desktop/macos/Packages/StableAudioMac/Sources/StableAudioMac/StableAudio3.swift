import Foundation
import MLX

/// Stable Audio 3 text to music on the Apple GPU: prompt, T5Gemma, DiT ping-pong sampler (8 steps), SAME decoder,
/// 44.1 kHz stereo. A port of Stability AI's official MLX runtime (scripts/sa3_mlx.py and models/defs/sa3_pipeline.py
/// in https://github.com/Stability-AI/stable-audio-3, MIT License, Copyright (c) 2026 Stability AI) for text to audio
/// without CFG, the runtime's defaults. The model weights are under the Stability AI Community License.
/// Changes: Swift and mlx-swift instead of Python; npz read directly; one class that keeps the loaded models.
public final class StableAudio3: @unchecked Sendable {
    public enum Model: String, CaseIterable, Sendable {
        case smallMusic = "sm-music"
        case medium

        /// Longest clip the model was trained for (docs/guides/model-overview.md: ~2 min and ~4.75 min).
        public var maxSeconds: Double { self == .smallMusic ? 120 : 285 }
        public var ditFile: String { self == .smallMusic ? "dit_sm-music_f16.npz" : "dit_medium_f16.npz" }
        public var decoderFile: String { self == .smallMusic ? "same_s_decoder_f32.npz" : "same_l_decoder_f32.npz" }
        /// The files this model needs, with their sizes in the pinned Hugging Face revision.
        public var files: [(name: String, size: Int64)] {
            let model: [(name: String, size: Int64)] = self == .smallMusic
                ? [(ditFile, Int64(919_193_814)), (decoderFile, Int64(218_090_820))]
                : [(ditFile, Int64(2_907_300_946)), (decoderFile, Int64(1_704_311_976))]
            return [(StableAudio3.textEncoderFile, Int64(567_443_068))] + model
        }
        public var downloadSize: Int64 { files.reduce(0) { $0 + $1.size } }
    }

    public static let sampleRate = 44100
    public static let samplesPerLatent = 4096
    public static let textEncoderFile = "t5gemma_f16.npz"
    /// stabilityai/stable-audio-3-optimized (ungated), pinned; files live under MLX/.
    public static let repoBase = "https://huggingface.co/stabilityai/stable-audio-3-optimized/resolve/08b17423649b5714b22eab20f96bf00751f2358a/MLX/"

    public enum Stage: Sendable {
        case loading
        case encoding
        case sampling(step: Int, of: Int)
        case decoding
    }

    public struct Timings: Sendable {
        public var load = 0.0
        public var encode = 0.0
        public var sample = 0.0
        public var decode = 0.0
    }

    private let folder: URL
    private let lock = NSLock()
    private var textEncoder: T5GemmaEncoder?
    private var dit: DiT?
    private var decoder: SameDecoder?
    private var loadedModel: Model?

    public init(folder: URL) {
        self.folder = folder
    }

    public static func isInstalled(_ model: Model, in folder: URL) -> Bool {
        model.files.allSatisfy { file in
            let size = (try? FileManager.default.attributesOfItem(atPath: folder.appendingPathComponent(file.name).path)[.size] as? NSNumber)?.int64Value
            return size == file.size
        }
    }

    /// Frees the models (they reload on the next generation).
    public func unload() {
        lock.lock()
        defer { lock.unlock() }
        textEncoder = nil
        dit = nil
        decoder = nil
        loadedModel = nil
        MLX.GPU.clearCache()
    }

    private func load(_ model: Model) throws {
        if textEncoder == nil { textEncoder = try T5GemmaEncoder(url: folder.appendingPathComponent(Self.textEncoderFile)) }
        guard loadedModel != model else { return }
        dit = nil
        decoder = nil
        MLX.GPU.clearCache()
        dit = try DiT(url: folder.appendingPathComponent(model.ditFile), config: model == .smallMusic ? .small : .medium)
        decoder = try SameDecoder(url: folder.appendingPathComponent(model.decoderFile), kind: model == .smallMusic ? .small : .large)
        loadedModel = model
    }

    /// Linear t from 1 to 0 in steps + 1 points, warped through log-SNR (anchor -6.2, end 2), start kept at 1.
    static func schedule(steps: Int, sigmaMax: Float = 1) -> MLXArray {
        let t = MLXArray.linspace(sigmaMax, Float(0), count: steps + 1)
        let logsnr = 2.0 - t * Float(2.0 - -6.2)
        var out = sigmoid(-logsnr)
        out = which(t .<= 0, MLXArray.zeros(like: out), out)
        out = which(t .>= 1, MLXArray.ones(like: out), out)
        return concatenated([MLXArray([sigmaMax]), out[1...]], axis: 0)
    }

    /// Generates `seconds` of stereo audio for `prompt`. Returns interleaved left/right samples at 44.1 kHz (unclipped).
    public func generate(_ model: Model, prompt: String, seconds: Double, seed: UInt64, steps: Int = 8,
                         stage: (Stage) -> Void = { _ in }, isCancelled: () -> Bool = { false }) throws -> (samples: [Float], timings: Timings) {
        lock.lock()
        defer {
            lock.unlock()
            MLX.GPU.clearCache()
        }
        var timings = Timings()
        var clock = Date()
        stage(.loading)
        try load(model)
        guard let textEncoder, let dit, let decoder else { throw CancellationError() }
        timings.load = Date().timeIntervalSince(clock)
        if isCancelled() { throw CancellationError() }

        // 1. Text encoder.
        clock = Date()
        stage(.encoding)
        let (embeds, mask) = textEncoder.encode(prompt)
        eval(embeds, mask)
        timings.encode = Date().timeIntervalSince(clock)

        // 2. Conditioning: prompt tokens plus the seconds_total token for cross-attention; seconds alone is global.
        let dtype = DType.float16
        let length = max(1, Int((seconds * Double(Self.sampleRate) / Double(Self.samplesPerLatent)).rounded(.up)))
        let padded = dit.applyPromptPadding(embeds.asType(dtype), mask: mask)
        let secondsEmbed = dit.secondsEmbedding(seconds).asType(dtype)
        let cross = concatenated([padded, secondsEmbed], axis: 1)
        let global = secondsEmbed[0..., 0, 0...]

        // 3. Ping-pong sampler (sample_flow_pingpong), noise and step noise from the seed exactly as the Python runtime.
        clock = Date()
        let sigmas = Self.schedule(steps: steps)
        var x = MLXRandom.normal([1, DiT.ioChannels, length], dtype: dtype, key: MLXRandom.key(seed))
        var key = MLXRandom.key(seed &+ 1)
        eval(x, sigmas, cross, global)
        for i in 0 ..< steps {
            if isCancelled() { throw CancellationError() }
            stage(.sampling(step: i + 1, of: steps))
            let tCurr = sigmas[i], tNext = sigmas[i + 1]
            let t = tCurr * MLXArray.ones([x.dim(0)], dtype: x.dtype)
            let v = dit(x, t: t, cross: cross, global: global)
            let denoised = x - tCurr.asType(x.dtype) * v
            if i < steps - 1 && tNext.item(Float.self) > 0 {
                let (next, sub) = MLXRandom.split(key: key)
                key = next
                let noise = MLXRandom.normal(x.shape, dtype: x.dtype, key: sub)
                x = (1.0 - tNext).asType(x.dtype) * denoised + tNext.asType(x.dtype) * noise
            } else {
                x = denoised
            }
            eval(x)
        }
        timings.sample = Date().timeIntervalSince(clock)

        // 4. Decode (fp32) and unpatch: [1, 512, T*16] to [1, 2, T*4096].
        clock = Date()
        stage(.decoding)
        let patches = try decoder.decode(x.asType(.float32), isCancelled: isCancelled)
        let frames = patches.dim(2) * 256
        let audio = patches.reshaped(1, 2, 256, patches.dim(2)).transposed(0, 1, 3, 2).reshaped(2, frames)
        let wanted = min(frames, Int((seconds * Double(Self.sampleRate)).rounded()))
        // Interleave on the GPU, then copy out once.
        let interleaved = audio[0..., ..<wanted].transposed(1, 0).asType(.float32).reshaped(-1)
        eval(interleaved)
        let samples = interleaved.asArray(Float.self)
        timings.decode = Date().timeIntervalSince(clock)
        return (samples, timings)
    }
}
