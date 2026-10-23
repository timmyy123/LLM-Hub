import Foundation
import MLX

/// Live driver for Magenta RealTime 2's exported stateful MLX function, following the iOS app's MagentaRealtimeEngine
/// (one 1,920-sample stereo frame at 48 kHz, 40 ms, per call; every returned state tensor feeds the next call), with
/// the style tokens and sampling settings swappable between frames so the music follows the sliders while it plays.
public final class MagentaLive: @unchecked Sendable {
    public static let sampleRate = 48_000
    public static let frameSamples = 1_920
    private static let musicCoCaLevels = 12
    private static let conditionLength = 141
    private static let reservedTokens: Int32 = 7

    /// Sampling settings; the defaults are magenta-realtime's MLX CLI defaults.
    public struct Controls: Sendable, Equatable {
        public var temperature: Float = 1.3
        public var topK: Int32 = 40
        public var guidance: Float = 3.0
        public init() {}
    }

    private let function: ImportedFunction
    private let initialState: [MLXArray]
    private let keyIndex: Int
    private let lock = NSLock()
    private var pendingTokens: [Int32]?
    private var controls = Controls()
    private var controlsChanged = false

    public init(function functionURL: URL, state stateURL: URL) throws {
        function = try importFunction(from: functionURL)
        let dictionary = try loadArrays(url: stateURL)
        var state: [MLXArray] = []
        while let value = dictionary["state_\(state.count)"] { state.append(value) }
        guard state.count == 165 || state.count == 229 else { throw EngineError.unsupportedStateCount(state.count) }
        // The RNG key is the one uint32 state tensor whose last dimension is two, for either model size.
        let keys = state.indices.filter { state[$0].dtype == .uint32 && state[$0].shape.last == 2 }
        guard keys.count == 1 else { throw EngineError.invalidRandomKeyCount(keys.count) }
        try withError { eval(state) }
        initialState = state
        keyIndex = keys[0]
    }

    /// New style tokens (from MusicCoCa.tokens) take effect on the next 40 ms frame.
    public func update(tokens: [Int32]) {
        lock.lock()
        pendingTokens = tokens
        lock.unlock()
    }

    public func update(controls: Controls) {
        lock.lock()
        self.controls = controls
        controlsChanged = true
        lock.unlock()
    }

    /// Generates until `frames` frames are done (nil: until `shouldStop`), handing each frame to `onFrame` as
    /// interleaved little-endian Int16 stereo PCM. Returns the number of frames made.
    @discardableResult
    public func run(tokens: [Int32], controls initial: Controls, seed: UInt64? = nil, frames: Int? = nil,
                    shouldStop: () -> Bool, onFrame: (Data) throws -> Void) throws -> Int {
        lock.lock()
        controls = initial
        controlsChanged = false
        pendingTokens = nil
        lock.unlock()

        let stateCount = initialState.count
        let condition = Self.conditionArguments(tokens)
        let values = Self.controlArguments(initial)
        // Condition, temperature, top-k, guidance, 1, 1, negative music, negative notes, empty audio prompt, state.
        var arguments = [condition[0], values[0], values[1], values[2], values[3], values[4], condition[1], condition[2],
                         MLX.zeros([1, 0, Self.musicCoCaLevels], dtype: .int32)]
        arguments.append(contentsOf: initialState)
        let generationSeed = seed ?? UInt64.random(in: 0...UInt64.max)
        let savedKey = initialState[keyIndex]
        let keys = (0..<(savedKey.size / 2)).map { MLXRandom.key(generationSeed &+ UInt64($0)) }
        arguments[9 + keyIndex] = stacked(keys).reshaped(savedKey.shape)
        try withError { eval(arguments) }

        var frame = 0
        while frames.map({ frame < $0 }) ?? true {
            if shouldStop() { break }
            lock.lock()
            let tokens = pendingTokens
            pendingTokens = nil
            let newControls = controlsChanged ? controls : nil
            controlsChanged = false
            lock.unlock()
            if let tokens {
                let condition = Self.conditionArguments(tokens)
                arguments[0] = condition[0]
                arguments[6] = condition[1]
                arguments[7] = condition[2]
            }
            if let newControls {
                let values = Self.controlArguments(newControls)
                arguments[1] = values[0]
                arguments[2] = values[1]
                arguments[3] = values[2]
            }

            let outputs = try Self.invoke(function, arguments)
            guard outputs.count == stateCount + 1 else { throw EngineError.invalidOutputCount(outputs.count, stateCount + 1) }
            try withError { eval(outputs) }
            try onFrame(try Self.pcm(outputs[0]))
            for index in 0..<stateCount { arguments[9 + index] = outputs[1 + index] }
            frame += 1
        }
        return frame
    }

    /// Condition, negative-music and negative-notes inputs for 12 MusicCoCa tokens.
    private static func conditionArguments(_ tokens: [Int32]) -> [MLXArray] {
        var condition = [Int32](repeating: reservedTokens - 1, count: conditionLength)
        for index in 0..<musicCoCaLevels { condition[index] = tokens[index] + reservedTokens }
        condition[conditionLength - 1] = reservedTokens - 1
        var negativeMusic = condition
        for index in 0..<musicCoCaLevels { negativeMusic[index] = reservedTokens - 1 }
        var negativeNotes = condition
        for index in musicCoCaLevels..<(musicCoCaLevels + 128) { negativeNotes[index] = reservedTokens - 1 }
        let arrays = [MLXArray(condition, [1, 1, conditionLength]), MLXArray(negativeMusic, [1, 1, conditionLength]),
                      MLXArray(negativeNotes, [1, 1, conditionLength])]
        eval(arrays)
        return arrays
    }

    private static func controlArguments(_ c: Controls) -> [MLXArray] {
        let arrays = [MLXArray([c.temperature], [1]), MLXArray([c.topK], [1]), MLXArray([c.guidance], [1]),
                      MLXArray([Float(1.0)], [1]), MLXArray([Float(1.0)], [1])]
        eval(arrays)
        return arrays
    }

    private static func pcm(_ audio: MLXArray) throws -> Data {
        var data = Data(count: frameSamples * 4)
        switch audio.dtype {
        case .int16:
            // MRT2's decoder emits PCM-scale Int16: keep the samples as they are.
            let samples = audio.asArray(Int16.self)
            guard samples.count >= frameSamples * 2 else { throw EngineError.invalidAudioShape(audio.shape) }
            data.withUnsafeMutableBytes { bytes in
                let out = bytes.bindMemory(to: Int16.self)
                for i in 0..<frameSamples {
                    out[i * 2] = samples[i].littleEndian
                    out[i * 2 + 1] = samples[frameSamples + i].littleEndian
                }
            }
        case .float32:
            let samples = audio.asArray(Float.self)
            guard samples.count >= frameSamples * 2 else { throw EngineError.invalidAudioShape(audio.shape) }
            func scale(_ v: Float) -> Int16 { Int16(clamping: Int((min(max(v, -1), 1) * 32_767).rounded())).littleEndian }
            data.withUnsafeMutableBytes { bytes in
                let out = bytes.bindMemory(to: Int16.self)
                for i in 0..<frameSamples {
                    out[i * 2] = scale(samples[i])
                    out[i * 2 + 1] = scale(samples[frameSamples + i])
                }
            }
        default:
            throw EngineError.unsupportedAudioType(String(describing: audio.dtype))
        }
        return data
    }

    public enum EngineError: LocalizedError {
        case unsupportedStateCount(Int)
        case invalidRandomKeyCount(Int)
        case invalidOutputCount(Int, Int)
        case invalidAudioShape([Int])
        case unsupportedAudioType(String)

        public var errorDescription: String? {
            switch self {
            case .unsupportedStateCount(let count): return "Unsupported Magenta state layout (\(count) arrays)"
            case .invalidRandomKeyCount(let count): return "Magenta state must contain one RNG key tensor; found \(count)"
            case .invalidOutputCount(let actual, let expected): return "Magenta returned \(actual) tensors; expected \(expected)"
            case .invalidAudioShape(let shape): return "Magenta returned invalid audio shape \(shape)"
            case .unsupportedAudioType(let type): return "Magenta returned unsupported audio type \(type)"
            }
        }
    }
}
