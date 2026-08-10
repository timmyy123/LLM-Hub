#if canImport(MLX)
import Foundation
import MLX

/// Offline iOS driver for Magenta RealTime 2's exported stateful MLX function.
///
/// This follows magenta-realtime/core/src/mlx_engine.cpp: one 1,920-sample
/// stereo frame at 48 kHz per transformer invocation, carrying every returned
/// state tensor into the next invocation.
public enum MagentaRealtimeEngine {
    public static let sampleRate = 48_000
    private static let frameSamples = 1_920
    private static let musicCoCaLevels = 12
    private static let conditionLength = 141
    private static let reservedTokens: Int32 = 7
    // Match magenta-realtime's MLX CLI and hello_mrt2 defaults exactly.
    private static let promptAdherence: Float = 3.0

    public final class Session: @unchecked Sendable {
        fileprivate let function: ImportedFunction
        fileprivate let initialState: [MLXArray]

        fileprivate init(function: ImportedFunction, initialState: [MLXArray]) {
            self.function = function
            self.initialState = initialState
        }
    }

    public static func load(functionURL: URL, stateURL: URL) throws -> Session {
        let function = try importFunction(from: functionURL)
        let stateDictionary = try loadArrays(url: stateURL)
        var state: [MLXArray] = []
        var index = 0
        while let value = stateDictionary["state_\(index)"] {
            state.append(value)
            index += 1
        }
        guard state.count == 165 || state.count == 229 else {
            throw EngineError.unsupportedStateCount(state.count)
        }
        eval(state)
        return Session(function: function, initialState: state)
    }

    public static func generate(
        functionURL: URL,
        stateURL: URL,
        prompt: String,
        resourceDirectory: URL,
        durationSeconds: Double,
        progress: @Sendable (Double) -> Void
    ) throws -> Data {
        let session = try load(functionURL: functionURL, stateURL: stateURL)
        return try generate(
            session: session,
            prompt: prompt,
            resourceDirectory: resourceDirectory,
            durationSeconds: durationSeconds,
            progress: progress
        )
    }

    public static func generate(
        session: Session,
        prompt: String,
        resourceDirectory: URL,
        durationSeconds: Double,
        progress: @Sendable (Double) -> Void
    ) throws -> Data {
        let function = session.function
        var state = session.initialState

        let frameCount = max(1, Int(ceil(durationSeconds * 25.0)))
        var pcm = Data(capacity: frameCount * frameSamples * 4)
        let musicCoCaTokens = try MagentaPromptEncoder.encode(
            prompt: prompt,
            resourceDirectory: resourceDirectory
        )
        let condition = makeCondition(musicCoCaTokens: musicCoCaTokens)
        let negativeMusic = makeNegativeMusic(from: condition)
        let negativeNotes = makeNegativeNotes(from: condition)

        for frame in 0..<frameCount {
            var arguments: [MLXArray] = [
                MLXArray(condition, [1, 1, conditionLength]),
                // Match `mrt mlx generate`, the repository's documented MLX path.
                MLXArray([Float(1.3)], [1]),
                MLXArray([Int32(40)], [1]),
                MLXArray([promptAdherence], [1]),
                MLXArray([Float(1.0)], [1]),
                MLXArray([Float(1.0)], [1]),
                MLXArray(negativeMusic, [1, 1, conditionLength]),
                MLXArray(negativeNotes, [1, 1, conditionLength]),
                MLXArray([Int32](), [1, 0, musicCoCaLevels]),
            ]
            arguments.append(contentsOf: state)
            let outputs = try invoke(function, arguments: arguments, stateCount: state.count)
            guard outputs.count == state.count + 1 else {
                throw EngineError.invalidOutputCount(outputs.count, state.count + 1)
            }
            eval(outputs)
            let audioTensor = outputs[0]
            if frame == 0 {
                print("[LLMHub][MusicGen] audioOutput dtype=\(audioTensor.dtype) shape=\(audioTensor.shape)")
            }
            switch audioTensor.dtype {
            case .int16:
                // MRT2's exported decoder normally emits PCM-scale Int16. Keep
                // those samples verbatim; converting to Float and clamping to
                // [-1, 1] destroys the waveform through full-scale clipping.
                let audio = audioTensor.asArray(Int16.self)
                guard audio.count >= frameSamples * 2 else {
                    throw EngineError.invalidAudioShape(audioTensor.shape)
                }
                for sample in 0..<frameSamples {
                    var left = audio[sample].littleEndian
                    var right = audio[frameSamples + sample].littleEndian
                    withUnsafeBytes(of: &left) { pcm.append(contentsOf: $0) }
                    withUnsafeBytes(of: &right) { pcm.append(contentsOf: $0) }
                }
            case .float32:
                // Some exports emit normalized Float32, matching the C++
                // engine's float-output branch.
                let audio = audioTensor.asArray(Float.self)
                guard audio.count >= frameSamples * 2 else {
                    throw EngineError.invalidAudioShape(audioTensor.shape)
                }
                for sample in 0..<frameSamples {
                    var left = Int16(clamping: Int((audio[sample].clamped(to: -1...1) * 32_767).rounded())).littleEndian
                    var right = Int16(clamping: Int((audio[frameSamples + sample].clamped(to: -1...1) * 32_767).rounded())).littleEndian
                    withUnsafeBytes(of: &left) { pcm.append(contentsOf: $0) }
                    withUnsafeBytes(of: &right) { pcm.append(contentsOf: $0) }
                }
            default:
                throw EngineError.unsupportedAudioType(String(describing: audioTensor.dtype))
            }
            state = Array(outputs.dropFirst())
            progress(Double(frame + 1) / Double(frameCount))
        }
        return pcm
    }

    private static func makeCondition(musicCoCaTokens: [Int32]) -> [Int32] {
        var condition = [Int32](repeating: reservedTokens - 1, count: conditionLength)
        for index in 0..<musicCoCaLevels {
            // `mrt mlx generate` passes all 12 MusicCoCa RVQ levels. Masking the
            // finer six levels is a realtime C++ UI optimization and noticeably
            // weakens text-prompt adherence for offline generation.
            condition[index] = musicCoCaTokens[index] + reservedTokens
        }
        condition[conditionLength - 1] = reservedTokens - 1
        return condition
    }

    private static func makeNegativeMusic(from condition: [Int32]) -> [Int32] {
        var result = condition
        for index in 0..<musicCoCaLevels { result[index] = reservedTokens - 1 }
        return result
    }

    private static func makeNegativeNotes(from condition: [Int32]) -> [Int32] {
        var result = condition
        for index in musicCoCaLevels..<(musicCoCaLevels + 128) {
            result[index] = reservedTokens - 1
        }
        return result
    }

    /// MLX Swift exposes imported functions through a variadic call. MRT2's
    /// published Small and Base checkpoints have fixed 165/229-state signatures.
    private static func invoke(
        _ function: ImportedFunction,
        arguments args: [MLXArray],
        stateCount: Int
    ) throws -> [MLXArray] {
        switch stateCount {
        case 165:
            return try function(
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5],
                args[6],
                args[7],
                args[8],
                args[9],
                args[10],
                args[11],
                args[12],
                args[13],
                args[14],
                args[15],
                args[16],
                args[17],
                args[18],
                args[19],
                args[20],
                args[21],
                args[22],
                args[23],
                args[24],
                args[25],
                args[26],
                args[27],
                args[28],
                args[29],
                args[30],
                args[31],
                args[32],
                args[33],
                args[34],
                args[35],
                args[36],
                args[37],
                args[38],
                args[39],
                args[40],
                args[41],
                args[42],
                args[43],
                args[44],
                args[45],
                args[46],
                args[47],
                args[48],
                args[49],
                args[50],
                args[51],
                args[52],
                args[53],
                args[54],
                args[55],
                args[56],
                args[57],
                args[58],
                args[59],
                args[60],
                args[61],
                args[62],
                args[63],
                args[64],
                args[65],
                args[66],
                args[67],
                args[68],
                args[69],
                args[70],
                args[71],
                args[72],
                args[73],
                args[74],
                args[75],
                args[76],
                args[77],
                args[78],
                args[79],
                args[80],
                args[81],
                args[82],
                args[83],
                args[84],
                args[85],
                args[86],
                args[87],
                args[88],
                args[89],
                args[90],
                args[91],
                args[92],
                args[93],
                args[94],
                args[95],
                args[96],
                args[97],
                args[98],
                args[99],
                args[100],
                args[101],
                args[102],
                args[103],
                args[104],
                args[105],
                args[106],
                args[107],
                args[108],
                args[109],
                args[110],
                args[111],
                args[112],
                args[113],
                args[114],
                args[115],
                args[116],
                args[117],
                args[118],
                args[119],
                args[120],
                args[121],
                args[122],
                args[123],
                args[124],
                args[125],
                args[126],
                args[127],
                args[128],
                args[129],
                args[130],
                args[131],
                args[132],
                args[133],
                args[134],
                args[135],
                args[136],
                args[137],
                args[138],
                args[139],
                args[140],
                args[141],
                args[142],
                args[143],
                args[144],
                args[145],
                args[146],
                args[147],
                args[148],
                args[149],
                args[150],
                args[151],
                args[152],
                args[153],
                args[154],
                args[155],
                args[156],
                args[157],
                args[158],
                args[159],
                args[160],
                args[161],
                args[162],
                args[163],
                args[164],
                args[165],
                args[166],
                args[167],
                args[168],
                args[169],
                args[170],
                args[171],
                args[172],
                args[173]
            )
        case 229:
            return try function(
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5],
                args[6],
                args[7],
                args[8],
                args[9],
                args[10],
                args[11],
                args[12],
                args[13],
                args[14],
                args[15],
                args[16],
                args[17],
                args[18],
                args[19],
                args[20],
                args[21],
                args[22],
                args[23],
                args[24],
                args[25],
                args[26],
                args[27],
                args[28],
                args[29],
                args[30],
                args[31],
                args[32],
                args[33],
                args[34],
                args[35],
                args[36],
                args[37],
                args[38],
                args[39],
                args[40],
                args[41],
                args[42],
                args[43],
                args[44],
                args[45],
                args[46],
                args[47],
                args[48],
                args[49],
                args[50],
                args[51],
                args[52],
                args[53],
                args[54],
                args[55],
                args[56],
                args[57],
                args[58],
                args[59],
                args[60],
                args[61],
                args[62],
                args[63],
                args[64],
                args[65],
                args[66],
                args[67],
                args[68],
                args[69],
                args[70],
                args[71],
                args[72],
                args[73],
                args[74],
                args[75],
                args[76],
                args[77],
                args[78],
                args[79],
                args[80],
                args[81],
                args[82],
                args[83],
                args[84],
                args[85],
                args[86],
                args[87],
                args[88],
                args[89],
                args[90],
                args[91],
                args[92],
                args[93],
                args[94],
                args[95],
                args[96],
                args[97],
                args[98],
                args[99],
                args[100],
                args[101],
                args[102],
                args[103],
                args[104],
                args[105],
                args[106],
                args[107],
                args[108],
                args[109],
                args[110],
                args[111],
                args[112],
                args[113],
                args[114],
                args[115],
                args[116],
                args[117],
                args[118],
                args[119],
                args[120],
                args[121],
                args[122],
                args[123],
                args[124],
                args[125],
                args[126],
                args[127],
                args[128],
                args[129],
                args[130],
                args[131],
                args[132],
                args[133],
                args[134],
                args[135],
                args[136],
                args[137],
                args[138],
                args[139],
                args[140],
                args[141],
                args[142],
                args[143],
                args[144],
                args[145],
                args[146],
                args[147],
                args[148],
                args[149],
                args[150],
                args[151],
                args[152],
                args[153],
                args[154],
                args[155],
                args[156],
                args[157],
                args[158],
                args[159],
                args[160],
                args[161],
                args[162],
                args[163],
                args[164],
                args[165],
                args[166],
                args[167],
                args[168],
                args[169],
                args[170],
                args[171],
                args[172],
                args[173],
                args[174],
                args[175],
                args[176],
                args[177],
                args[178],
                args[179],
                args[180],
                args[181],
                args[182],
                args[183],
                args[184],
                args[185],
                args[186],
                args[187],
                args[188],
                args[189],
                args[190],
                args[191],
                args[192],
                args[193],
                args[194],
                args[195],
                args[196],
                args[197],
                args[198],
                args[199],
                args[200],
                args[201],
                args[202],
                args[203],
                args[204],
                args[205],
                args[206],
                args[207],
                args[208],
                args[209],
                args[210],
                args[211],
                args[212],
                args[213],
                args[214],
                args[215],
                args[216],
                args[217],
                args[218],
                args[219],
                args[220],
                args[221],
                args[222],
                args[223],
                args[224],
                args[225],
                args[226],
                args[227],
                args[228],
                args[229],
                args[230],
                args[231],
                args[232],
                args[233],
                args[234],
                args[235],
                args[236],
                args[237]
            )
        default:
            throw EngineError.unsupportedStateCount(stateCount)
        }
    }

    enum EngineError: LocalizedError {
        case unsupportedStateCount(Int)
        case invalidOutputCount(Int, Int)
        case invalidAudioShape([Int])
        case unsupportedAudioType(String)

        var errorDescription: String? {
            switch self {
            case .unsupportedStateCount(let count):
                return "Unsupported Magenta state layout (\(count) arrays)"
            case .invalidOutputCount(let actual, let expected):
                return "Magenta returned \(actual) tensors; expected \(expected)"
            case .invalidAudioShape(let shape):
                return "Magenta returned invalid audio shape \(shape)"
            case .unsupportedAudioType(let type):
                return "Magenta returned unsupported audio type \(type)"
            }
        }
    }
}

private extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
#endif
