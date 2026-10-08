import Foundation
import NNC

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#endif

public struct DeepSeek4_1EngramConfiguration: Codable, Sendable {
  public var layers = [1, 14]
  public var rows = [384_006_168, 384_016_682]
  public var heads = 8
  public var headDim = 256
  public var maxNgramSize = 4
  public var padID = 2

  public var compressedVocabularySize: Int { 99_092 }
  public var rowBytes: Int { headDim + headDim / 32 }
  public var tableOffsets: [UInt64] {
    var offset: UInt64 = 0
    return rows.map { rows in
      defer { offset += UInt64(rows) * UInt64(rowBytes) }
      return offset
    }
  }
  public var fileSize: UInt64 {
    rows.reduce(0) { $0 + UInt64($1) * UInt64(rowBytes) }
  }
  // Exact draws from default_rng(10007 * layer), with compressed vocabulary 99,092.
  public var multipliers: [[UInt64]] {
    [
      [76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391],
      [67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591],
    ]
  }
  public var primes: [[[UInt64]]] {
    [
      [
        [
          16_000_057, 16_000_079, 16_000_081, 16_000_097, 16_000_121, 16_000_129, 16_000_133,
          16_000_183,
        ],
        [
          16_000_189, 16_000_207, 16_000_211, 16_000_253, 16_000_277, 16_000_289, 16_000_307,
          16_000_321,
        ],
        [
          16_000_339, 16_000_381, 16_000_393, 16_000_399, 16_000_403, 16_000_409, 16_000_447,
          16_000_463,
        ],
      ],
      [
        [
          16_000_477, 16_000_487, 16_000_499, 16_000_507, 16_000_511, 16_000_573, 16_000_609,
          16_000_627,
        ],
        [
          16_000_667, 16_000_669, 16_000_693, 16_000_697, 16_000_711, 16_000_729, 16_000_759,
          16_000_769,
        ],
        [
          16_000_781, 16_000_799, 16_000_813, 16_000_819, 16_000_841, 16_000_877, 16_000_879,
          16_000_889,
        ],
      ],
    ]
  }
  public var bucketOffsets: [[UInt64]] {
    primes.map { layer in
      var offset: UInt64 = 0
      return layer.flatMap { $0 }.map { prime in
        defer { offset += prime }
        return offset
      }
    }
  }

  public var embeddingSize: Int { (maxNgramSize - 1) * heads * headDim }

  public init() {}

  public static let deepSeekV4_1Flash = DeepSeek4_1EngramConfiguration()
}

/// Hash text n-grams on CPU using the vocabulary-derived token map and fixed
/// model constants. `previousTokens` holds up to the preceding three original token
/// IDs; keep the returned history with the KV state for continuation/rollback.
/// The returned tensor is [tokens, Engram layers, n-gram/head columns].
public func DeepSeek4_1EngramHash(
  tokens: [Int32], previousTokens: [Int32] = [], tokenMap: [Int32],
  configuration: DeepSeek4_1EngramConfiguration = .deepSeekV4_1Flash
) -> (hashes: Tensor<Int32>, history: [Int32]) {
  let ngram = configuration.maxNgramSize
  let layers = configuration.layers.count
  let heads = configuration.heads
  let columns = (ngram - 1) * heads
  precondition(tokenMap.count == DeepSeek4_1ModelConfiguration.deepSeekV4_1Flash.vocabularySize)
  let multipliers = configuration.multipliers
  let primes = configuration.primes
  let offsets = configuration.bucketOffsets
  precondition(!tokens.isEmpty && previousTokens.count < ngram)
  let combined = previousTokens + tokens
  let compressed: [UInt64] = combined.map { token in
    precondition(token >= 0 && Int(token) < tokenMap.count)
    let compressed = tokenMap[Int(token)]
    precondition(compressed >= 0 && compressed < configuration.compressedVocabularySize)
    return UInt64(compressed)
  }
  let padding = tokenMap[configuration.padID]
  precondition(padding >= 0 && padding < configuration.compressedVocabularySize)
  let pad = UInt64(padding)
  var values = [Int32](repeating: 0, count: tokens.count * layers * columns)
  for token in 0..<tokens.count {
    let position = previousTokens.count + token
    for layer in 0..<layers {
      var rolling = compressed[position] * multipliers[layer][0]
      for lookback in 1..<ngram {
        let previous = position >= lookback ? compressed[position - lookback] : pad
        rolling ^= previous * multipliers[layer][lookback]
        for head in 0..<heads {
          let column = (lookback - 1) * heads + head
          let row =
            rolling % primes[layer][lookback - 1][head] + offsets[layer][column]
          values[(token * layers + layer) * columns + column] = Int32(row)
        }
      }
    }
  }
  return (
    Tensor(values, .CPU, .HWC(tokens.count, layers, columns)), Array(combined.suffix(ngram - 1))
  )
}

// E4M3 lookup values match F8_E4M3 in Diffusion's SafeTensors.swift.
private let FP8_E4M3: [Float] = [
  0.0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875, 0.013671875,
  0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375, 0.029296875,
  0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375, 0.0625,
  0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625,
  0.15625,
  0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625,
  0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25,
  1.375,
  1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0,
  6.5,
  7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0,
  28.0,
  30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0,
  112.0,
  120.0, 128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 288.0, 320.0, 352.0,
  384.0,
  416.0, 448.0, .nan, -0.0, -0.001953125, -0.00390625, -0.005859375, -0.0078125, -0.009765625,
  -0.01171875, -0.013671875, -0.015625, -0.017578125, -0.01953125, -0.021484375, -0.0234375,
  -0.025390625, -0.02734375, -0.029296875, -0.03125, -0.03515625, -0.0390625, -0.04296875,
  -0.046875, -0.05078125, -0.0546875, -0.05859375, -0.0625, -0.0703125, -0.078125, -0.0859375,
  -0.09375, -0.1015625, -0.109375, -0.1171875, -0.125, -0.140625, -0.15625, -0.171875, -0.1875,
  -0.203125, -0.21875, -0.234375, -0.25, -0.28125, -0.3125, -0.34375, -0.375, -0.40625, -0.4375,
  -0.46875, -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375, -1.0, -1.125, -1.25,
  -1.375, -1.5, -1.625, -1.75, -1.875, -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0,
  -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0,
  -16.0, -18.0, -20.0, -22.0, -24.0, -26.0, -28.0, -30.0, -32.0, -36.0, -40.0, -44.0, -48.0,
  -52.0,
  -56.0, -60.0, -64.0, -72.0, -80.0, -88.0, -96.0, -104.0, -112.0, -120.0, -128.0, -144.0, -160.0,
  -176.0, -192.0, -208.0, -224.0, -240.0, -256.0, -288.0, -320.0, -352.0, -384.0, -416.0, -448,
  .nan,
]

/// The headerless file contains layer 1 followed by layer 14. Each row stores
/// 256 E4M3 bytes followed by eight E8M0 scales. Copies share the file handle;
/// positional reads do not mutate its seek position. Scheduling and persistent
/// row caches belong to the caller.
public struct DeepSeek4_1EngramReader<FloatType: TensorNumeric & BinaryFloatingPoint> {
  private let file: FileHandle

  public init(filePath: String) {
    guard let file = FileHandle(forReadingAtPath: filePath) else {
      preconditionFailure("Cannot open Engram file: \(filePath)")
    }
    precondition(
      (try? file.seekToEnd()) == DeepSeek4_1EngramConfiguration.deepSeekV4_1Flash.fileSize,
      "The Engram file does not have the expected size.")
    self.file = file
  }

  public func read(layerIndex: Int, rows: [Int32]) -> Tensor<FloatType> {
    let configuration = DeepSeek4_1EngramConfiguration.deepSeekV4_1Flash
    precondition(configuration.layers.contains(layerIndex))
    let layer = configuration.layers.firstIndex(of: layerIndex)!
    let width = configuration.headDim
    let rowBytes = configuration.rowBytes
    let tableOffset = configuration.tableOffsets[layer]
    let rowCount = configuration.rows[layer]
    var packed = [UInt8](repeating: 0, count: rowBytes)
    var output = [FloatType](repeating: 0, count: rows.count * width)
    var decoded = [Int32: Int]()
    output.withUnsafeMutableBufferPointer { destination in
      for (index, row) in rows.enumerated() {
        precondition(row >= 0 && Int(row) < rowCount)
        if let previous = decoded[row] {
          destination.baseAddress!.advanced(by: index * width).update(
            from: destination.baseAddress!.advanced(by: previous * width), count: width)
          continue
        }
        let offset = tableOffset + UInt64(row) * UInt64(rowBytes)
        packed.withUnsafeMutableBytes { buffer in
          var readCount = 0
          while readCount < rowBytes {
            let result = pread(
              file.fileDescriptor, buffer.baseAddress!.advanced(by: readCount),
              rowBytes - readCount, off_t(offset + UInt64(readCount)))
            if result < 0 && errno == EINTR { continue }
            precondition(
              result > 0, "Cannot read Engram row \(row) (errno \(result < 0 ? errno : 0)).")
            readCount += result
          }
        }
        for group in 0..<(width / 32) {
          let scaleByte = packed[width + group]
          precondition(scaleByte != 255, "Engram row \(row) contains an invalid scale.")
          let scale = Float(sign: .plus, exponent: Int(scaleByte) - 127, significand: 1)
          for column in (group * 32)..<((group + 1) * 32) {
            // Apply the E8M0 scale in FP32 before converting to the requested type.
            let value = FloatType(FP8_E4M3[Int(packed[column])] * scale)
            precondition(
              value.isFinite, "Engram row \(row) contains an invalid or overflowing value.")
            destination[index * width + column] = value
          }
        }
        decoded[row] = index
      }
    }
    return Tensor(output, .CPU, .WC(rows.count, width))
  }
}
