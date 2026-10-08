import Foundation

// See https://github.com/dsnz/random/blob/master/philox.py for philox 4 32 configuration.
public struct NVRandomSource {
  public let seed: UInt64
  private var offset: UInt32

  /// Initialize with a random seed
  ///
  /// - Parameters
  ///     - seed: Seed for underlying Mersenne Twister 19937 generator
  /// - Returns random source
  public init(seed: UInt64) {
    self.seed = seed
    offset = 0
  }

  static private let PHILOX_M4_32: (UInt32, UInt32) = (0xD251_1F53, 0xCD9E_8D57)
  static private let PHILOX_W_32: (UInt32, UInt32) = (0x9E37_79B9, 0xBB67_AE85)

  static private func philox4Round(counter: inout [[UInt32]], key: [[UInt32]]) {
    for i in 0..<counter[0].count {
      let v1: UInt64 = UInt64(counter[0][i]) * UInt64(PHILOX_M4_32.0)
      let v2: UInt64 = UInt64(counter[2][i]) * UInt64(PHILOX_M4_32.1)
      counter[0][i] = UInt32(v2 >> 32) ^ counter[1][i] ^ key[0][i]
      counter[1][i] = UInt32(v2 & 0xffff_ffff)
      counter[2][i] = UInt32(v1 >> 32) ^ counter[3][i] ^ key[1][i]
      counter[3][i] = UInt32(v1 & 0xffff_ffff)
    }
  }

  static private func philox4Bumpkey(key: inout [[UInt32]]) {
    for (i, element) in key[0].enumerated() {
      key[0][i] = element &+ PHILOX_W_32.0
    }
    for (i, element) in key[1].enumerated() {
      key[1][i] = element &+ PHILOX_W_32.1
    }
  }

  static private func philox4_32(counter: inout [[UInt32]], key: inout [[UInt32]], rounds: Int = 10)
  {
    for _ in 0..<(rounds - 1) {
      philox4Round(counter: &counter, key: key)
      philox4Bumpkey(key: &key)
    }
    philox4Round(counter: &counter, key: key)
  }

  private func boxMuller(_ counter1: [UInt32], _ counter2: [UInt32]) -> [Float] {
    // Box-Muller transform
    return zip(counter1, counter2).map {
      let u: Double = Double($0) / 4294967296.0 + (1.0 / 8589934592.0)
      let v: Double = Double($1) * (.pi / 2147483648.0) + (.pi / 4294967296.0)
      let radius = sqrt(-2.0 * log(u))
      return Float(radius * sin(v))
    }
  }

  public mutating func normalArray(count: Int, mean: Float = 0.0, stdev: Float = 1.0) -> [Float] {
    var counter: [[UInt32]] = [
      Array(repeating: offset, count: count),
      Array(repeating: 0, count: count),
      Array(0..<UInt32(count)),
      Array(repeating: 0, count: count),
    ]
    offset += 1
    var key: [[UInt32]] = [
      Array(repeating: UInt32(seed & 0xffff_ffff), count: count),
      Array(repeating: UInt32(seed >> 32), count: count),
    ]
    Self.philox4_32(counter: &counter, key: &key)
    return boxMuller(counter[0], counter[1])
  }
}
