// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation

/// A random source consistent with PyTorch
///
///  This implementation matches:
///  [NumPy's older randomkit.c](https://github.com/numpy/numpy/blob/v1.0/numpy/random/mtrand/randomkit.c)
///
public struct TorchRandomSource: RandomNumberGenerator {

  struct State {
    var key = [UInt32](repeating: 0, count: 624)
    var pos: Int = 0
    var nextGauss: Double? = nil
  }

  var state: State

  /// Initialize with a random seed
  ///
  /// - Parameters
  ///     - seed: Seed for underlying Mersenne Twister 19937 generator
  /// - Returns random source
  public init(seed: UInt32) {
    state = .init()
    var s = seed & 0xffff_ffff
    for i in 0..<state.key.count {
      state.key[i] = s
      s = UInt32((UInt64(1_812_433_253) * UInt64(s ^ (s >> 30)) + UInt64(i) + 1) & 0xffff_ffff)
    }
    state.pos = state.key.count
    state.nextGauss = nil
  }

  /// Generate next UInt32 using fast 32bit Mersenne Twister
  mutating func nextUInt32() -> UInt32 {
    let n = 624
    let m = 397
    let matrixA: UInt64 = 0x9908_b0df
    let upperMask: UInt32 = 0x8000_0000
    let lowerMask: UInt32 = 0x7fff_ffff

    var y: UInt32
    if state.pos == state.key.count {
      for i in 0..<(n - m) {
        y = (state.key[i] & upperMask) | (state.key[i + 1] & lowerMask)
        state.key[i] = state.key[i + m] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
      }
      for i in (n - m)..<(n - 1) {
        y = (state.key[i] & upperMask) | (state.key[i + 1] & lowerMask)
        state.key[i] = state.key[i + (m - n)] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
      }
      y = (state.key[n - 1] & upperMask) | (state.key[0] & lowerMask)
      state.key[n - 1] = state.key[m - 1] ^ (y >> 1) ^ UInt32((UInt64(~(y & 1)) + 1) & matrixA)
      state.pos = 0
    }
    y = state.key[state.pos]
    state.pos += 1

    y ^= (y >> 11)
    y ^= (y << 7) & 0x9d2c_5680
    y ^= (y << 15) & 0xefc6_0000
    y ^= (y >> 18)

    return y
  }

  public mutating func next() -> UInt64 {
    let high = nextUInt32()
    let low = nextUInt32()
    return (UInt64(high) << 32) | UInt64(low)
  }

  /// Generate next random double value
  mutating func nextDouble() -> Double {
    let a = next()
    return Double(a & 9_007_199_254_740_991) * (1.0 / 9007199254740992.0)
  }

  /// Generate next random float value
  mutating func nextFloat() -> Float {
    let a = nextUInt32()
    return Float(a & 16_777_215) * (1.0 / 16777216.0)
  }

  /// Generate next random value from a standard normal
  mutating func nextGauss() -> Double {
    if let nextGauss = state.nextGauss {
      state.nextGauss = nil
      return nextGauss
    }
    // Box-Muller transform
    let u1: Double = nextDouble()
    let u2: Double = 1 - nextDouble()
    let radius = sqrt(-2.0 * log(u2))
    let theta = 2.0 * .pi * u1
    state.nextGauss = radius * sin(theta)
    return radius * cos(theta)
  }

  mutating func normalArrayDouble(count: Int) -> [Double] {
    (0..<count).map { _ in nextGauss() }
  }

  /// Generates an array of random values from a normal distribution.
  public mutating func normalArray(count: Int) -> [Float] {
    guard count >= 16 else {
      return normalArrayDouble(count: count).map { Float($0) }
    }
    var data = (0..<count).map { _ in nextFloat() }
    for i in stride(from: 0, to: count - 15, by: 16) {
      for j in 0..<8 {
        let u1 = 1 - data[i + j]
        let u2 = data[i + j + 8]
        let radius = sqrt(-2.0 * log(u1))
        let theta = 2.0 * .pi * u2
        data[i + j] = radius * cos(theta)
        data[i + j + 8] = radius * sin(theta)
      }
    }
    if count % 16 != 0 {
      for i in (count - 16)..<count {
        data[i] = nextFloat()
      }
      let i = count - 16
      for j in 0..<8 {
        let u1 = 1 - data[i + j]
        let u2 = data[i + j + 8]
        let radius = sqrt(-2.0 * log(u1))
        let theta = 2.0 * .pi * u2
        data[i + j] = radius * cos(theta)
        data[i + j + 8] = radius * sin(theta)
      }
    }
    return data
  }
}
