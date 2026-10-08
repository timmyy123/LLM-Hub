//
//  Number+Extensions.swift
//  Utils
//
//  Created by Gong Chen on 9/6/23.
//

import Foundation

extension UInt16 {
  public static func gcd(_ a: UInt16, _ b: UInt16) -> UInt16 {
    guard a != 0 && b != 0 else { return 1 }
    var common = Swift.max(a, b)
    var remainder = Swift.min(a, b)

    while remainder != 0 {
      let temp = remainder
      remainder = common % remainder
      common = temp
    }

    return common
  }
}

extension Int {
  public static func gcd(_ a: Int, _ b: Int) -> Int {
    guard a != 0 && b != 0 else { return 1 }
    var common = Swift.max(abs(a), abs(b))
    var remainder = Swift.min(abs(a), abs(b))

    while remainder != 0 {
      let temp = remainder
      remainder = common % remainder
      common = temp
    }

    return common
  }
}
