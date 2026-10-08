import Foundation
#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif

enum InsdiffPatcher {
  enum Error: Swift.Error {
    case invalidPatch(String)
  }

  private struct PatchFile: Decodable {
    let format: String
    let base_sha256: String
    let target_sha256: String
    let base_size: Int
    let target_size: Int
    let ops: [PatchOp]
  }

  private struct PatchOp: Decodable {
    let type: String
    let src_offset: Int?
    let payload_offset: Int?
    let length: Int
  }

  static func apply(baseData: Data, patchData: Data) throws -> Data {
    guard patchData.count >= 8 else {
      throw Error.invalidPatch("Patch data is too small.")
    }

    let metadataLength = patchData.withUnsafeBytes { bytes -> Int in
      let u8 = bytes.bindMemory(to: UInt8.self)
      var value: UInt64 = 0
      for i in 0..<8 {
        value |= UInt64(u8[i]) << UInt64(8 * i)
      }
      return Int(value)
    }
    guard metadataLength >= 0, 8 + metadataLength <= patchData.count else {
      throw Error.invalidPatch("Invalid metadata length in patch.")
    }

    let metadataData = patchData.subdata(in: 8..<(8 + metadataLength))
    let payloadData = patchData.subdata(in: (8 + metadataLength)..<patchData.count)
    let patch = try JSONDecoder().decode(PatchFile.self, from: metadataData)

    guard patch.format == "insdiff-v1" else {
      throw Error.invalidPatch("Unsupported patch format: \(patch.format)")
    }
    guard baseData.count == patch.base_size else {
      throw Error.invalidPatch(
        "Base size mismatch. expected=\(patch.base_size) actual=\(baseData.count)")
    }

    let baseHash = sha256Hex(baseData)
    guard baseHash == patch.base_sha256 else {
      throw Error.invalidPatch(
        "Base SHA256 mismatch. expected=\(patch.base_sha256) actual=\(baseHash)")
    }

    var output = Data()
    output.reserveCapacity(patch.target_size)

    for op in patch.ops {
      guard op.length >= 0 else {
        throw Error.invalidPatch("Invalid op length: \(op.length)")
      }
      switch op.type {
      case "copy":
        guard let srcOffset = op.src_offset,
          let range = validatedRange(offset: srcOffset, length: op.length, limit: baseData.count)
        else {
          throw Error.invalidPatch(
            "copy out of range: src=\(op.src_offset ?? -1) len=\(op.length)")
        }
        output.append(baseData.subdata(in: range))
      case "insert":
        guard let payloadOffset = op.payload_offset,
          let range = validatedRange(
            offset: payloadOffset, length: op.length, limit: payloadData.count)
        else {
          throw Error.invalidPatch(
            "insert out of range: off=\(op.payload_offset ?? -1) len=\(op.length)")
        }
        output.append(payloadData.subdata(in: range))
      default:
        throw Error.invalidPatch("Unknown op type: \(op.type)")
      }
    }

    guard output.count == patch.target_size else {
      throw Error.invalidPatch(
        "Output size mismatch. expected=\(patch.target_size) actual=\(output.count)")
    }
    let outputHash = sha256Hex(output)
    guard outputHash == patch.target_sha256 else {
      throw Error.invalidPatch(
        "Output SHA256 mismatch. expected=\(patch.target_sha256) actual=\(outputHash)")
    }

    return output
  }

  private static func validatedRange(offset: Int, length: Int, limit: Int) -> Range<Int>? {
    guard offset >= 0, length >= 0, offset <= limit, length <= limit - offset else {
      return nil
    }
    return offset..<(offset + length)
  }

  private static func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
}
