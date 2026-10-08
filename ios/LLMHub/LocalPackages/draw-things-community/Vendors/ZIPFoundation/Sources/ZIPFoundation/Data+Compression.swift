//
//  Data+Compression.swift
//  ZIPFoundation
//
//  Copyright © 2017-2021 Thomas Zoechling, https://www.peakstep.com and the ZIP Foundation project authors.
//  Released under the MIT License.
//
//  See https://github.com/weichsel/ZIPFoundation/blob/master/LICENSE for license information.
//

import Foundation

#if canImport(zlib)
  import zlib
#endif

/// The compression method of an `Entry` in a ZIP `Archive`.
public enum CompressionMethod: UInt16 {
  /// Indicates that an `Entry` has no compression applied to its contents.
  case none = 0
  /// Indicates that contents of an `Entry` have been compressed with a zlib compatible Deflate algorithm.
  case deflate = 8
}

/// An unsigned 32-Bit Integer representing a checksum.
public typealias CRC32 = UInt32
/// A custom handler that consumes a `Data` object containing partial entry data.
/// - Parameters:
///   - data: A chunk of `Data` to consume.
/// - Throws: Can throw to indicate errors during data consumption.
public typealias Consumer = (_ data: Data) throws -> Void
/// A custom handler that receives a position and a size that can be used to provide data from an arbitrary source.
/// - Parameters:
///   - position: The current read position.
///   - size: The size of the chunk to provide.
/// - Returns: A chunk of `Data`.
/// - Throws: Can throw to indicate errors in the data source.
public typealias Provider = (_ position: Int64, _ size: Int) throws -> Data

extension Data {
  enum CompressionError: Error {
    case invalidStream
    case corruptedData
  }

  /// Calculate the `CRC32` checksum of the receiver.
  ///
  /// - Parameter checksum: The starting seed.
  /// - Returns: The checksum calculated from the bytes of the receiver and the starting seed.
  public func crc32(checksum: CRC32) -> CRC32 {
    #if canImport(zlib)
      return withUnsafeBytes { bufferPointer in
        let length = UInt32(count)
        return CRC32(
          zlib.crc32(UInt(checksum), bufferPointer.bindMemory(to: UInt8.self).baseAddress, length))
      }
    #else
      return self.builtInCRC32(checksum: checksum)
    #endif
  }

  /// Compress the output of `provider` and pass it to `consumer`.
  /// - Parameters:
  ///   - size: The uncompressed size of the data to be compressed.
  ///   - bufferSize: The maximum size of the compression buffer.
  ///   - provider: A closure that accepts a position and a chunk size. Returns a `Data` chunk.
  ///   - consumer: A closure that processes the result of the compress operation.
  /// - Returns: The checksum of the processed content.
  public static func compress(size: Int64, bufferSize: Int, provider: Provider, consumer: Consumer)
    throws -> CRC32
  {
    #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS) || os(visionOS)
      return try self.process(
        operation: COMPRESSION_STREAM_ENCODE, size: size, bufferSize: bufferSize,
        provider: provider, consumer: consumer)
    #else
      return try self.encode(
        size: size, bufferSize: bufferSize, provider: provider, consumer: consumer)
    #endif
  }

  /// Decompress the output of `provider` and pass it to `consumer`.
  /// - Parameters:
  ///   - size: The compressed size of the data to be decompressed.
  ///   - bufferSize: The maximum size of the decompression buffer.
  ///   - skipCRC32: Optional flag to skip calculation of the CRC32 checksum to improve performance.
  ///   - provider: A closure that accepts a position and a chunk size. Returns a `Data` chunk.
  ///   - consumer: A closure that processes the result of the decompress operation.
  /// - Returns: The checksum of the processed content.
  public static func decompress(
    size: Int64, bufferSize: Int, skipCRC32: Bool,
    provider: Provider, consumer: Consumer
  ) throws -> CRC32 {
    #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS) || os(visionOS)
      return try self.process(
        operation: COMPRESSION_STREAM_DECODE, size: size, bufferSize: bufferSize,
        skipCRC32: skipCRC32, provider: provider, consumer: consumer)
    #else
      return try self.decode(
        bufferSize: bufferSize, skipCRC32: skipCRC32, provider: provider, consumer: consumer)
    #endif
  }
}

// MARK: - Apple Platforms

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS) || os(visionOS)
  import Compression

  extension Data {

    static func process(
      operation: compression_stream_operation, size: Int64, bufferSize: Int,
      skipCRC32: Bool = false,
      provider: Provider, consumer: Consumer
    ) throws -> CRC32 {
      var crc32 = CRC32(0)
      let destPointer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
      defer { destPointer.deallocate() }
      let streamPointer = UnsafeMutablePointer<compression_stream>.allocate(capacity: 1)
      defer { streamPointer.deallocate() }
      var stream = streamPointer.pointee
      var status = compression_stream_init(&stream, operation, COMPRESSION_ZLIB)
      guard status != COMPRESSION_STATUS_ERROR else { throw CompressionError.invalidStream }
      defer { compression_stream_destroy(&stream) }
      stream.src_size = 0
      stream.dst_ptr = destPointer
      stream.dst_size = bufferSize
      var position: Int64 = 0
      var sourceData: Data?
      repeat {
        let isExhausted = stream.src_size == 0
        if isExhausted {
          do {
            sourceData = try provider(
              position, Int(Swift.min((size - position), Int64(bufferSize))))
            position += Int64(stream.prepare(for: sourceData))
          } catch { throw error }
        }
        if let sourceData = sourceData {
          sourceData.withUnsafeBytes { rawBufferPointer in
            if let baseAddress = rawBufferPointer.baseAddress {
              let pointer = baseAddress.assumingMemoryBound(to: UInt8.self)
              stream.src_ptr = pointer.advanced(by: sourceData.count - stream.src_size)
              let flags =
                sourceData.count < bufferSize ? Int32(COMPRESSION_STREAM_FINALIZE.rawValue) : 0
              status = compression_stream_process(&stream, flags)
            }
          }
          if operation == COMPRESSION_STREAM_ENCODE && isExhausted && skipCRC32 == false {
            crc32 = sourceData.crc32(checksum: crc32)
          }
        }
        switch status {
        case COMPRESSION_STATUS_OK, COMPRESSION_STATUS_END:
          let outputData = Data(
            bytesNoCopy: destPointer, count: bufferSize - stream.dst_size, deallocator: .none)
          try consumer(outputData)
          if operation == COMPRESSION_STREAM_DECODE && !skipCRC32 {
            crc32 = outputData.crc32(checksum: crc32)
          }
          stream.dst_ptr = destPointer
          stream.dst_size = bufferSize
        default: throw CompressionError.corruptedData
        }
      } while status == COMPRESSION_STATUS_OK
      return crc32
    }
  }

  extension compression_stream {

    fileprivate mutating func prepare(for sourceData: Data?) -> Int {
      guard let sourceData = sourceData else { return 0 }

      self.src_size = sourceData.count
      return sourceData.count
    }
  }

// MARK: - Linux

#else
  import CZlib

  extension Data {
    static func encode(size: Int64, bufferSize: Int, provider: Provider, consumer: Consumer) throws
      -> CRC32
    {
      var stream = z_stream()
      let streamSize = Int32(MemoryLayout<z_stream>.size)
      var result = deflateInit2_(
        &stream, Z_DEFAULT_COMPRESSION,
        Z_DEFLATED, -MAX_WBITS, 9, Z_DEFAULT_STRATEGY, ZLIB_VERSION, streamSize)
      defer { deflateEnd(&stream) }
      guard result == Z_OK else { throw CompressionError.invalidStream }
      var flush = Z_NO_FLUSH
      var position: Int64 = 0
      var zipCRC32 = CRC32(0)
      repeat {
        let readSize = Int(Swift.min((size - position), Int64(bufferSize)))
        var inputChunk = try provider(position, readSize)
        zipCRC32 = inputChunk.crc32(checksum: zipCRC32)
        stream.avail_in = UInt32(inputChunk.count)
        try inputChunk.withUnsafeMutableBytes { (rawBufferPointer) in
          if let baseAddress = rawBufferPointer.baseAddress {
            let pointer = baseAddress.assumingMemoryBound(to: UInt8.self)
            stream.next_in = pointer
            flush = position + Int64(bufferSize) >= size ? Z_FINISH : Z_NO_FLUSH
          } else if rawBufferPointer.count > 0 {
            throw CompressionError.corruptedData
          } else {
            stream.next_in = nil
            flush = Z_FINISH
          }
          var outputChunk = Data(count: bufferSize)
          repeat {
            stream.avail_out = UInt32(bufferSize)
            try outputChunk.withUnsafeMutableBytes { (rawBufferPointer) in
              guard let baseAddress = rawBufferPointer.baseAddress, rawBufferPointer.count > 0
              else {
                throw CompressionError.corruptedData
              }
              let pointer = baseAddress.assumingMemoryBound(to: UInt8.self)
              stream.next_out = pointer
              result = deflate(&stream, flush)
            }
            guard result >= Z_OK else { throw CompressionError.corruptedData }

            outputChunk.count = bufferSize - Int(stream.avail_out)
            try consumer(outputChunk)
          } while stream.avail_out == 0
        }
        position += Int64(readSize)
      } while flush != Z_FINISH
      return zipCRC32
    }

    static func decode(bufferSize: Int, skipCRC32: Bool, provider: Provider, consumer: Consumer)
      throws -> CRC32
    {
      var stream = z_stream()
      let streamSize = Int32(MemoryLayout<z_stream>.size)
      var result = inflateInit2_(&stream, -MAX_WBITS, ZLIB_VERSION, streamSize)
      defer { inflateEnd(&stream) }
      guard result == Z_OK else { throw CompressionError.invalidStream }
      var unzipCRC32 = CRC32(0)
      var position: Int64 = 0
      repeat {
        stream.avail_in = UInt32(bufferSize)
        var chunk = try provider(position, bufferSize)
        position += Int64(chunk.count)
        try chunk.withUnsafeMutableBytes { (rawBufferPointer) in
          if let baseAddress = rawBufferPointer.baseAddress, rawBufferPointer.count > 0 {
            let pointer = baseAddress.assumingMemoryBound(to: UInt8.self)
            stream.next_in = pointer
            repeat {
              var outputData = Data(count: bufferSize)
              stream.avail_out = UInt32(bufferSize)
              try outputData.withUnsafeMutableBytes { (rawBufferPointer) in
                if let baseAddress = rawBufferPointer.baseAddress, rawBufferPointer.count > 0 {
                  let pointer = baseAddress.assumingMemoryBound(to: UInt8.self)
                  stream.next_out = pointer
                } else {
                  throw CompressionError.corruptedData
                }
                result = inflate(&stream, Z_NO_FLUSH)
                guard result != Z_NEED_DICT && result != Z_DATA_ERROR && result != Z_MEM_ERROR
                else {
                  throw CompressionError.corruptedData
                }
              }
              let remainingLength = UInt32(bufferSize) - stream.avail_out
              outputData.count = Int(remainingLength)
              try consumer(outputData)
              if !skipCRC32 { unzipCRC32 = outputData.crc32(checksum: unzipCRC32) }
            } while stream.avail_out == 0
          }
        }
      } while result != Z_STREAM_END
      return unzipCRC32
    }
  }

#endif

/// The lookup table used to calculate `CRC32` checksums when using the built-in
/// CRC32 implementation.
private let crcTable: [CRC32] = [
  0x0000_0000, 0x7707_3096, 0xee0e_612c, 0x9909_51ba, 0x076d_c419, 0x706a_f48f, 0xe963_a535,
  0x9e64_95a3, 0x0edb_8832,
  0x79dc_b8a4, 0xe0d5_e91e, 0x97d2_d988, 0x09b6_4c2b, 0x7eb1_7cbd, 0xe7b8_2d07, 0x90bf_1d91,
  0x1db7_1064, 0x6ab0_20f2,
  0xf3b9_7148, 0x84be_41de, 0x1ada_d47d, 0x6ddd_e4eb, 0xf4d4_b551, 0x83d3_85c7, 0x136c_9856,
  0x646b_a8c0, 0xfd62_f97a,
  0x8a65_c9ec, 0x1401_5c4f, 0x6306_6cd9, 0xfa0f_3d63, 0x8d08_0df5, 0x3b6e_20c8, 0x4c69_105e,
  0xd560_41e4, 0xa267_7172,
  0x3c03_e4d1, 0x4b04_d447, 0xd20d_85fd, 0xa50a_b56b, 0x35b5_a8fa, 0x42b2_986c, 0xdbbb_c9d6,
  0xacbc_f940, 0x32d8_6ce3,
  0x45df_5c75, 0xdcd6_0dcf, 0xabd1_3d59, 0x26d9_30ac, 0x51de_003a, 0xc8d7_5180, 0xbfd0_6116,
  0x21b4_f4b5, 0x56b3_c423,
  0xcfba_9599, 0xb8bd_a50f, 0x2802_b89e, 0x5f05_8808, 0xc60c_d9b2, 0xb10b_e924, 0x2f6f_7c87,
  0x5868_4c11, 0xc161_1dab,
  0xb666_2d3d, 0x76dc_4190, 0x01db_7106, 0x98d2_20bc, 0xefd5_102a, 0x71b1_8589, 0x06b6_b51f,
  0x9fbf_e4a5, 0xe8b8_d433,
  0x7807_c9a2, 0x0f00_f934, 0x9609_a88e, 0xe10e_9818, 0x7f6a_0dbb, 0x086d_3d2d, 0x9164_6c97,
  0xe663_5c01, 0x6b6b_51f4,
  0x1c6c_6162, 0x8565_30d8, 0xf262_004e, 0x6c06_95ed, 0x1b01_a57b, 0x8208_f4c1, 0xf50f_c457,
  0x65b0_d9c6, 0x12b7_e950,
  0x8bbe_b8ea, 0xfcb9_887c, 0x62dd_1ddf, 0x15da_2d49, 0x8cd3_7cf3, 0xfbd4_4c65, 0x4db2_6158,
  0x3ab5_51ce, 0xa3bc_0074,
  0xd4bb_30e2, 0x4adf_a541, 0x3dd8_95d7, 0xa4d1_c46d, 0xd3d6_f4fb, 0x4369_e96a, 0x346e_d9fc,
  0xad67_8846, 0xda60_b8d0,
  0x4404_2d73, 0x3303_1de5, 0xaa0a_4c5f, 0xdd0d_7cc9, 0x5005_713c, 0x2702_41aa, 0xbe0b_1010,
  0xc90c_2086, 0x5768_b525,
  0x206f_85b3, 0xb966_d409, 0xce61_e49f, 0x5ede_f90e, 0x29d9_c998, 0xb0d0_9822, 0xc7d7_a8b4,
  0x59b3_3d17, 0x2eb4_0d81,
  0xb7bd_5c3b, 0xc0ba_6cad, 0xedb8_8320, 0x9abf_b3b6, 0x03b6_e20c, 0x74b1_d29a, 0xead5_4739,
  0x9dd2_77af, 0x04db_2615,
  0x73dc_1683, 0xe363_0b12, 0x9464_3b84, 0x0d6d_6a3e, 0x7a6a_5aa8, 0xe40e_cf0b, 0x9309_ff9d,
  0x0a00_ae27, 0x7d07_9eb1,
  0xf00f_9344, 0x8708_a3d2, 0x1e01_f268, 0x6906_c2fe, 0xf762_575d, 0x8065_67cb, 0x196c_3671,
  0x6e6b_06e7, 0xfed4_1b76,
  0x89d3_2be0, 0x10da_7a5a, 0x67dd_4acc, 0xf9b9_df6f, 0x8ebe_eff9, 0x17b7_be43, 0x60b0_8ed5,
  0xd6d6_a3e8, 0xa1d1_937e,
  0x38d8_c2c4, 0x4fdf_f252, 0xd1bb_67f1, 0xa6bc_5767, 0x3fb5_06dd, 0x48b2_364b, 0xd80d_2bda,
  0xaf0a_1b4c, 0x3603_4af6,
  0x4104_7a60, 0xdf60_efc3, 0xa867_df55, 0x316e_8eef, 0x4669_be79, 0xcb61_b38c, 0xbc66_831a,
  0x256f_d2a0, 0x5268_e236,
  0xcc0c_7795, 0xbb0b_4703, 0x2202_16b9, 0x5505_262f, 0xc5ba_3bbe, 0xb2bd_0b28, 0x2bb4_5a92,
  0x5cb3_6a04, 0xc2d7_ffa7,
  0xb5d0_cf31, 0x2cd9_9e8b, 0x5bde_ae1d, 0x9b64_c2b0, 0xec63_f226, 0x756a_a39c, 0x026d_930a,
  0x9c09_06a9, 0xeb0e_363f,
  0x7207_6785, 0x0500_5713, 0x95bf_4a82, 0xe2b8_7a14, 0x7bb1_2bae, 0x0cb6_1b38, 0x92d2_8e9b,
  0xe5d5_be0d, 0x7cdc_efb7,
  0x0bdb_df21, 0x86d3_d2d4, 0xf1d4_e242, 0x68dd_b3f8, 0x1fda_836e, 0x81be_16cd, 0xf6b9_265b,
  0x6fb0_77e1, 0x18b7_4777,
  0x8808_5ae6, 0xff0f_6a70, 0x6606_3bca, 0x1101_0b5c, 0x8f65_9eff, 0xf862_ae69, 0x616b_ffd3,
  0x166c_cf45, 0xa00a_e278,
  0xd70d_d2ee, 0x4e04_8354, 0x3903_b3c2, 0xa767_2661, 0xd060_16f7, 0x4969_474d, 0x3e6e_77db,
  0xaed1_6a4a, 0xd9d6_5adc,
  0x40df_0b66, 0x37d8_3bf0, 0xa9bc_ae53, 0xdebb_9ec5, 0x47b2_cf7f, 0x30b5_ffe9, 0xbdbd_f21c,
  0xcaba_c28a, 0x53b3_9330,
  0x24b4_a3a6, 0xbad0_3605, 0xcdd7_0693, 0x54de_5729, 0x23d9_67bf, 0xb366_7a2e, 0xc461_4ab8,
  0x5d68_1b02, 0x2a6f_2b94,
  0xb40b_be37, 0xc30c_8ea1, 0x5a05_df1b, 0x2d02_ef8d,
]

extension Data {

  /// Lookup table-based CRC32 implenetation that is used
  /// if `zlib` isn't available.
  /// - Parameter checksum: Running checksum or `0` for the initial run.
  /// - Returns: The calculated checksum of the receiver.
  func builtInCRC32(checksum: CRC32) -> CRC32 {
    // The typecast is necessary on 32-bit platforms because of
    // https://bugs.swift.org/browse/SR-1774
    let mask = 0xffff_ffff as CRC32
    var result = checksum ^ mask
    #if swift(>=5.0)
      crcTable.withUnsafeBufferPointer { crcTablePointer in
        self.withUnsafeBytes { bufferPointer in
          var bufferIndex = 0
          while bufferIndex < self.count {
            let byte = bufferPointer[bufferIndex]
            let index = Int((result ^ CRC32(byte)) & 0xff)
            result = (result >> 8) ^ crcTablePointer[index]
            bufferIndex += 1
          }
        }
      }
    #else
      self.withUnsafeBytes { (bytes) in
        let bins = stride(from: 0, to: self.count, by: 256)
        for bin in bins {
          for binIndex in 0..<256 {
            let byteIndex = bin + binIndex
            guard byteIndex < self.count else { break }

            let byte = bytes[byteIndex]
            let index = Int((result ^ CRC32(byte)) & 0xff)
            result = (result >> 8) ^ crcTable[index]
          }
        }
      }
    #endif
    return result ^ mask
  }
}

#if !swift(>=5.0)

  // Since Swift 5.0, `Data.withUnsafeBytes()` passes an `UnsafeRawBufferPointer` instead of an `UnsafePointer<UInt8>`
  // into `body`.
  // We provide a compatible method for targets that use Swift 4.x so that we can use the new version
  // across all language versions.

  extension Data {
    func withUnsafeBytes<T>(_ body: (UnsafeRawBufferPointer) throws -> T) rethrows -> T {
      let count = self.count
      return try withUnsafeBytes { (pointer: UnsafePointer<UInt8>) throws -> T in
        try body(UnsafeRawBufferPointer(start: pointer, count: count))
      }
    }

    #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS) || os(visionOS)
    #else
      mutating func withUnsafeMutableBytes<T>(_ body: (UnsafeMutableRawBufferPointer) throws -> T)
        rethrows -> T
      {
        let count = self.count
        guard count > 0 else {
          return try body(UnsafeMutableRawBufferPointer(start: nil, count: count))
        }
        return try withUnsafeMutableBytes { (pointer: UnsafeMutablePointer<UInt8>) throws -> T in
          try body(UnsafeMutableRawBufferPointer(start: pointer, count: count))
        }
      }
    #endif
  }
#endif
