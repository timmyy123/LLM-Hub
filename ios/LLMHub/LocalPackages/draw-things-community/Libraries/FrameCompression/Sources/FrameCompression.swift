import Diffusion
import Foundation
import NNC

#if canImport(CoreMedia) && canImport(CoreVideo) && canImport(VideoToolbox)
  import CoreMedia
  import CoreVideo
  import VideoToolbox
#endif
#if canImport(CoreGraphics) && canImport(ImageIO) && canImport(UniformTypeIdentifiers)
  import CoreGraphics
  import ImageIO
  import UniformTypeIdentifiers
#endif
#if canImport(CoreImage)
  import CoreImage
#endif

public enum CompressionCodec: String, CaseIterable {
  case h264
  case h265
  case jpeg
}

public enum FrameCompression {
  public enum Error: Swift.Error, CustomStringConvertible {
    case unsupportedPlatform
    case invalidQuality(Float)
    case tensorMustBeCPU
    case tensorMustBeNHWC
    case invalidTensorShape(String)
    case unsupportedDimensions(codec: CompressionCodec, width: Int, height: Int)
    case tensorStorageUnavailable
    case pixelBufferCreate(OSStatus)
    case pixelBufferNoBaseAddress
    case codecUnavailable(CompressionCodec, OSStatus)
    case compressionPropertySet(String, OSStatus)
    case encode(OSStatus)
    case encodedFrameUnavailable
    case decompressionSessionCreate(OSStatus)
    case decode(OSStatus)
    case decodedFrameUnavailable
    case jpegEncodingFailed
    case jpegDecodingFailed

    public var description: String {
      switch self {
      case .unsupportedPlatform:
        return "Compression artifacts are not available on this platform."
      case .invalidQuality(let quality):
        return "Quality \(quality) is outside the valid range of 0...100."
      case .tensorMustBeCPU:
        return "Input tensor must be on CPU."
      case .tensorMustBeNHWC:
        return "Input tensor must use NHWC format."
      case .invalidTensorShape(let shape):
        return "Input tensor must have shape NHWC(1, height, width, 3), got \(shape)."
      case .unsupportedDimensions(let codec, let width, let height):
        return "\(codec.rawValue.uppercased()) requires even width/height. Got \(width)x\(height)."
      case .tensorStorageUnavailable:
        return "Tensor storage is unavailable."
      case .pixelBufferCreate(let status):
        return "Failed to create CVPixelBuffer (status: \(status))."
      case .pixelBufferNoBaseAddress:
        return "CVPixelBuffer has no base address."
      case .codecUnavailable(let codec, let status):
        return "Codec \(codec.rawValue.uppercased()) is unavailable (status: \(status))."
      case .compressionPropertySet(let key, let status):
        return "Failed to set compression property '\(key)' (status: \(status))."
      case .encode(let status):
        return "Video encoding failed (status: \(status))."
      case .encodedFrameUnavailable:
        return "Encoder did not return a compressed sample buffer."
      case .decompressionSessionCreate(let status):
        return "Failed to create VTDecompressionSession (status: \(status))."
      case .decode(let status):
        return "Video decoding failed (status: \(status))."
      case .decodedFrameUnavailable:
        return "Decoder did not return a pixel buffer."
      case .jpegEncodingFailed:
        return "Failed to encode JPEG data."
      case .jpegDecodingFailed:
        return "Failed to decode JPEG data."
      }
    }
  }

  /// Applies compression/decompression to inject codec artifacts into an image tensor.
  /// - Parameters:
  ///   - tensor: CPU tensor in NHWC(1, height, width, 3), value range approximately [-1, 1].
  ///   - codec: Compression codec used to introduce artifacts.
  ///   - quality: 0...100 where lower quality introduces stronger artifacts.
  /// - Returns: Tensor with compression artifacts introduced by encode/decode round-trip.
  public static func applyCompressionArtifacts(
    to tensor: Tensor<FloatType>, codec: CompressionCodec, quality: Float
  ) throws -> Tensor<FloatType> {
    let (width, height) = try validateInput(tensor, codec: codec, quality: quality)
    let pixelBuffer = try createPixelBuffer(from: tensor, width: width, height: height)
    switch codec {
    case .jpeg:
      return try jpegArtifacts(from: pixelBuffer, quality: quality)
    case .h264, .h265:
      #if canImport(CoreMedia) && canImport(CoreVideo) && canImport(VideoToolbox)
        return try videoToolboxArtifacts(
          from: pixelBuffer, width: width, height: height, codec: codec, quality: quality)
      #else
        throw Error.unsupportedPlatform
      #endif
    }
  }

  private static func validateInput(
    _ tensor: Tensor<FloatType>, codec: CompressionCodec, quality: Float
  ) throws -> (width: Int, height: Int) {
    guard quality >= 0, quality <= 100 else {
      throw Error.invalidQuality(quality)
    }
    guard tensor.kind == .CPU else {
      throw Error.tensorMustBeCPU
    }
    guard tensor.format == .NHWC else {
      throw Error.tensorMustBeNHWC
    }
    let shape = tensor.shape
    guard shape.count == 4, shape[0] == 1, shape[3] == 3 else {
      throw Error.invalidTensorShape(shape.map(String.init).joined(separator: "x"))
    }
    let height = shape[1]
    let width = shape[2]
    guard width > 0, height > 0 else {
      throw Error.invalidTensorShape(shape.map(String.init).joined(separator: "x"))
    }
    if codec == .h264 || codec == .h265 {
      guard width % 2 == 0, height % 2 == 0 else {
        throw Error.unsupportedDimensions(codec: codec, width: width, height: height)
      }
    }
    return (width, height)
  }

  private static func createPixelBuffer(
    from tensor: Tensor<FloatType>, width: Int, height: Int
  ) throws -> CVPixelBuffer {
    var pixelBuffer: CVPixelBuffer?
    let attributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
      attributes as CFDictionary, &pixelBuffer)
    guard status == kCVReturnSuccess, let pixelBuffer else {
      throw Error.pixelBufferCreate(status)
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw Error.pixelBufferNoBaseAddress
    }

    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let dst = baseAddress.assumingMemoryBound(to: UInt8.self)
    var storageUnavailable = false

    tensor.withUnsafeBytes { buffer in
      guard let src = buffer.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
        storageUnavailable = true
        return
      }
      for y in 0..<height {
        for x in 0..<width {
          let srcOffset = y * width * 3 + x * 3
          let dstOffset = y * bytesPerRow + x * 4
          let r = max(-1, min(1, Float(src[srcOffset])))
          let g = max(-1, min(1, Float(src[srcOffset + 1])))
          let b = max(-1, min(1, Float(src[srcOffset + 2])))
          dst[dstOffset] = UInt8(min(max(Int(((b + 1) * 127.5).rounded()), 0), 255))
          dst[dstOffset + 1] = UInt8(min(max(Int(((g + 1) * 127.5).rounded()), 0), 255))
          dst[dstOffset + 2] = UInt8(min(max(Int(((r + 1) * 127.5).rounded()), 0), 255))
          dst[dstOffset + 3] = 255
        }
      }
    }

    if storageUnavailable {
      throw Error.tensorStorageUnavailable
    }
    return pixelBuffer
  }

  fileprivate static func createTensor(from pixelBuffer: CVPixelBuffer) throws -> Tensor<FloatType>
  {
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, height, width, 3))

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw Error.pixelBufferNoBaseAddress
    }

    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let src = baseAddress.assumingMemoryBound(to: UInt8.self)
    var storageUnavailable = false

    tensor.withUnsafeMutableBytes { buffer in
      guard let dst = buffer.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
        storageUnavailable = true
        return
      }
      for y in 0..<height {
        for x in 0..<width {
          let srcOffset = y * bytesPerRow + x * 4
          let dstOffset = y * width * 3 + x * 3
          dst[dstOffset] = FloatType(Float(src[srcOffset + 2]) * (2.0 / 255.0) - 1.0)
          dst[dstOffset + 1] = FloatType(Float(src[srcOffset + 1]) * (2.0 / 255.0) - 1.0)
          dst[dstOffset + 2] = FloatType(Float(src[srcOffset]) * (2.0 / 255.0) - 1.0)
        }
      }
    }

    if storageUnavailable {
      throw Error.tensorStorageUnavailable
    }
    return tensor
  }
}

#if canImport(CoreGraphics) && canImport(ImageIO) && canImport(UniformTypeIdentifiers) && canImport(CoreImage)
  extension FrameCompression {
    private static func jpegArtifacts(from pixelBuffer: CVPixelBuffer, quality: Float) throws
      -> Tensor<FloatType>
    {
      let width = CVPixelBufferGetWidth(pixelBuffer)
      let height = CVPixelBufferGetHeight(pixelBuffer)
      let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
      let ciContext = CIContext(options: nil)
      guard
        let cgImage = ciContext.createCGImage(
          ciImage, from: CGRect(x: 0, y: 0, width: width, height: height))
      else {
        throw Error.jpegEncodingFailed
      }

      let data = NSMutableData()
      guard
        let destination = CGImageDestinationCreateWithData(
          data, UTType.jpeg.identifier as CFString, 1, nil)
      else {
        throw Error.jpegEncodingFailed
      }
      let options: CFDictionary =
        [
          kCGImageDestinationLossyCompressionQuality as String: NSNumber(value: quality / 100.0)
        ] as CFDictionary
      CGImageDestinationAddImage(destination, cgImage, options)
      guard CGImageDestinationFinalize(destination) else {
        throw Error.jpegEncodingFailed
      }

      guard
        let source = CGImageSourceCreateWithData(data as CFData, nil),
        let decodedImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
      else {
        throw Error.jpegDecodingFailed
      }

      guard
        let bitmapContext = CGContext(
          data: nil,
          width: width,
          height: height,
          bitsPerComponent: 8,
          bytesPerRow: width * 4,
          space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue
            | CGImageAlphaInfo.premultipliedLast.rawValue,
          releaseCallback: nil,
          releaseInfo: nil)
      else {
        throw Error.jpegDecodingFailed
      }
      bitmapContext.draw(decodedImage, in: CGRect(x: 0, y: 0, width: width, height: height))
      guard let contextData = bitmapContext.data else {
        throw Error.jpegDecodingFailed
      }

      let bytes = contextData.assumingMemoryBound(to: UInt8.self)
      var tensor = Tensor<FloatType>(.CPU, .NHWC(1, height, width, 3))
      var storageUnavailable = false
      tensor.withUnsafeMutableBytes { buffer in
        guard let dst = buffer.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
          storageUnavailable = true
          return
        }
        for y in 0..<height {
          for x in 0..<width {
            let srcOffset = y * width * 4 + x * 4
            let dstOffset = y * width * 3 + x * 3
            dst[dstOffset] = FloatType(Float(bytes[srcOffset]) * (2.0 / 255.0) - 1.0)
            dst[dstOffset + 1] = FloatType(Float(bytes[srcOffset + 1]) * (2.0 / 255.0) - 1.0)
            dst[dstOffset + 2] = FloatType(Float(bytes[srcOffset + 2]) * (2.0 / 255.0) - 1.0)
          }
        }
      }
      if storageUnavailable {
        throw Error.tensorStorageUnavailable
      }
      return tensor
    }
  }
#endif

#if canImport(CoreMedia) && canImport(CoreVideo) && canImport(VideoToolbox)
  extension FrameCompression {
    fileprivate static func videoToolboxArtifacts(
      from pixelBuffer: CVPixelBuffer, width: Int, height: Int, codec: CompressionCodec,
      quality: Float
    ) throws -> Tensor<FloatType> {
      let sampleBuffer = try encodeFrame(
        pixelBuffer: pixelBuffer, width: width, height: height, codec: codec, quality: quality)
      let decodedPixelBuffer = try decodeFrame(sampleBuffer: sampleBuffer)
      return try createTensor(from: decodedPixelBuffer)
    }

    fileprivate static func encodeFrame(
      pixelBuffer: CVPixelBuffer, width: Int, height: Int, codec: CompressionCodec,
      quality: Float
    ) throws -> CMSampleBuffer {
      var compressionSession: VTCompressionSession?
      let createStatus = VTCompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        width: Int32(width),
        height: Int32(height),
        codecType: cmCodecType(for: codec),
        encoderSpecification: nil,
        imageBufferAttributes: nil,
        compressedDataAllocator: nil,
        outputCallback: nil,
        refcon: nil,
        compressionSessionOut: &compressionSession)
      guard createStatus == noErr, let compressionSession else {
        throw Error.codecUnavailable(codec, createStatus)
      }
      defer { VTCompressionSessionInvalidate(compressionSession) }

      try setCompressionProperties(
        compressionSession: compressionSession, codec: codec, quality: quality)

      let prepareStatus = VTCompressionSessionPrepareToEncodeFrames(compressionSession)
      guard prepareStatus == noErr else {
        throw Error.encode(prepareStatus)
      }

      let frameProperties: CFDictionary =
        [
          kVTEncodeFrameOptionKey_ForceKeyFrame as String: true
        ] as CFDictionary
      var infoFlags = VTEncodeInfoFlags(rawValue: 0)
      var encodedSampleBuffer: CMSampleBuffer?
      var callbackError: Error?

      let encodeStatus = VTCompressionSessionEncodeFrame(
        compressionSession,
        imageBuffer: pixelBuffer,
        presentationTimeStamp: .zero,
        duration: .invalid,
        frameProperties: frameProperties,
        infoFlagsOut: &infoFlags
      ) { status, _, sampleBuffer in
        guard status == noErr else {
          callbackError = .encode(status)
          return
        }
        guard let sampleBuffer, CMSampleBufferDataIsReady(sampleBuffer) else {
          callbackError = .encodedFrameUnavailable
          return
        }
        encodedSampleBuffer = sampleBuffer
      }
      guard encodeStatus == noErr else {
        throw Error.encode(encodeStatus)
      }

      let completeStatus = VTCompressionSessionCompleteFrames(
        compressionSession, untilPresentationTimeStamp: .invalid)
      guard completeStatus == noErr else {
        throw Error.encode(completeStatus)
      }

      if let callbackError {
        throw callbackError
      }
      guard let encodedSampleBuffer else {
        throw Error.encodedFrameUnavailable
      }
      return encodedSampleBuffer
    }

    fileprivate static func decodeFrame(sampleBuffer: CMSampleBuffer) throws -> CVPixelBuffer {
      guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else {
        throw Error.encodedFrameUnavailable
      }

      var decompressionSession: VTDecompressionSession?
      let attributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      ]
      let createStatus = VTDecompressionSessionCreate(
        allocator: kCFAllocatorDefault,
        formatDescription: formatDescription,
        decoderSpecification: nil,
        imageBufferAttributes: attributes as CFDictionary,
        outputCallback: nil,
        decompressionSessionOut: &decompressionSession)
      guard createStatus == noErr, let decompressionSession else {
        throw Error.decompressionSessionCreate(createStatus)
      }
      defer { VTDecompressionSessionInvalidate(decompressionSession) }

      var decodeInfoFlags = VTDecodeInfoFlags(rawValue: 0)
      var decodedPixelBuffer: CVPixelBuffer?
      var callbackError: Error?

      let decodeStatus = VTDecompressionSessionDecodeFrame(
        decompressionSession,
        sampleBuffer: sampleBuffer,
        flags: [],
        infoFlagsOut: &decodeInfoFlags
      ) { status, _, imageBuffer, _, _ in
        guard status == noErr else {
          callbackError = .decode(status)
          return
        }
        guard let imageBuffer else {
          callbackError = .decodedFrameUnavailable
          return
        }
        decodedPixelBuffer = imageBuffer
      }
      guard decodeStatus == noErr else {
        throw Error.decode(decodeStatus)
      }

      let waitStatus = VTDecompressionSessionWaitForAsynchronousFrames(decompressionSession)
      guard waitStatus == noErr else {
        throw Error.decode(waitStatus)
      }

      if let callbackError {
        throw callbackError
      }
      guard let decodedPixelBuffer else {
        throw Error.decodedFrameUnavailable
      }
      return decodedPixelBuffer
    }

    fileprivate static func setCompressionProperties(
      compressionSession: VTCompressionSession, codec: CompressionCodec, quality: Float
    ) throws {
      let crf = crfValue(fromQuality: quality)
      let constantQuality = 1.0 - (Float(crf) / 51.0)

      try setProperty(
        for: compressionSession,
        key: kVTCompressionPropertyKey_ProfileLevel,
        value: profileLevel(for: codec))
      try setProperty(
        for: compressionSession,
        key: kVTCompressionPropertyKey_AllowFrameReordering,
        value: kCFBooleanFalse)
      try setProperty(
        for: compressionSession,
        key: kVTCompressionPropertyKey_MaxKeyFrameInterval,
        value: NSNumber(value: 1))
      try setProperty(
        for: compressionSession,
        key: kVTCompressionPropertyKey_ExpectedFrameRate,
        value: NSNumber(value: 1))
      try setProperty(
        for: compressionSession,
        key: kVTCompressionPropertyKey_RealTime,
        value: kCFBooleanFalse)
      try setProperty(
        for: compressionSession,
        key: kVTCompressionPropertyKey_Quality,
        value: NSNumber(value: constantQuality))

      // Approximate x264 "preset=veryfast" by preferring speed when available.
      try trySetPropertyIfSupported(
        for: compressionSession,
        key: kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality,
        value: kCFBooleanTrue)
    }

    fileprivate static func setProperty(
      for session: VTCompressionSession, key: CFString, value: CFTypeRef
    ) throws {
      let status = VTSessionSetProperty(session, key: key, value: value)
      guard status == noErr else {
        throw Error.compressionPropertySet(key as String, status)
      }
    }

    fileprivate static func trySetPropertyIfSupported(
      for session: VTCompressionSession, key: CFString, value: CFTypeRef
    ) throws {
      let status = VTSessionSetProperty(session, key: key, value: value)
      if status == noErr || status == kVTPropertyNotSupportedErr {
        return
      }
      throw Error.compressionPropertySet(key as String, status)
    }

    fileprivate static func cmCodecType(for codec: CompressionCodec) -> CMVideoCodecType {
      switch codec {
      case .h264:
        return kCMVideoCodecType_H264
      case .h265:
        return kCMVideoCodecType_HEVC
      case .jpeg:
        return kCMVideoCodecType_H264
      }
    }

    fileprivate static func profileLevel(for codec: CompressionCodec) -> CFTypeRef {
      switch codec {
      case .h264:
        return kVTProfileLevel_H264_High_AutoLevel
      case .h265:
        return kVTProfileLevel_HEVC_Main_AutoLevel
      case .jpeg:
        return kVTProfileLevel_H264_High_AutoLevel
      }
    }

    fileprivate static func crfValue(fromQuality quality: Float) -> Int {
      let normalizedQuality = max(0, min(100, quality))
      return Int(roundf((100.0 - normalizedQuality) / 100.0 * 51.0))
    }
  }
#endif
