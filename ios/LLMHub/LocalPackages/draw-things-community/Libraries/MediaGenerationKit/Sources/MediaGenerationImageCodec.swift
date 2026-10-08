import CoreGraphics
import Diffusion
import Foundation
import ImageIO
import LocalImageGenerator
import NNC
import UniformTypeIdentifiers

#if canImport(UIKit)
  import UIKit
#endif

internal enum MediaGenerationImageCodec {
  static func decode(_ data: Data) throws -> Tensor<FloatType> {
    guard
      let cgImage = ImageConverter.cgImage(from: data),
      let bitmapContext = ImageConverter.bitmapContext(from: cgImage),
      let tensor = ImageConverter.tensor(from: bitmapContext).0
    else {
      throw MediaGenerationKitError.generationFailed("result payload is not a decodable image")
    }
    return tensor
  }

  static func encode(_ tensor: Tensor<FloatType>, type: UTType) throws -> Data {
    switch type {
    case .png:
      guard let pngData = pngData(from: tensor) else {
        throw MediaGenerationKitError.generationFailed("failed to encode result tensor as PNG")
      }
      return pngData
    default:
      guard let pngData = pngData(from: tensor) else {
        throw MediaGenerationKitError.generationFailed("failed to encode result tensor as PNG")
      }
      guard let source = CGImageSourceCreateWithData(pngData as CFData, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
        let mutableData = CFDataCreateMutable(nil, 0),
        let destination = CGImageDestinationCreateWithData(
          mutableData,
          type.identifier as CFString,
          1,
          nil
        )
      else {
        throw MediaGenerationKitError.generationFailed(
          "failed to encode result tensor as \(type.identifier)"
        )
      }
      CGImageDestinationAddImage(destination, image, nil)
      guard CGImageDestinationFinalize(destination) else {
        throw MediaGenerationKitError.generationFailed(
          "failed to encode result tensor as \(type.identifier)"
        )
      }
      return mutableData as Data
    }
  }

  static func width(for tensor: Tensor<FloatType>) -> Int {
    if tensor.shape.count >= 3 {
      return tensor.shape[tensor.shape.count - 2]
    }
    return 0
  }

  static func height(for tensor: Tensor<FloatType>) -> Int {
    if tensor.shape.count >= 3 {
      return tensor.shape[tensor.shape.count - 3]
    }
    return 0
  }

  static func previewImage(from tensor: Tensor<FloatType>, version: ModelVersion) -> CGImage? {
    ImageConverter.cgImages(
      fromLatent: tensor,
      canUseTAESD: true,
      version: version
    ).0.first
  }

  private static func pngData(from tensor: Tensor<FloatType>) -> Data? {
    #if canImport(UIKit)
      let image = ImageConverter.image(from: tensor, scaleFactor: 1.0)
      return image.pngData()
    #else
      let imageHeight = height(for: tensor)
      let imageWidth = width(for: tensor)
      guard imageWidth > 0, imageHeight > 0 else {
        return nil
      }

      let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
      defer { bytes.deallocate() }

      tensor.withUnsafeBytes {
        guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
        for i in 0..<(imageHeight * imageWidth) {
          let r = (fp16[i * 3] + 1) * 127.5
          let g = (fp16[i * 3 + 1] + 1) * 127.5
          let b = (fp16[i * 3 + 2] + 1) * 127.5
          bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
          bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
          bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
          bytes[i * 4 + 3] = 255
        }
      }

      guard
        let cgImage = CGImage(
          width: imageWidth,
          height: imageHeight,
          bitsPerComponent: 8,
          bitsPerPixel: 32,
          bytesPerRow: 4 * imageWidth,
          space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder32Big.rawValue
              | CGImageAlphaInfo.noneSkipLast.rawValue
          ),
          provider: CGDataProvider(
            dataInfo: nil,
            data: bytes,
            size: imageWidth * imageHeight * 4,
            releaseData: { _, _, _ in }
          )!,
          decode: nil,
          shouldInterpolate: false,
          intent: .defaultIntent
        ),
        let mutableData = CFDataCreateMutable(nil, 0),
        let destination = CGImageDestinationCreateWithData(
          mutableData,
          UTType.png.identifier as CFString,
          1,
          nil
        )
      else {
        return nil
      }

      CGImageDestinationAddImage(destination, cgImage, nil)
      guard CGImageDestinationFinalize(destination) else {
        return nil
      }

      return mutableData as Data
    #endif
  }
}
