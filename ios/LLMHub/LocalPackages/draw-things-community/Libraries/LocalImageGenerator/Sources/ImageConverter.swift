import CoreGraphics
import DataModels
import Diffusion
import ImageGenerator
import ImageIO
import ModelZoo
import NNC
import ScriptDataModels
import UniformTypeIdentifiers

#if canImport(UIKit)
  import UIKit
#endif
#if canImport(CoreML)
  import CoreML
  import NNCCoreMLConversion
  import DiffusionCoreML
  import ZIPFoundation
#endif

public enum ImageConverter {
  public static func cgImage(from data: Data) -> CGImage? {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
      return nil
    }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
  }

  public static func bitmapContext(from cgImage: CGImage) -> CGContext? {
    guard
      let bitmapContext = CGContext(
        data: nil, width: cgImage.width, height: cgImage.height, bitsPerComponent: 8,
        bytesPerRow: cgImage.width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue
          | CGImageAlphaInfo.premultipliedLast.rawValue, releaseCallback: nil, releaseInfo: nil)
    else {
      return nil
    }
    bitmapContext.draw(
      cgImage,
      in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
    return bitmapContext
  }

  public static func resize(
    from cgImage: CGImage, imageWidth: Int, imageHeight: Int,
    interpolationQuality: CGInterpolationQuality = .high
  ) -> CGContext? {
    guard
      let bitmapContext = CGContext(
        data: nil, width: imageWidth, height: imageHeight, bitsPerComponent: 8,
        bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue
          | CGImageAlphaInfo.premultipliedLast.rawValue, releaseCallback: nil, releaseInfo: nil)
    else {
      return nil
    }
    bitmapContext.interpolationQuality = interpolationQuality
    let scaledWidth: Int
    let scaledHeight: Int
    if cgImage.width * imageHeight > cgImage.height * imageWidth {
      scaledHeight = imageHeight
      scaledWidth = cgImage.width * imageHeight / cgImage.height
    } else {
      scaledWidth = imageWidth
      scaledHeight = cgImage.height * imageWidth / cgImage.width
    }
    bitmapContext.draw(
      cgImage,
      in: CGRect(
        x: (imageWidth - scaledWidth) / 2, y: (imageHeight - scaledHeight) / 2,
        width: scaledWidth, height: scaledHeight))
    return bitmapContext
  }
  #if canImport(UIKit)
    public static func resize(
      from image: UIImage, imageWidth: Int, imageHeight: Int,
      interpolationQuality: CGInterpolationQuality = .high
    ) -> CGContext? {
      guard let cgImage = image.cgImage,
        let bitmapContext = CGContext(
          data: nil, width: imageWidth, height: imageHeight, bitsPerComponent: 8,
          bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue
            | CGImageAlphaInfo.premultipliedLast.rawValue, releaseCallback: nil, releaseInfo: nil)
      else {
        return nil
      }
      bitmapContext.interpolationQuality = interpolationQuality
      // Scale to fill.
      var cgImageWidth = cgImage.width
      var cgImageHeight = cgImage.height
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&cgImageWidth, &cgImageHeight)
      }
      var scaledWidth: Int
      var scaledHeight: Int
      if cgImageWidth * imageHeight > cgImageHeight * imageWidth {
        scaledHeight = imageHeight
        scaledWidth = cgImageWidth * imageHeight / cgImageHeight
      } else {
        scaledWidth = imageWidth
        scaledHeight = cgImageHeight * imageWidth / cgImageWidth
      }
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&scaledWidth, &scaledHeight)
      }
      // Set rotation angle with image orientation. Don't deal with mirrored just yet.
      switch image.imageOrientation {
      case .up:
        break
      case .upMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        bitmapContext.scaleBy(x: -1, y: 1)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .down, .downMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .downMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .left, .rightMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .rightMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .right, .leftMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .leftMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi * 3 / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      @unknown default:
        break
      }
      bitmapContext.draw(
        cgImage,
        in: CGRect(
          x: (imageWidth - scaledWidth) / 2, y: (imageHeight - scaledHeight) / 2,
          width: scaledWidth, height: scaledHeight))
      return bitmapContext
    }
    public static func grayscaleTensor(from image: UIImage, imageWidth: Int, imageHeight: Int)
      -> Tensor<
        UInt8
      >?
    {
      guard let cgImage = image.cgImage,
        let bitmapContext = CGContext(
          data: nil, width: imageWidth, height: imageHeight, bitsPerComponent: 8,
          bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue
            | CGImageAlphaInfo.premultipliedLast.rawValue, releaseCallback: nil, releaseInfo: nil)
      else {
        return nil
      }
      bitmapContext.interpolationQuality = .high
      // Scale to fill.
      var cgImageWidth = cgImage.width
      var cgImageHeight = cgImage.height
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&cgImageWidth, &cgImageHeight)
      }
      var scaledWidth: Int
      var scaledHeight: Int
      if cgImageWidth * imageHeight > cgImageHeight * imageWidth {
        scaledHeight = imageHeight
        scaledWidth = cgImageWidth * imageHeight / cgImageHeight
      } else {
        scaledWidth = imageWidth
        scaledHeight = cgImageHeight * imageWidth / cgImageWidth
      }
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&scaledWidth, &scaledHeight)
      }
      // Set rotation angle with image orientation. Don't deal with mirrored just yet.
      switch image.imageOrientation {
      case .up:
        break
      case .upMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        bitmapContext.scaleBy(x: -1, y: 1)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .down, .downMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .downMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .left, .rightMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .rightMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .right, .leftMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .leftMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi * 3 / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      @unknown default:
        break
      }
      bitmapContext.draw(
        cgImage,
        in: CGRect(
          x: (imageWidth - scaledWidth) / 2, y: (imageHeight - scaledHeight) / 2,
          width: scaledWidth, height: scaledHeight))
      return grayscaleTensor(from: bitmapContext)
    }
  #endif
  public static func grayscaleTensor(from bitmapContext: CGContext) -> Tensor<UInt8>? {
    precondition(bitmapContext.bytesPerRow >= bitmapContext.width * 4)
    guard let data = bitmapContext.data else {
      return nil
    }
    let bytes = data.assumingMemoryBound(to: UInt8.self)
    let bytesPerRow = bitmapContext.bytesPerRow
    let imageHeight = bitmapContext.height
    let imageWidth = bitmapContext.width
    func fillGrayscaleTensor(destination u8: UnsafeMutablePointer<UInt8>) {
      for y in 0..<imageHeight {
        let srcRow = y * bytesPerRow
        let dstRow = y * imageWidth
        for x in 0..<imageWidth {
          let srcOffset = srcRow + x * 4
          let alpha = bytes[srcOffset + 3]
          if alpha < 128 {
            u8[dstRow + x] = 255  // We treat transparent as white.
          } else {
            let r = Int32(bytes[srcOffset])
            let g = Int32(bytes[srcOffset + 1])
            let b = Int32(bytes[srcOffset + 2])
            u8[dstRow + x] = UInt8((r * 6969 + g * 23434 + b * 2365) >> 15)
          }
        }
      }
    }
    var tensor = Tensor<UInt8>(.CPU, .NC(bitmapContext.height, bitmapContext.width))
    tensor.withUnsafeMutableBytes {
      (unsafeMutableRawBufferPointer: UnsafeMutableRawBufferPointer) in
      guard
        let u8: UnsafeMutablePointer<UInt8> = unsafeMutableRawBufferPointer.baseAddress?
          .assumingMemoryBound(to: UInt8.self)
      else { return }
      fillGrayscaleTensor(destination: u8)
    }
    return tensor
  }
  public static func scribbleRGB(from tensor: Tensor<UInt8>) -> Tensor<FloatType> {
    let imageHeight = tensor.shape[0]
    let imageWidth = tensor.shape[1]
    var rgb = Tensor<FloatType>(.CPU, .NHWC(1, imageHeight, imageWidth, 3))
    rgb.withUnsafeMutableBytes {
      guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      tensor.withUnsafeBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for i in 0..<imageHeight * imageWidth {
          if u8[i] > 128 {
            fp16[i * 3] = 0
            fp16[i * 3 + 1] = 0
            fp16[i * 3 + 2] = 0
          } else {
            fp16[i * 3] = 1
            fp16[i * 3 + 1] = 1
            fp16[i * 3 + 2] = 1
          }
        }
      }
    }
    return rgb
  }
  static func scribble(from binaryMask: Tensor<UInt8>) -> Tensor<UInt8>? {
    let maskHeight = binaryMask.shape[0]
    let maskWidth = binaryMask.shape[1]
    var flag = true
    var paintCount = 0
    binaryMask.withUnsafeBytes {
      guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      for y in 0..<maskHeight {
        for x in 0..<maskWidth {
          let byte = u8[y * maskWidth + x]
          if byte == 0 {
            paintCount += 1
          } else if byte == 1 {
            // Empty.
          } else {
            flag = false
            break
          }
        }
        if !flag {
          break
        }
      }
    }
    // If we haven't paint even half of the screen, this must be scribble.
    guard flag && paintCount * 2 < maskHeight * maskWidth else {
      return nil
    }
    var scribble = Tensor<UInt8>(.CPU, .NC(maskHeight, maskWidth))
    scribble.withUnsafeMutableBytes {
      guard let s8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      binaryMask.withUnsafeBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for y in 0..<maskHeight {
          for x in 0..<maskWidth {
            let byte = u8[y * maskWidth + x]
            s8[y * maskWidth + x] = byte == 0 ? 0 : 255
          }
        }
      }
    }
    return scribble
  }
  #if canImport(UIKit)
    public static func grayscaleImage(binaryMask: Tensor<UInt8>) -> UIImage {
      let imageHeight = binaryMask.shape[0]
      let imageWidth = binaryMask.shape[1]
      let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
      binaryMask.withUnsafeBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for i in 0..<imageHeight * imageWidth {
          let byteMask = u8[i]
          if (byteMask & 7) != 0 && (byteMask & 7) != 3 {
            let alpha = 0xff - (byteMask & 0xf8)
            bytes[i * 4] = alpha
            bytes[i * 4 + 1] = alpha
            bytes[i * 4 + 2] = alpha
            bytes[i * 4 + 3] = 255
          } else {
            bytes[i * 4] = 0
            bytes[i * 4 + 1] = 0
            bytes[i * 4 + 2] = 0
            bytes[i * 4 + 3] = 255
          }
        }
      }
      return UIImage(
        cgImage: CGImage(
          width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
          bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
          provider: CGDataProvider(
            dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
            releaseData: { _, p, _ in
              p.deallocate()
            })!, decode: nil, shouldInterpolate: false,
          intent: CGColorRenderingIntent.defaultIntent)!)
    }
    public static func grayscaleImage(from tensor: Tensor<UInt8>) -> UIImage {
      let imageHeight = tensor.shape[0]
      let imageWidth = tensor.shape[1]
      let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
      tensor.withUnsafeBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for i in 0..<imageHeight * imageWidth {
          let byte = u8[i]
          bytes[i * 4] = byte
          bytes[i * 4 + 1] = byte
          bytes[i * 4 + 2] = byte
          bytes[i * 4 + 3] = 255
        }
      }
      return UIImage(
        cgImage: CGImage(
          width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
          bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
          provider: CGDataProvider(
            dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
            releaseData: { _, p, _ in
              p.deallocate()
            })!, decode: nil, shouldInterpolate: false,
          intent: CGColorRenderingIntent.defaultIntent)!)
    }
    public static func grayscaleImage(from tensor: Tensor<FloatType>) -> UIImage {
      let imageHeight = tensor.shape[1]
      let imageWidth = tensor.shape[2]
      let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
      tensor.withUnsafeBytes {
        guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
        for i in 0..<imageHeight * imageWidth {
          let byte = UInt8(min(max(Int((fp16[i] + 1) * 127.5), 0), 255))
          bytes[i * 4] = byte
          bytes[i * 4 + 1] = byte
          bytes[i * 4 + 2] = byte
          bytes[i * 4 + 3] = 255
        }
      }
      return UIImage(
        cgImage: CGImage(
          width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
          bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
          provider: CGDataProvider(
            dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
            releaseData: { _, p, _ in
              p.deallocate()
            })!, decode: nil, shouldInterpolate: false,
          intent: CGColorRenderingIntent.defaultIntent)!)
    }
    public static func positiveRangeTensor(
      from image: UIImage, imageWidth: Int, imageHeight: Int
    ) -> Tensor<
      FloatType
    >? {
      let bitmapInfo =
        CGBitmapInfo.byteOrderDefault.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
      guard let cgImage = image.cgImage,
        let bitmapContext = CGContext(
          data: nil, width: imageWidth, height: imageHeight, bitsPerComponent: 8,
          bytesPerRow: imageWidth * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: bitmapInfo, releaseCallback: nil, releaseInfo: nil)
      else {
        return nil
      }
      bitmapContext.interpolationQuality = .high
      // Scale to fill.
      var cgImageWidth = cgImage.width
      var cgImageHeight = cgImage.height
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&cgImageWidth, &cgImageHeight)
      }
      var scaledWidth: Int
      var scaledHeight: Int
      if cgImageWidth * imageHeight > cgImageHeight * imageWidth {
        scaledHeight = imageHeight
        scaledWidth = cgImageWidth * imageHeight / cgImageHeight
      } else {
        scaledWidth = imageWidth
        scaledHeight = cgImageHeight * imageWidth / cgImageWidth
      }
      if image.imageOrientation == .left || image.imageOrientation == .leftMirrored
        || image.imageOrientation == .right || image.imageOrientation == .rightMirrored
      {
        swap(&scaledWidth, &scaledHeight)
      }
      // Set rotation angle with image orientation. Don't deal with mirrored just yet.
      switch image.imageOrientation {
      case .up:
        break
      case .upMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        bitmapContext.scaleBy(x: -1, y: 1)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .down, .downMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .downMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .left, .rightMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .rightMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      case .right, .leftMirrored:
        bitmapContext.translateBy(x: CGFloat(imageWidth) / 2, y: CGFloat(imageHeight) / 2)
        if image.imageOrientation == .leftMirrored {
          bitmapContext.scaleBy(x: -1, y: 1)
        }
        bitmapContext.rotate(by: .pi * 3 / 2)
        bitmapContext.translateBy(x: -CGFloat(imageWidth) / 2, y: -CGFloat(imageHeight) / 2)
      @unknown default:
        break
      }
      bitmapContext.draw(
        cgImage,
        in: CGRect(
          x: (imageWidth - scaledWidth) / 2, y: (imageHeight - scaledHeight) / 2,
          width: scaledWidth, height: scaledHeight))
      return positiveRangeTensor(from: bitmapContext)
    }
  #endif
  public static func positiveRangeTensor(from bitmapContext: CGContext) -> Tensor<FloatType>? {
    precondition(bitmapContext.bytesPerRow >= bitmapContext.width * 4)
    guard let data = bitmapContext.data else {
      return nil
    }
    let bytes = data.assumingMemoryBound(to: UInt8.self)
    let bytesPerRow = bitmapContext.bytesPerRow
    let imageHeight = bitmapContext.height
    let imageWidth = bitmapContext.width
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, imageHeight, imageWidth, 3))
    tensor.withUnsafeMutableBytes {
      guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for y in 0..<imageHeight {
        for x in 0..<imageWidth {
          f16[y * imageWidth * 3 + x * 3] = FloatType(Float(bytes[y * bytesPerRow + x * 4]) / 255.0)
          f16[y * imageWidth * 3 + x * 3 + 1] = FloatType(
            Float(bytes[y * bytesPerRow + x * 4 + 1]) / 255.0)
          f16[y * imageWidth * 3 + x * 3 + 2] = FloatType(
            Float(bytes[y * bytesPerRow + x * 4 + 2]) / 255.0)
        }
      }
    }
    return tensor
  }
  public static func imageAndMask(from argbTensor: Tensor<FloatType>) -> (
    Tensor<FloatType>, Tensor<UInt8>?
  ) {
    let shape = argbTensor.shape
    guard shape[3] == 4 else {
      precondition(shape[3] == 3)
      return (argbTensor, nil)
    }
    let rgbTensor = argbTensor[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<shape[3]].copied()
    var binaryMask = Tensor<UInt8>(.CPU, .NC(shape[1], shape[2]))
    let height = shape[1]
    let width = shape[2]
    argbTensor.withUnsafeBytes {
      guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      binaryMask.withUnsafeMutableBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for y in 0..<height {
          for x in 0..<width {
            let alpha = min(max(Int((f16[y * width * 4 + x * 4] * 255).rounded()), 0), 255)
            if alpha != 255 {
              u8[y * width + x] = (1 | UInt8(alpha & 0xf8))
            } else {
              u8[y * width + x] = 3
            }
          }
        }
      }
    }
    return (rgbTensor, binaryMask)
  }
  #if canImport(UIKit)

    public static func image(
      from tensor: Tensor<FloatType>, scaleFactor: CGFloat, binaryMask: Tensor<UInt8>? = nil,
      only1: Bool = false, overlayRects: [CGRect] = []
    ) -> UIImage {
      let imageHeight = tensor.shape[1]
      let imageWidth = tensor.shape[2]
      let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
      tensor.withUnsafeBytes {
        guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
        for i in 0..<imageHeight * imageWidth {
          let r = (fp16[i * 3] + 1) * 127.5
          let g = (fp16[i * 3 + 1] + 1) * 127.5
          let b = (fp16[i * 3 + 2] + 1) * 127.5
          bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
          bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
          bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
          bytes[i * 4 + 3] = 255
        }
      }
      let invScale = 1 / scaleFactor
      if let binaryMask = binaryMask {
        let maskHeight = binaryMask.shape[0]
        let maskWidth = binaryMask.shape[1]
        if only1 {
          let overlayIntegralRects: [(minX: Int, minY: Int, maxX: Int, maxY: Int)] =
            overlayRects.map {
              (
                minX: Int($0.minX.rounded(.up)), minY: Int($0.minY.rounded(.up)),
                maxX: Int(($0.maxX - 1).rounded(.down)), maxY: Int(($0.maxY - 1).rounded(.down))
              )
            }
          binaryMask.withUnsafeBytes {
            guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for i in 0..<imageHeight {
              let ii = max(
                0, min(maskHeight - 1, Int(((CGFloat(i) + 0.5) * invScale - 0.5).rounded())))
              for j in 0..<imageWidth {
                let ij = max(
                  0, min(maskWidth - 1, Int(((CGFloat(j) + 0.5) * invScale - 0.5).rounded())))
                let byteMask = u8[ii * maskWidth + ij]
                let flag = overlayIntegralRects.contains {
                  ii >= $0.minY && ii <= $0.maxY && ij >= $0.minX && ij <= $0.maxX
                }
                // We make it transparent if it is 1 or it is overlapped by the overlay rects.
                if (byteMask & 7) == 1 || (flag && (byteMask & 7) != 0 && (byteMask & 7) != 3) {
                  let alpha = byteMask & 0xf8
                  if alpha > 0 {
                    bytes[i * imageWidth * 4 + j * 4] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 1] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4 + 1]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 2] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4 + 2]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 3] = alpha
                  } else {
                    bytes[i * imageWidth * 4 + j * 4] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 1] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 2] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 3] = 0
                  }
                }
              }
            }
          }
        } else {
          binaryMask.withUnsafeBytes {
            guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for i in 0..<imageHeight {
              let ii = max(
                0, min(maskHeight - 1, Int(((CGFloat(i) + 0.5) * invScale - 0.5).rounded())))
              for j in 0..<imageWidth {
                let ij = max(
                  0, min(maskWidth - 1, Int(((CGFloat(j) + 0.5) * invScale - 0.5).rounded())))
                let byteMask = u8[ii * maskWidth + ij]
                if (byteMask & 7) != 0 && (byteMask & 7) != 3 {
                  let alpha = byteMask & 0xf8
                  if alpha > 0 {
                    bytes[i * imageWidth * 4 + j * 4] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 1] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4 + 1]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 2] = UInt8(
                      min((Int32(bytes[i * imageWidth * 4 + j * 4 + 2]) * Int32(alpha)) >> 8, 255))
                    bytes[i * imageWidth * 4 + j * 4 + 3] = alpha
                  } else {
                    bytes[i * imageWidth * 4 + j * 4] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 1] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 2] = 0
                    bytes[i * imageWidth * 4 + j * 4 + 3] = 0
                  }
                }
              }
            }
          }
        }
        return UIImage(
          cgImage: CGImage(
            width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(
              rawValue: CGBitmapInfo.byteOrder32Big.rawValue
                | CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: CGDataProvider(
              dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
              releaseData: { _, p, _ in
                p.deallocate()
              })!, decode: nil, shouldInterpolate: false,
            intent: CGColorRenderingIntent.defaultIntent)!)
      } else {
        return UIImage(
          cgImage: CGImage(
            width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(
              rawValue: CGBitmapInfo.byteOrder32Big.rawValue
                | CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: CGDataProvider(
              dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
              releaseData: { _, p, _ in
                p.deallocate()
              })!, decode: nil, shouldInterpolate: false,
            intent: CGColorRenderingIntent.defaultIntent)!)
      }
    }
  #endif
  @inline(__always)
  static private func OKlabToLinearsRGB(L: Float, a: Float, b: Float) -> (Float, Float, Float) {
    let l_ = L + 0.3963377774 * a + 0.2158037573 * b
    let m_ = L - 0.1055613458 * a - 0.0638541728 * b
    let s_ = L - 0.0894841775 * a - 1.2914855480 * b
    let l = l_ * l_ * l_
    let m = m_ * m_ * m_
    let s = s_ * s_ * s_
    return (
      +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    )
  }
  @inline(__always)
  static private func linearsRGBTosRGB(x: Float) -> Float {
    if x >= 0.04045 {
      return pow((x + 0.055) / (1 + 0.055), 2.4)
    } else {
      return x / 12.92
    }
  }
  #if canImport(CoreML)
    public static func imagesWithTAESD(fromLatent tensor: Tensor<FloatType>, version: ModelVersion)
      -> [CGImage]
    {
      var tensor = tensor
      var shape = tensor.shape
      let imageHeight: Int
      let imageWidth: Int
      let imagePaddingSize: Int
      let taesd: ManagedMLModel?
      let isVideo: Bool
      switch version {
      case .hiDreamO1:
        return []
      case .flux1, .hiDreamI1, .zImage:
        imageHeight = shape[1] * 8
        imageWidth = shape[2] * 8
        isVideo = false
        switch max(imageWidth, imageHeight) {
        case 0...768:
          taesd = flux1TinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = flux1TinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = flux1TinyDecoderFor1280
          imagePaddingSize = 1280
        case 1281...1536:
          taesd = flux1TinyDecoderFor1536
          imagePaddingSize = 1536
        case 1537...1792:
          taesd = flux1TinyDecoderFor1792
          imagePaddingSize = 1792
        case 1793...2048:
          taesd = flux1TinyDecoderFor2048
          imagePaddingSize = 2048
        default:
          return []
        }
      case .v1, .v2, .svdI2v:
        imageHeight = shape[1] * 8
        imageWidth = shape[2] * 8
        isVideo = false
        switch max(imageWidth, imageHeight) {
        case 0...768:
          taesd = sdTinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = sdTinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = sdTinyDecoderFor1280
          imagePaddingSize = 1280
        case 1281...1536:
          taesd = sdTinyDecoderFor1536
          imagePaddingSize = 1536
        case 1537...1792:
          taesd = sdTinyDecoderFor1792
          imagePaddingSize = 1792
        case 1793...2048:
          taesd = sdTinyDecoderFor2048
          imagePaddingSize = 2048
        default:
          return []
        }
      case .sd3, .sd3Large:
        imageHeight = shape[1] * 8
        imageWidth = shape[2] * 8
        isVideo = false
        switch max(imageWidth, imageHeight) {
        case 0...768:
          taesd = sd3TinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = sd3TinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = sd3TinyDecoderFor1280
          imagePaddingSize = 1280
        case 1281...1536:
          taesd = sd3TinyDecoderFor1536
          imagePaddingSize = 1536
        case 1537...1792:
          taesd = sd3TinyDecoderFor1792
          imagePaddingSize = 1792
        case 1793...2048:
          taesd = sd3TinyDecoderFor2048
          imagePaddingSize = 2048
        default:
          return []
        }
      case .sdxlBase, .sdxlRefiner, .ssd1b, .pixart, .auraflow:
        imageHeight = shape[1] * 8
        imageWidth = shape[2] * 8
        isVideo = false
        switch max(imageWidth, imageHeight) {
        case 0...768:
          taesd = sdxlTinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = sdxlTinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = sdxlTinyDecoderFor1280
          imagePaddingSize = 1280
        case 1281...1536:
          taesd = sdxlTinyDecoderFor1536
          imagePaddingSize = 1536
        case 1537...1792:
          taesd = sdxlTinyDecoderFor1792
          imagePaddingSize = 1792
        case 1793...2048:
          taesd = sdxlTinyDecoderFor2048
          imagePaddingSize = 2048
        default:
          return []
        }
      case .qwenImage2_1:
        imageHeight = shape[1] * 16
        imageWidth = shape[2] * 16
        isVideo = false
        switch max(imageWidth, imageHeight) {
        case 0...768:
          taesd = qwenImage2_1TinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = qwenImage2_1TinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = qwenImage2_1TinyDecoderFor1280
          imagePaddingSize = 1280
        case 1281...1536:
          taesd = qwenImage2_1TinyDecoderFor1536
          imagePaddingSize = 1536
        case 1537...1792:
          taesd = qwenImage2_1TinyDecoderFor1792
          imagePaddingSize = 1792
        case 1793...2048:
          taesd = qwenImage2_1TinyDecoderFor2048
          imagePaddingSize = 2048
        default:
          return []
        }
      case .qwenImage, .cosmos2_5_2b, .krea2:
        imageHeight = shape[1] * 8
        imageWidth = shape[2] * 8
        isVideo = false
        switch max(imageWidth, imageHeight) {
        case 0...768:
          taesd = qwenImageTinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = qwenImageTinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = qwenImageTinyDecoderFor1280
          imagePaddingSize = 1280
        case 1281...1536:
          taesd = qwenImageTinyDecoderFor1536
          imagePaddingSize = 1536
        case 1537...1792:
          taesd = qwenImageTinyDecoderFor1792
          imagePaddingSize = 1792
        case 1793...2048:
          taesd = qwenImageTinyDecoderFor2048
          imagePaddingSize = 2048
        default:
          return []
        }
      case .wan21_1_3b, .wan21_14b, .longcatVideoAvatar1_5:
        imageHeight = shape[1] * 8
        imageWidth = shape[2] * 8
        isVideo = true
        switch max(imageWidth, imageHeight) {
        case 0...512:
          taesd = wan21TinyDecoderFor512
          imagePaddingSize = 512
        case 0...768:
          taesd = wan21TinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = wan21TinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = wan21TinyDecoderFor1280
          imagePaddingSize = 1280
        default:
          return []
        }
      case .ernieImage, .flux2, .flux2_4b, .flux2_9b, .ideogram4:
        imageHeight = shape[1] * 8
        imageWidth = shape[2] * 8
        isVideo = false
        switch max(imageWidth, imageHeight) {
        case 0...768:
          taesd = flux2TinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = flux2TinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = flux2TinyDecoderFor1280
          imagePaddingSize = 1280
        case 1281...1536:
          taesd = flux2TinyDecoderFor1536
          imagePaddingSize = 1536
        case 1537...1792:
          taesd = flux2TinyDecoderFor1792
          imagePaddingSize = 1792
        case 1793...2048:
          taesd = flux2TinyDecoderFor2048
          imagePaddingSize = 2048
        default:
          return []
        }
      case .ltx2:
        let (_, audioHeight) = LTX2ExtractAudioFramesAndHeight(shape)
        tensor = tensor[0..<shape[0], 0..<(shape[1] - audioHeight), 0..<shape[2], 0..<shape[3]]
          .copied()
        shape = tensor.shape
        imageHeight = shape[1] * 32
        imageWidth = shape[2] * 32
        isVideo = true
        switch max(imageWidth, imageHeight) {
        case 0...768:
          taesd = LTX2TinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = LTX2TinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = LTX2TinyDecoderFor1280
          imagePaddingSize = 1280
        case 1281...1536:
          taesd = LTX2TinyDecoderFor1536
          imagePaddingSize = 1536
        case 1537...1792:
          taesd = LTX2TinyDecoderFor1792
          imagePaddingSize = 1792
        case 1793...2048:
          taesd = LTX2TinyDecoderFor2048
          imagePaddingSize = 2048
        default:
          return []
        }
      case .ltx2_3:
        let (_, audioHeight) = LTX2ExtractAudioFramesAndHeight(shape)
        tensor = tensor[0..<shape[0], 0..<(shape[1] - audioHeight), 0..<shape[2], 0..<shape[3]]
          .copied()
        shape = tensor.shape
        imageHeight = shape[1] * 32
        imageWidth = shape[2] * 32
        isVideo = true
        switch max(imageWidth, imageHeight) {
        case 0...768:
          taesd = LTX2_3TinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = LTX2_3TinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = LTX2_3TinyDecoderFor1280
          imagePaddingSize = 1280
        case 1281...1536:
          taesd = LTX2_3TinyDecoderFor1536
          imagePaddingSize = 1536
        case 1537...1792:
          taesd = LTX2_3TinyDecoderFor1792
          imagePaddingSize = 1792
        case 1793...2048:
          taesd = LTX2_3TinyDecoderFor2048
          imagePaddingSize = 2048
        default:
          return []
        }
      case .minimaxH3:
        let audioHeight = MiniMaxH3AudioHeight(
          videoLatentFrames: shape[0], latentWidth: shape[2])
        guard shape[3] == 24, shape[1] > audioHeight else { return [] }
        tensor = tensor[0..<shape[0], 0..<(shape[1] - audioHeight), 0..<shape[2], 0..<shape[3]]
          .copied()
        shape = tensor.shape
        imageHeight = shape[1] * 16
        imageWidth = shape[2] * 16
        isVideo = true
        switch max(imageWidth, imageHeight) {
        case 0...768:
          taesd = MiniMaxH3TinyDecoderFor768
          imagePaddingSize = 768
        case 769...1024:
          taesd = MiniMaxH3TinyDecoderFor1024
          imagePaddingSize = 1024
        case 1025...1280:
          taesd = MiniMaxH3TinyDecoderFor1280
          imagePaddingSize = 1280
        case 1281...1536:
          taesd = MiniMaxH3TinyDecoderFor1536
          imagePaddingSize = 1536
        case 1537...1792:
          taesd = MiniMaxH3TinyDecoderFor1792
          imagePaddingSize = 1792
        case 1793...2048:
          taesd = MiniMaxH3TinyDecoderFor2048
          imagePaddingSize = 2048
        default:
          return []
        }
      case .hunyuanVideo, .kandinsky21, .wan22_5b, .seedvr2_3b, .seedvr2_7b, .ideogram4,
        .wurstchenStageB, .wurstchenStageC:
        return []
      }
      guard #available(iOS 16.0, *), let taesd = taesd else {
        return []
      }
      do {
        if isVideo {
          try taesd.loadResources()
          return try taesd.perform { model in
            func zeroFill(_ array: MLMultiArray) {
              guard array.dataType == .float16 else { return }
              let ptr = array.dataPointer.bindMemory(to: FloatType.self, capacity: array.count)
              ptr.initialize(repeating: 0, count: array.count)
            }
            func imageFromBytes(_ bytes: UnsafeMutablePointer<UInt8>) -> CGImage {
              CGImage(
                width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(
                  rawValue: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: CGDataProvider(
                  dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
                  releaseData: { _, p, _ in
                    p.deallocate()
                  })!, decode: nil, shouldInterpolate: false,
                intent: CGColorRenderingIntent.defaultIntent)!
            }
            guard
              let latentConstraint = model.modelDescription.inputDescriptionsByName["latent"]?
                .multiArrayConstraint,
              let imageConstraint = model.modelDescription.outputDescriptionsByName["image"]?
                .multiArrayConstraint
            else {
              return []
            }
            let latentModelShape = latentConstraint.shape.map { $0.intValue }
            let imageModelShape = imageConstraint.shape.map { $0.intValue }
            guard latentConstraint.dataType == .float16, imageConstraint.dataType == .float16,
              latentModelShape.count == 5, imageModelShape.count == 5
            else {
              return []
            }
            let chunkT = latentModelShape[1]
            let paddedLatentHeight = latentModelShape[2]
            let paddedLatentWidth = latentModelShape[3]
            let latentChannels = latentModelShape[4]
            let outputFramesPerCall = imageModelShape[1]
            let outputHeight = imageModelShape[2]
            let outputWidth = imageModelShape[3]
            let outputChannels = imageModelShape[4]
            guard chunkT > 0, outputFramesPerCall > 0, outputFramesPerCall % chunkT == 0 else {
              return []
            }
            let tUpscale = outputFramesPerCall / chunkT
            let framesToTrim = max(0, tUpscale - 1)
            guard shape.count == 4, shape[1] <= paddedLatentHeight, shape[2] <= paddedLatentWidth,
              shape[3] == latentChannels, outputHeight >= imageHeight, outputWidth >= imageWidth,
              outputChannels >= 3
            else {
              return []
            }
            let actInputNames = (0...8).map { "act_\($0)" }
            let actOutputNames = (0...8).map { "act_\($0)_out" }
            var states = [MLMultiArray]()
            states.reserveCapacity(actInputNames.count)
            for name in actInputNames {
              guard
                let constraint = model.modelDescription.inputDescriptionsByName[name]?
                  .multiArrayConstraint,
                constraint.dataType == .float16
              else {
                return []
              }
              guard
                let state = try? MLMultiArray(
                  shape: constraint.shape, dataType: constraint.dataType)
              else {
                return []
              }
              zeroFill(state)
              states.append(state)
            }
            func copyLatentChunk(from sourceStartFrame: Int, realFrames: Int) -> MLMultiArray {
              var latentChunkTensor = Tensor<FloatType>(
                Array(
                  repeating: 0,
                  count: chunkT * paddedLatentHeight * paddedLatentWidth * shape[3]),
                .CPU, .NHWC(chunkT, paddedLatentHeight, paddedLatentWidth, shape[3]))
              latentChunkTensor[0..<realFrames, 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                tensor[
                  sourceStartFrame..<(sourceStartFrame + realFrames), 0..<shape[1], 0..<shape[2],
                  0..<shape[3]
                ]
              return MLMultiArray(
                MLShapedArray(
                  latentChunkTensor.reshaped(
                    format: .NHWC,
                    shape: [
                      1, chunkT, paddedLatentHeight, paddedLatentWidth, shape[3],
                    ])))
            }
            func frameImage(from outputArray: MLMultiArray, frameIndex: Int) -> CGImage? {
              let strides = outputArray.strides.map { $0.intValue }
              guard strides.count == 5 else { return nil }
              let tStride = strides[1]
              let hStride = strides[2]
              let wStride = strides[3]
              let cStride = strides[4]
              let bytes = UnsafeMutablePointer<UInt8>.allocate(
                capacity: imageWidth * imageHeight * 4)
              guard outputArray.dataType == .float16 else {
                bytes.deallocate()
                return nil
              }
              let ptr = outputArray.dataPointer.bindMemory(
                to: FloatType.self, capacity: outputArray.count)
              var o = bytes
              let frameBase = frameIndex * tStride
              for y in 0..<imageHeight {
                for x in 0..<imageWidth {
                  let pixelBase = frameBase + y * hStride + x * wStride
                  let r = 255.0 * Float(ptr[pixelBase])
                  let g = 255.0 * Float(ptr[pixelBase + cStride])
                  let b = 255.0 * Float(ptr[pixelBase + 2 * cStride])
                  o[0] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
                  o[1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
                  o[2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
                  o[3] = 255
                  o += 4
                }
              }
              return imageFromBytes(bytes)
            }
            do {
              var images = [CGImage]()
              images.reserveCapacity(max(0, shape[0] * tUpscale - framesToTrim))
              var producedRawFrames = 0
              for start in stride(from: 0, to: shape[0], by: chunkT) {
                let realFrames = min(chunkT, shape[0] - start)
                let chunkLatent = copyLatentChunk(from: start, realFrames: realFrames)
                var features: [String: MLFeatureValue] = [
                  "latent": MLFeatureValue(multiArray: chunkLatent)
                ]
                for (i, name) in actInputNames.enumerated() {
                  features[name] = MLFeatureValue(multiArray: states[i])
                }
                let out = try model.prediction(
                  from: MLDictionaryFeatureProvider(dictionary: features))
                guard let outputArray = out.featureValue(for: "image")?.multiArrayValue else {
                  return []
                }
                let rawFramesInChunk = realFrames * tUpscale
                for localFrame in 0..<rawFramesInChunk {
                  let globalRawFrame = producedRawFrames + localFrame
                  if globalRawFrame < framesToTrim {
                    continue
                  }
                  guard let image = frameImage(from: outputArray, frameIndex: localFrame) else {
                    return []
                  }
                  images.append(image)
                }
                let nextStates = actOutputNames.compactMap {
                  out.featureValue(for: $0)?.multiArrayValue
                }
                guard nextStates.count == actOutputNames.count,
                  nextStates.allSatisfy({ $0.dataType == .float16 })
                else {
                  return []
                }
                states = nextStates
                producedRawFrames += rawFramesInChunk
              }
              return images
            } catch {
              return []
            }
          }
        } else {
          // Qwen Image 2.1 uses 16x spatial compression; other image TAEs use 8x.
          let spatialScale = version == .qwenImage2_1 ? 16 : 8
          let latentPaddingSize = imagePaddingSize / spatialScale
          var images = [Tensor<FloatType>]()
          if imageWidth != imagePaddingSize || imageHeight != imagePaddingSize {
            for i in 0..<shape[0] {
              var image = Tensor<FloatType>(
                Array(
                  repeating: 0, count: latentPaddingSize * latentPaddingSize * shape[3]),
                .CPU, .NHWC(1, latentPaddingSize, latentPaddingSize, shape[3]))
              image[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                tensor[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
              images.append(image)
            }
          } else if shape[0] > 1 {
            for i in 0..<shape[0] {
              images.append(
                tensor[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].contiguous())
            }
          } else {
            images.append(tensor)
          }
          try taesd.loadResources()
          let array = images.map { MLMultiArray(MLShapedArray($0)) }
          return try taesd.perform {
            let provider = MLArrayBatchProvider(
              array: try array.map {
                try MLDictionaryFeatureProvider(dictionary: [
                  "latent": MLFeatureValue(multiArray: $0)
                ]
                )
              })
            let name = $0.modelDescription.outputDescriptionsByName.keys.first!
            let predictions = try $0.predictions(fromBatch: provider)
            return (0..<predictions.count).compactMap {
              let outFeatures = predictions.features(at: $0)
              let outputArray = outFeatures.featureValue(for: name)!.multiArrayValue!
              let rowStride = outputArray.strides[1].intValue
              let stride = outputArray.strides[2].intValue
              return outputArray.withUnsafeBytes {
                guard var fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
                  return nil
                }
                let bytes = UnsafeMutablePointer<UInt8>.allocate(
                  capacity: imageWidth * imageHeight * 4)
                var o = bytes
                for _ in 0..<imageHeight {
                  var i = fp16
                  for _ in 0..<imageWidth {
                    // We need to do some computations from the latent values.
                    let (v0, v1, v2) = (i[0], i[1], i[2])
                    let r = 255.0 * v0
                    let g = 255.0 * v1
                    let b = 255.0 * v2
                    o[0] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
                    o[1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
                    o[2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
                    o[3] = 255
                    i += stride
                    o += 4
                  }
                  fp16 += rowStride
                }
                return CGImage(
                  width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
                  bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGBitmapInfo(
                    rawValue: CGBitmapInfo.byteOrder32Big.rawValue
                      | CGImageAlphaInfo.noneSkipLast.rawValue),
                  provider: CGDataProvider(
                    dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
                    releaseData: { _, p, _ in
                      p.deallocate()
                    })!, decode: nil, shouldInterpolate: false,
                  intent: CGColorRenderingIntent.defaultIntent
                )
              }
            }
          }
        }
      } catch {
        return []
      }
    }
  #endif
  public static func cgImages(
    fromLatent tensor: Tensor<FloatType>, canUseTAESD: Bool, version: ModelVersion
  )
    -> ([CGImage], Bool)
  {
    let shape = tensor.shape
    let batchSize = shape[0]
    let imageHeight = shape[1]
    let imageWidth = shape[2]
    let channels = shape[3]
    if version == .hiDreamO1 {
      let patchSize = 32
      guard channels == 3 * patchSize * patchSize else { return ([], false) }
      let outputWidth = imageWidth * patchSize
      let outputHeight = imageHeight * patchSize
      return tensor.withUnsafeBytes {
        guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
          return ([], false)
        }
        var images = [CGImage]()
        for b in 0..<batchSize {
          let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: outputWidth * outputHeight * 4)
          let fp16 = fp16 + b * imageHeight * imageWidth * channels
          for y in 0..<outputHeight {
            let patchY = y / patchSize
            let patchYOffset = y % patchSize
            for x in 0..<outputWidth {
              let patchX = x / patchSize
              let patchXOffset = x % patchSize
              let patchPixelOffset = patchYOffset * patchSize + patchXOffset
              let patchOffset =
                ((patchY * imageWidth + patchX) * channels
                  + patchPixelOffset)
              let pixelOffset = (y * outputWidth + x) * 4
              let r = (fp16[patchOffset] + 1) * 127.5
              let g = (fp16[patchOffset + patchSize * patchSize] + 1) * 127.5
              let b = (fp16[patchOffset + patchSize * patchSize * 2] + 1) * 127.5
              bytes[pixelOffset] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
              bytes[pixelOffset + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
              bytes[pixelOffset + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
              bytes[pixelOffset + 3] = 255
            }
          }
          images.append(
            CGImage(
              width: outputWidth, height: outputHeight, bitsPerComponent: 8, bitsPerPixel: 32,
              bytesPerRow: 4 * outputWidth, space: CGColorSpaceCreateDeviceRGB(),
              bitmapInfo: CGBitmapInfo(
                rawValue: CGBitmapInfo.byteOrder32Big.rawValue
                  | CGImageAlphaInfo.noneSkipLast.rawValue),
              provider: CGDataProvider(
                dataInfo: nil, data: bytes, size: outputWidth * outputHeight * 4,
                releaseData: { _, p, _ in
                  p.deallocate()
                })!, decode: nil, shouldInterpolate: false,
              intent: CGColorRenderingIntent.defaultIntent
            )!)
        }
        return (images, false)
      }
    }
    guard
      channels == 4 || channels == 3 || channels == 16 || channels == 24 || channels == 32
        || channels == 48 || channels == 64
        || channels == 128
    else { return ([], false) }
    #if canImport(CoreML)
      if canUseTAESD
        && (version == .flux1 || version == .hiDreamI1 || version == .zImage
          || version == .ernieImage
          || version == .flux2 || version == .flux2_4b || version == .flux2_9b
          || version == .ideogram4
          || version == .qwenImage || version == .qwenImage2_1
          || version == .krea2
          || version == .cosmos2_5_2b
          || version == .wan21_1_3b || version == .wan21_14b || version == .ltx2
          || version == .ltx2_3 || version == .minimaxH3
          || version == .v1 || version == .v2 || version == .svdI2v || version == .sd3
          || version == .sd3Large || version == .sdxlBase || version == .sdxlRefiner
          || version == .ssd1b || version == .pixart || version == .auraflow)
      {
        let images = imagesWithTAESD(fromLatent: tensor, version: version)
        guard images.isEmpty else {
          return (images, version != .ltx2 && version != .ltx2_3)  // It is still fairly low quality for LTX-2.
        }
      }
    #endif
    return tensor.withUnsafeBytes {
      guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
        return ([], false)
      }
      // MiniMax H3 packs audio rows below the 24-channel video latent.
      let imageHeight =
        channels == 24
        ? shape[1] - MiniMaxH3AudioHeight(videoLatentFrames: batchSize, latentWidth: imageWidth)
        : shape[1]
      guard imageHeight > 0 else { return ([], false) }
      var images = [CGImage]()
      for b in 0..<batchSize {
        let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
        let fp16 = fp16 + b * shape[1] * imageWidth * channels
        switch version {
        case _ where channels == 24:
          for i in 0..<imageHeight * imageWidth {
            let offset = i * channels
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11) = (
              fp16[offset], fp16[offset + 1], fp16[offset + 2],
              fp16[offset + 3], fp16[offset + 4], fp16[offset + 5],
              fp16[offset + 6], fp16[offset + 7], fp16[offset + 8],
              fp16[offset + 9], fp16[offset + 10], fp16[offset + 11]
            )
            let (v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23) = (
              fp16[offset + 12], fp16[offset + 13], fp16[offset + 14],
              fp16[offset + 15], fp16[offset + 16], fp16[offset + 17],
              fp16[offset + 18], fp16[offset + 19], fp16[offset + 20],
              fp16[offset + 21], fp16[offset + 22], fp16[offset + 23]
            )
            let r: FloatType =
              (-0.018555 * v0 + 0.150164 * v1 + 0.027367 * v2 - 0.000793 * v3 - 0.048556 * v4
                + 0.011740 * v5 + 0.061517 * v6 + 0.035321 * v7 - 0.017426 * v8 + 0.531539
                * v9 - 0.024968 * v10 - 0.032549 * v11 + 0.022609 * v12 - 0.084001 * v13
                - 0.018830 * v14 + 0.020777 * v15 - 0.008390 * v16 - 0.013281 * v17 + 0.000260
                * v18 + 0.105471 * v19 + 0.016529 * v20 - 0.014015 * v21 - 0.033787 * v22
                + 0.004224 * v23 + 0.057426) * 127.5 + 127.5
            let g: FloatType =
              (0.024344 * v0 + 0.137244 * v1 - 0.050369 * v2 - 0.164622 * v3 + 0.013970 * v4
                + 0.014172 * v5 + 0.061212 * v6 + 0.086879 * v7 + 0.002997 * v8 + 0.548819
                * v9 - 0.040234 * v10 - 0.029096 * v11 + 0.020286 * v12 - 0.038131 * v13
                + 0.010412 * v14 + 0.011196 * v15 - 0.012201 * v16 - 0.002924 * v17 + 0.001833
                * v18 + 0.100482 * v19 + 0.015213 * v20 - 0.017438 * v21 - 0.009984 * v22
                + 0.017284 * v23 - 0.022078) * 127.5 + 127.5
            let b: FloatType =
              (-0.017536 * v0 + 0.129221 * v1 - 0.208606 * v2 - 0.323161 * v3 - 0.074286 * v4
                - 0.006906 * v5 + 0.110025 * v6 + 0.110059 * v7 + 0.035356 * v8 + 0.624404
                * v9 - 0.034302 * v10 - 0.017221 * v11 + 0.050661 * v12 - 0.020805 * v13
                + 0.061120 * v14 - 0.030994 * v15 - 0.025687 * v16 + 0.006331 * v17 - 0.011038
                * v18 + 0.132106 * v19 + 0.009999 * v20 - 0.019134 * v21 - 0.019725 * v22
                + 0.027196 * v23 - 0.071449) * 127.5 + 127.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .v1, .v2, .svdI2v:
          for i in 0..<imageHeight * imageWidth {
            // We need to do some computations from the latent values.
            let (v0, v1, v2, v3) = (
              fp16[i * 4], fp16[i * 4 + 1], fp16[i * 4 + 2], fp16[i * 4 + 3]
            )
            let r = 49.5210 * v0 + 29.0283 * v1 - 23.9673 * v2 - 39.4981 * v3 + 99.9368
            let g = 41.1373 * v0 + 42.4951 * v1 + 24.7349 * v2 - 50.8279 * v3 + 99.8421
            let b = 40.2919 * v0 + 18.9304 * v1 + 30.0236 * v2 - 81.9976 * v3 + 99.5384
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .sd3, .sd3Large:
          for i in 0..<imageHeight * imageWidth {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) = (
              fp16[i * 16], fp16[i * 16 + 1], fp16[i * 16 + 2], fp16[i * 16 + 3],
              fp16[i * 16 + 4],
              fp16[i * 16 + 5], fp16[i * 16 + 6], fp16[i * 16 + 7], fp16[i * 16 + 8],
              fp16[i * 16 + 9], fp16[i * 16 + 10], fp16[i * 16 + 11], fp16[i * 16 + 12],
              fp16[i * 16 + 13], fp16[i * 16 + 14], fp16[i * 16 + 15]
            )
            // TBD, I may want to regress these coefficients myself.
            let r: FloatType =
              (-0.0922 * v0 + 0.0311 * v1 + 0.1994 * v2 + 0.0856 * v3 + 0.0587 * v4 - 0.0006 * v5
                + 0.0978 * v6 - 0.0042 * v7 - 0.0194 * v8 - 0.0488 * v9 + 0.0922 * v10 - 0.0278
                * v11 + 0.0332 * v12 - 0.0069 * v13 - 0.0596 * v14 - 0.1448 * v15 + 0.2394)
              * 127.5
              + 127.5
            let g: FloatType =
              (-0.0175 * v0 + 0.0633 * v1 + 0.0927 * v2 + 0.0339 * v3 + 0.0272 * v4 + 0.1104 * v5
                + 0.0306 * v6 + 0.1038 * v7 + 0.0020 * v8 + 0.0130 * v9 + 0.0988 * v10 + 0.0524
                * v11 + 0.0456 * v12 - 0.0030 * v13 - 0.0465 * v14 - 0.1463 * v15 + 0.2135)
              * 127.5
              + 127.5
            let b: FloatType =
              (0.0749 * v0 + 0.0954 * v1 + 0.0458 * v2 + 0.0902 * v3 - 0.0496 * v4 + 0.0309 * v5
                + 0.0427 * v6 + 0.1358 * v7 + 0.0669 * v8 - 0.0268 * v9 + 0.0951 * v10 - 0.0542
                * v11 + 0.0895 * v12 - 0.0810 * v13 - 0.0293 * v14 - 0.1189 * v15 + 0.1925)
              * 127.5
              + 127.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .flux1, .hiDreamI1, .zImage, .seedvr2_3b, .seedvr2_7b:
          for i in 0..<imageHeight * imageWidth {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) = (
              fp16[i * 16], fp16[i * 16 + 1], fp16[i * 16 + 2], fp16[i * 16 + 3],
              fp16[i * 16 + 4],
              fp16[i * 16 + 5], fp16[i * 16 + 6], fp16[i * 16 + 7], fp16[i * 16 + 8],
              fp16[i * 16 + 9], fp16[i * 16 + 10], fp16[i * 16 + 11], fp16[i * 16 + 12],
              fp16[i * 16 + 13], fp16[i * 16 + 14], fp16[i * 16 + 15]
            )
            // TBD, I may want to regress these coefficients myself.
            let r: FloatType =
              (-0.0346 * v0 + 0.0034 * v1 + 0.0275 * v2 - 0.0174 * v3 + 0.0859 * v4 + 0.0004 * v5
                + 0.0405 * v6 - 0.0236 * v7 - 0.0245 * v8 + 0.1008 * v9 - 0.0515 * v10 + 0.0428
                * v11 + 0.0817 * v12 - 0.1264 * v13 - 0.0280 * v14 - 0.1262 * v15 - 0.0329)
              * 127.5
              + 127.5
            let g: FloatType =
              (0.0244 * v0 + 0.0210 * v1 - 0.0668 * v2 + 0.0160 * v3 + 0.0721 * v4 + 0.0383 * v5
                + 0.0861 * v6 - 0.0185 * v7 + 0.0250 * v8 + 0.0755 * v9 + 0.0201 * v10 - 0.0012
                * v11 + 0.0765 * v12 - 0.0522 * v13 - 0.0881 * v14 - 0.0982 * v15 - 0.0718)
              * 127.5
              + 127.5
            let b: FloatType =
              (0.0681 * v0 + 0.0687 * v1 - 0.0433 * v2 + 0.0617 * v3 + 0.0329 * v4 + 0.0115 * v5
                + 0.0915 * v6 - 0.0259 * v7 + 0.1180 * v8 - 0.0421 * v9 + 0.0011 * v10 - 0.0036
                * v11 + 0.0749 * v12 - 0.1103 * v13 - 0.0499 * v14 - 0.0778 * v15 - 0.0851)
              * 127.5
              + 127.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ideogram4:
          for i in 0..<imageHeight * imageWidth {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) = (
              fp16[i * 32], fp16[i * 32 + 1], fp16[i * 32 + 2], fp16[i * 32 + 3],
              fp16[i * 32 + 4], fp16[i * 32 + 5], fp16[i * 32 + 6], fp16[i * 32 + 7],
              fp16[i * 32 + 8], fp16[i * 32 + 9], fp16[i * 32 + 10], fp16[i * 32 + 11],
              fp16[i * 32 + 12], fp16[i * 32 + 13], fp16[i * 32 + 14], fp16[i * 32 + 15]
            )
            let (v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31) =
              (
                fp16[i * 32 + 16], fp16[i * 32 + 17], fp16[i * 32 + 18], fp16[i * 32 + 19],
                fp16[i * 32 + 20], fp16[i * 32 + 21], fp16[i * 32 + 22], fp16[i * 32 + 23],
                fp16[i * 32 + 24], fp16[i * 32 + 25], fp16[i * 32 + 26], fp16[i * 32 + 27],
                fp16[i * 32 + 28], fp16[i * 32 + 29], fp16[i * 32 + 30], fp16[i * 32 + 31]
              )
            let r: FloatType =
              (0.0058 * v0 + 0.0495 * v1 - 0.0099 * v2 + 0.2144 * v3 + 0.0166 * v4 + 0.0157 * v5
                - 0.0398 * v6 - 0.0052 * v7 - 0.3527 * v8 - 0.0301 * v9 - 0.0107 * v10 + 0.0746
                * v11 + 0.0156 * v12 - 0.0034 * v13 + 0.0032 * v14 - 0.0939 * v15 + 0.0018 * v16
                + 0.0284 * v17 - 0.0024 * v18 + 0.1207 * v19 + 0.0128 * v20 + 0.0137 * v21
                + 0.0095 * v22 + 0.0000 * v23 - 0.0465 * v24 + 0.0095 * v25 + 0.0290 * v26
                + 0.0220 * v27 - 0.0332 * v28 - 0.0085 * v29 - 0.0076 * v30 - 0.0111 * v31
                - 0.0329) * 127.5 + 127.5
            let g: FloatType =
              (0.0113 * v0 + 0.0443 * v1 + 0.0096 * v2 + 0.3009 * v3 - 0.0039 * v4 + 0.0103 * v5
                + 0.0902 * v6 + 0.0095 * v7 - 0.2712 * v8 - 0.0356 * v9 + 0.0078 * v10 + 0.0090
                * v11 + 0.0169 * v12 - 0.0040 * v13 + 0.0181 * v14 - 0.0008 * v15 + 0.0043 * v16
                + 0.0056 * v17 - 0.0022 * v18 - 0.0026 * v19 + 0.0101 * v20 - 0.0072 * v21
                + 0.0092 * v22 - 0.0077 * v23 - 0.0204 * v24 + 0.0012 * v25 - 0.0034 * v26
                + 0.0169 * v27 - 0.0457 * v28 + 0.0389 * v29 + 0.0003 * v30 - 0.0460 * v31
                - 0.0718)
              * 127.5
              + 127.5
            let b: FloatType =
              (0.0073 * v0 + 0.0836 * v1 + 0.0644 * v2 + 0.3652 * v3 - 0.0054 * v4 - 0.0160 * v5
                - 0.0235 * v6 + 0.0109 * v7 - 0.1666 * v8 - 0.0180 * v9 + 0.0013 * v10 - 0.0941
                * v11 + 0.0070 * v12 - 0.0114 * v13 + 0.0080 * v14 + 0.0186 * v15 + 0.0104 * v16
                - 0.0127 * v17 - 0.0030 * v18 + 0.0065 * v19 + 0.0142 * v20 - 0.0007 * v21
                - 0.0059 * v22 - 0.0049 * v23 - 0.0312 * v24 - 0.0066 * v25 + 0.0025 * v26
                - 0.0048 * v27 - 0.0468 * v28 + 0.0609 * v29 - 0.0043 * v30 - 0.0614 * v31
                - 0.0851)
              * 127.5
              + 127.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .qwenImage2_1:
          for i in 0..<imageHeight * imageWidth {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) = (
              fp16[i * 64], fp16[i * 64 + 1], fp16[i * 64 + 2], fp16[i * 64 + 3],
              fp16[i * 64 + 4], fp16[i * 64 + 5], fp16[i * 64 + 6], fp16[i * 64 + 7],
              fp16[i * 64 + 8], fp16[i * 64 + 9], fp16[i * 64 + 10], fp16[i * 64 + 11],
              fp16[i * 64 + 12], fp16[i * 64 + 13], fp16[i * 64 + 14], fp16[i * 64 + 15]
            )
            let (v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31) =
              (
                fp16[i * 64 + 16], fp16[i * 64 + 17], fp16[i * 64 + 18], fp16[i * 64 + 19],
                fp16[i * 64 + 20], fp16[i * 64 + 21], fp16[i * 64 + 22], fp16[i * 64 + 23],
                fp16[i * 64 + 24], fp16[i * 64 + 25], fp16[i * 64 + 26], fp16[i * 64 + 27],
                fp16[i * 64 + 28], fp16[i * 64 + 29], fp16[i * 64 + 30], fp16[i * 64 + 31]
              )
            let (v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47) =
              (
                fp16[i * 64 + 32], fp16[i * 64 + 33], fp16[i * 64 + 34], fp16[i * 64 + 35],
                fp16[i * 64 + 36], fp16[i * 64 + 37], fp16[i * 64 + 38], fp16[i * 64 + 39],
                fp16[i * 64 + 40], fp16[i * 64 + 41], fp16[i * 64 + 42], fp16[i * 64 + 43],
                fp16[i * 64 + 44], fp16[i * 64 + 45], fp16[i * 64 + 46], fp16[i * 64 + 47]
              )
            let (v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63) =
              (
                fp16[i * 64 + 48], fp16[i * 64 + 49], fp16[i * 64 + 50], fp16[i * 64 + 51],
                fp16[i * 64 + 52], fp16[i * 64 + 53], fp16[i * 64 + 54], fp16[i * 64 + 55],
                fp16[i * 64 + 56], fp16[i * 64 + 57], fp16[i * 64 + 58], fp16[i * 64 + 59],
                fp16[i * 64 + 60], fp16[i * 64 + 61], fp16[i * 64 + 62], fp16[i * 64 + 63]
              )
            // ComfyUI comfy/latent_formats.py: QwenImage21, applied to normalized latents.
            let r: FloatType =
              (-0.0158 * v0 + 0.0030 * v1 + 0.0637 * v2 + 0.0360 * v3 + 0.0159 * v4
                + 0.0132 * v5 + 0.0191 * v6 - 0.0146 * v7 + 0.0187 * v8 - 0.1059 * v9
                - 0.0195 * v10 - 0.0295 * v11 + 0.0191 * v12 - 0.0144 * v13 + 0.0389 * v14
                - 0.0153 * v15 + 0.0339 * v16 - 0.0136 * v17 - 0.0340 * v18 - 0.0133 * v19
                + 0.0109 * v20 - 0.0010 * v21 + 0.0301 * v22 - 0.0281 * v23 - 0.0725 * v24
                - 0.0036 * v25 + 0.0087 * v26 - 0.0260 * v27 - 0.0039 * v28 - 0.0026 * v29
                + 0.0088 * v30 - 0.0087 * v31 - 0.0027 * v32 - 0.0064 * v33 + 0.0594 * v34
                + 0.0103 * v35 - 0.0091 * v36 + 0.0178 * v37 - 0.0063 * v38 + 0.0452 * v39
                + 0.0149 * v40 + 0.1484 * v41 - 0.0120 * v42 + 0.0181 * v43 + 0.0132 * v44
                + 0.0291 * v45 + 0.0066 * v46 - 0.1153 * v47 + 0.0258 * v48 + 0.0375 * v49
                - 0.0142 * v50 + 0.0339 * v51 + 0.0346 * v52 + 0.0369 * v53 - 0.0052 * v54
                - 0.0279 * v55 + 0.0036 * v56 - 0.0441 * v57 - 0.0001 * v58 - 0.0222 * v59
                + 0.0039 * v60 - 0.0094 * v61 - 0.0066 * v62 + 0.0220 * v63 - 0.1228)
              * 127.5
              + 127.5
            let g: FloatType =
              (-0.0115 * v0 + 0.0120 * v1 + 0.0470 * v2 + 0.0661 * v3 + 0.0181 * v4
                + 0.0326 * v5 + 0.0261 * v6 - 0.0276 * v7 - 0.0024 * v8 - 0.0090 * v9
                - 0.0226 * v10 + 0.0024 * v11 - 0.0393 * v12 - 0.0166 * v13 + 0.0430 * v14
                - 0.0336 * v15 + 0.0122 * v16 - 0.0078 * v17 - 0.0282 * v18 - 0.0176 * v19
                - 0.0087 * v20 + 0.0044 * v21 + 0.0053 * v22 - 0.0205 * v23 + 0.0002 * v24
                + 0.0158 * v25 + 0.0040 * v26 + 0.0183 * v27 - 0.0035 * v28 + 0.0172 * v29
                + 0.0078 * v30 - 0.0310 * v31 + 0.0018 * v32 + 0.0292 * v33 + 0.1049 * v34
                - 0.0103 * v35 + 0.0025 * v36 + 0.0243 * v37 - 0.0012 * v38 + 0.0246 * v39
                + 0.0270 * v40 + 0.0801 * v41 + 0.0040 * v42 + 0.0051 * v43 + 0.0050 * v44
                + 0.0020 * v45 - 0.0410 * v46 - 0.0629 * v47 + 0.0378 * v48 + 0.1139 * v49
                - 0.0126 * v50 + 0.0153 * v51 + 0.0211 * v52 - 0.0431 * v53 - 0.0092 * v54
                + 0.0410 * v55 + 0.0017 * v56 - 0.0367 * v57 - 0.0092 * v58 - 0.0183 * v59
                + 0.0053 * v60 - 0.0075 * v61 - 0.0088 * v62 + 0.0074 * v63 - 0.1869)
              * 127.5
              + 127.5
            let b: FloatType =
              (-0.0174 * v0 + 0.0027 * v1 - 0.0127 * v2 - 0.0030 * v3 + 0.0082 * v4
                + 0.0169 * v5 + 0.0136 * v6 - 0.0361 * v7 - 0.0072 * v8 + 0.0350 * v9
                - 0.0138 * v10 - 0.0215 * v11 - 0.0001 * v12 - 0.0272 * v13 + 0.0445 * v14
                + 0.0031 * v15 + 0.0220 * v16 - 0.0120 * v17 - 0.0245 * v18 - 0.0133 * v19
                + 0.0096 * v20 + 0.0016 * v21 + 0.0361 * v22 - 0.0032 * v23 + 0.0160 * v24
                + 0.0807 * v25 - 0.0053 * v26 - 0.0077 * v27 - 0.0107 * v28 + 0.0237 * v29
                + 0.0078 * v30 - 0.0122 * v31 + 0.0094 * v32 - 0.0256 * v33 + 0.1180 * v34
                - 0.0026 * v35 - 0.0015 * v36 + 0.0292 * v37 + 0.0202 * v38 + 0.0143 * v39
                + 0.0052 * v40 + 0.0804 * v41 + 0.0010 * v42 - 0.0021 * v43 + 0.0019 * v44
                + 0.0092 * v45 - 0.1314 * v46 - 0.0802 * v47 + 0.0298 * v48 + 0.0468 * v49
                - 0.0276 * v50 + 0.0138 * v51 + 0.0267 * v52 - 0.0993 * v53 + 0.0056 * v54
                - 0.0357 * v55 - 0.0083 * v56 - 0.0454 * v57 - 0.0001 * v58 - 0.0051 * v59
                - 0.0184 * v60 - 0.0143 * v61 - 0.0063 * v62 + 0.0100 * v63 - 0.3083)
              * 127.5
              + 127.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .hiDreamO1, .ltx2, .ltx2_3, .minimaxH3:
          bytes.deallocate()
          continue
        case .hunyuanVideo:
          // Need to update the coefficients.
          for i in 0..<imageHeight * imageWidth {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) = (
              fp16[i * 16], fp16[i * 16 + 1], fp16[i * 16 + 2], fp16[i * 16 + 3],
              fp16[i * 16 + 4],
              fp16[i * 16 + 5], fp16[i * 16 + 6], fp16[i * 16 + 7], fp16[i * 16 + 8],
              fp16[i * 16 + 9], fp16[i * 16 + 10], fp16[i * 16 + 11], fp16[i * 16 + 12],
              fp16[i * 16 + 13], fp16[i * 16 + 14], fp16[i * 16 + 15]
            )
            // TBD, I may want to regress these coefficients myself.
            let r: FloatType =
              (-0.0395 * v0 + 0.0696 * v1 + 0.0135 * v2 + 0.0108 * v3 - 0.0209 * v4 - 0.0804 * v5
                - 0.0991 * v6 - 0.0646 * v7 - 0.0696 * v8 - 0.0799 * v9 + 0.1166 * v10 + 0.1165
                * v11 - 0.2315 * v12 - 0.0270 * v13 - 0.0616 * v14 + 0.0249 * v15 + 0.0249)
              * 127.5
              + 127.5
            let g: FloatType =
              (-0.0331 * v0 + 0.0795 * v1 - 0.0945 * v2 - 0.0250 * v3 + 0.0032 * v4 - 0.0254 * v5
                + 0.0271 * v6 - 0.0422 * v7 - 0.0595 * v8 - 0.0208 * v9 + 0.1627 * v10 + 0.0432
                * v11 - 0.1920 * v12 + 0.0401 * v13 - 0.0997 * v14 - 0.0469 * v15 - 0.0192)
              * 127.5
              + 127.5
            let b: FloatType =
              (0.0445 * v0 + 0.0518 * v1 - 0.0282 * v2 - 0.0765 * v3 + 0.0224 * v4 - 0.0639 * v5
                - 0.0669 * v6 - 0.0400 * v7 - 0.0894 * v8 - 0.0375 * v9 + 0.0962 * v10 + 0.0407
                * v11 - 0.1355 * v12 - 0.0821 * v13 - 0.0727 * v14 - 0.1703 * v15 - 0.0761)
              * 127.5
              + 127.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .wan21_1_3b, .wan21_14b, .qwenImage, .cosmos2_5_2b, .krea2,
          .longcatVideoAvatar1_5:
          for i in 0..<imageHeight * imageWidth {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) = (
              fp16[i * 16], fp16[i * 16 + 1], fp16[i * 16 + 2], fp16[i * 16 + 3],
              fp16[i * 16 + 4],
              fp16[i * 16 + 5], fp16[i * 16 + 6], fp16[i * 16 + 7], fp16[i * 16 + 8],
              fp16[i * 16 + 9], fp16[i * 16 + 10], fp16[i * 16 + 11], fp16[i * 16 + 12],
              fp16[i * 16 + 13], fp16[i * 16 + 14], fp16[i * 16 + 15]
            )
            // TBD, I may want to regress these coefficients myself.
            let r: FloatType =
              (-0.1299 * v0 + 0.0671 * v1 + 0.3568 * v2 + 0.0372 * v3 + 0.0313 * v4 + 0.0296 * v5
                - 0.3477 * v6 + 0.0166 * v7 - 0.0412 * v8 - 0.1293 * v9 + 0.0680 * v10 + 0.0032
                * v11 - 0.1251 * v12 + 0.0060 * v13 + 0.3477 * v14 + 0.1984 * v15 - 0.1835)
              * 127.5
              + 127.5
            let g: FloatType =
              (-0.1692 * v0 + 0.0406 * v1 + 0.2548 * v2 + 0.2344 * v3 + 0.0189 * v4 - 0.0956 * v5
                - 0.4059 * v6 + 0.1902 * v7 + 0.0267 * v8 + 0.0740 * v9 + 0.3019 * v10 + 0.0581
                * v11 + 0.0927 * v12 - 0.0633 * v13 + 0.2275 * v14 + 0.0913 * v15 - 0.0868)
              * 127.5
              + 127.5
            let b: FloatType =
              (0.2932 * v0 + 0.0442 * v1 + 0.1747 * v2 + 0.1420 * v3 - 0.0328 * v4 - 0.0665 * v5
                - 0.2925 * v6 + 0.1975 * v7 - 0.1364 * v8 + 0.1636 * v9 + 0.1128 * v10 + 0.0639
                * v11 + 0.1699 * v12 + 0.0005 * v13 + 0.2950 * v14 + 0.1861 * v15 - 0.336) * 127.5
              + 127.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .wan22_5b:
          // Need to update the coefficients.
          for i in 0..<imageHeight * imageWidth {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) = (
              fp16[i * 48], fp16[i * 48 + 1], fp16[i * 48 + 2], fp16[i * 48 + 3],
              fp16[i * 48 + 4],
              fp16[i * 48 + 5], fp16[i * 48 + 6], fp16[i * 48 + 7], fp16[i * 48 + 8],
              fp16[i * 48 + 9], fp16[i * 48 + 10], fp16[i * 48 + 11], fp16[i * 48 + 12],
              fp16[i * 48 + 13], fp16[i * 48 + 14], fp16[i * 48 + 15]
            )
            let (v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31) =
              (
                fp16[i * 48 + 16], fp16[i * 48 + 17], fp16[i * 48 + 18], fp16[i * 48 + 19],
                fp16[i * 48 + 20],
                fp16[i * 48 + 21], fp16[i * 48 + 22], fp16[i * 48 + 23], fp16[i * 48 + 24],
                fp16[i * 48 + 25], fp16[i * 48 + 26], fp16[i * 48 + 27], fp16[i * 48 + 28],
                fp16[i * 48 + 29], fp16[i * 48 + 30], fp16[i * 48 + 31]
              )
            let (v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47) =
              (
                fp16[i * 48 + 32], fp16[i * 48 + 33], fp16[i * 48 + 34], fp16[i * 48 + 35],
                fp16[i * 48 + 36],
                fp16[i * 48 + 37], fp16[i * 48 + 38], fp16[i * 48 + 39], fp16[i * 48 + 40],
                fp16[i * 48 + 41], fp16[i * 48 + 42], fp16[i * 48 + 43], fp16[i * 48 + 44],
                fp16[i * 48 + 45], fp16[i * 48 + 46], fp16[i * 48 + 47]
              )
            // TBD, I may want to regress these coefficients myself.
            let r: FloatType =
              (0.0119 * v0 - 0.1062 * v1 + 0.0140 * v2 - 0.0813 * v3 + 0.0656 * v4 + 0.0264 * v5
                + 0.0295 * v6 - 0.0244 * v7 + 0.0443 * v8 - 0.0465 * v9 + 0.0359 * v10 - 0.0776
                * v11 + 0.0564 * v12 + 0.0006 * v13 - 0.0319 * v14 - 0.0268 * v15 + 0.0539 * v16
                - 0.0359 * v17 - 0.0285 * v18 + 0.1041 * v19 - 0.0086 * v20 + 0.0390 * v21
                + 0.0069 * v22 + 0.0006 * v23 + 0.0313 * v24 - 0.1454 * v25 + 0.0714 * v26
                - 0.0304 * v27 + 0.0401 * v28 - 0.0758 * v29 + 0.0568 * v30 - 0.0055 * v31
                + 0.0239 * v32 - 0.0663 * v33 - 0.0416 * v34 + 0.0166 * v35 - 0.0211 * v36
                + 0.1833 * v37 - 0.0368 * v38 - 0.3441 * v39 - 0.0479 * v40 - 0.0660 * v41
                - 0.0101 * v42 - 0.0690 * v43 - 0.0145 * v44 + 0.0421 * v45 + 0.0504 * v46
                - 0.0837 * v47) * 127.5 + 127.5
            let g: FloatType =
              (0.0103 * v0 - 0.0504 * v1 + 0.0409 * v2 - 0.0677 * v3 + 0.0851 * v4 + 0.0463 * v5
                + 0.0326 * v6 - 0.0270 * v7 - 0.0102 * v8 - 0.0090 * v9 + 0.0236 * v10 + 0.0854
                * v11 + 0.0264 * v12 + 0.0594 * v13 - 0.0542 * v14 + 0.0024 * v15 + 0.0265 * v16
                - 0.0312 * v17 - 0.1032 * v18 + 0.0537 * v19 - 0.0374 * v20 + 0.0670 * v21
                + 0.0144 * v22 - 0.0167 * v23 - 0.0574 * v24 - 0.0902 * v25 + 0.0827 * v26
                - 0.0574 * v27 + 0.0384 * v28 - 0.0297 * v29 + 0.1307 * v30 - 0.0310 * v31
                - 0.0305 * v32 - 0.0673 * v33 - 0.0047 * v34 + 0.0112 * v35 + 0.0011 * v36
                + 0.1466 * v37 + 0.0370 * v38 - 0.3543 * v39 - 0.0489 * v40 - 0.0153 * v41
                + 0.0068 * v42 - 0.0452 * v43 + 0.0041 * v44 + 0.0451 * v45 - 0.0483 * v46
                + 0.0168 * v47) * 127.5 + 127.5
            let b: FloatType =
              (0.0046 * v0 + 0.0165 * v1 + 0.0491 * v2 + 0.0607 * v3 + 0.0808 * v4 + 0.0912 * v5
                + 0.0590 * v6 + 0.0025 * v7 + 0.0288 * v8 - 0.0205 * v9 + 0.0082 * v10 + 0.1048
                * v11 + 0.0561 * v12 + 0.0418 * v13 - 0.0637 * v14 + 0.0260 * v15 + 0.0358 * v16
                - 0.0287 * v17 - 0.1237 * v18 + 0.0622 * v19 - 0.0051 * v20 + 0.2863 * v21
                + 0.0082 * v22 + 0.0079 * v23 - 0.0232 * v24 - 0.0481 * v25 + 0.0447 * v26
                - 0.0196 * v27 + 0.0204 * v28 - 0.0014 * v29 + 0.1372 * v30 - 0.0380 * v31
                + 0.0325 * v32 - 0.0140 * v33 - 0.0023 * v34 - 0.0093 * v35 + 0.0331 * v36
                + 0.2250 * v37 + 0.0295 * v38 - 0.2008 * v39 - 0.0420 * v40 + 0.0800 * v41
                + 0.0156 * v42 - 0.0927 * v43 + 0.0015 * v44 + 0.0373 * v45 - 0.0356 * v46
                + 0.0055 * v47) * 127.5
              + 127.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .sdxlBase, .sdxlRefiner, .ssd1b, .pixart, .auraflow:
          for i in 0..<imageHeight * imageWidth {
            // We need to do some computations from the latent values.
            let (v0, v1, v2, v3) = (
              fp16[i * 4], fp16[i * 4 + 1], fp16[i * 4 + 2], fp16[i * 4 + 3]
            )
            let r = 47.195 * v0 - 29.114 * v1 + 11.883 * v2 - 38.063 * v3 + 141.64
            let g = 53.237 * v0 - 1.4623 * v1 + 12.991 * v2 - 28.043 * v3 + 127.46
            let b = 58.182 * v0 + 4.3734 * v1 - 3.3735 * v2 - 26.722 * v3 + 114.5
            bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .kandinsky21:
          for i in 0..<imageHeight * imageWidth {
            // We need to do some computations from the latent values.
            let (v0, v1, v2, v3) = (
              Float(fp16[i * 4]), Float(fp16[i * 4 + 1]), Float(fp16[i * 4 + 2]),
              Float(fp16[i * 4 + 3])
            )
            let L = -0.051509 * v0 + 0.039954 * v1 + 0.039893 * v2 - 0.087302 * v3 + 0.88591
            let a = -0.028686 * v0 - 0.0061331 * v1 - 0.016837 * v2 + 0.016139 * v3 + 0.0018263
            let b = -0.0068242 * v0 + 0.0068562 * v1 - 0.03415 * v2 + 0.00056286 * v3 + 0.0096209
            var (sr, sg, sb) = OKlabToLinearsRGB(L: L, a: a, b: b)
            sr = linearsRGBTosRGB(x: sr) * 255
            sg = linearsRGBTosRGB(x: sg) * 255
            sb = linearsRGBTosRGB(x: sb) * 255
            bytes[i * 4] = UInt8(min(max(Int(sr.isFinite ? sr : 0), 0), 255))
            bytes[i * 4 + 1] = UInt8(min(max(Int(sg.isFinite ? sg : 0), 0), 255))
            bytes[i * 4 + 2] = UInt8(min(max(Int(sb.isFinite ? sb : 0), 0), 255))
            bytes[i * 4 + 3] = 255
          }
        case .wurstchenStageC, .wurstchenStageB:
          if channels == 3 {
            for i in 0..<imageHeight * imageWidth {
              // We need to do some computations from the latent values.
              let (r, g, b) = (fp16[i * 3], fp16[i * 3 + 1], fp16[i * 3 + 2])
              bytes[i * 4] = UInt8(
                min(max(Int(r.isFinite ? (Float(r) * 255).rounded() : 0), 0), 255))
              bytes[i * 4 + 1] = UInt8(
                min(max(Int(g.isFinite ? (Float(g) * 255).rounded() : 0), 0), 255))
              bytes[i * 4 + 2] = UInt8(
                min(max(Int(b.isFinite ? (Float(b) * 255).rounded() : 0), 0), 255))
              bytes[i * 4 + 3] = 255
            }
          } else {
            for i in 0..<imageHeight * imageWidth {
              // We need to do some computations from the latent values.
              let (v0, v1, v2, v3) = (
                fp16[i * 4], fp16[i * 4 + 1], fp16[i * 4 + 2], fp16[i * 4 + 3]
              )
              let r = 10.175 * v0 - 20.807 * v1 - 27.834 * v2 - 2.0577 * v3 + 143.39
              let g = 21.07 * v0 - 4.3022 * v1 - 11.258 * v2 - 18.8 * v3 + 131.53
              let b = 7.8454 * v0 - 2.3713 * v1 - 0.45565 * v2 - 41.648 * v3 + 120.76
              bytes[i * 4] = UInt8(min(max(Int(r.isFinite ? r : 0), 0), 255))
              bytes[i * 4 + 1] = UInt8(min(max(Int(g.isFinite ? g : 0), 0), 255))
              bytes[i * 4 + 2] = UInt8(min(max(Int(b.isFinite ? b : 0), 0), 255))
              bytes[i * 4 + 3] = 255
            }
          }
        }
        guard
          let cgImage = CGImage(
            width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(
              rawValue: CGBitmapInfo.byteOrder32Big.rawValue
                | CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: CGDataProvider(
              dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
              releaseData: { _, p, _ in
                p.deallocate()
              })!, decode: nil, shouldInterpolate: false,
            intent: CGColorRenderingIntent.defaultIntent)
        else {
          bytes.deallocate()
          continue
        }
        images.append(cgImage)
      }
      return (images, false)
    }
  }
  #if canImport(UIKit)
    public static func images(
      fromLatent tensor: Tensor<FloatType>, canUseTAESD: Bool, version: ModelVersion
    )
      -> ([UIImage], Bool)
    {
      let (cgImages, usedTAESD) = cgImages(
        fromLatent: tensor, canUseTAESD: canUseTAESD, version: version)
      return (cgImages.map(UIImage.init(cgImage:)), usedTAESD)
    }
  #endif
  #if canImport(UIKit)
    public static func imageData(
      from tensor: Tensor<FloatType>, binaryMask: Tensor<UInt8>?,
      configuration: GenerationConfiguration,
      savedPositivePrompt: String, savedNegativePrompt: String,
      memorizedBy: Set<String>, profile: GenerationProfile? = nil
    ) -> Data? {
      return autoreleasepool { () -> Data? in
        guard let cgImage = image(from: tensor, scaleFactor: 1, binaryMask: binaryMask).cgImage
        else { return nil }
        let data = NSMutableData()
        guard
          let imageDestination = CGImageDestinationCreateWithData(
            data as CFMutableData, UTType.png.identifier as CFString, 1, nil)
        else { return nil }
        var description = "\(savedPositivePrompt.trimmingCharacters(in: .whitespacesAndNewlines))\n"
        if savedNegativePrompt.count > 0 {
          description += "-\(savedNegativePrompt.trimmingCharacters(in: .whitespacesAndNewlines))\n"
        }
        let model = configuration.model ?? ModelZoo.defaultSpecification.file
        let modelVersion = ModelZoo.versionForModel(model)
        let hasTeaCache = ModelZoo.teaCacheCoefficientsForModel(model) != nil
        let modifier = ImageGeneratorUtils.modifierForModel(
          model, LoRAs: configuration.loras.compactMap(\.file))
        let seedMode: String
        switch configuration.seedMode {
        case .legacy:
          seedMode = "Legacy"
        case .torchCpuCompatible:
          seedMode = "Torch CPU Compatible"
        case .scaleAlike:
          seedMode = "Scale Alike"
        case .nvidiaGpuCompatible:
          seedMode = "NVIDIA GPU Compatible"
        }
        let sampler = configuration.sampler.description
        description +=
          "Steps: \(configuration.steps), Sampler: \(sampler), Guidance Scale: \(configuration.guidanceScale), Seed: \(configuration.seed), Size: \(configuration.startWidth * 64)x\(configuration.startHeight * 64), Model: \(model), Strength: \(configuration.strength), Seed Mode: \(seedMode)"
        var json = [String: Any]()
        let jsonEncoder = JSONEncoder()
        if let data = try? jsonEncoder.encode(
          JSGenerationConfiguration(configuration: configuration))
        {
          json["v2"] = try? JSONSerialization.jsonObject(
            with: data, options: [.topLevelDictionaryAssumed])
        }
        json["c"] = savedPositivePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        json["uc"] = savedNegativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        json["steps"] = configuration.steps
        json["sampler"] = sampler
        json["scale"] = configuration.guidanceScale
        json["seed"] = configuration.seed
        json["size"] = "\(configuration.startWidth * 64)x\(configuration.startHeight * 64)"
        json["model"] = model
        json["strength"] = configuration.strength
        json["seed_mode"] = seedMode
        if let upscaler = configuration.upscaler,
          UpscalerZoo.isModelDownloaded(upscaler, memorizedBy: memorizedBy)
        {
          json["upscaler"] = upscaler
          description += ", Upscaler: \(upscaler)"
        }
        if configuration.shift != 1 {
          json["shift"] = configuration.shift
          description += ", Shift: \(configuration.shift)"
        }
        if configuration.clipSkip > 1 {
          json["clip_skip"] = configuration.clipSkip
          description += ", CLIP Skip: \(configuration.clipSkip)"
        }
        if configuration.sharpness > 0 {
          json["sharpness"] = configuration.sharpness
          description += ", Sharpness: \(configuration.sharpness)"
        }
        if configuration.maskBlur > 0 {
          json["mask_blur"] = configuration.maskBlur
        }
        if configuration.maskBlurOutset != 0 {
          json["mask_blur_outset"] = configuration.maskBlurOutset
          description += ", Mask Blur Outset: \(configuration.maskBlurOutset)"
        }
        if modifier == .editing {
          json["image_guidance"] = configuration.imageGuidanceScale
          description += ", Image Guidance: \(configuration.imageGuidanceScale)"
        }
        if configuration.hiresFix && modelVersion != .wurstchenStageC {
          description +=
            ", Hires Fix: true, First Stage Size: \(configuration.hiresFixStartWidth * 64)x\(configuration.hiresFixStartHeight * 64), Second Stage Strength: \(configuration.hiresFixStrength)"
          json["hires_fix"] = configuration.hiresFix
          json["first_stage_size"] =
            "\(configuration.hiresFixStartWidth * 64)x\(configuration.hiresFixStartHeight * 64)"
          json["second_stage_strength"] = configuration.hiresFixStrength
        }
        let refinerVersion: ModelVersion? = configuration.refinerModel.flatMap {
          guard $0 != configuration.model, ModelZoo.isModelDownloaded($0, memorizedBy: memorizedBy)
          else { return nil }
          let version = ModelZoo.versionForModel($0)
          guard ModelZoo.isCompatibleRefiner(modelVersion, refinerVersion: version) else {
            return nil
          }
          return ModelZoo.versionForModel($0)
        }
        let loras: [LoRAConfiguration] = configuration.loras.compactMap {
          guard let file = $0.file, LoRAZoo.isModelDownloaded(file, memorizedBy: memorizedBy) else {
            return nil
          }
          let loraVersion = LoRAZoo.versionForModel(file)
          guard modelVersion == loraVersion || refinerVersion == loraVersion
          else { return nil }
          return LoRAConfiguration(
            file: file, weight: $0.weight, version: modelVersion,
            isLoHa: LoRAZoo.isLoHaForModel(file), modifier: LoRAZoo.modifierForModel(file),
            mode: .init(from: $0.mode))
        }
        if modelVersion == .kandinsky21 {
          json["clip_weight"] = configuration.clipWeight
          description +=
            ", CLIP Weight: \(configuration.clipWeight.formatted(.percent.precision(.fractionLength(0))))"
          json["image_prior_steps"] = configuration.imagePriorSteps
          description += ", Image Prior Steps: \(configuration.imagePriorSteps)"
          json["negative_prompt_for_image_prior"] = configuration.negativePromptForImagePrior
          description +=
            ", Negative Prompt for Image Prior: \(configuration.negativePromptForImagePrior ? "true" : "false")"
        }
        if modelVersion == .minimaxH3 {
          json["shift_for_audio"] = configuration.shiftForAudio
          description += ", Audio Shift: \(configuration.shiftForAudio)"
        }
        if modelVersion == .ideogram4 {
          json["expand_prompt_to_json"] = configuration.expandPromptToJson
          description +=
            ", Expand Prompt to JSON: \(configuration.expandPromptToJson ? "true" : "false")"
        }
        if modelVersion == .wurstchenStageC {
          let width = Int(
            ((Double(
              configuration.hiresFixStartWidth > 0 && configuration.hiresFix
                ? configuration.hiresFixStartWidth : configuration.startWidth) * 3) / 2).rounded(
                .up))
          let height = Int(
            ((Double(
              configuration.hiresFixStartHeight > 0 && configuration.hiresFix
                ? configuration.hiresFixStartHeight : configuration.startHeight) * 3) / 2).rounded(
                .up))
          json["stage_1_latents_size"] = "\(width)x\(height)"
          description += ", Stage 1 Latents Size: \(width)x\(height)"
          json["stage_2_steps"] = configuration.stage2Steps
          description += ", Stage 2 Steps: \(configuration.stage2Steps)"
          json["stage_2_guidance"] = configuration.stage2Cfg
          description += ", Stage 2 Guidance: \(configuration.stage2Cfg)"
          if configuration.stage2Shift != 1 {
            json["stage_2_shift"] = configuration.stage2Shift
            description += ", Stage 2 Shift: \(configuration.stage2Shift)"
          }
        }
        if let refiner = configuration.refinerModel {
          json["refiner"] = refiner
          json["refiner_start"] = configuration.refinerStart
          description +=
            ", Refiner: \(refiner), Refiner Start: \(configuration.refinerStart.formatted(.percent.precision(.fractionLength(0))))"
        }
        if modelVersion == .sdxlBase || modelVersion == .sdxlRefiner || modelVersion == .ssd1b {
          json["target_size"] =
            "\(configuration.targetImageWidth)x\(configuration.targetImageHeight)"
          json["crop_top"] = configuration.cropTop
          json["crop_left"] = configuration.cropLeft
          json["original_size"] =
            "\(configuration.originalImageWidth)x\(configuration.originalImageHeight)"
          json["aesthetic_score"] = configuration.aestheticScore
          json["negative_aesthetic_score"] = configuration.negativeAestheticScore
          json["zero_negative_prompt"] = configuration.zeroNegativePrompt
          json["negative_original_size"] =
            "\(configuration.negativeOriginalImageWidth)x\(configuration.negativeOriginalImageHeight)"
          description +=
            ", Target Size: \(configuration.targetImageWidth)x\(configuration.targetImageHeight), Crop: (\(configuration.cropTop), \(configuration.cropLeft)), Original Size: \(configuration.originalImageWidth)x\(configuration.originalImageHeight), Negative Original Size: \(configuration.negativeOriginalImageWidth)x\(configuration.negativeOriginalImageHeight), Aesthetic Score: \(configuration.aestheticScore.formatted(.number.precision(.fractionLength(1)))), Negative Aesthetic Score: \(configuration.negativeAestheticScore.formatted(.number.precision(.fractionLength(1)))), Zero Negative Prompt: \(configuration.zeroNegativePrompt ? "true" : "false")"
        }
        if modelVersion == .svdI2v {
          json["num_frames"] = configuration.numFrames
          json["fps"] = configuration.fpsId
          json["motion_scale"] = configuration.motionBucketId
          json["cond_aug"] = configuration.condAug
          json["min_cfg"] = configuration.startFrameCfg
          description +=
            ", Number of Frames: \(configuration.numFrames), FPS: \(configuration.fpsId), Motion Scale: \(configuration.motionBucketId), Guiding Frame Noise: \(configuration.condAug.formatted(.number.precision(.fractionLength(2)))), Start Frame Guidance: \(configuration.startFrameCfg.formatted(.number.precision(.fractionLength(1))))"
        } else if modelVersion == .hunyuanVideo {
          json["num_frames"] = configuration.numFrames
          description += ", Number of Frames: \(configuration.numFrames)"
        }
        if configuration.tiledDecoding
          && (configuration.startWidth > configuration.decodingTileWidth
            || configuration.startHeight > configuration.decodingTileHeight)
        {
          json["tiled_decoding"] = configuration.tiledDecoding
          json["decoding_tile_width"] = configuration.decodingTileWidth * 64
          json["decoding_tile_height"] = configuration.decodingTileHeight * 64
          json["decoding_tile_overlap"] = configuration.decodingTileOverlap * 64
          description +=
            ", Tiled Decoding Enabled: \(configuration.decodingTileWidth * 64)x\(configuration.decodingTileHeight * 64) [\(configuration.decodingTileOverlap * 64)]"
        }
        if configuration.tiledDiffusion
          && (configuration.startWidth > configuration.diffusionTileWidth
            || configuration.startHeight > configuration.diffusionTileHeight)
        {
          json["tiled_diffusion"] = configuration.tiledDiffusion
          json["diffusion_tile_width"] = configuration.diffusionTileWidth * 64
          json["diffusion_tile_height"] = configuration.diffusionTileHeight * 64
          json["diffusion_tile_overlap"] = configuration.diffusionTileOverlap * 64
          description +=
            ", Tiled Diffusion Enabled: \(configuration.diffusionTileWidth * 64)x\(configuration.diffusionTileHeight * 64) [\(configuration.diffusionTileOverlap * 64)]"
        }
        if modelVersion == .minimaxH3 && configuration.usesSolAttention {
          json["uses_sol_attention"] = true
          json["sol_attention_start"] = configuration.solAttentionStart
          json["sol_attention_tau"] = configuration.solAttentionTau
          description +=
            ", Sol Attention Enabled: \(configuration.solAttentionStart), \(configuration.solAttentionTau.formatted(.number.precision(.fractionLength(2))))"
        }
        if hasTeaCache && configuration.teaCache && configuration.teaCacheThreshold > 0 {
          json["tea_cache"] = configuration.teaCache
          json["tea_cache_start"] = configuration.teaCacheStart
          json["tea_cache_end"] = configuration.teaCacheEnd
          json["tea_cache_threshold"] = configuration.teaCacheThreshold
          json["tea_cache_max_skip_steps"] = configuration.teaCacheMaxSkipSteps
          var teaCacheEnd =
            configuration.teaCacheEnd < 0
            ? Int32(configuration.steps) + 1 + configuration.teaCacheEnd : configuration.teaCacheEnd
          let teaCacheStart = min(
            max(configuration.teaCacheStart, 0), Int32(configuration.steps))
          teaCacheEnd = min(
            max(max(teaCacheStart, teaCacheEnd), 0), Int32(configuration.steps))
          description +=
            ", TeaCache Enabled: \(min(teaCacheStart, teaCacheEnd)) - \(max(teaCacheStart, teaCacheEnd)), \(configuration.teaCacheThreshold.formatted(.number.precision(.fractionLength(2)))), \(configuration.teaCacheMaxSkipSteps)"
        }
        if configuration.causalInferenceEnabled && configuration.causalInference > 0 {
          json["causal_inference"] = configuration.causalInference
          description += ", Causal Inference: \(configuration.causalInference)"
        }
        if configuration.sampler == .TCD || configuration.sampler == .tCDTrailing {
          json["stochastic_sampling_gamma"] = configuration.stochasticSamplingGamma
          description += ", Strategic Stochastic Sampling: \(configuration.stochasticSamplingGamma)"
        }
        if loras.count > 0 {
          if let firstLoRA = loras.first, loras.count == 1 {
            description += ", LoRA Model: \(firstLoRA.file), LoRA Weight: \(firstLoRA.weight)"
          } else {
            description +=
              ", \(loras.enumerated().map({ "LoRA \($0 + 1) Model: \($1.file), LoRA \($0 + 1) Weight: \($1.weight)" }).joined(separator: ", "))"
          }
          json["lora"] = loras.map {
            ["model": $0.file, "weight": $0.weight] as [String: Any]
          }
        }
        let (
          canInjectControls, canInjectT2IAdapters, canInjectAttentionKVs, canInjectPrompts,
          injectIPAdapterLengths, _
        ) =
          ImageGeneratorUtils.canInjectControls(
            hasImage: true, hasDepth: true, hasHints: Set([.scribble, .pose, .color]),
            hasCustom: true, shuffleCount: 1, controls: configuration.controls,
            version: modelVersion, memorizedBy: memorizedBy)
        if canInjectControls || canInjectT2IAdapters || canInjectAttentionKVs || canInjectPrompts
          || !injectIPAdapterLengths.isEmpty
        {
          let controls:
            [(
              file: String, weight: Float, guidanceStart: Float, guidanceEnd: Float, noPrompt: Bool,
              globalAveragePooling: Bool, inputOverride: ControlInputType
            )] =
              configuration.controls.compactMap {
                guard let file = $0.file,
                  let specification = ControlNetZoo.specificationForModel(file),
                  ControlNetZoo.isModelDownloaded(specification, memorizedBy: memorizedBy),
                  let modifier = ControlNetZoo.modifierForModel(file)
                    ?? ControlHintType(from: $0.inputOverride)
                else { return nil }
                switch modifier {
                case .canny, .custom, .depth, .scribble, .pose, .color, .normalbae, .lineart,
                  .softedge, .seg, .inpaint, .ip2p, .shuffle, .mlsd, .tile, .blur, .gray,
                  .lowquality:
                  return (
                    file, $0.weight, $0.guidanceStart, $0.guidanceEnd, $0.noPrompt,
                    $0.globalAveragePooling, $0.inputOverride
                  )
                case .audio:
                  return nil
                }
              }
          if controls.count > 1 {
            var jsonControls = [[String: Any]]()
            for (i, control) in controls.enumerated() {
              description +=
                ", Control \(i + 1): \(control.file), Control \(i + 1) No Prompt: \(control.noPrompt ? "true" : "false"), Control \(i + 1) Weight: \(control.weight), Control \(i + 1) Start: \(control.guidanceStart), Control \(i + 1) End: \(control.guidanceEnd)"
              var subjson: [String: Any] = [
                "file": control.file, "weight": control.weight, "no_prompt": control.noPrompt,
                "guidance_start": control.guidanceStart, "guidance_end": control.guidanceEnd,
              ]

              let modifier =
                ControlNetZoo.modifierForModel(control.file)
                ?? ControlHintType(from: control.inputOverride)
              if modifier == .shuffle {
                subjson["global_average_pooling"] = control.globalAveragePooling
                description +=
                  ", Control \(i + 1) Global Average Pooling: \(control.globalAveragePooling ? "true" : "false")"
              }
              jsonControls.append(subjson)
            }
            json["control"] = jsonControls
          } else if let control = controls.first {
            description +=
              ", Control: \(control.file), Control No Prompt: \(control.noPrompt ? "true" : "false"), Control Weight: \(control.weight), Control Start: \(control.guidanceStart), Control End: \(control.guidanceEnd)"
            json["control"] = [
              [
                "file": control.file, "weight": control.weight, "no_prompt": control.noPrompt,
                "guidance_start": control.guidanceStart, "guidance_end": control.guidanceEnd,
              ] as [String: Any]
            ]
          }
        }
        if let profile = profile {
          let jsonEncoder = JSONEncoder()
          jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
          if let data = try? jsonEncoder.encode(profile),
            let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
          {
            json["profile"] = dict
          }
        }
        var info = [String: Any]()
        info[kCGImagePropertyPNGDescription as String] = description
        if let jsonData = try? JSONSerialization.data(withJSONObject: json, options: .sortedKeys) {
          info[kCGImagePropertyPNGComment as String] = String(data: jsonData, encoding: .utf8)
        }
        info[kCGImagePropertyPNGSoftware as String] = "Draw Things"
        var metadata = [String: Any]()
        metadata[kCGImagePropertyPNGDictionary as String] = info
        CGImageDestinationAddImage(imageDestination, cgImage, metadata as CFDictionary)
        CGImageDestinationFinalize(imageDestination)
        return data as Data
      }
    }
  #endif
  public static func tensor(from bitmapContext: CGContext) -> (
    Tensor<FloatType>?, Tensor<UInt8>?, Bool
  ) {
    precondition(bitmapContext.bytesPerRow >= bitmapContext.width * 4)
    guard let data = bitmapContext.data else {
      return (nil, nil, false)
    }
    let bytes = data.assumingMemoryBound(to: UInt8.self)
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, bitmapContext.height, bitmapContext.width, 3))
    let bytesPerRow = bitmapContext.bytesPerRow
    var hasTransparent = false
    var hasNonTransparent = true
    let imageHeight = bitmapContext.height
    let imageWidth = bitmapContext.width
    tensor.withUnsafeMutableBytes {
      guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for y in 0..<imageHeight {
        for x in 0..<imageWidth {
          let alpha = bytes[y * bytesPerRow + x * 4 + 3]
          if alpha != 255 {
            hasTransparent = true
            if alpha == 0 {
              fp16[y * imageWidth * 3 + x * 3] = 0
              fp16[y * imageWidth * 3 + x * 3 + 1] = 0
              fp16[y * imageWidth * 3 + x * 3 + 2] = 0
            } else {
              let invAlpha = 255.0 / FloatType(alpha)
              let r = FloatType(bytes[y * bytesPerRow + x * 4]) * 2 / 255
              let g = FloatType(bytes[y * bytesPerRow + x * 4 + 1]) * 2 / 255
              let b = FloatType(bytes[y * bytesPerRow + x * 4 + 2]) * 2 / 255
              fp16[y * imageWidth * 3 + x * 3] = min(r * invAlpha, 2) - 1
              fp16[y * imageWidth * 3 + x * 3 + 1] = min(g * invAlpha, 2) - 1
              fp16[y * imageWidth * 3 + x * 3 + 2] = min(b * invAlpha, 2) - 1
            }
          } else {
            hasNonTransparent = true
            fp16[y * imageWidth * 3 + x * 3] =
              FloatType(bytes[y * bytesPerRow + x * 4]) * 2 / 255 - 1
            fp16[y * imageWidth * 3 + x * 3 + 1] =
              FloatType(bytes[y * bytesPerRow + x * 4 + 1]) * 2 / 255 - 1
            fp16[y * imageWidth * 3 + x * 3 + 2] =
              FloatType(bytes[y * bytesPerRow + x * 4 + 2]) * 2 / 255 - 1
          }
        }
      }
    }
    var newBinaryMask: Tensor<UInt8>? = nil
    if hasTransparent && hasNonTransparent {
      // Create binaryMask.
      var binaryMask = Tensor<UInt8>(.CPU, .NC(bitmapContext.height, bitmapContext.width))
      binaryMask.withUnsafeMutableBytes {
        guard let u8 = $0.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
        for y in 0..<imageHeight {
          for x in 0..<imageWidth {
            // 3 is to retain everything, 1 is to not.
            let alpha = bytes[y * bytesPerRow + x * 4 + 3]
            let flag: UInt8 = alpha != 255 ? 1 : 3
            u8[y * imageWidth + x] = (flag | (alpha & 0xf8))
          }
        }
      }
      newBinaryMask = binaryMask
    }
    return (tensor, newBinaryMask, hasTransparent)
  }

  public static func configuration(
    from imageProperties: [String: Any]?, configuration: GenerationConfiguration,
    memorizedBy: Set<String>
  ) -> (
    String?, String?, GenerationConfiguration?
  ) {
    var savedPositivePrompt: String?
    var savedNegativePrompt: String?
    var previousConfiguration: GenerationConfiguration?
    if let imageProperties = imageProperties {
      if let imageData = imageProperties[kCGImagePropertyExifDictionary as String]
        as? [String: Any],
        let userComments = imageData[kCGImagePropertyExifUserComment as String] as? String,
        let commentsJsonData = userComments.data(using: .utf8),
        let commentsJsonDictionary = try? JSONSerialization.jsonObject(with: commentsJsonData)
          as? [String: Any]
      {
        var configurationBuilder = GenerationConfigurationBuilder(from: configuration)

        if let positivePrompt = commentsJsonDictionary["c"] as? String {
          savedPositivePrompt = positivePrompt
        }

        if let negativePrompt = commentsJsonDictionary["uc"] as? String {
          savedNegativePrompt = negativePrompt
        }

        if let steps = commentsJsonDictionary["steps"] as? UInt32 {
          configurationBuilder.steps = steps
        }

        if let sampler = commentsJsonDictionary["sampler"] as? String, !sampler.isEmpty {
          configurationBuilder.sampler = SamplerType(from: sampler)
        }

        if let guidanceScale = commentsJsonDictionary["scale"] as? Float32 {
          configurationBuilder.guidanceScale = guidanceScale
        }

        if let seed = commentsJsonDictionary["seed"] as? Int64 {
          configurationBuilder.seed = UInt32(seed)
        }

        if let size = commentsJsonDictionary["size"] as? String, !size.isEmpty {
          let components = size.split(separator: "x").compactMap { Int($0) }
          if components.count == 2 {
            var width = components[0]
            var height = components[1]

            let maxSupportLength = Int(DeviceCapability.maxSupportScale() * 64)
            if width > maxSupportLength || height > maxSupportLength {
              if width > height {
                height = maxSupportLength * height / width
                width = maxSupportLength
              } else {
                width = maxSupportLength * width / height
                height = maxSupportLength
              }
            }
            configurationBuilder.startWidth = UInt16(width / 64)
            configurationBuilder.startHeight = UInt16(height / 64)
          }
        }

        if let model = commentsJsonDictionary["model"] as? String {
          configurationBuilder.model = model
        }

        if let strength = commentsJsonDictionary["strength"] as? Float {
          configurationBuilder.strength = strength
        }

        if let seedModeString = commentsJsonDictionary["seed_mode"] as? String {
          configurationBuilder.seedMode = SeedMode(from: seedModeString)
        }

        if let upscaler = commentsJsonDictionary["upscaler"] as? String,
          UpscalerZoo.isModelDownloaded(upscaler, memorizedBy: memorizedBy)
        {
          configurationBuilder.upscaler = upscaler
        }

        if let clipSkip = commentsJsonDictionary["clip_skip"] as? UInt32 {
          configurationBuilder.clipSkip = clipSkip
        }

        if let imageGuidance = commentsJsonDictionary["image_guidance"] as? Float32 {
          configurationBuilder.imageGuidanceScale = imageGuidance
        }

        if let hiresFix = commentsJsonDictionary["hires_fix"] as? Bool {
          configurationBuilder.hiresFix = hiresFix
        }

        if let hiresFixSize = commentsJsonDictionary["first_stage_size"] as? String {
          let components = hiresFixSize.split(separator: "x").compactMap { UInt16($0) }
          if components.count == 2 {
            configurationBuilder.hiresFixStartWidth = components[0] / 64
            configurationBuilder.hiresFixStartHeight = components[1] / 64
          }
        }

        if let hiresFixStrength = commentsJsonDictionary["second_stage_strength"] as? Float32 {
          configurationBuilder.hiresFixStrength = hiresFixStrength
        }

        if let clipWeight = commentsJsonDictionary["clip_weight"] as? Float32 {
          configurationBuilder.clipWeight = clipWeight
        }

        if let imagePriorSteps = commentsJsonDictionary["image_prior_steps"] as? UInt32 {
          configurationBuilder.imagePriorSteps = imagePriorSteps
        }

        if let negativePromptImagePrior = commentsJsonDictionary[
          "negative_prompt_for_image_prior"] as? Bool
        {
          configurationBuilder.negativePromptForImagePrior = negativePromptImagePrior
        }

        if let refinerModel = commentsJsonDictionary["refiner"] as? String, !refinerModel.isEmpty {
          configurationBuilder.refinerModel = refinerModel
        }

        if let refinerStart = commentsJsonDictionary["refiner_start"] as? Float32 {
          configurationBuilder.refinerStart = refinerStart
        }

        if let targetSize = commentsJsonDictionary["target_size"] as? String {
          let components = targetSize.split(separator: "x").compactMap { UInt32($0) }
          if components.count == 2 {
            configurationBuilder.targetImageWidth = components[0]
            configurationBuilder.targetImageHeight = components[1]
          }
        }

        if let cropTop = commentsJsonDictionary["crop_top"] as? Int32 {
          configurationBuilder.cropTop = cropTop
        }

        if let cropLeft = commentsJsonDictionary["crop_left"] as? Int32 {
          configurationBuilder.cropLeft = cropLeft
        }

        if let originalSize = commentsJsonDictionary["original_size"] as? String {
          let components = originalSize.split(separator: "x").compactMap { UInt32($0) }
          if components.count == 2 {
            configurationBuilder.originalImageWidth = components[0]
            configurationBuilder.originalImageHeight = components[1]
          }
        }

        if let aestheticScore = commentsJsonDictionary["aesthetic_score"] as? Int64 {
          configurationBuilder.aestheticScore = Float(aestheticScore)
        }

        if let negtiveAestheticScore = commentsJsonDictionary["negative_aesthetic_score"]
          as? Float
        {
          configurationBuilder.negativeAestheticScore = negtiveAestheticScore
        }

        if let zeroNegativePrompt = commentsJsonDictionary["zero_negative_prompt"] as? Bool {
          configurationBuilder.zeroNegativePrompt = zeroNegativePrompt
        }

        if let usesSolAttention = commentsJsonDictionary["uses_sol_attention"] as? Bool {
          configurationBuilder.usesSolAttention = usesSolAttention
        }
        if let solAttentionStart = commentsJsonDictionary["sol_attention_start"] as? Int32 {
          configurationBuilder.solAttentionStart = max(0, solAttentionStart)
        }
        if let solAttentionTau = commentsJsonDictionary["sol_attention_tau"] as? Float,
          solAttentionTau.isFinite
        {
          configurationBuilder.solAttentionTau = min(max(solAttentionTau, 0), 4)
        }

        if let shiftForAudio = commentsJsonDictionary["shift_for_audio"] as? Float,
          shiftForAudio.isFinite && shiftForAudio > 0
        {
          configurationBuilder.shiftForAudio = shiftForAudio
        }

        if let expandPromptToJson = commentsJsonDictionary["expand_prompt_to_json"] as? Bool {
          configurationBuilder.expandPromptToJson = expandPromptToJson
        }

        if let negativeOriginalSize = commentsJsonDictionary["negative_original_size"] as? String {
          let components = negativeOriginalSize.split(separator: "x").compactMap { UInt32($0) }
          if components.count == 2 {
            configurationBuilder.negativeOriginalImageWidth = components[0]
            configurationBuilder.negativeOriginalImageHeight = components[1]
          }
        }

        if let loraArray = commentsJsonDictionary["lora"] as? [[String: Any]] {
          let loras = loraArray.compactMap { (lora) -> DataModels.LoRA? in
            if let file = lora["model"] as? String,
              LoRAZoo.isModelDownloaded(file, memorizedBy: memorizedBy),
              let weight = lora["weight"] as? Float
            {
              let mode: DataModels.LoRAMode = {
                switch lora["mode"] as? String {
                case "base":
                  return .base
                case "refiner":
                  return .refiner
                case "all":
                  return .all
                default:
                  return .all
                }
              }()
              return DataModels.LoRA(file: file, weight: weight, mode: mode)
            }
            return nil
          }
          configurationBuilder.loras = loras
        }

        if let controlsArray = commentsJsonDictionary["control"] as? [[String: Any]] {
          let controls = controlsArray.compactMap { (control) -> DataModels.Control? in
            if let file = control["file"] as? String,
              let specification = ControlNetZoo.specificationForModel(file),
              ControlNetZoo.isModelDownloaded(specification, memorizedBy: memorizedBy),
              let weight = control["weight"] as? Float32,
              let guidanceStart = control["guidance_start"] as? Float32,
              let guidanceEnd = control["guidance_end"] as? Float32,
              let noPrompt = control["no_prompt"] as? Bool
            {
              var controlObject = DataModels.Control(
                file: file, weight: weight, guidanceStart: guidanceStart,
                guidanceEnd: guidanceEnd, noPrompt: noPrompt)
              if let globalAveragePooling = control["global_average_pooling"] as? Bool {
                controlObject.globalAveragePooling = globalAveragePooling
              }
              return controlObject
            }
            return nil
          }
          configurationBuilder.controls = controls
        }

        previousConfiguration = configurationBuilder.build()
      }
    }
    return (savedPositivePrompt, savedNegativePrompt, previousConfiguration)
  }
}
