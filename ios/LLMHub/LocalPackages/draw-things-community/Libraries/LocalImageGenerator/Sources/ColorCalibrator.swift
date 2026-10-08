import DataModels
import Diffusion
import Foundation
import NNC

#if canImport(C_ccv)
  import C_ccv
#elseif canImport(C_swiftpm_ccv)
  import C_swiftpm_ccv
#endif

enum ColorCalibrator {
  private static let histogramBinCount = 4096
  private static let maskGenerated: UInt8 = 1
  private static let maskRetained: UInt8 = 2

  private struct MaskFlags {
    var flags: Tensor<UInt8>
    var generatedCount: Int
    var retainedCount: Int
  }

  private struct TransferTables {
    var l: [Float]
    var a: [Float]
    var b: [Float]
  }

  static func calibrate(
    to images: [Tensor<FloatType>], reference: Tensor<FloatType>, mask: Tensor<UInt8>?,
    colorCalibration: ColorCalibration, isTransparentDecoder: Bool
  ) -> [Tensor<FloatType>] {
    return images.enumerated().map { index, image in
      calibrate(
        to: image, reference: reference, referenceFrameOffset: index, mask: mask,
        isTransparentDecoder: isTransparentDecoder)
    }
  }

  private static func calibrate(
    to image: Tensor<FloatType>, reference: Tensor<FloatType>, referenceFrameOffset: Int,
    mask: Tensor<UInt8>?, isTransparentDecoder: Bool
  ) -> Tensor<FloatType> {
    let shape = image.shape
    let referenceShape = reference.shape
    guard shape.count == 4, referenceShape.count == 4, shape[0] > 0, shape[1] > 0, shape[2] > 0,
      shape[3] >= 3, referenceShape[0] > 0, referenceShape[1] > 0, referenceShape[2] > 0,
      referenceShape[3] >= 3
    else {
      return image
    }
    let batch = shape[0]
    let height = shape[1]
    let width = shape[2]
    let channels = shape[3]
    let channelStart = channels - 3
    let referenceChannelStart = referenceShape[3] - 3
    let maskFlags = mask.flatMap {
      makeMaskFlags(
        mask: $0, width: width, height: height, isTransparentDecoder: isTransparentDecoder)
    }
    if let maskFlags, maskFlags.generatedCount == 0 || maskFlags.retainedCount == 0 {
      return image
    }
    var output = image.copied()
    for n in 0..<batch {
      let content = Tensor<Float>(
        from: image[
          n..<(n + 1), 0..<height, 0..<width, channelStart..<(channelStart + 3)
        ].copied())
      let referenceFrame = min(referenceFrameOffset + n, referenceShape[0] - 1)
      let referenceRGB = Tensor<Float>(
        from: reference[
          referenceFrame..<(referenceFrame + 1), 0..<referenceShape[1], 0..<referenceShape[2],
          referenceChannelStart..<(referenceChannelStart + 3)
        ].copied())
      let style = resizedRGB(referenceRGB, width: width, height: height)
      let reconstructed = waveletReconstruct(content: content, style: style)
      var contentLab = Tensor<Float>(.CPU, .NHWC(1, height, width, 3))
      var styleLab = Tensor<Float>(.CPU, .NHWC(1, height, width, 3))
      rgbToLab(reconstructed, into: &contentLab)
      rgbToLab(style, into: &styleLab)
      guard
        let transfers = transferTables(
          contentLab: contentLab, styleLab: styleLab, maskFlags: maskFlags,
          pixelCount: width * height)
      else { continue }
      writeCalibratedRGB(
        into: &output, batch: n, contentLab: contentLab, transfers: transfers,
        maskFlags: maskFlags, width: width, height: height, channels: channels,
        channelStart: channelStart)
    }
    return output
  }

  private static func resizedRGB(_ rgb: Tensor<Float>, width: Int, height: Int) -> Tensor<Float> {
    let shape = rgb.shape
    guard shape.count == 4, shape[1] != height || shape[2] != width else { return rgb }
    let sourceHeight = shape[1]
    let sourceWidth = shape[2]
    let f32 = rgb.reshaped(.HWC(sourceHeight, sourceWidth, 3))
    var output: UnsafeMutablePointer<ccv_dense_matrix_t>? = ccv_dense_matrix_new(
      Int32(height), Int32(width), Int32(CCV_C3 | CCV_32F), nil, 0)
    ccv_resample(
      UnsafeMutableRawPointer(f32.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self),
      &output, 0, Double(height) / Double(sourceHeight), Double(width) / Double(sourceWidth),
      Int32(CCV_INTER_AREA | CCV_INTER_CUBIC))
    guard let output else { return rgb }
    let resized = Tensor<Float>(
      .CPU, format: .NHWC, shape: [1, height, width, 3],
      unsafeMutablePointer: output.pointee.data.f32, bindLifetimeOf: output
    ).copied()
    ccv_matrix_free(output)
    return resized
  }

  private static func waveletReconstruct(content: Tensor<Float>, style: Tensor<Float>)
    -> Tensor<Float>
  {
    let shape = content.shape
    guard shape.count == 4, shape == style.shape else { return content }
    let height = shape[1]
    let width = shape[2]
    let contentHWC = content.reshaped(.HWC(height, width, 3))
    let styleHWC = style.reshaped(.HWC(height, width, 3))
    var contentHigh: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
    var contentLow: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
    var styleHigh: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
    var styleLow: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
    ccv_wavelet_decompose(
      UnsafeMutableRawPointer(contentHWC.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self),
      &contentHigh, &contentLow, 0)
    ccv_wavelet_decompose(
      UnsafeMutableRawPointer(styleHWC.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self),
      &styleHigh, &styleLow, 0)
    defer {
      ccv_matrix_free(contentHigh)
      ccv_matrix_free(contentLow)
      ccv_matrix_free(styleHigh)
      ccv_matrix_free(styleLow)
    }
    guard let contentHigh, let styleLow else { return content }
    var reconstructed = Tensor<Float>(.CPU, .NHWC(1, height, width, 3))
    reconstructed.withUnsafeMutableBytes { outputBuffer in
      guard let outputPtr = outputBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else {
        return
      }
      let contentHighStep = Int(contentHigh.pointee.step) / MemoryLayout<Float>.size
      let styleLowStep = Int(styleLow.pointee.step) / MemoryLayout<Float>.size
      let rowSize = width * 3
      for y in 0..<height {
        let contentPtr = contentHigh.pointee.data.f32.advanced(by: y * contentHighStep)
        let stylePtr = styleLow.pointee.data.f32.advanced(by: y * styleLowStep)
        let outputPtr = outputPtr.advanced(by: y * rowSize)
        for i in 0..<rowSize {
          outputPtr[i] = clamp(contentPtr[i] + stylePtr[i], -1, 1)
        }
      }
    }
    return reconstructed
  }

  private static func rgbToLab(_ rgb: Tensor<Float>, into lab: inout Tensor<Float>) {
    let pixelCount = rgb.shape[1] * rgb.shape[2]
    lab.withUnsafeMutableBytes { labBuffer in
      guard let labPtr = labBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else {
        return
      }
      rgb.withUnsafeBytes { rgbBuffer in
        guard let rgbPtr = rgbBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else {
          return
        }
        for i in 0..<pixelCount {
          let offset = i * 3
          let r = srgbToLinear(clamp((rgbPtr[offset] + 1) * 0.5, 0, 1))
          let g = srgbToLinear(clamp((rgbPtr[offset + 1] + 1) * 0.5, 0, 1))
          let blue = srgbToLinear(clamp((rgbPtr[offset + 2] + 1) * 0.5, 0, 1))
          let x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * blue) / 0.95047
          let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * blue
          let z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * blue) / 1.08883
          let fx = labF(x)
          let fy = labF(y)
          let fz = labF(z)
          labPtr[offset] = 116 * fy - 16
          labPtr[offset + 1] = 500 * (fx - fy)
          labPtr[offset + 2] = 200 * (fy - fz)
        }
      }
    }
  }

  private static func transferTables(
    contentLab: Tensor<Float>, styleLab: Tensor<Float>, maskFlags: MaskFlags?, pixelCount: Int
  ) -> TransferTables? {
    var tables: TransferTables?
    contentLab.withUnsafeBytes { contentBuffer in
      guard let contentPtr = contentBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else {
        return
      }
      styleLab.withUnsafeBytes { styleBuffer in
        guard let stylePtr = styleBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else {
          return
        }
        if let maskFlags {
          maskFlags.flags.withUnsafeBytes { maskBuffer in
            guard let maskPtr = maskBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
              return
            }
            tables = transferTables(
              contentPtr: contentPtr, stylePtr: stylePtr, maskPtr: maskPtr,
              pixelCount: pixelCount, generatedCount: maskFlags.generatedCount,
              retainedCount: maskFlags.retainedCount)
          }
        } else {
          tables = transferTables(
            contentPtr: contentPtr, stylePtr: stylePtr, maskPtr: nil, pixelCount: pixelCount,
            generatedCount: pixelCount, retainedCount: pixelCount)
        }
      }
    }
    return tables
  }

  private static func transferTables(
    contentPtr: UnsafePointer<Float>, stylePtr: UnsafePointer<Float>,
    maskPtr: UnsafePointer<UInt8>?,
    pixelCount: Int, generatedCount: Int, retainedCount: Int
  ) -> TransferTables? {
    guard generatedCount > 0, retainedCount > 0 else { return nil }
    guard
      let l = histogramMatchBinned(
        sourcePtr: contentPtr, referencePtr: stylePtr, channel: 0, maskPtr: maskPtr,
        pixelCount: pixelCount, sourceCount: generatedCount, referenceCount: retainedCount,
        lowerBound: 0, upperBound: 100),
      let a = histogramMatchBinned(
        sourcePtr: contentPtr, referencePtr: stylePtr, channel: 1, maskPtr: maskPtr,
        pixelCount: pixelCount, sourceCount: generatedCount, referenceCount: retainedCount,
        lowerBound: -128, upperBound: 127),
      let b = histogramMatchBinned(
        sourcePtr: contentPtr, referencePtr: stylePtr, channel: 2, maskPtr: maskPtr,
        pixelCount: pixelCount, sourceCount: generatedCount, referenceCount: retainedCount,
        lowerBound: -128, upperBound: 127)
    else { return nil }
    return TransferTables(l: l, a: a, b: b)
  }

  private static func histogramMatchBinned(
    sourcePtr: UnsafePointer<Float>, referencePtr: UnsafePointer<Float>, channel: Int,
    maskPtr: UnsafePointer<UInt8>?, pixelCount: Int, sourceCount: Int, referenceCount: Int,
    lowerBound: Float, upperBound: Float
  ) -> [Float]? {
    guard sourceCount > 0, referenceCount > 0 else { return nil }
    var sourceHistogram = [Int](repeating: 0, count: histogramBinCount)
    var referenceHistogram = [Int](repeating: 0, count: histogramBinCount)
    for i in 0..<pixelCount {
      if let maskPtr, maskPtr[i] != maskGenerated { continue }
      let bin = histogramBin(
        sourcePtr[i * 3 + channel], lowerBound: lowerBound, upperBound: upperBound)
      sourceHistogram[bin] += 1
    }
    for i in 0..<pixelCount {
      if let maskPtr, maskPtr[i] != maskRetained { continue }
      let bin = histogramBin(
        referencePtr[i * 3 + channel], lowerBound: lowerBound, upperBound: upperBound)
      referenceHistogram[bin] += 1
    }
    var transfer = [Float](repeating: 0, count: histogramBinCount)
    var sourcePrefix = 0
    var referencePrefix = referenceHistogram[0]
    var referenceBin = 0
    for sourceBin in 0..<histogramBinCount {
      sourcePrefix += sourceHistogram[sourceBin]
      if sourcePrefix == 0 {
        transfer[sourceBin] = binValue(
          sourceBin, lowerBound: lowerBound, upperBound: upperBound)
        continue
      }
      while referenceBin < histogramBinCount - 1
        && referencePrefix * sourceCount < sourcePrefix * referenceCount
      {
        referenceBin += 1
        referencePrefix += referenceHistogram[referenceBin]
      }
      transfer[sourceBin] = binValue(
        referenceBin, lowerBound: lowerBound, upperBound: upperBound)
    }
    return transfer
  }

  private static func writeCalibratedRGB<FloatType: TensorNumeric & BinaryFloatingPoint>(
    into output: inout Tensor<FloatType>, batch: Int, contentLab: Tensor<Float>,
    transfers: TransferTables, maskFlags: MaskFlags?, width: Int, height: Int, channels: Int,
    channelStart: Int
  ) {
    let pixelCount = width * height
    output.withUnsafeMutableBytes { outputBuffer in
      guard let outputPtr = outputBuffer.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
        return
      }
      contentLab.withUnsafeBytes { labBuffer in
        guard let labPtr = labBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else {
          return
        }
        if let maskFlags {
          maskFlags.flags.withUnsafeBytes { maskBuffer in
            guard let maskPtr = maskBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
              return
            }
            writeCalibratedRGB(
              outputPtr: outputPtr, labPtr: labPtr, maskPtr: maskPtr, batch: batch,
              transfers: transfers, pixelCount: pixelCount, channels: channels,
              channelStart: channelStart)
          }
        } else {
          writeCalibratedRGB(
            outputPtr: outputPtr, labPtr: labPtr, maskPtr: nil, batch: batch,
            transfers: transfers, pixelCount: pixelCount, channels: channels,
            channelStart: channelStart)
        }
      }
    }
  }

  private static func writeCalibratedRGB<FloatType: TensorNumeric & BinaryFloatingPoint>(
    outputPtr: UnsafeMutablePointer<FloatType>, labPtr: UnsafePointer<Float>,
    maskPtr: UnsafePointer<UInt8>?, batch: Int, transfers: TransferTables, pixelCount: Int,
    channels: Int, channelStart: Int
  ) {
    let outputBatchBase = batch * pixelCount * channels
    for i in 0..<pixelCount {
      if let maskPtr, maskPtr[i] != maskGenerated { continue }
      let labOffset = i * 3
      let l0 = labPtr[labOffset]
      let a0 = labPtr[labOffset + 1]
      let b0 = labPtr[labOffset + 2]
      let lBin = histogramBin(l0, lowerBound: 0, upperBound: 100)
      let aBin = histogramBin(a0, lowerBound: -128, upperBound: 127)
      let bBin = histogramBin(b0, lowerBound: -128, upperBound: 127)
      let l = 0.8 * l0 + 0.2 * transfers.l[lBin]
      let (r, g, b) = labToRGB(l: l, a: transfers.a[aBin], b: transfers.b[bBin])
      let outputOffset = outputBatchBase + i * channels + channelStart
      outputPtr[outputOffset] = FloatType(r)
      outputPtr[outputOffset + 1] = FloatType(g)
      outputPtr[outputOffset + 2] = FloatType(b)
    }
  }

  private static func makeMaskFlags(
    mask: Tensor<UInt8>, width: Int, height: Int, isTransparentDecoder: Bool
  ) -> MaskFlags? {
    let shape = mask.shape
    guard shape.count == 2, shape[0] > 0, shape[1] > 0 else { return nil }
    let maskHeight = shape[0]
    let maskWidth = shape[1]
    var exists0 = false
    var exists1 = false
    var exists2 = false
    var exists3 = false
    var scanned = false
    mask.withUnsafeBytes { maskBuffer in
      guard let maskPtr = maskBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
        return
      }
      scanned = true
      for i in 0..<(maskHeight * maskWidth) {
        switch normalizedMaskValue(maskPtr[i]) {
        case 0:
          exists0 = true
        case 1:
          exists1 = true
        case 2:
          exists2 = true
        case 3:
          exists3 = true
        default:
          break
        }
      }
    }
    guard scanned, exists0 || exists1 || exists2 || exists3 else { return nil }
    let overwrite0 = exists0 && (!exists2 || (exists3 && exists2 && exists1))
    var flags = Tensor<UInt8>(.CPU, .C(width * height))
    var generatedCount = 0
    var retainedCount = 0
    var wrote = false
    flags.withUnsafeMutableBytes { flagsBuffer in
      guard let flagsPtr = flagsBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
        return
      }
      mask.withUnsafeBytes { maskBuffer in
        guard let maskPtr = maskBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
          return
        }
        wrote = true
        for y in 0..<height {
          let maskY = min(y * maskHeight / height, maskHeight - 1)
          for x in 0..<width {
            let maskX = min(x * maskWidth / width, maskWidth - 1)
            let value = normalizedMaskValue(maskPtr[maskY * maskWidth + maskX])
            let isRetained =
              value == 3 || (value == 1 && isTransparentDecoder) || (value == 0 && !overwrite0)
            let index = y * width + x
            if isRetained {
              flagsPtr[index] = maskRetained
              retainedCount += 1
            } else {
              flagsPtr[index] = maskGenerated
              generatedCount += 1
            }
          }
        }
      }
    }
    guard wrote else { return nil }
    return MaskFlags(flags: flags, generatedCount: generatedCount, retainedCount: retainedCount)
  }

  private static func normalizedMaskValue(_ value: UInt8) -> UInt8 {
    let masked = value & 7
    return masked == 4 ? 2 : masked
  }

  private static func labToRGB(l: Float, a: Float, b: Float) -> (Float, Float, Float) {
    let fy = (l + 16) / 116
    let fx = fy + a / 500
    let fz = fy - b / 200
    let x = 0.95047 * labFInverse(fx)
    let y = labFInverse(fy)
    let z = 1.08883 * labFInverse(fz)
    let linearR = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    let linearG = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    let linearB = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    let r = linearToSrgb(linearR) * 2 - 1
    let g = linearToSrgb(linearG) * 2 - 1
    let blue = linearToSrgb(linearB) * 2 - 1
    return (clamp(r, -1, 1), clamp(g, -1, 1), clamp(blue, -1, 1))
  }

  private static func srgbToLinear(_ value: Float) -> Float {
    value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
  }

  private static func linearToSrgb(_ value: Float) -> Float {
    let value = clamp(value, 0, 1)
    return value <= 0.0031308 ? 12.92 * value : 1.055 * pow(value, 1.0 / 2.4) - 0.055
  }

  private static func labF(_ value: Float) -> Float {
    let epsilon: Float = 216.0 / 24_389.0
    let kappa: Float = 24_389.0 / 27.0
    return value > epsilon ? pow(value, 1.0 / 3.0) : (kappa * value + 16) / 116
  }

  private static func labFInverse(_ value: Float) -> Float {
    let value3 = value * value * value
    let epsilon: Float = 216.0 / 24_389.0
    let kappa: Float = 24_389.0 / 27.0
    return value3 > epsilon ? value3 : (116 * value - 16) / kappa
  }

  private static func histogramBin(
    _ value: Float, lowerBound: Float, upperBound: Float
  ) -> Int {
    let scale = Float(histogramBinCount - 1) / (upperBound - lowerBound)
    return min(
      max(Int((clamp(value, lowerBound, upperBound) - lowerBound) * scale), 0),
      histogramBinCount - 1)
  }

  private static func binValue(
    _ bin: Int, lowerBound: Float, upperBound: Float
  ) -> Float {
    lowerBound + Float(bin) * (upperBound - lowerBound) / Float(histogramBinCount - 1)
  }

  private static func clamp(_ value: Float, _ lowerBound: Float, _ upperBound: Float) -> Float {
    min(max(value, lowerBound), upperBound)
  }
}
