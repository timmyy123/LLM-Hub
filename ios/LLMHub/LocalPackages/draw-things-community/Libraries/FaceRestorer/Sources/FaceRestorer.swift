import CoreGraphics
import NNC
import Vision

#if canImport(C_ccv)
  import C_ccv
#elseif canImport(C_swiftpm_ccv)
  import C_swiftpm_ccv
#endif

public struct FaceRestorer<FloatType: TensorNumeric & BinaryFloatingPoint> {
  private let filePath: String
  private let parseFilePath: String
  public init(filePath: String, parseFilePath: String) {
    self.filePath = filePath
    self.parseFilePath = parseFilePath
  }

  public func enhance(
    _ x: DynamicGraph.Tensor<FloatType>, restoreFormer: Model? = nil,
    embedding: DynamicGraph.Tensor<FloatType>? = nil, parsenet: Model? = nil
  ) -> (DynamicGraph.Tensor<FloatType>, Model?, DynamicGraph.Tensor<FloatType>?, Model?) {
    let shape = x.shape
    let batchSize = shape[0]
    guard batchSize > 1 else {
      return enhanceOne(x, restoreFormer: restoreFormer, embedding: embedding, parsenet: parsenet)
    }
    var x = x
    var restoreFormer = restoreFormer
    var embedding = embedding
    var parsenet = parsenet
    for i in 0..<batchSize {
      let v = x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied()
      (
        x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]], restoreFormer, embedding, parsenet
      ) = enhanceOne(v, restoreFormer: restoreFormer, embedding: embedding, parsenet: parsenet)
    }
    return (x, restoreFormer, embedding, parsenet)
  }

  private func averagePoints(_ region: VNFaceLandmarkRegion2D, observation: VNFaceObservation)
    -> CGPoint?
  {
    guard region.normalizedPoints.count > 0 else { return nil }
    var avgPoint = CGPoint.zero
    for point in region.normalizedPoints {
      avgPoint.x += observation.boundingBox.minX + point.x * observation.boundingBox.width
      avgPoint.y += observation.boundingBox.minY + point.y * observation.boundingBox.height
    }
    avgPoint.x /= CGFloat(region.normalizedPoints.count)
    avgPoint.y /= CGFloat(region.normalizedPoints.count)
    return avgPoint
  }

  public func enhanceOne(
    _ x: DynamicGraph.Tensor<FloatType>, restoreFormer: Model?,
    embedding: DynamicGraph.Tensor<FloatType>?, parsenet: Model?
  ) -> (DynamicGraph.Tensor<FloatType>, Model?, DynamicGraph.Tensor<FloatType>?, Model?) {
    var image = x.toCPU().rawValue
    let cgImage = Self.image(from: image)
    let imageWidth = cgImage.width
    let imageHeight = cgImage.height
    let handler = VNImageRequestHandler(cgImage: cgImage)
    let request = VNDetectFaceLandmarksRequest()
    var restoreFormer = restoreFormer
    var embedding = embedding
    var parsenet = parsenet
    request.revision = VNDetectFaceLandmarksRequestRevision3
    let graph = x.graph
    do {
      try handler.perform([request])
      let observations = request.results ?? []
      guard observations.count > 0 else {
        return (x, restoreFormer, embedding, parsenet)
      }
      for observation in observations {
        // RestoreFormer is trained from FFHQ, this alignment code should match that dataset: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
        guard let landmarks = observation.landmarks, let leftEye = landmarks.leftEye,
          let rightEye = landmarks.rightEye, let mouth = landmarks.outerLips,
          let leftEyePoint = averagePoints(leftEye, observation: observation),
          let rightEyePoint = averagePoints(rightEye, observation: observation)
        else { continue }
        let eyeLeft = CGPoint(
          x: leftEyePoint.x * CGFloat(imageWidth), y: (1 - leftEyePoint.y) * CGFloat(imageHeight))
        let eyeRight = CGPoint(
          x: rightEyePoint.x * CGFloat(imageWidth), y: (1 - rightEyePoint.y) * CGFloat(imageHeight))
        let eyeAvg = CGPoint(x: (eyeLeft.x + eyeRight.x) * 0.5, y: (eyeLeft.y + eyeRight.y) * 0.5)
        let eyeToEye = CGPoint(x: eyeRight.x - eyeLeft.x, y: eyeRight.y - eyeLeft.y)
        let eyeDist: CGFloat = (eyeToEye.x * eyeToEye.x + eyeToEye.y * eyeToEye.y).squareRoot()
        guard eyeDist >= 5 else { continue }  // threshold on eye distance.
        let mouthLeft = CGPoint(
          x: (observation.boundingBox.minX + mouth.normalizedPoints[0].x
            * observation.boundingBox.width) * CGFloat(imageWidth),
          y: (1
            - (observation.boundingBox.minY + mouth.normalizedPoints[0].y
              * observation.boundingBox.height))
            * CGFloat(imageHeight))
        let mouthRight = CGPoint(
          x: (observation.boundingBox.minX + mouth.normalizedPoints[6].x
            * observation.boundingBox.width) * CGFloat(imageWidth),
          y: (1
            - (observation.boundingBox.minY + mouth.normalizedPoints[6].y
              * observation.boundingBox.height))
            * CGFloat(imageHeight))
        let mouthAvg = CGPoint(
          x: (mouthLeft.x + mouthRight.x) * 0.5, y: (mouthLeft.y + mouthRight.y) * 0.5)
        let eyeToMouth = CGPoint(x: mouthAvg.x - eyeAvg.x, y: mouthAvg.y - eyeAvg.y)
        // If eye to mouth is not perpendicular with eye to eye, this will try to average out and find the level vector.
        let xVec = CGPoint(x: eyeToEye.x + eyeToMouth.y, y: eyeToEye.y - eyeToMouth.x)
        let xHypot: CGFloat = (xVec.x * xVec.x + xVec.y * xVec.y).squareRoot()
        let xUnitVec = CGPoint(x: xVec.x / xHypot, y: xVec.y / xHypot)
        let faceSize =
          max(
            eyeDist * 2,
            (eyeToMouth.x * eyeToMouth.x + eyeToMouth.y * eyeToMouth.y).squareRoot() * 1.8) * 2
        let center = CGPoint(x: eyeAvg.x + eyeToMouth.x * 0.1, y: eyeAvg.y + eyeToMouth.y * 0.1)
        guard faceSize <= 512 else { continue }
        var croppedFaceTensor = Tensor<FloatType>(.CPU, .NHWC(1, 512, 512, 3))
        let m00 = Float(xUnitVec.x) * Float(faceSize) / 512
        let m01 = Float(-xUnitVec.y) * Float(faceSize) / 512
        let m02 = Float(center.x)
        let m10 = Float(xUnitVec.y) * Float(faceSize) / 512
        let m11 = Float(xUnitVec.x) * Float(faceSize) / 512
        let m12 = Float(center.y)
        croppedFaceTensor.withUnsafeMutableBytes {
          guard var croppedFacePtr = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
            return
          }
          image.withUnsafeBytes {
            guard let imagePtr = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
              return
            }
            for i in 0..<512 {
              let cy = Float(i) - 256
              let crx = cy * m01 + m02
              let cry = cy * m11 + m12
              for j in 0..<512 {
                let cx = Float(j) - 256
                let wx = cx * m00 + crx
                let wy = cx * m10 + cry
                let iwx = Int(wx)
                let iwy = Int(wy)
                let iwx1 = iwx + 1
                let iwy1 = iwy + 1
                let wrx = wx - Float(iwx)
                let wry = wy - Float(iwy)
                guard
                  iwx >= 0 && iwx < imageWidth && iwx1 >= 0 && iwx1 < imageWidth && iwy >= 0
                    && iwy < imageHeight && iwy1 >= 0 && iwy1 < imageHeight
                else {
                  croppedFacePtr[j * 3] = 0
                  croppedFacePtr[j * 3 + 1] = 0
                  croppedFacePtr[j * 3 + 2] = 0
                  continue
                }
                let r00 = FloatType((1 - wrx) * (1 - wry))
                let r01 = FloatType(wrx * (1 - wry))
                let r10 = FloatType((1 - wrx) * wry)
                let r11 = FloatType(wrx * wry)
                croppedFacePtr[j * 3] =
                  FloatType(
                    imagePtr[iwy * imageWidth * 3 + iwx * 3] * r00 + imagePtr[
                      iwy * imageWidth * 3 + iwx1 * 3] * r01)
                  + FloatType(
                    imagePtr[iwy1 * imageWidth * 3 + iwx * 3] * r10 + imagePtr[
                      iwy1 * imageWidth * 3 + iwx1 * 3] * r11)
                croppedFacePtr[j * 3 + 1] =
                  FloatType(
                    imagePtr[iwy * imageWidth * 3 + iwx * 3 + 1] * r00 + imagePtr[
                      iwy * imageWidth * 3 + iwx1 * 3 + 1] * r01)
                  + FloatType(
                    imagePtr[iwy1 * imageWidth * 3 + iwx * 3 + 1] * r10 + imagePtr[
                      iwy1 * imageWidth * 3 + iwx1 * 3 + 1] * r11)
                croppedFacePtr[j * 3 + 2] =
                  FloatType(
                    imagePtr[iwy * imageWidth * 3 + iwx * 3 + 2] * r00 + imagePtr[
                      iwy * imageWidth * 3 + iwx1 * 3 + 2] * r01)
                  + FloatType(
                    imagePtr[iwy1 * imageWidth * 3 + iwx * 3 + 2] * r10 + imagePtr[
                      iwy1 * imageWidth * 3 + iwx1 * 3 + 2] * r11)
              }
              croppedFacePtr = croppedFacePtr.advanced(by: 512 * 3)
            }
          }
        }
        var croppedFace = graph.variable(croppedFaceTensor.toGPU(0))
        croppedFace = croppedFace.permuted(0, 3, 1, 2).reshaped(.NCHW(1, 3, 512, 512)).copied()
        var restoredMask: UnsafeMutablePointer<ccv_dense_matrix_t>
        var restoredFace: Tensor<FloatType>
        (restoredFace, restoredMask, restoreFormer, embedding, parsenet) = restoreFace(
          croppedFace, restoreFormer: restoreFormer, embedding: embedding, parsenet: parsenet)
        if faceSize < 256 {
          // Resize the restoredFace and restoredMask down to only 2 larger. In this way, we can do bilinear properly to reconstruct the face.
          restoredFace = Upsample(
            .bilinear, widthScale: Float(faceSize) / 256, heightScale: Float(faceSize) / 256)(
              graph.variable(restoredFace.toGPU(0))
            ).rawValue.toCPU()
          var resizedMask: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
          ccv_resample(
            restoredMask, &resizedMask, 0,
            Double(restoredFace.shape[1]) / Double(restoredMask.pointee.rows),
            Double(restoredFace.shape[2]) / Double(restoredMask.pointee.cols), Int32(CCV_INTER_AREA)
          )
          ccv_matrix_free(restoredMask)
          restoredMask = resizedMask!
        }
        let restoredFaceSize = restoredFace.shape[1]
        precondition(restoredFaceSize == restoredFace.shape[2])
        let restoredBytesPerRow = ((restoredFaceSize + 3) / 4) * 4  // Align to 4 bytes for the blur mask.
        image.withUnsafeMutableBytes {
          guard var imagePtr = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
            return
          }
          restoredFace.withUnsafeBytes {
            guard let croppedFacePtr = $0.baseAddress?.assumingMemoryBound(to: FloatType.self)
            else { return }
            let croppedFaceMaskU8 = restoredMask.pointee.data.u8!
            let m00 = Float(xUnitVec.x) * Float(restoredFaceSize) / Float(faceSize)
            let m01 = Float(xUnitVec.y) * Float(restoredFaceSize) / Float(faceSize)
            let icx = Float(center.x)
            let m10 = Float(-xUnitVec.y) * Float(restoredFaceSize) / Float(faceSize)
            let m11 = Float(xUnitVec.x) * Float(restoredFaceSize) / Float(faceSize)
            let icy = Float(center.y)
            let c0 = Float(restoredFaceSize) * 0.5
            for i in 0..<imageHeight {
              let cy = Float(i) - icy
              let crx = cy * m01 + c0
              let cry = cy * m11 + c0
              for j in 0..<imageWidth {
                let cx = Float(j) - icx
                let wx = cx * m00 + crx
                let wy = cx * m10 + cry
                let iwx = Int(wx)
                let iwy = Int(wy)
                let iwx1 = iwx + 1
                let iwy1 = iwy + 1
                let wrx = wx - Float(iwx)
                let wry = wy - Float(iwy)
                guard
                  iwx >= 0 && iwx < restoredFaceSize && iwx1 >= 0 && iwx1 < restoredFaceSize
                    && iwy >= 0 && iwy < restoredFaceSize && iwy1 >= 0 && iwy1 < restoredFaceSize
                else {
                  continue
                }
                guard
                  croppedFaceMaskU8[iwy * restoredBytesPerRow + iwx] > 0
                    || croppedFaceMaskU8[iwy * restoredBytesPerRow + iwx1] > 0
                    || croppedFaceMaskU8[iwy1 * restoredBytesPerRow + iwx] > 0
                    || croppedFaceMaskU8[iwy1 * restoredBytesPerRow + iwx1] > 0
                else { continue }
                let a00 = FloatType(croppedFaceMaskU8[iwy * restoredBytesPerRow + iwx]) / 255
                let a01 = FloatType(croppedFaceMaskU8[iwy * restoredBytesPerRow + iwx1]) / 255
                let a10 = FloatType(croppedFaceMaskU8[iwy1 * restoredBytesPerRow + iwx]) / 255
                let a11 = FloatType(croppedFaceMaskU8[iwy1 * restoredBytesPerRow + iwx1]) / 255
                let r00 = FloatType((1 - wrx) * (1 - wry)) * a00
                let r01 = FloatType(wrx * (1 - wry)) * a01
                let r10 = FloatType((1 - wrx) * wry) * a10
                let r11 = FloatType(wrx * wry) * a11
                let _1ma = 1 - (r00 + r01 + r10 + r11)
                imagePtr[j * 3] =
                  imagePtr[j * 3] * _1ma
                  + FloatType(
                    croppedFacePtr[iwy * restoredFaceSize * 3 + iwx * 3] * r00 + croppedFacePtr[
                      iwy * restoredFaceSize * 3 + iwx1 * 3] * r01)
                  + FloatType(
                    croppedFacePtr[iwy1 * restoredFaceSize * 3 + iwx * 3] * r10 + croppedFacePtr[
                      iwy1 * restoredFaceSize * 3 + iwx1 * 3] * r11)
                imagePtr[j * 3 + 1] =
                  imagePtr[j * 3 + 1] * _1ma
                  + FloatType(
                    croppedFacePtr[iwy * restoredFaceSize * 3 + iwx * 3 + 1] * r00 + croppedFacePtr[
                      iwy * restoredFaceSize * 3 + iwx1 * 3 + 1] * r01)
                  + FloatType(
                    croppedFacePtr[iwy1 * restoredFaceSize * 3 + iwx * 3 + 1] * r10
                      + croppedFacePtr[iwy1 * restoredFaceSize * 3 + iwx1 * 3 + 1] * r11)
                imagePtr[j * 3 + 2] =
                  imagePtr[j * 3 + 2] * _1ma
                  + FloatType(
                    croppedFacePtr[iwy * restoredFaceSize * 3 + iwx * 3 + 2] * r00 + croppedFacePtr[
                      iwy * restoredFaceSize * 3 + iwx1 * 3 + 1] * r01)
                  + FloatType(
                    croppedFacePtr[iwy1 * restoredFaceSize * 3 + iwx * 3 + 2] * r10
                      + croppedFacePtr[iwy1 * restoredFaceSize * 3 + iwx1 * 3 + 2] * r11)
              }
              imagePtr = imagePtr.advanced(by: imageWidth * 3)
            }
          }
        }
        ccv_matrix_free(restoredMask)
      }
    } catch {
      // TODO: log error.
    }
    return (graph.variable(image.toGPU(0)), restoreFormer, embedding, parsenet)
  }
}

extension FaceRestorer {
  static func image(from tensor: Tensor<FloatType>) -> CGImage {
    let imageHeight = tensor.shape[1]
    let imageWidth = tensor.shape[2]
    let bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: imageWidth * imageHeight * 4)
    tensor.withUnsafeBytes {
      guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for i in 0..<imageHeight * imageWidth {
        bytes[i * 4] = UInt8(min(max(Int((fp16[i * 3] + 1) * 127.5), 0), 255))
        bytes[i * 4 + 1] = UInt8(min(max(Int((fp16[i * 3 + 1] + 1) * 127.5), 0), 255))
        bytes[i * 4 + 2] = UInt8(min(max(Int((fp16[i * 3 + 2] + 1) * 127.5), 0), 255))
        bytes[i * 4 + 3] = 255
      }
    }
    return CGImage(
      width: imageWidth, height: imageHeight, bitsPerComponent: 8, bitsPerPixel: 32,
      bytesPerRow: 4 * imageWidth, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo(
        rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue),
      provider: CGDataProvider(
        dataInfo: nil, data: bytes, size: imageWidth * imageHeight * 4,
        releaseData: { _, p, _ in
          p.deallocate()
        })!, decode: nil, shouldInterpolate: false,
      intent: CGColorRenderingIntent.defaultIntent)!
  }

  private func restoreFace(
    _ x: DynamicGraph.Tensor<FloatType>, restoreFormer: Model? = nil,
    embedding: DynamicGraph.Tensor<FloatType>? = nil, parsenet: Model? = nil
  ) -> (
    Tensor<FloatType>, UnsafeMutablePointer<ccv_dense_matrix_t>, Model,
    DynamicGraph.Tensor<FloatType>, Model
  ) {
    let shape = x.shape
    let width = shape[3]
    let height = shape[2]
    let graph = x.graph
    let hasRestoreFormer = (restoreFormer != nil)
    let hasEmbedding = (embedding != nil)
    let hasParseNet = (parsenet != nil)
    let restoreFormer =
      restoreFormer
      ?? RestoreFormer(
        nEmbed: 1024, embedDim: 256, ch: 64, chMult: [1, 2, 2, 4, 4, 8], zChannels: 256,
        numHeads: 8,
        numResBlocks: 2)
    let embedding = embedding ?? graph.variable(.GPU(0), .NC(1024, 256), of: FloatType.self)
    if !hasEmbedding {
      graph.openStore(filePath, flags: .readOnly) {
        $0.read("embedding", variable: embedding)
      }
    }
    let parsenet = parsenet ?? ParseNet()
    if !hasRestoreFormer {
      restoreFormer.maxConcurrency = .limit(4)
      restoreFormer.compile(inputs: x, embedding)
      graph.openStore(filePath, flags: .readOnly) {
        $0.read("restoreformer", model: restoreFormer)
      }
    }
    if !hasParseNet {
      parsenet.maxConcurrency = .limit(4)
      parsenet.compile(inputs: x)
      graph.openStore(parseFilePath, flags: .readOnly) {
        $0.read("parsenet", model: parsenet)
      }
    }
    precondition(x.kind == .GPU(0))
    let restored = restoreFormer(inputs: x, embedding)[0].as(of: FloatType.self)
    let mask = parsenet(inputs: restored)[0].as(of: FloatType.self)
    let index = Functional.argmax(mask, axis: 1).toCPU()
    let maskBeforeBlurred = ccv_dense_matrix_new(
      Int32(height), Int32(width), Int32(CCV_8U | CCV_C1), nil, 0)!
    let u8 = maskBeforeBlurred.pointee.data.u8!
    let maskMap: [UInt8] = [
      0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0, 0, 0,
    ]
    for y in 0..<height {
      for x in 0..<width {
        guard x >= 10 && x < width - 10 && y >= 10 && y < height - 10 else {
          u8[y * width + x] = 0
          continue
        }
        let i = Int(index[0, 0, y, x])
        u8[y * width + x] = maskMap[i]
      }
    }
    var blurred: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
    ccv_blur(maskBeforeBlurred, &blurred, Int32(CCV_8U | CCV_C1), 22)
    ccv_matrix_free(maskBeforeBlurred)
    let restoredTensor = restored.permuted(0, 2, 3, 1).reshaped(.NHWC(1, 512, 512, 3)).copied()
      .rawValue.toCPU()
    return (restoredTensor, blurred!, restoreFormer, embedding, parsenet)
  }
}
