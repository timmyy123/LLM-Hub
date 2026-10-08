import NNC

#if canImport(C_ccv)
  import C_ccv
#elseif canImport(C_swiftpm_ccv)
  import C_swiftpm_ccv
#endif

public struct FaceExtractor<FloatType: TensorNumeric & BinaryFloatingPoint> {
  let filePath: String
  let imageEncoderVersion: ImageEncoderVersion
  public init(filePath: String, imageEncoderVersion: ImageEncoderVersion) {
    self.filePath = filePath
    self.imageEncoderVersion = imageEncoderVersion
  }
}

extension FaceExtractor {
  private func resize(_ x: [DynamicGraph.Tensor<FloatType>]) -> [(
    DynamicGraph.Tensor<FloatType>, Float
  )] {
    x.map {
      precondition($0.kind == .CPU)
      let shape = $0.shape
      guard shape[1] >= 640 || shape[2] >= 640 else {
        let height = (shape[1] / 32) * 32
        let width = (shape[2] / 32) * 32
        guard height != shape[1] || width != shape[2] else {
          return ($0.toGPU(0), 1)
        }
        return ($0[0..<shape[0], 0..<height, 0..<width, 0..<shape[3]].contiguous().toGPU(0), 1)
      }
      let zoomScale = 640 / Double(max(shape[1], shape[2]))
      let resizedHeight = Int((Double(shape[1]) * zoomScale).rounded())
      let resizedWidth = Int((Double(shape[2]) * zoomScale).rounded())
      let f32 = Tensor<Float>(
        from: $0.rawValue.reshaped(.HWC(shape[1], shape[2], shape[3])).contiguous())
      var b: UnsafeMutablePointer<ccv_dense_matrix_t>? = ccv_dense_matrix_new(
        Int32(resizedHeight), Int32(resizedWidth), Int32(CCV_C3 | CCV_32F), nil, 0)
      ccv_resample(
        UnsafeMutableRawPointer(f32.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self), &b,
        0, zoomScale, zoomScale, Int32(CCV_INTER_AREA | CCV_INTER_CUBIC))
      // Now this is properly resized, we can claim a few things:
      // We can shift the viewModel to the new one, and update the image to use the new one as well.
      let image = Tensor<FloatType>(
        from: Tensor<Float>(
          .CPU, format: .NHWC, shape: [1, resizedHeight, resizedWidth, shape[3]],
          unsafeMutablePointer: b!.pointee.data.f32, bindLifetimeOf: b!
        ).copied())
      let height = (resizedHeight / 32) * 32
      let width = (resizedWidth / 32) * 32
      guard height != resizedHeight || width != resizedWidth else {
        return ($0.graph.variable(image.toGPU(0)), Float(zoomScale))
      }
      return (
        $0.graph.variable(
          image[0..<shape[0], 0..<height, 0..<width, 0..<shape[3]].contiguous().toGPU(0)),
        Float(zoomScale)
      )
    }
  }

  private func ipAdapterRGB(
    x: DynamicGraph.Tensor<FloatType>, imageEncoderVersion: ImageEncoderVersion
  ) -> DynamicGraph.Tensor<FloatType> {
    let graph = x.graph
    // IP-Adapter requires image to be normalized to the format CLIP model requires.
    let mean = graph.variable(
      Tensor<FloatType>(
        [
          FloatType(2 * 0.48145466 - 1), FloatType(2 * 0.4578275 - 1),
          FloatType(2 * 0.40821073 - 1),
        ], .GPU(0), .NHWC(1, 1, 1, 3)))
    let invStd = graph.variable(
      Tensor<FloatType>(
        [
          FloatType(0.5 / 0.26862954), FloatType(0.5 / 0.26130258), FloatType(0.5 / 0.27577711),
        ],
        .GPU(0), .NHWC(1, 1, 1, 3)))
    let input = x.toGPU(0)
    let inputHeight = input.shape[1]
    let inputWidth = input.shape[2]
    precondition(input.shape[3] == 3)
    let imageSize: Int
    switch imageEncoderVersion {
    case .siglipL27_384:
      imageSize = 378
    case .siglip2L27_512:
      imageSize = 512
    case .clipL14_336, .eva02L14_336:
      imageSize = 336
    case .openClipH14:
      imageSize = 224
    }
    if inputHeight != imageSize || inputWidth != imageSize {
      return
        (Upsample(
          .bilinear, widthScale: Float(imageSize) / Float(inputWidth),
          heightScale: Float(imageSize) / Float(inputHeight))(input) - mean) .* invStd
    } else {
      return (input - mean) .* invStd
    }
  }

  private func faceIDRGB(
    x: DynamicGraph.Tensor<FloatType>
  ) -> DynamicGraph.Tensor<FloatType> {
    let input = x.toGPU(0)
    let inputHeight = input.shape[1]
    let inputWidth = input.shape[2]
    precondition(input.shape[3] == 3)
    let imageSize = 112
    if inputHeight != imageSize || inputWidth != imageSize {
      return Upsample(
        .bilinear, widthScale: Float(imageSize) / Float(inputWidth),
        heightScale: Float(imageSize) / Float(inputHeight))(input)
    } else {
      return input
    }
  }

  private func ipAdapterRGB(
    x: DynamicGraph.Tensor<FloatType>, imageEncoderVersion: ImageEncoderVersion,
    boundingBox: (Double, Double, Double, Double)
  ) -> DynamicGraph.Tensor<FloatType> {
    let shape = x.shape
    let centerX = (boundingBox.0 + boundingBox.2) / 2
    let centerY = (boundingBox.1 + boundingBox.3) / 2
    let width = boundingBox.2 - boundingBox.0
    let height = boundingBox.3 - boundingBox.1
    let rect = max(width, height) / 2
    let l0 = min(max(Int((centerX - rect).rounded()), 0), shape[2] - 1)
    let r0 = min(max(Int((centerX + rect).rounded()), 1), shape[2])
    let t0 = min(max(Int((centerY - rect).rounded()), 0), shape[1] - 1)
    let b0 = min(max(Int((centerY + rect).rounded()), 1), shape[1])
    return ipAdapterRGB(
      x: x[0..<shape[0], t0..<b0, l0..<r0, 0..<shape[3]].contiguous(),
      imageEncoderVersion: imageEncoderVersion)
  }

  public func extract(_ x: [DynamicGraph.Tensor<FloatType>]) -> [(
    DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>
  )] {
    precondition(x.count > 0)
    let graph = x[0].graph
    return graph.withNoGrad {
      let retinaFace = RetinaFace(batchSize: 1)
      let images = resize(x)
      retinaFace.maxConcurrency = .limit(4)
      retinaFace.compile(inputs: images[0].0)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        $0.read("retinaface", model: retinaFace, codec: [.ezm7, .q6p, .q8p, .jit, .externalData])
      }
      return zip(images, x).map {
        let outputs = retinaFace(inputs: $0.0.0).map { $0.as(of: FloatType.self).toCPU() }
        var maximum: (Int, Int, Int, Int)? = nil
        var maximumValue = -FloatType.greatestFiniteMagnitude
        for i in 0..<3 {
          let cls = outputs[i * 3].rawValue
          let shape = cls.shape
          for y in 0..<shape[1] {
            for x in 0..<shape[2] {
              let cls0 = cls[0, y, x, 0]
              if cls0 >= 0.5 && cls0 > maximumValue {
                maximumValue = cls0
                maximum = (i, y, x, 0)
              }
              let cls1 = cls[0, y, x, 1]
              if cls1 >= 0.5 && cls1 > maximumValue {
                maximumValue = cls1
                maximum = (i, y, x, 1)
              }
            }
          }
        }
        guard let maximum = maximum else {
          // If no face found, just treat the existing image as a whole for reference.
          return (
            ipAdapterRGB(x: $0.1, imageEncoderVersion: imageEncoderVersion), faceIDRGB(x: $0.1)
          )
        }
        let stride = [32, 16, 8][maximum.0]
        let anchor = (maximum.2 * stride, maximum.1 * stride)  // x, y
        let reg = outputs[maximum.0 * 3 + 1]
        let kps = outputs[maximum.0 * 3 + 2]
        let x1 =
          (Double(anchor.0) - Double(reg[0, maximum.1, maximum.2, maximum.3 * 4]) * Double(stride))
          / Double($0.0.1)
        let y1 =
          (Double(anchor.1) - Double(reg[0, maximum.1, maximum.2, maximum.3 * 4 + 1])
            * Double(stride)) / Double($0.0.1)
        let x2 =
          (Double(anchor.0) + Double(reg[0, maximum.1, maximum.2, maximum.3 * 4 + 2])
            * Double(stride)) / Double($0.0.1)
        let y2 =
          (Double(anchor.1) + Double(reg[0, maximum.1, maximum.2, maximum.3 * 4 + 3])
            * Double(stride)) / Double($0.0.1)
        let ipAdapter = ipAdapterRGB(
          x: $0.1, imageEncoderVersion: imageEncoderVersion, boundingBox: (x1, y1, x2, y2))
        var kp: [Point] = Array(repeating: Point(x: 0, y: 0), count: 5)
        for i in 0..<5 {
          let x =
            (Double(anchor.0) + Double(kps[0, maximum.1, maximum.2, maximum.3 * 10 + i * 2])
              * Double(stride)) / Double($0.0.1)
          let y =
            (Double(anchor.1) + Double(kps[0, maximum.1, maximum.2, maximum.3 * 10 + i * 2 + 1])
              * Double(stride)) / Double($0.0.1)
          kp[i] = Point(x: x, y: y)
        }
        let dst: [Point] = [
          Point(x: 38.2946, y: 51.6963), Point(x: 73.5318, y: 51.5014),
          Point(x: 56.0252, y: 71.7366), Point(x: 41.5493, y: 92.3655),
          Point(x: 70.7299, y: 92.2041),
        ]
        var similarityTransform = UmeyamaSimilarityTransform()
        similarityTransform.estimate(dst, kp)  // reverse the estimation, because our m matrix for ccv_perspective_transform is reversed too.
        let shape = $0.1.shape
        let f32 = Tensor<Float>(
          from: $0.1.rawValue.reshaped(.HWC(shape[1], shape[2], shape[3])).contiguous())
        let originalFaceSize = Int((112 * similarityTransform.scale).rounded())
        var b: UnsafeMutablePointer<ccv_dense_matrix_t>? = ccv_dense_matrix_new(
          Int32(shape[1]), Int32(shape[2]), Int32(CCV_C3 | CCV_32F), nil, 0)
        // Only rotate and translate, don't scale.
        let m00 = Float(similarityTransform.rotation[0][0])
        let m01 = Float(similarityTransform.rotation[0][1])
        let m10 = Float(similarityTransform.rotation[1][0])
        let m11 = Float(similarityTransform.rotation[1][1])
        let tx = Float(
          similarityTransform.translation.x + Double(shape[1]) * 0.5
            * similarityTransform.rotation[0][1] + Double(shape[2]) * 0.5
            * similarityTransform.rotation[0][0] - Double(shape[2]) * 0.5)
        let ty = Float(
          similarityTransform.translation.y + Double(shape[1]) * 0.5
            * similarityTransform.rotation[1][1] + Double(shape[2]) * 0.5
            * similarityTransform.rotation[1][0] - Double(shape[1]) * 0.5)
        ccv_perspective_transform(
          UnsafeMutableRawPointer(f32.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self), &b,
          0, m00, m01, tx, m10, m11, ty, 0, 0, 1)
        let faceID = faceIDRGB(
          x: $0.1.graph.variable(
            Tensor<FloatType>(
              from: Tensor<Float>(
                .CPU, format: .NHWC, shape: [1, shape[1], shape[2], shape[3]],
                unsafeMutablePointer: b!.pointee.data.f32, bindLifetimeOf: b!
              ).copied()))[
              0..<1, 0..<min(shape[1], originalFaceSize), 0..<min(shape[2], originalFaceSize),
              0..<shape[3]
            ].contiguous())
        ccv_matrix_free(b)
        return (ipAdapter, faceID)
      }
    }
  }
}

extension FaceExtractor {
  struct Point {
    var x: Double
    var y: Double
  }

  // Claude AI implemented this, not me. I think there might be degenerated cases when rank = 1.
  struct UmeyamaSimilarityTransform {
    var scale: Double = 1.0
    var rotation: [[Double]] = [[1, 0], [0, 1]]
    var translation: Point = Point(x: 0, y: 0)

    mutating func estimate(_ src: [Point], _ dst: [Point]) {
      precondition(src.count == dst.count && src.count >= 2)

      let n = Double(src.count)

      // Compute means
      let srcMean = meanPoint(src)
      let dstMean = meanPoint(dst)

      // Compute variance of src
      var srcVar: Double = 0
      for point in src {
        srcVar +=
          (point.x - srcMean.x) * (point.x - srcMean.x) + (point.y - srcMean.y)
          * (point.y - srcMean.y)
      }
      srcVar /= n

      // Compute covariance matrix
      var cov = [[Double]](repeating: [Double](repeating: 0, count: 2), count: 2)
      for i in 0..<src.count {
        let srcDiff = Point(x: src[i].x - srcMean.x, y: src[i].y - srcMean.y)
        let dstDiff = Point(x: dst[i].x - dstMean.x, y: dst[i].y - dstMean.y)
        cov[0][0] += srcDiff.x * dstDiff.x
        cov[0][1] += srcDiff.x * dstDiff.y
        cov[1][0] += srcDiff.y * dstDiff.x
        cov[1][1] += srcDiff.y * dstDiff.y
      }
      cov[0][0] /= n
      cov[0][1] /= n
      cov[1][0] /= n
      cov[1][1] /= n

      // Compute SVD of covariance matrix
      let (u, s, vt) = svd(cov)

      // Compute rotation matrix
      var d = [[1.0, 0.0], [0.0, 1.0]]
      if determinant(matrixMultiply(u, vt)) < 0 {
        d[1][1] = -1
      }
      self.rotation = matrixMultiply(matrixMultiply(u, d), vt)

      // Compute scale
      let srcVarPseudoinverse = srcVar > 1e-10 ? 1.0 / srcVar : 0.0
      self.scale = (s[0] + s[1]) * srcVarPseudoinverse

      // Compute translation
      self.translation.x =
        dstMean.x - self.scale * (self.rotation[0][0] * srcMean.x + self.rotation[0][1] * srcMean.y)
      self.translation.y =
        dstMean.y - self.scale * (self.rotation[1][0] * srcMean.x + self.rotation[1][1] * srcMean.y)
    }

    func apply(_ point: Point) -> Point {
      let x =
        self.scale * (self.rotation[0][0] * point.x + self.rotation[0][1] * point.y)
        + self.translation.x
      let y =
        self.scale * (self.rotation[1][0] * point.x + self.rotation[1][1] * point.y)
        + self.translation.y
      return Point(x: x, y: y)
    }

    // Helper functions

    private func meanPoint(_ points: [Point]) -> Point {
      let sum = points.reduce(Point(x: 0, y: 0)) { Point(x: $0.x + $1.x, y: $0.y + $1.y) }
      return Point(x: sum.x / Double(points.count), y: sum.y / Double(points.count))
    }

    private func determinant(_ matrix: [[Double]]) -> Double {
      return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    }

    private func matrixMultiply(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
      var result = [[Double]](repeating: [Double](repeating: 0, count: 2), count: 2)
      for i in 0..<2 {
        for j in 0..<2 {
          for k in 0..<2 {
            result[i][j] += a[i][k] * b[k][j]
          }
        }
      }
      return result
    }

    // Simplified SVD for 2x2 matrices
    private func svd(_ matrix: [[Double]]) -> ([[Double]], [Double], [[Double]]) {
      let a = matrix[0][0]
      let b = matrix[0][1]
      let c = matrix[1][0]
      let d = matrix[1][1]

      let e = (a + d) / 2
      let f = (a - d) / 2
      let g = (c + b) / 2
      let h = (c - b) / 2

      let q = sqrt(e * e + h * h)
      let r = sqrt(f * f + g * g)

      let s1 = q + r
      let s2 = abs(q - r)

      let a1 = atan2(g, f)
      let a2 = atan2(h, e)

      let phi = (a2 - a1) / 2
      let theta = (a2 + a1) / 2

      let u = [[cos(theta), -sin(theta)], [-sin(theta), -cos(theta)]]
      let v = [[cos(phi), sin(phi)], [sin(phi), -cos(phi)]]

      return (u, [s1, s2], v)
    }
  }
}
