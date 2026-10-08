import NNC

#if canImport(C_ccv)
  import C_ccv
#elseif canImport(C_swiftpm_ccv)
  import C_swiftpm_ccv
#endif

public struct MoondreamEncode<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public enum Version {
    case moondream1
    case moondream2_240306
    case moondream2_240520
  }
  let filePaths: [String]
  let usesFlashAttention: Bool
  let version: Version
  public init(filePaths: [String], usesFlashAttention: Bool, version: Version) {
    self.filePaths = filePaths
    self.usesFlashAttention = usesFlashAttention
    self.version = version
  }
}

extension MoondreamEncode {
  public func preprocess(_ input: Tensor<FloatType>) -> (Tensor<FloatType>, Tensor<FloatType>?) {
    var x = input
    let shape = input.shape
    precondition(shape.count == 4)
    precondition(shape[0] == 1)
    precondition(shape[3] == 3)
    switch version {
    case .moondream1, .moondream2_240306:
      if shape[1] != 378 || shape[2] != 378 {
        let f32 = Tensor<Float>(from: input.reshaped(.HWC(shape[1], shape[2], shape[3])))
        var b: UnsafeMutablePointer<ccv_dense_matrix_t>? = ccv_dense_matrix_new(
          378, 378, Int32(CCV_C3 | CCV_32F), nil, 0)
        ccv_resample(
          UnsafeMutableRawPointer(f32.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self),
          &b,
          0, 378 / Double(shape[1]), 378 / Double(shape[2]),
          Int32(CCV_INTER_AREA | CCV_INTER_CUBIC)
        )
        x = Tensor<FloatType>(
          from: Tensor<Float>(
            .CPU, format: .NHWC, shape: [1, 378, 378, 3],
            unsafeMutablePointer: b!.pointee.data.f32, bindLifetimeOf: b!
          ).copied())
        ccv_matrix_free(b)
      }
      return (x, nil)
    case .moondream2_240520:
      // First resize the whole image.
      if shape[1] != 378 || shape[2] != 378 {
        let f32 = Tensor<Float>(from: input.reshaped(.HWC(shape[1], shape[2], shape[3])))
        var b: UnsafeMutablePointer<ccv_dense_matrix_t>? = ccv_dense_matrix_new(
          378, 378, Int32(CCV_C3 | CCV_32F), nil, 0)
        ccv_resample(
          UnsafeMutableRawPointer(f32.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self),
          &b,
          0, 378 / Double(shape[1]), 378 / Double(shape[2]),
          Int32(CCV_INTER_AREA | CCV_INTER_CUBIC)
        )
        x = Tensor<FloatType>(
          from: Tensor<Float>(
            .CPU, format: .NHWC, shape: [1, 378, 378, 3],
            unsafeMutablePointer: b!.pointee.data.f32, bindLifetimeOf: b!
          ).copied())
        ccv_matrix_free(b)
      }
      // Then find the patch and resize, cut into patches.
      let supportedSizes: [(Int, Int)] = [(378, 378), (756, 756), (756, 378), (378, 756)]
      let aspectRatio = Double(shape[1]) / Double(shape[2])
      if let targetSize =
        (supportedSizes.min { a, b in
          let aApsectRatio = Double(a.0) / Double(a.1)
          let bAspectRatio = Double(b.0) / Double(b.1)
          if abs(aApsectRatio - aspectRatio) < abs(bAspectRatio - aspectRatio) {
            return true
          } else if abs(aApsectRatio - aspectRatio) > abs(bAspectRatio - aspectRatio) {
            return false
          }
          return abs(a.0 - shape[1]) + abs(a.1 - shape[2]) < abs(b.0 - shape[1])
            + abs(b.1 - shape[2])
        }), targetSize.0 != 378 || targetSize.1 != 378
      {
        let f32 = Tensor<Float>(from: input.reshaped(.HWC(shape[1], shape[2], shape[3])))
        var b: UnsafeMutablePointer<ccv_dense_matrix_t>? = ccv_dense_matrix_new(
          Int32(targetSize.0), Int32(targetSize.1), Int32(CCV_C3 | CCV_32F), nil, 0)
        ccv_resample(
          UnsafeMutableRawPointer(f32.cTensor).assumingMemoryBound(to: ccv_dense_matrix_t.self),
          &b,
          0, Double(targetSize.0) / Double(shape[1]), Double(targetSize.1) / Double(shape[2]),
          Int32(CCV_INTER_AREA | CCV_INTER_CUBIC)
        )
        let y = Tensor<FloatType>(
          from: Tensor<Float>(
            .CPU, format: .NHWC, shape: [1, targetSize.0, targetSize.1, 3],
            unsafeMutablePointer: b!.pointee.data.f32, bindLifetimeOf: b!
          ).copied())
        ccv_matrix_free(b)
        return (x, y)
      }
      return (x, nil)
    }
  }

  public func encode(
    _ x: (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>?),
    vit existingVit: Model? = nil,
    visionProj existingVisionProj: Model? = nil
  ) -> (DynamicGraph.Tensor<FloatType>, Model, Model) {
    // We only support 378x378.
    var input = x.0
    precondition(input.shape[1] == 378)
    precondition(input.shape[2] == 378)
    let graph = input.graph
    let patchShape = x.1?.shape
    return graph.withNoGrad {
      let vit: Model
      switch version {
      case .moondream1, .moondream2_240306:
        vit =
          existingVit
          ?? SigLIPVisionTransformer(
            FloatType.self, gridX: 27, gridY: 27, filterSize: 14, width: 1152, layers: 27,
            heads: 16, MLP: 4304, batchSize: 1, usesFlashAttention: usesFlashAttention,
            approximate: .none)
      case .moondream2_240520:
        let patchCount =
          patchShape.map {
            ($0[1] / 378) * ($0[2] / 378)
          } ?? 0
        vit =
          existingVit
          ?? SigLIPVisionTransformer(
            FloatType.self, gridX: 27, gridY: 27, filterSize: 14, width: 1152, layers: 27,
            heads: 16, MLP: 4304, batchSize: 1 + patchCount, usesFlashAttention: usesFlashAttention,
            approximate: .tanh)
        if let patch = x.1 {
          input = graph.variable(.GPU(0), .NHWC(1 + patchCount, 378, 378, 3))
          input[0..<1, 0..<378, 0..<378, 0..<3] = x.0
          var idx = 1
          for i in stride(from: 0, to: patch.shape[1], by: 378) {
            for j in stride(from: 0, to: patch.shape[2], by: 378) {
              input[idx..<(idx + 1), 0..<378, 0..<378, 0..<3] =
                patch[0..<1, i..<(i + 378), j..<(j + 378), 0..<3]
              idx += 1
            }
          }
        }
      }
      if existingVit == nil {
        vit.maxConcurrency = .limit(4)
        vit.compile(inputs: input)
        graph.openStore(
          filePaths[0], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[0])
        ) {
          $0.read("vit", model: vit, codec: [.jit, .q8p, .ezm7, .externalData])
        }
      }
      var out = vit(inputs: input)[0].as(of: FloatType.self)
      let visionProj: Model
      switch version {
      case .moondream1:
        visionProj =
          existingVisionProj
          ?? MoondreamVisionProjection(layers: 2, approximate: .none)
      case .moondream2_240306:
        visionProj =
          existingVisionProj
          ?? MoondreamVisionProjection(layers: 1, approximate: .none)
      case .moondream2_240520:
        visionProj =
          existingVisionProj
          ?? MoondreamVisionProjection(layers: 1, approximate: .tanh)
        if let patchShape = patchShape {
          // Need to stitch and then sample down
          var idx = 1
          let channels = out.shape[2]
          var sampleFeatures = graph.variable(
            .GPU(0), .HWC(patchShape[1] / 14, patchShape[2] / 14, channels), of: FloatType.self)
          for i in stride(from: 0, to: patchShape[1] / 14, by: 27) {
            for j in stride(from: 0, to: patchShape[2] / 14, by: 27) {
              sampleFeatures[i..<(i + 27), j..<(j + 27), 0..<channels] = out[
                idx..<(idx + 1), 0..<(27 * 27), 0..<channels
              ].reshaped(.HWC(27, 27, channels), strides: [27 * channels, channels, 1])
              idx += 1
            }
          }
          let patchFeatures = Upsample(
            .bilinear, widthScale: Float(27) / Float(patchShape[2] / 14),
            heightScale: Float(27) / Float(patchShape[1] / 14))(sampleFeatures).reshaped(
              .HWC(1, 27 * 27, channels), strides: [27 * 27 * channels, channels, 1])
          out = Functional.concat(axis: 2, out[0..<1, 0..<(27 * 27), 0..<channels], patchFeatures)
        } else {
          out = Functional.concat(axis: 2, out, out)
        }
      }
      if existingVisionProj == nil {
        visionProj.compile(inputs: out)
        graph.openStore(
          filePaths[1], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[1])
        ) {
          $0.read("vision_proj", model: visionProj, codec: [.q8p, .ezm7, .externalData])
        }
      }
      out = visionProj(inputs: out)[0].as(of: FloatType.self)
      out = out.reshaped(.WC(out.shape[1], out.shape[2]))
      return (out, vit, visionProj)
    }
  }
}
