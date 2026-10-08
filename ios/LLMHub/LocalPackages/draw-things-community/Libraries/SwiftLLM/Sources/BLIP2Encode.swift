import NNC

public struct BLIP2Encode<T: TensorNumeric & BinaryFloatingPoint> {
  let filePaths: [String]
  let usesFlashAttention: Bool
  public init(filePaths: [String], usesFlashAttention: Bool) {
    self.filePaths = filePaths
    self.usesFlashAttention = usesFlashAttention
  }
}

extension BLIP2Encode {
  public func encode(
    _ x: DynamicGraph.Tensor<T>, vit existingVit: Model? = nil,
    lnVision existingLnVision: Model? = nil, qformer existingQformer: Model? = nil,
    optProj existingOptProj: Model? = nil,
    queryTokens existingQueryTokens: DynamicGraph.Tensor<T>? = nil, returningModels: Bool = false
  ) -> (DynamicGraph.Tensor<T>, Model?, Model?, Model?, Model?, DynamicGraph.Tensor<T>?) {
    // We only support 364x364 for now with BLIP2 caption_coco_2.7b model.
    precondition(x.shape[1] == 364)
    precondition(x.shape[2] == 364)
    let graph = x.graph
    var outVit: Model? = nil
    let out = graph.withNoGrad {
      let vit =
        existingVit
        ?? EvaVisionTransformer(
          T.self, grid: 26, width: 1408, MLP: 6144, layers: 39, heads: 16, batchSize: 1,
          usesFlashAttention: usesFlashAttention)
      if existingVit == nil {
        vit.maxConcurrency = .limit(4)
        vit.compile(inputs: x)
        graph.openStore(
          filePaths[0], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[0])
        ) {
          $0.read("vit", model: vit, codec: [.jit, .q8p, .ezm7, .externalData])
        }
      }
      let out = vit(inputs: x)[0].as(of: T.self)
      if returningModels {
        outVit = vit
      }
      return out
    }
    return graph.withNoGrad {
      let lnVision = existingLnVision ?? LayerNorm(epsilon: 1e-5, axis: [1])
      if existingLnVision == nil {
        lnVision.compile(inputs: out)
        graph.openStore(
          filePaths[1], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[1])
        ) {
          $0.read("ln_vision", model: lnVision, codec: [.q8p, .ezm7, .externalData])
        }
      }
      var out = lnVision(inputs: out)[0].as(of: T.self)
      let queryTokens = existingQueryTokens ?? graph.variable(.GPU(0), .WC(32, 768), of: T.self)
      let qformer =
        existingQformer
        ?? BertModel(
          width: 768, queryEmbeddingLength: 32, imageEmbeddingLength: 26 * 26 + 1, MLP: 768 * 4,
          layers: 12, heads: 12, batchSize: 1, crossAttentionFreq: 2,
          usesFlashAttention: usesFlashAttention)
      if existingQformer == nil || existingQueryTokens == nil {
        qformer.maxConcurrency = .limit(4)
        qformer.compile(inputs: queryTokens, out)
        graph.openStore(
          filePaths[1], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[1])
        ) {
          $0.read("query_tokens", variable: queryTokens, codec: [.q8p, .ezm7, .externalData])
          $0.read("qformer", model: qformer, codec: [.q8p, .ezm7, .externalData])
        }
      }
      out = qformer(inputs: queryTokens, out)[0].as(of: T.self)
      let proj = existingOptProj ?? Dense(count: 2560)
      if existingOptProj == nil {
        proj.compile(inputs: out)
        graph.openStore(
          filePaths[1], flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: filePaths[1])
        ) {
          $0.read("opt_proj", model: proj, codec: [.q8p, .ezm7, .externalData])
        }
      }
      if returningModels {
        return (proj(inputs: out)[0].as(of: T.self), outVit, lnVision, qformer, proj, queryTokens)
      } else {
        return (proj(inputs: out)[0].as(of: T.self), nil, nil, nil, nil, nil)
      }
    }
  }
}
