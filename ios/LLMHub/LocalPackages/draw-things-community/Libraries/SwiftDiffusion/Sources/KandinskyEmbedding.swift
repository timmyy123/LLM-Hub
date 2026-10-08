import NNC

public struct KandinskyEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePath: String
  public init(filePath: String) {
    self.filePath = filePath
  }
}

extension KandinskyEmbedding {
  public func encode(
    textEncoding: DynamicGraph.Tensor<FloatType>, textEmbedding: DynamicGraph.Tensor<FloatType>,
    imageEmbedding: DynamicGraph.Tensor<FloatType>
  ) -> (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>) {
    let graph = textEncoding.graph
    let imageAndTextEmbedding = ImageAndTextEmbedding(batchSize: 2)
    imageAndTextEmbedding.compile(inputs: textEmbedding, textEncoding, imageEmbedding)
    graph.openStore(
      filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
    ) {
      $0.read(
        "image_and_text_embed", model: imageAndTextEmbedding,
        codec: [.q6p, .q8p, .ezm7, .externalData])
    }
    let outputs = imageAndTextEmbedding(inputs: textEmbedding, textEncoding, imageEmbedding).map {
      $0.as(of: FloatType.self)
    }
    return (outputs[0], outputs[1])
  }
}
