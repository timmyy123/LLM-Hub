import Diffusion
import NNC

#if canImport(CoreGraphics)
  import CoreGraphics
#endif

#if canImport(CoreML)
  import CoreML
#endif

#if canImport(NNCCoreMLConversion)
  import NNCCoreMLConversion
#endif

extension ControlModel {
  private static func decodeOutputScoresAndPoints(_ tpMap: DynamicGraph.Tensor<Float>)
    -> ([(Int, Int)], [Float], DynamicGraph.Tensor<Float>)
  {
    let ksize = 3
    let topk_n = 200
    let h = tpMap.shape[2]
    let w = tpMap.shape[3]
    let displacement = tpMap[0..<1, 1..<5, 0..<h, 0..<w]  // 1, 4, 256, 256
    let center = tpMap[0..<1, 0..<1, 0..<h, 0..<w]  // 1, 1, 256, 256
    var heat = Sigmoid()(center)
    let maxPool = MaxPool(
      filterSize: [ksize, ksize],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
    let hmax = maxPool(heat).toCPU()  // 1, 1, 256, 256
    heat = heat.copied().toCPU()

    //  keep = (hmax == heat).float()
    //  heat = heat * keep
    //  reshape heat to one dimension
    var reshapedHeat: [Float] = []
    for i in 0..<h {
      for j in 0..<w {
        if heat[0, 0, i, j] == hmax[0, 0, i, j] {
          heat[0, 0, i, j] = heat[0, 0, i, j]
        } else {
          heat[0, 0, i, j] = 0.0
        }
        reshapedHeat.append(heat[0, 0, i, j])
      }
    }
    // scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    // yy = torch.floor_divide(indices, w).unsqueeze(-1)
    // xx = torch.fmod(indices, w).unsqueeze(-1)
    // ptss = torch.cat((yy, xx),dim=-1)
    let sortedReshaped = reshapedHeat.enumerated().sorted { $0.element > $1.element }
    let indices = sortedReshaped.map { $0.offset }[0..<topk_n]
    let scores = sortedReshaped.map { $0.element }[0..<topk_n]
    var ptss: [(Int, Int)] = []
    for i in 0..<topk_n {
      let x = Float(indices[i] % w).rounded(.down)
      let y = Float(indices[i] / w).rounded(.down)
      ptss.append((Int(y), Int(x)))
    }

    return (ptss, Array(scores), displacement)
  }

  public static func mlsd(_ inputTensor: DynamicGraph.Tensor<FloatType>)
    -> DynamicGraph.Tensor<FloatType>?
  {
    #if !os(Linux)

      let graph = inputTensor.graph
      let height = inputTensor.shape[1]
      let width = inputTensor.shape[2]
      let scaled = Upsample(
        .bilinear, widthScale: Float(512) / Float(width),
        heightScale: Float(512) / Float(height))(inputTensor)
      assert(scaled.shape[1] == 512)
      assert(scaled.shape[2] == 512)
      var tensor = Tensor<FloatType>(.CPU, .NHWC(1, 512, 512, 4))
      tensor[0..<1, 0..<512, 0..<512, 0..<3] = scaled.rawValue.toCPU()
      for i in 0..<512 {
        for j in 0..<512 {
          tensor[0, i, j, 3] = FloatType(-0.9922)
        }
      }
      tensor = tensor.permuted(0, 3, 1, 2).copied()
      // outputs = model(batch_image)
      guard let url = Bundle.main.url(forResource: "mlsd_v1", withExtension: "mlmodelc") else {
        return nil
      }
      let configuration = MLModelConfiguration()
      guard let loadedModel = try? MLModel(contentsOf: url, configuration: configuration) else {
        return nil
      }

      var inputs = [MLDictionaryFeatureProvider]()
      inputs.append(
        try! MLDictionaryFeatureProvider(dictionary: [
          "x_1": MLMultiArray(
            MLShapedArray<Float>(
              Tensor<Float>(from: tensor)
            )
          )
        ])
      )
      let batch: MLArrayBatchProvider
      batch = MLArrayBatchProvider(array: inputs)
      let resultBatch = try! loadedModel.predictions(fromBatch: batch)
      let ouputs = resultBatch.features(at: 0).featureValue(for: "var_1034")!  // 1, 9, 256, 256
      let tensorOutputs = Tensor<Float>(from: Tensor(ouputs.shapedArrayValue(of: Float.self)!))

      // pts, pts_score, vmap = deccode_output_score_and_ptss(outputs, 200, 3)
      var (pts, ptsScore, vmap) = Self.decodeOutputScoresAndPoints(
        graph.variable(tensorOutputs).toGPU(0))
      vmap = vmap.toCPU()
      //vmap, 1, 4, 256, 256
      let start = vmap[0..<1, 0..<2, 0..<256, 0..<256]  // start = vmap[:, :, :2] 1, 2, 256, 256 NCHW
      let end = vmap[0..<1, 2..<4, 0..<256, 0..<256]  // end = vmap[:, :, 2:] 1, 2, 256, 256 NCHW
      var distMap = Tensor<Float>(.CPU, .NCHW(1, 1, 256, 256))
      for i in 0..<256 {
        for j in 0..<256 {
          var distance =
            (start[0, 0, i, j] - end[0, 0, i, j]) * (start[0, 0, i, j] - end[0, 0, i, j])
          distance +=
            (start[0, 1, i, j] - end[0, 1, i, j]) * (start[0, 1, i, j] - end[0, 1, i, j])
          distMap[0, 0, i, j] = distance.squareRoot()
        }
      }

      var segmentsList: [[Float]] = []
      let widthScale = Float(width) / Float(256)
      let heightScale = Float(height) / Float(256)
      for (center, score) in zip(pts, ptsScore) {
        let y = center.0
        let x = center.1
        let distance = distMap[0, 0, y, x]
        if score > 0.1 && distance > 20 {
          let xStart = Float(x) + vmap[0, 0, y, x]
          let yStart = Float(y) + vmap[0, 1, y, x]
          let xEnd = Float(x) + vmap[0, 2, y, x]
          let yEnd = Float(y) + vmap[0, 3, y, x]
          // 256 > 512
          segmentsList.append([
            xStart * widthScale, Float(height) - yStart * heightScale, xEnd * widthScale,
            Float(height) - yEnd * heightScale,
          ])
        }
      }

      guard
        let hintTensor = drawLine(
          segmentsList, imageSize: CGSize(width: CGFloat(width), height: CGFloat(height)),
          lineWidth: CGFloat(max(widthScale, heightScale)))
      else {
        return nil
      }
      return graph.variable(hintTensor)
    #else
      return inputTensor
    #endif
  }

  #if !os(Linux)
    private static func drawLine(_ points: [[Float]], imageSize: CGSize, lineWidth: CGFloat)
      -> Tensor<FloatType>?
    {
      let colorSpace = CGColorSpaceCreateDeviceRGB()
      let bytesPerPixel = 4
      let bitsPerComponent = 8
      let bytesPerRow = bytesPerPixel * Int(imageSize.width)

      guard
        let context = CGContext(
          data: nil,
          width: Int(imageSize.width),
          height: Int(imageSize.height),
          bitsPerComponent: bitsPerComponent,
          bytesPerRow: bytesPerRow,
          space: colorSpace,
          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
      else {
        print("Failed to create bitmap context.")
        return nil
      }

      // Set up drawing properties
      context.setLineWidth(lineWidth)
      context.setStrokeColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)

      // Draw lines connecting the points
      if !points.isEmpty {
        for point in points {
          let x0 = point[0]
          let y0 = point[1]
          let x1 = point[2]
          let y1 = point[3]

          context.move(to: CGPointMake(CGFloat(x0), CGFloat(y0)))
          context.addLine(to: CGPointMake(CGFloat(x1), CGFloat(y1)))

        }
        // Stroke the path
        context.strokePath()
      }

      // Access the pixel data directly from the bitmap context
      guard let data = context.data?.assumingMemoryBound(to: UInt8.self) else {
        return nil
      }

      let width = Int(imageSize.width)
      let height = Int(imageSize.height)

      var tensor = Tensor<FloatType>(.CPU, .NHWC(1, height, width, 3))

      for i in 0..<height {
        for j in 0..<width {
          let index = (width * i + j) * 4
          let alpha = data[index + 3]
          if alpha == 0 {
            tensor[0, i, j, 0] = 0
            tensor[0, i, j, 1] = 0
            tensor[0, i, j, 2] = 0
          } else {
            tensor[0, i, j, 0] = 1
            tensor[0, i, j, 1] = 1
            tensor[0, i, j, 2] = 1
          }
        }
      }

      return tensor
    }
  #endif

}
