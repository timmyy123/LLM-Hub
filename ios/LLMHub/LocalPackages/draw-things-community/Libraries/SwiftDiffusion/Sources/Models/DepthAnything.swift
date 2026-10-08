import NNC

func DinoSelfAttention(k: Int, h: Int, b: Int, t: Int, usesFlashAttention: Bool) -> Model {
  let x = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  if usesFlashAttention {
    let queries = toqueries(x).reshaped([b, t, h, k]).identity().identity()
    let keys = tokeys(x).reshaped([b, t, h, k]).identity()
    let values = tovalues(x).reshaped([b, t, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), flags: [.Float32], multiHeadOutputProjectionFused: true)
    let out = scaledDotProductAttention(queries, keys, values).reshaped([
      b * t, h * k,
    ])
    return Model([x], [out])
  } else {
    let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
      .transposed(1, 2)
    let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queries, keys)
    dot = dot.reshaped([b * h * t, t])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t, t])
    var out = dot * values
    out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
    let unifyheads = Dense(count: k * h)
    out = unifyheads(out)
    return Model([x], [out])
  }
}

func DinoResidualAttentionBlock<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, k: Int, h: Int, b: Int, t: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-6, axis: [2])
  let attention = DinoSelfAttention(
    k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention)
  let ls1 = Parameter<T>(.GPU(0), .NC(1, h * k), initBound: 1)
  var out = x.reshaped([b * t, h * k]) + (attention(ln1(x)) .* ls1)
  let ln2 = LayerNorm(epsilon: 1e-6, axis: [1])
  let fc = Dense(count: k * h * 4)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  let ls2 = Parameter<T>(.GPU(0), .NC(1, h * k), initBound: 1)
  out = out + (proj(gelu(fc(ln2(out)))) .* ls2)
  return Model([x], [out])
}

func DinoVisionTransformer<T: TensorNumeric>(
  _ dataType: T.Type,
  gridX: Int, gridY: Int, width: Int, layers: Int, heads: Int, batchSize: Int,
  intermediateLayers: [Int], usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let classEmbedding = Parameter<T>(.GPU(0), .HWC(1, 1, 1024), initBound: 1, name: "cls_token")
  let positionalEmbedding = Parameter<T>(
    .GPU(0), .HWC(1, 37 * 37 + 1, 1024), initBound: 1, name: "pos_embed")
  let clsPositionalEmbedding = positionalEmbedding.reshaped([1, 1, 1024])
  let patchPositionalEmbedding = positionalEmbedding.reshaped(
    [1, 37 * 37, 1024], offset: [0, 1, 0], strides: [(37 * 37 + 1) * 1024, 1024, 1]
  ).reshaped([1, 37, 37, 1024])
  let scaledPatchPositionalEmbedding = Upsample(
    .bilinear, widthScale: Float(gridX) / 37, heightScale: Float(gridY) / 37)(
      patchPositionalEmbedding
    ).reshaped([1, gridX * gridY, 1024])
  let posEmb = Functional.concat(axis: 1, clsPositionalEmbedding, scaledPatchPositionalEmbedding)

  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], hint: Hint(stride: [14, 14]), format: .OIHW)
  var out = conv1(x).reshaped([batchSize, gridX * gridY, width])
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + posEmb
  var outs = [Model.IO]()
  for i in 0..<layers {
    let block = DinoResidualAttentionBlock(
      T.self,
      prefix: "blocks.\(i)", k: width / heads, h: heads, b: batchSize,
      t: gridX * gridY + 1, usesFlashAttention: usesFlashAttention)
    out = block(out.reshaped([batchSize, gridX * gridY + 1, width]))
    if intermediateLayers.contains(i) {
      outs.append(out)
    }
  }
  let lnPost = LayerNorm(epsilon: 1e-6, axis: [1])
  outs = outs.map { lnPost($0) }
  return Model([x], outs)
}

func ResidualConvUnit(prefix: String) -> Model {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv1(x.ReLU())
  let conv2 = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = x + conv2(out.ReLU())
  return Model([x], [out])
}

func DepthHead(gridX: Int, gridY: Int, paddingFinalConvLayer: Bool) -> Model {
  let x0 = Input()
  let proj0 = Convolution(groups: 1, filters: 256, filterSize: [1, 1], format: .OIHW)
  let conv0 = ConvolutionTranspose(
    groups: 1, filters: 256, filterSize: [4, 4], hint: Hint(stride: [4, 4]), format: .OIHW)
  var out0 = conv0(proj0(x0))
  let x1 = Input()
  let proj1 = Convolution(groups: 1, filters: 512, filterSize: [1, 1], format: .OIHW)
  let conv1 = ConvolutionTranspose(
    groups: 1, filters: 512, filterSize: [2, 2], hint: Hint(stride: [2, 2]), format: .OIHW)
  var out1 = conv1(proj1(x1))
  let x2 = Input()
  let proj2 = Convolution(groups: 1, filters: 1024, filterSize: [1, 1], format: .OIHW)
  var out2 = proj2(x2)
  let x3 = Input()
  let proj3 = Convolution(groups: 1, filters: 1024, filterSize: [1, 1], format: .OIHW)
  let conv3 = Convolution(
    groups: 1, filters: 1024, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out3 = conv3(proj3(x3))

  let layer1_rn = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out0 = layer1_rn(out0)
  let layer2_rn = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out1 = layer2_rn(out1)
  let layer3_rn = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out2 = layer3_rn(out2)
  let layer4_rn = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out3 = layer4_rn(out3)

  let refinenet4 = ResidualConvUnit(prefix: "scratch.refinenet4.resConfUnit2")
  out3 = Upsample(
    .bilinear, widthScale: Float(gridX) / Float((gridX + 1) / 2),
    heightScale: Float(gridY) / Float((gridY + 1) / 2), alignCorners: true)(
      refinenet4(out3))
  let refinenet4OutConv = Convolution(groups: 1, filters: 256, filterSize: [1, 1], format: .OIHW)
  out3 = refinenet4OutConv(out3)
  let refinenet3Unit1 = ResidualConvUnit(
    prefix: "scratch.refinenet3.resConfUnit1")
  out2 = out3 + refinenet3Unit1(out2)
  let refinenet3Unit2 = ResidualConvUnit(
    prefix: "scratch.refinenet3.resConfUnit2")
  out2 = Upsample(.bilinear, widthScale: 2, heightScale: 2, alignCorners: true)(
    refinenet3Unit2(out2))
  let refinenet3OutConv = Convolution(groups: 1, filters: 256, filterSize: [1, 1], format: .OIHW)
  out2 = refinenet3OutConv(out2)
  let refinenet2Unit1 = ResidualConvUnit(
    prefix: "scratch.refinenet2.resConfUnit1")
  out1 = out2 + refinenet2Unit1(out1)
  let refinenet2Unit2 = ResidualConvUnit(
    prefix: "scratch.refinenet2.resConfUnit2")
  out1 = Upsample(.bilinear, widthScale: 2, heightScale: 2, alignCorners: true)(
    refinenet2Unit2(out1))
  let refinenet2OutConv = Convolution(groups: 1, filters: 256, filterSize: [1, 1], format: .OIHW)
  out1 = refinenet2OutConv(out1)
  let refinenet1Unit1 = ResidualConvUnit(
    prefix: "scratch.refinenet1.resConfUnit1")
  out0 = out1 + refinenet1Unit1(out0)
  let refinenet1Unit2 = ResidualConvUnit(
    prefix: "scratch.refinenet1.resConfUnit2")
  out0 = Upsample(.bilinear, widthScale: 2, heightScale: 2, alignCorners: true)(
    refinenet1Unit2(out0))
  let refinenet1OutConv = Convolution(groups: 1, filters: 256, filterSize: [1, 1], format: .OIHW)
  out0 = refinenet1OutConv(out0)

  let outputConv1 = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out0 = Upsample(
    .bilinear, widthScale: 14.0 / 8.0, heightScale: 14.0 / 8.0, alignCorners: true)(
      outputConv1(out0))

  let outputConv20 = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out0 = outputConv20(out0).ReLU()
  let outputConv22 = Convolution(
    groups: 1, filters: paddingFinalConvLayer ? 4 : 1, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  out0 = outputConv22(out0).ReLU()

  return Model([x0, x1, x2, x3], [out0])
}
