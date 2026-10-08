import DiffusionMappings
import Foundation
import NNC

// Frozen full-sequence baseline for fixed-prefix parity tests.
@testable import Diffusion

private enum QwenImage2_1Configuration {
  static let width = 4_096
  static let heads = 32
  static let headDim = 128
  static let layers = 32
  static let intermediate = 12_288
  static let latentChannels = 64
  static let contextWidth = 4_096
  static let epsilon: Float = 1e-6
  static let ropeAxes = [16, 56, 56]
}

private struct QwenImage2_1Binding {
  let model: Model
  let key: String
  var bias = false
  var zeroCentered = false
  var parameter = false
  var chunk: Int? = nil
}

func QwenImage2_1UnsplitTimeEmbedding(_ timestep: Float) -> Tensor<Float> {
  var result = Tensor<Float>(.CPU, .NC(2, 256))
  for row in 0..<2 {
    for i in 0..<128 {
      let frequency = exp(-log(Float(10_000)) * Float(i) / 128)
      let phase = (row == 0 ? timestep * 1_000 : 0) * frequency
      result[row, i] = cos(phase)
      result[row, i + 128] = sin(phase)
    }
  }
  return result
}

private func QwenImage2_1Core<T: TensorNumeric>(
  _ type: T.Type, text: Int, height: Int, width: Int,
  layers: Int = QwenImage2_1Configuration.layers,
  device: Int = 0, referenceLength: Int, usesFlashAttention: Bool
) -> (Model, [QwenImage2_1Binding]) {
  let image = Input()
  let context = Input()
  let time = Input()
  let rotary = Input()
  let mask = Input()
  let pixels = height * width
  let indices = Input()
  let count = text + referenceLength + pixels
  let prefixLength = count - pixels
  let d = QwenImage2_1Configuration.width
  let heads = QwenImage2_1Configuration.heads
  let k = QwenImage2_1Configuration.headDim
  var bindings = [QwenImage2_1Binding]()
  func dense(_ key: String, _ output: Int, name: String, chunk: Int? = nil) -> Model {
    let model = Dense(
      count: output, noBias: true, name: name)
    bindings.append(.init(model: model, key: key, chunk: chunk))
    return model
  }
  func rms(_ key: String, axis: Int, name: String, zeroCentered: Bool = false) -> Model {
    let model = RMSNorm(
      epsilon: QwenImage2_1Configuration.epsilon, axis: [axis],
      name: name)
    bindings.append(.init(model: model, key: key, zeroCentered: zeroCentered))
    return model
  }
  func norm(_ x: Model.IO) -> Model.IO {
    LayerNorm(epsilon: QwenImage2_1Configuration.epsilon, axis: [1], elementwiseAffine: false)(x)
  }
  func swish(_ x: Model.IO) -> Model.IO { x.to(.Float32).swish().to(T.dataType) }
  func attentionNorm(_ x: Model.IO, key: String, name: String) -> Model.IO {
    if T.dataType == .Float32 { return rms(key, axis: 3, name: name)(x) }
    // Diffusers rounds the normalized vector BEFORE applying the learned gain.
    let gain = Parameter<T>(
      .GPU(device), .NHWC(1, 1, 1, k), name: name)
    bindings.append(.init(model: gain, key: key, parameter: true))
    let normalized = RMSNorm(
      epsilon: QwenImage2_1Configuration.epsilon, axis: [3], elementwiseAffine: false)(
        x.to(.Float32)
      ).to(T.dataType)
    return normalized .* gain()
  }
  func slice(_ x: Model.IO, start: Int, length: Int, columns: Int = QwenImage2_1Configuration.width)
    -> Model.IO
  {
    x.reshaped([length, columns], offset: [start, 0], strides: [columns, 1]).contiguous()
  }
  // Modulation has two rows: noisy target at row 0, invariant prefix at row 1.
  func modulate(_ x: Model.IO, _ table: Model.IO, columns: Int = QwenImage2_1Configuration.width)
    -> Model.IO
  {
    let prefix =
      slice(x, start: 0, length: prefixLength, columns: columns)
      .* slice(table, start: 1, length: 1, columns: columns)
    let target =
      slice(x, start: prefixLength, length: pixels, columns: columns)
      .* slice(table, start: 0, length: 1, columns: columns)
    return Functional.concat(axis: 0, prefix, target)
  }
  let textNorm = rms("txt_in.text_norm", axis: 1, name: "context_norm", zeroCentered: true)
  // Effective zero-centered weights are kept in FP32, as in the reference.
  let projectionType = T.dataType
  let normalizedText = textNorm(context.to(.Float32)).to(projectionType)
  let textIn = dense("txt_in.in_layer", d, name: "context_embedder_0")
  let textOut = dense("txt_in.out_layer", d, name: "context_embedder_1")
  let encodedText = textOut(GELU(approximate: .tanh)(textIn(normalizedText)))
  let imageIn = dense("img_in", d, name: "x_embedder")
  let encodedImage = imageIn(image.to(projectionType))
  var x = IndexSelect()(Functional.concat(axis: 0, encodedText, encodedImage), indices)
  x = x.to(.Float32)
  let time0 = dense("time_text_embed.timestep_embedder.linear_1", d, name: "t_embedder_0")
  let time2 = dense("time_text_embed.timestep_embedder.linear_2", d, name: "t_embedder_1")
  let temb = time2(time0(time.to(projectionType)).to(.Float32).swish().to(projectionType))
  let modulationInput = temb.to(.Float32).swish().to(projectionType)
  // Shared across all 32 blocks; each Dense loads its own rows of modulation.1.weight.
  let mods = (0..<4).map { (chunk: Int) in
    dense("modulation.1", d, name: "x_ada_ln_\(chunk)", chunk: chunk)(modulationInput)
  }
  let residualType: DataType = .Float32
  let scale1 = (1 + mods[0].to(.Float32)).to(residualType)
  let gate1 = mods[1].to(.Float32).tanh().to(residualType)
  let scale2 = (1 + mods[2].to(.Float32)).to(residualType)
  let gate2 = mods[3].to(.Float32).tanh().to(residualType)
  for layer in 0..<layers {
    let prefix = "transformer_blocks.\(layer)"
    let attentionInput = modulate(norm(x), scale1).to(T.dataType)
    let qProj = dense("\(prefix).attn.to_q", d, name: "x_q")
    let kProj = dense("\(prefix).attn.to_k", d, name: "x_k")
    let vProj = dense("\(prefix).attn.to_v", d, name: "x_v")
    let q = Functional.cmul(
      left: attentionNorm(
        qProj(attentionInput).reshaped(.NHWC(1, count, heads, k)),
        key: "\(prefix).attn.norm_q", name: "x_norm_q"
      ).to(.Float32),
      right: rotary
    ).to(T.dataType)
    let keys = Functional.cmul(
      left: attentionNorm(
        kProj(attentionInput).reshaped(.NHWC(1, count, heads, k)),
        key: "\(prefix).attn.norm_k", name: "x_norm_k"
      ).to(.Float32),
      right: rotary
    ).to(T.dataType)
    let values = vProj(attentionInput).reshaped(
      .NHWC(1, count, heads, k))
    let attended: Model.IO
    if !usesFlashAttention {
      let scores =
        Matmul(transposeB: (2, 3))(
          q.to(.Float32).transposed(1, 2), keys.to(.Float32).transposed(1, 2))
        * (1 / Float(k).squareRoot())
      let probabilities = (scores + mask.to(.Float32)).reshaped([heads * count, count]).softmax()
        .reshaped([
          1, heads, count, count,
        ])
      attended = Matmul()(probabilities, values.to(.Float32).transposed(1, 2)).transposed(1, 2)
        .reshaped([
          count, d,
        ]).to(T.dataType)
    } else {
      attended = ScaledDotProductAttention(
        scale: 1 / Float(k).squareRoot(), isCausal: false, hasAttentionMask: true)(
          q, keys, values, mask
        ).reshaped([count, d])
    }
    let attentionOut = dense("\(prefix).attn.to_out.0", d, name: "x_o")(attended).to(
      residualType)
    x = x + modulate(attentionOut, gate1)
    let ffnInput = modulate(norm(x), scale2).to(T.dataType)
    let up = dense(
      "\(prefix).img_mlp.proj", QwenImage2_1Configuration.intermediate, name: "ffn_up_proj")
    let gate = dense(
      "\(prefix).img_mlp.gate_layer", QwenImage2_1Configuration.intermediate, name: "ffn_gate_proj")
    let down = dense("\(prefix).img_mlp.out", d, name: "ffn_down_proj")
    let upValue = up(ffnInput)
    let gateValue = gate(ffnInput)
    let product = swish(gateValue) .* upValue
    let branch = down(product).to(residualType)
    x = x + modulate(branch, gate2)

  }
  let finalScale =
    (1
    + dense("norm_out.linear", d, name: "ada_ln_0")(temb.to(.Float32).swish().to(projectionType))
    .to(.Float32)).to(
      residualType)
  let out = dense("proj_out", 64, name: "linear")(modulate(norm(x), finalScale).to(projectionType))
  let target = slice(out, start: prefixLength, length: pixels, columns: 64)
  return (
    Model(
      [image, context, time, rotary, mask, indices],
      [target.to(.Float32)]), bindings
  )
}

// One shared graph is applied to each CFG / image batch row. The prefix uses t=0
// on every step; caching that invariant prefix is deliberately left for a later pass.
func QwenImage2_1Unsplit<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, height: Int, width: Int, textLength: Int,
  referenceLength: Int, usesFlashAttention: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let time = Input()
  let context = Input()
  let rotary = Input()
  let causalAttentionMask = Input()
  let indices = Input()
  let reference = Input()
  let count = textLength + referenceLength + height * width
  let (core, bindings) = QwenImage2_1Core(
    dataType, text: textLength, height: height, width: width,
    referenceLength: referenceLength, usesFlashAttention: usesFlashAttention)
  var outputs = [Model.IO]()
  for batch in 0..<batchSize {
    let target = x.reshaped(
      [height * width, 64], offset: [batch * height * width, 0], strides: [64, 1]
    ).contiguous()
    let image = referenceLength > 0 ? Functional.concat(axis: 0, reference, target) : target
    let text = context.reshaped(
      [textLength, 4096], offset: [batch * textLength, 0], strides: [4096, 1]
    ).contiguous()
    let rot = rotary.reshaped(
      [1, count, 1, 128], offset: [batch, 0, 0, 0], strides: [count * 128, 128, 128, 1]
    ).contiguous()
    let mask = causalAttentionMask.reshaped(
      [1, 1, count, count], offset: [batch, 0, 0, 0],
      strides: [count * count, count * count, count, 1]
    ).contiguous()
    outputs.append(
      core(image, text, time, rot, mask, indices).reshaped([1, height, width, 64]).to(
        dataType.dataType))
  }
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    for binding in bindings where binding.chunk == nil {
      mapping[binding.key + ".weight"] = [
        binding.parameter ? binding.model.parameters.name : binding.model.weight.name
      ]
    }
    mapping["modulation.1.weight"] = ModelWeightElement(
      bindings.filter { $0.chunk != nil }.sorted { $0.chunk! < $1.chunk! }.map {
        $0.model.weight.name
      })
    return mapping
  }
  let out = outputs.count == 1 ? outputs[0] : Concat(axis: 0)(outputs)
  return (
    mapper,
    Model(
      [x, time, context, rotary, causalAttentionMask, indices]
        + (referenceLength > 0 ? [reference] : []), [out])
  )
}
