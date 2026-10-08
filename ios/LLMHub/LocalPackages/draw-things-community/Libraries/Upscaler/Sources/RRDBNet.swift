import NNC

typealias ModelWeightMapper = () -> [String: [String]]

private func ResidualDenseBlock(prefix: [String], numberOfFeatures: Int, numberOfGrowChannels: Int)
  -> (
    ModelWeightMapper, Model
  )
{
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "rdb_conv1")
  let x1 = conv1(x).leakyReLU(negativeSlope: 0.2)
  let conv2 = (0..<2).map { k in
    Convolution(
      groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3], noBias: k != 0,
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
      name: "rdb_conv2_\(k)")
  }
  let x2 = (conv2[0](x) + conv2[1](x1)).leakyReLU(negativeSlope: 0.2)
  let conv3 = (0..<3).map { k in
    Convolution(
      groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3], noBias: k != 0,
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
      name: "rdb_conv3_\(k)")
  }
  let x3 = (conv3[0](x) + conv3[1](x1) + conv3[2](x2)).leakyReLU(negativeSlope: 0.2)
  let conv4 = (0..<4).map { k in
    Convolution(
      groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3], noBias: k != 0,
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
      name: "rdb_conv4_\(k)")
  }
  let x4 = (conv4[0](x) + conv4[1](x1) + conv4[2](x2) + conv4[3](x3)).leakyReLU(negativeSlope: 0.2)
  let conv5 = (0..<5).map { k in
    Convolution(
      groups: 1, filters: numberOfFeatures, filterSize: [3, 3], noBias: k != 0,
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
      name: "rdb_conv5_\(k)")
  }
  let x5 = conv5[0](x) + conv5[1](x1) + conv5[2](x2) + conv5[3](x3) + conv5[4](x4)
  let out = 0.2 * x5 + x
  let mapper: ModelWeightMapper = {
    var mapping = [String: [String]]()
    for prefix in prefix {
      mapping["\(prefix).conv1.weight"] = [conv1.weight.name]
      mapping["\(prefix).conv1.bias"] = [conv1.bias.name]
      mapping["\(prefix).conv2.weight"] = conv2.map { $0.weight.name }
      mapping["\(prefix).conv2.bias"] = [conv2[0].bias.name]
      mapping["\(prefix).conv3.weight"] = conv3.map { $0.weight.name }
      mapping["\(prefix).conv3.bias"] = [conv3[0].bias.name]
      mapping["\(prefix).conv4.weight"] = conv4.map { $0.weight.name }
      mapping["\(prefix).conv4.bias"] = [conv4[0].bias.name]
      mapping["\(prefix).conv5.weight"] = conv5.map { $0.weight.name }
      mapping["\(prefix).conv5.bias"] = [conv5[0].bias.name]
      // More formats.
      mapping["\(prefix).conv1.0.weight"] = [conv1.weight.name]
      mapping["\(prefix).conv1.0.bias"] = [conv1.bias.name]
      mapping["\(prefix).conv2.0.weight"] = conv2.map { $0.weight.name }
      mapping["\(prefix).conv2.0.bias"] = [conv2[0].bias.name]
      mapping["\(prefix).conv3.0.weight"] = conv3.map { $0.weight.name }
      mapping["\(prefix).conv3.0.bias"] = [conv3[0].bias.name]
      mapping["\(prefix).conv4.0.weight"] = conv4.map { $0.weight.name }
      mapping["\(prefix).conv4.0.bias"] = [conv4[0].bias.name]
      mapping["\(prefix).conv5.0.weight"] = conv5.map { $0.weight.name }
      mapping["\(prefix).conv5.0.bias"] = [conv5[0].bias.name]
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func RRDB(prefix: [String], numberOfFeatures: Int, numberOfGrowChannels: Int) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let (rdb1Mapper, rdb1) = ResidualDenseBlock(
    prefix: prefix.flatMap { ["\($0).rdb1", "\($0).RDB1"] }, numberOfFeatures: numberOfFeatures,
    numberOfGrowChannels: numberOfGrowChannels)
  var out = rdb1(x)
  let (rdb2Mapper, rdb2) = ResidualDenseBlock(
    prefix: prefix.flatMap { ["\($0).rdb2", "\($0).RDB2"] }, numberOfFeatures: numberOfFeatures,
    numberOfGrowChannels: numberOfGrowChannels)
  out = rdb2(out)
  let (rdb3Mapper, rdb3) = ResidualDenseBlock(
    prefix: prefix.flatMap { ["\($0).rdb3", "\($0).RDB3"] }, numberOfFeatures: numberOfFeatures,
    numberOfGrowChannels: numberOfGrowChannels)
  out = 0.2 * rdb3(out) + x
  let mapper: ModelWeightMapper = {
    var mapping = [String: [String]]()
    mapping.merge(rdb1Mapper()) { v, _ in v }
    mapping.merge(rdb2Mapper()) { v, _ in v }
    mapping.merge(rdb3Mapper()) { v, _ in v }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

public func RRDBNet(
  numberOfOutputChannels: Int, numberOfFeatures: Int, numberOfBlocks: Int, numberOfGrowChannels: Int
) -> (() -> [String: [String]], Model) {
  let x = Input()
  let convFirst = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "conv_first")
  var out = convFirst(x)
  let feat = out
  var mappers = [ModelWeightMapper]()
  for i in 0..<numberOfBlocks {
    let (rrdbMapper, rrdb) = RRDB(
      prefix: ["body.\(i)", "model.1.sub.\(i)"], numberOfFeatures: numberOfFeatures,
      numberOfGrowChannels: numberOfGrowChannels)
    out = rrdb(out)
    mappers.append(rrdbMapper)
  }
  let convBody = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "conv_body")
  out = convBody(out)
  out = feat + out
  let convUp1 = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "conv_up1")
  out = convUp1(Upsample(.nearest, widthScale: 2, heightScale: 2)(out)).leakyReLU(
    negativeSlope: 0.2)
  let convUp2 = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "conv_up2")
  out = convUp2(Upsample(.nearest, widthScale: 2, heightScale: 2)(out)).leakyReLU(
    negativeSlope: 0.2)
  let convHr = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "conv_hr")
  out = convHr(out).leakyReLU(negativeSlope: 0.2)
  let convLast = Convolution(
    groups: 1, filters: numberOfOutputChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "conv_last")
  out = convLast(out)
  let mapper: ModelWeightMapper = {
    var mapping = [String: [String]]()
    mapping["conv_first.weight"] = [convFirst.weight.name]
    mapping["conv_first.bias"] = [convFirst.bias.name]
    for mapper in mappers {
      mapping.merge(mapper()) { v, _ in v }
    }
    mapping["conv_body.weight"] = [convBody.weight.name]
    mapping["conv_body.bias"] = [convBody.bias.name]
    mapping["conv_up1.weight"] = [convUp1.weight.name]
    mapping["conv_up1.bias"] = [convUp1.bias.name]
    mapping["conv_up2.weight"] = [convUp2.weight.name]
    mapping["conv_up2.bias"] = [convUp2.bias.name]
    mapping["conv_hr.weight"] = [convHr.weight.name]
    mapping["conv_hr.bias"] = [convHr.bias.name]
    mapping["conv_last.weight"] = [convLast.weight.name]
    mapping["conv_last.bias"] = [convLast.bias.name]
    // More formats.
    mapping["model.0.weight"] = [convFirst.weight.name]
    mapping["model.0.bias"] = [convFirst.bias.name]
    mapping["model.1.sub.23.weight"] = [convBody.weight.name]
    mapping["model.1.sub.23.bias"] = [convBody.bias.name]
    mapping["model.3.weight"] = [convUp1.weight.name]
    mapping["model.3.bias"] = [convUp1.bias.name]
    mapping["model.6.weight"] = [convUp2.weight.name]
    mapping["model.6.bias"] = [convUp2.bias.name]
    mapping["model.8.weight"] = [convHr.weight.name]
    mapping["model.8.bias"] = [convHr.bias.name]
    mapping["model.10.weight"] = [convLast.weight.name]
    mapping["model.10.bias"] = [convLast.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}
