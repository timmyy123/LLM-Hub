import NNC

func REBNCONV(outCh: Int, dirate: Int = 1, stride: Int = 1, dilation: [Int] = [], prefix: String)
  -> Model
{
  let x = Input()
  let convS1 = Convolution(
    groups: 1, filters: outCh, filterSize: [3, 3], dilation: dilation,
    hint: Hint(
      stride: [stride, stride],
      border: Hint.Border(begin: [dirate, dirate], end: [dirate, dirate])), format: .OIHW)
  var out = convS1(x)
  out = ReLU()(out)

  return Model([x], [out])
}

func RSU7(midCh: Int, outCh: Int, prefix: String) -> Model {
  let x = Input()

  let rebnconvin = REBNCONV(outCh: outCh, prefix: "\(prefix).rebnconvin")
  let hxin = rebnconvin(x)

  let rebnconv1 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv1")
  let hx1 = rebnconv1(hxin)
  var hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx1)

  let rebnconv2 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv2")
  let hx2 = rebnconv2(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx2)

  let rebnconv3 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv3")
  let hx3 = rebnconv3(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx3)

  let rebnconv4 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv4")
  let hx4 = rebnconv4(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx4)

  let rebnconv5 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv5")
  let hx5 = rebnconv5(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx5)

  let rebnconv6 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv6")
  let hx6 = rebnconv6(hx)

  let rebnconv7 = REBNCONV(
    outCh: midCh, dirate: 2, dilation: [2, 2], prefix: "\(prefix).rebnconv7")
  let hx7 = rebnconv7(hx6)

  let rebnconv6d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv6d")
  let hx6dIn = Functional.concat(axis: 3, hx7, hx6)
  let hx6d = rebnconv6d(hx6dIn)
  let hx6dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx6d)

  let rebnconv5d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv5d")
  let hx5dIn = Functional.concat(axis: 3, hx6dup, hx5)
  let hx5d = rebnconv5d(hx5dIn)
  let hx5dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx5d)

  let rebnconv4d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv4d")
  let hx4dIn = Functional.concat(axis: 3, hx5dup, hx4)
  let hx4d = rebnconv4d(hx4dIn)
  let hx4dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx4d)

  let rebnconv3d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv3d")
  let hx3dIn = Functional.concat(axis: 3, hx4dup, hx3)
  let hx3d = rebnconv3d(hx3dIn)
  let hx3dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx3d)

  let rebnconv2d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv2d")
  let hx2dIn = Functional.concat(axis: 3, hx3dup, hx2)
  let hx2d = rebnconv2d(hx2dIn)
  let hx2dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx2d)

  let rebnconv1d = REBNCONV(outCh: outCh, prefix: "\(prefix).rebnconv1d")
  let hx1dIn = Functional.concat(axis: 3, hx2dup, hx1)
  let hx1d = rebnconv1d(hx1dIn)

  let out = hx1d + hxin
  return Model([x], [out])
}

func RSU6(midCh: Int, outCh: Int, prefix: String) -> Model {
  let x = Input()

  let rebnconvin = REBNCONV(outCh: outCh, prefix: "\(prefix).rebnconvin")
  let hxin = rebnconvin(x)

  let rebnconv1 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv1")
  let hx1 = rebnconv1(hxin)
  var hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx1)

  let rebnconv2 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv2")
  let hx2 = rebnconv2(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx2)

  let rebnconv3 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv3")
  let hx3 = rebnconv3(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx3)

  let rebnconv4 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv4")
  let hx4 = rebnconv4(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx4)

  let rebnconv5 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv5")
  let hx5 = rebnconv5(hx)

  let rebnconv6 = REBNCONV(
    outCh: midCh, dirate: 2, dilation: [2, 2], prefix: "\(prefix).rebnconv6")
  let hx6 = rebnconv6(hx5)

  let rebnconv5d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv5d")
  let hx5dIn = Functional.concat(axis: 3, hx6, hx5)
  let hx5d = rebnconv5d(hx5dIn)
  let hx5dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx5d)

  let rebnconv4d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv4d")
  let hx4dIn = Functional.concat(axis: 3, hx5dup, hx4)
  let hx4d = rebnconv4d(hx4dIn)
  let hx4dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx4d)

  let rebnconv3d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv3d")
  let hx3dIn = Functional.concat(axis: 3, hx4dup, hx3)
  let hx3d = rebnconv3d(hx3dIn)
  let hx3dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx3d)

  let rebnconv2d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv2d")
  let hx2dIn = Functional.concat(axis: 3, hx3dup, hx2)
  let hx2d = rebnconv2d(hx2dIn)
  let hx2dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx2d)

  let rebnconv1d = REBNCONV(outCh: outCh, prefix: "\(prefix).rebnconv1d")
  let hx1dIn = Functional.concat(axis: 3, hx2dup, hx1)
  let hx1d = rebnconv1d(hx1dIn)

  let out = hx1d + hxin

  return Model([x], [out])
}

func RSU5(midCh: Int, outCh: Int, prefix: String) -> Model {
  let x = Input()

  let rebnconvin = REBNCONV(outCh: outCh, prefix: "\(prefix).rebnconvin")
  let hxin = rebnconvin(x)

  let rebnconv1 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv1")
  let hx1 = rebnconv1(hxin)
  var hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx1)

  let rebnconv2 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv2")
  let hx2 = rebnconv2(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx2)

  let rebnconv3 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv3")
  let hx3 = rebnconv3(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx3)

  let rebnconv4 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv4")
  let hx4 = rebnconv4(hx)

  let rebnconv5 = REBNCONV(
    outCh: midCh, dirate: 2, dilation: [2, 2], prefix: "\(prefix).rebnconv5")
  let hx5 = rebnconv5(hx4)

  let rebnconv4d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv4d")
  let hx4dIn = Functional.concat(axis: 3, hx5, hx4)
  let hx4d = rebnconv4d(hx4dIn)
  let hx4dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx4d)

  let rebnconv3d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv3d")
  let hx3dIn = Functional.concat(axis: 3, hx4dup, hx3)
  let hx3d = rebnconv3d(hx3dIn)
  let hx3dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx3d)

  let rebnconv2d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv2d")
  let hx2dIn = Functional.concat(axis: 3, hx3dup, hx2)
  let hx2d = rebnconv2d(hx2dIn)
  let hx2dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx2d)

  let rebnconv1d = REBNCONV(outCh: outCh, prefix: "\(prefix).rebnconv1d")
  let hx1dIn = Functional.concat(axis: 3, hx2dup, hx1)
  let hx1d = rebnconv1d(hx1dIn)

  let out = hx1d + hxin

  return Model([x], [out])
}

func RSU4(midCh: Int, outCh: Int, prefix: String) -> Model {
  let x = Input()

  let rebnconvin = REBNCONV(outCh: outCh, prefix: "\(prefix).rebnconvin")
  let hxin = rebnconvin(x)

  let rebnconv1 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv1")
  let hx1 = rebnconv1(hxin)
  var hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx1)

  let rebnconv2 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv2")
  let hx2 = rebnconv2(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx2)

  let rebnconv3 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv3")
  let hx3 = rebnconv3(hx)

  let rebnconv4 = REBNCONV(
    outCh: midCh, dirate: 2, dilation: [2, 2], prefix: "\(prefix).rebnconv4")
  let hx4 = rebnconv4(hx3)

  let rebnconv3d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv3d")
  let hx3dIn = Functional.concat(axis: 3, hx4, hx3)
  let hx3d = rebnconv3d(hx3dIn)
  let hx3dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx3d)

  let rebnconv2d = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv2d")
  let hx2dIn = Functional.concat(axis: 3, hx3dup, hx2)
  let hx2d = rebnconv2d(hx2dIn)
  let hx2dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx2d)

  let rebnconv1d = REBNCONV(outCh: outCh, prefix: "\(prefix).rebnconv1d")
  let hx1dIn = Functional.concat(axis: 3, hx2dup, hx1)
  let hx1d = rebnconv1d(hx1dIn)

  let out = hx1d + hxin

  return Model([x], [out])
}

func RSU4F(midCh: Int, outCh: Int, prefix: String) -> Model {
  let x = Input()

  let rebnconvin = REBNCONV(outCh: outCh, prefix: "\(prefix).rebnconvin")
  let hxin = rebnconvin(x)

  let rebnconv1 = REBNCONV(outCh: midCh, prefix: "\(prefix).rebnconv1")
  let hx1 = rebnconv1(hxin)

  let rebnconv2 = REBNCONV(
    outCh: midCh, dirate: 2, dilation: [2, 2], prefix: "\(prefix).rebnconv2")
  let hx2 = rebnconv2(hx1)

  let rebnconv3 = REBNCONV(
    outCh: midCh, dirate: 4, dilation: [4, 4], prefix: "\(prefix).rebnconv3")
  let hx3 = rebnconv3(hx2)

  let rebnconv4 = REBNCONV(
    outCh: midCh, dirate: 8, dilation: [8, 8], prefix: "\(prefix).rebnconv4")
  let hx4 = rebnconv4(hx3)

  let rebnconv3d = REBNCONV(
    outCh: midCh, dirate: 4, dilation: [4, 4], prefix: "\(prefix).rebnconv3d")
  let hx3dIn = Functional.concat(axis: 3, hx4, hx3)
  let hx3d = rebnconv3d(hx3dIn)

  let rebnconv2d = REBNCONV(
    outCh: midCh, dirate: 2, dilation: [2, 2], prefix: "\(prefix).rebnconv2d")
  let hx2dIn = Functional.concat(axis: 3, hx3d, hx2)
  let hx2d = rebnconv2d(hx2dIn)

  let rebnconv1d = REBNCONV(outCh: outCh, prefix: "\(prefix).rebnconv1d")
  let hx1dIn = Functional.concat(axis: 3, hx2d, hx1)
  let hx1d = rebnconv1d(hx1dIn)

  let out = hx1d + hxin
  return Model([x], [out])
}

public func ISNetDIS() -> Model {
  let x = Input()

  let convIn = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let hxin = convIn(x)

  let stage1 = RSU7(midCh: 32, outCh: 64, prefix: "stage1")
  let hx1 = stage1(hxin)
  var hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx1)

  let stage2 = RSU6(midCh: 32, outCh: 128, prefix: "stage2")
  let hx2 = stage2(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx2)

  let stage3 = RSU5(midCh: 64, outCh: 256, prefix: "stage3")
  let hx3 = stage3(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx3)

  let stage4 = RSU4(midCh: 128, outCh: 512, prefix: "stage4")
  let hx4 = stage4(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx4)

  let stage5 = RSU4F(midCh: 256, outCh: 512, prefix: "stage5")
  let hx5 = stage5(hx)
  hx = MaxPool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(hx5)

  let stage6 = RSU4F(midCh: 256, outCh: 512, prefix: "stage6")
  let hx6 = stage6(hx)
  let hx6up = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx6)

  let stage5d = RSU4F(midCh: 256, outCh: 512, prefix: "stage5d")
  let stage4d = RSU4(midCh: 128, outCh: 256, prefix: "stage4d")
  let stage3d = RSU5(midCh: 64, outCh: 128, prefix: "stage3d")
  let stage2d = RSU6(midCh: 32, outCh: 64, prefix: "stage2d")
  let stage1d = RSU7(midCh: 16, outCh: 64, prefix: "stage1d")
  let side1 = Convolution(
    groups: 1, filters: 1, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)

  let hx5d = stage5d(Functional.concat(axis: 3, hx6up, hx5))
  let hx5dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx5d)

  let hx4d = stage4d(Functional.concat(axis: 3, hx5dup, hx4))
  let hx4dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx4d)

  let hx3d = stage3d(Functional.concat(axis: 3, hx4dup, hx3))
  let hx3dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx3d)

  let hx2d = stage2d(Functional.concat(axis: 3, hx3dup, hx2))
  let hx2dup = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx2d)
  let hx1d = stage1d(Functional.concat(axis: 3, hx2dup, hx1))

  let hx1ds = side1(hx1d)
  let d1 = Upsample(.bilinear, widthScale: 2, heightScale: 2)(hx1ds)
  let out = Sigmoid()(d1)

  return Model([x], [out])
}
