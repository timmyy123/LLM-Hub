import NNC

public enum UpscaleFactor: Int, Codable {
  case x2 = 2
  case x4 = 4
}

public struct RealESRGANer<FloatType: TensorNumeric & BinaryFloatingPoint> {
  private let filePath: String
  private let nativeScaleFactor: UpscaleFactor
  private let forcedScaleFactor: UpscaleFactor
  private let numberOfBlocks: Int
  private let isNHWCPreferred: Bool
  private let tileSize: Int
  public init(
    filePath: String, nativeScaleFactor: UpscaleFactor, forcedScaleFactor: UpscaleFactor,
    numberOfBlocks: Int, isNHWCPreferred: Bool, tileSize: Int
  ) {
    self.filePath = filePath
    self.nativeScaleFactor = nativeScaleFactor
    self.forcedScaleFactor = forcedScaleFactor
    self.numberOfBlocks = numberOfBlocks
    self.isNHWCPreferred = isNHWCPreferred
    self.tileSize = tileSize
  }

  public static func downscale(
    _ x: DynamicGraph.Tensor<FloatType>, scaleFactor: Int, tileSize: Int = 1024
  ) -> DynamicGraph.Tensor<FloatType> {
    guard scaleFactor > 1 else { return x.toGPU(0) }
    let shape = x.shape
    let batchSize = shape[0]
    let height = shape[1]
    let width = shape[2]
    let channels = shape[3]
    let graph = x.graph
    precondition(height % scaleFactor == 0)
    precondition(width % scaleFactor == 0)
    let yTiles = (height + tileSize - 1) / tileSize
    let xTiles = (width + tileSize - 1) / tileSize
    guard yTiles > 1 || xTiles > 1 else {
      guard batchSize > 1 else {
        return Functional.averagePool(
          x.toGPU(0), filterSize: [scaleFactor, scaleFactor],
          hint: Hint(stride: [scaleFactor, scaleFactor]))
      }
      var output = graph.variable(
        .GPU(0), .NHWC(batchSize, height / scaleFactor, width / scaleFactor, channels),
        of: FloatType.self)
      for i in 0..<batchSize {
        let z = x[i..<(i + 1), 0..<height, 0..<width, 0..<channels].toGPU(0)
        output[i..<(i + 1), 0..<(height / scaleFactor), 0..<(width / scaleFactor), 0..<channels] =
          Functional.averagePool(
            z, filterSize: [scaleFactor, scaleFactor],
            hint: Hint(stride: [scaleFactor, scaleFactor]))
      }
      return output
    }
    let input = x
    var output = graph.variable(
      .GPU(0), .NHWC(batchSize, height / scaleFactor, width / scaleFactor, channels),
      of: FloatType.self)
    for i in 0..<batchSize {
      for y in 0..<yTiles {
        let yOfs = y * tileSize
        let inputStartY = yOfs
        let inputEndY = min(yOfs + tileSize, height)
        let outputStartY = inputStartY / scaleFactor
        let outputEndY = inputEndY / scaleFactor
        let outputStartYTile = 0
        let outputEndYTile = outputStartYTile + (inputEndY - inputStartY) / scaleFactor
        for x in 0..<xTiles {
          let xOfs = x * tileSize
          let inputStartX = xOfs
          let inputEndX = min(xOfs + tileSize, width)
          let outputStartX = inputStartX / scaleFactor
          let outputEndX = inputEndX / scaleFactor
          let outputStartXTile = 0
          let outputEndXTile = outputStartXTile + (inputEndX - inputStartX) / scaleFactor
          let z = input[i..<(i + 1), inputStartY..<inputEndY, inputStartX..<inputEndX, 0..<channels]
            .copied().toGPU(0)
          let outputTile = Functional.averagePool(
            z, filterSize: [scaleFactor, scaleFactor],
            hint: Hint(stride: [scaleFactor, scaleFactor]))
          output[i..<(i + 1), outputStartY..<outputEndY, outputStartX..<outputEndX, 0..<channels] =
            outputTile[
              0..<1, outputStartYTile..<outputEndYTile, outputStartXTile..<outputEndXTile,
              0..<channels]
        }
      }
    }
    return output
  }

  public func upscale(
    _ x: DynamicGraph.Tensor<FloatType>, rrdbnet: Model? = nil, overlapping: Int = 16
  ) -> (DynamicGraph.Tensor<FloatType>, Model) {
    let shape = x.shape
    let batchSize = shape[0]
    var height = shape[1]
    var width = shape[2]
    let graph = x.graph
    let hasRRDBNet = (rrdbnet != nil)
    let rrdbnet =
      rrdbnet
      ?? RRDBNet(
        numberOfOutputChannels: 3, numberOfFeatures: 64, numberOfBlocks: numberOfBlocks,
        numberOfGrowChannels: 32
      ).1
    var z: DynamicGraph.Tensor<FloatType>
    if nativeScaleFactor == .x2 {  // Need to do pixel unshuffle.
      if isNHWCPreferred {
        z = x.reshaped(format: .NHWC, shape: [batchSize, height / 2, 2, width / 2, 2, 3]).permuted(
          0, 1, 3, 5, 2, 4
        ).copied().reshaped(.NHWC(batchSize, height / 2, width / 2, 12))
      } else {
        z = x.reshaped(format: .NHWC, shape: [batchSize, height / 2, 2, width / 2, 2, 3]).permuted(
          0, 5, 2, 4, 1, 3
        ).copied().reshaped(.NCHW(batchSize, 12, height / 2, width / 2))
      }
      height = height / 2
      width = width / 2
    } else {
      if isNHWCPreferred {
        z = x
      } else {
        z = x.permuted(0, 3, 1, 2).copied().reshaped(.NCHW(batchSize, 3, height, width))
      }
    }
    z = (0.5 * (z + 1)).clamped(0...1)
    let yTiles = (height + tileSize - 1) / tileSize
    let xTiles = (width + tileSize - 1) / tileSize
    let input = z
    let inputShape = input.shape
    if !hasRRDBNet {
      // Load model from parameters.
      if isNHWCPreferred {
        z = graph.variable(
          .GPU(0),
          .NHWC(
            1, min(tileSize + 2 * overlapping, height),
            min(tileSize + 2 * overlapping, width), inputShape[3]))
      } else {
        z = graph.variable(
          .GPU(0),
          .NCHW(
            1, inputShape[1], min(tileSize + 2 * overlapping, height),
            min(tileSize + 2 * overlapping, width)))
      }
      rrdbnet.maxConcurrency = .limit(6)
      rrdbnet.compile(inputs: z)
      let legacyMapping: [String: Int] = [
        "conv_first": 0,
        "conv_body": numberOfBlocks * 15 + 1,
        "conv_up1": numberOfBlocks * 15 + 2,
        "conv_up2": numberOfBlocks * 15 + 3,
        "conv_hr": numberOfBlocks * 15 + 4,
        "conv_last": numberOfBlocks * 15 + 5,
      ]
      graph.openStore(filePath, flags: .readOnly) { store in
        let reader:
          (String, DataType, TensorFormat, TensorShape) -> DynamicGraph.Store.ModelReaderResult = {
            name, _, _, _ in
            var components = name.components(separatedBy: "-")
            guard components.count >= 4, let index = Int(components[2]) else {
              return .continue(name)
            }
            guard let legacyName = legacyMapping[components[1]] else {
              assert(components[1].hasPrefix("rdb_"))
              if components[1].hasPrefix("rdb_conv1") {
                components.remove(at: 1)
                components[1] = "\(1 + index * 5)"
                return .continue(components.joined(separator: "-"))
              } else if components[1].hasPrefix("rdb_conv2") {
                guard
                  let lastIndex =
                    (components[1].components(separatedBy: "_").last.flatMap { Int($0) })
                else { return .continue(name) }
                components.remove(at: 1)
                components[1] = "\(2 + index * 5)"
                let newName = components.joined(separator: "-")
                if components[2] == "1]" {
                  return .continue(newName)
                }
                guard let tensor = (store.read(newName).map { Tensor<FloatType>(from: $0).toCPU() })
                else { return .continue(name) }
                let shape = tensor.shape
                if lastIndex == 0 {
                  return .final(tensor[0..<shape[0], 0..<64, 0..<shape[2], 0..<shape[3]].copied())
                } else {
                  return .final(
                    tensor[
                      0..<shape[0], (64 + (lastIndex - 1) * 32)..<(64 + lastIndex * 32),
                      0..<shape[2], 0..<shape[3]
                    ].copied())
                }
              } else if components[1].hasPrefix("rdb_conv3") {
                guard
                  let lastIndex =
                    (components[1].components(separatedBy: "_").last.flatMap { Int($0) })
                else { return .continue(name) }
                components.remove(at: 1)
                components[1] = "\(3 + index * 5)"
                let newName = components.joined(separator: "-")
                if components[2] == "1]" {
                  return .continue(newName)
                }
                guard let tensor = (store.read(newName).map { Tensor<FloatType>(from: $0).toCPU() })
                else { return .continue(name) }
                let shape = tensor.shape
                if lastIndex == 0 {
                  return .final(tensor[0..<shape[0], 0..<64, 0..<shape[2], 0..<shape[3]].copied())
                } else {
                  return .final(
                    tensor[
                      0..<shape[0], (64 + (lastIndex - 1) * 32)..<(64 + lastIndex * 32),
                      0..<shape[2], 0..<shape[3]
                    ].copied())
                }
              } else if components[1].hasPrefix("rdb_conv4") {
                guard
                  let lastIndex =
                    (components[1].components(separatedBy: "_").last.flatMap { Int($0) })
                else { return .continue(name) }
                components.remove(at: 1)
                components[1] = "\(4 + index * 5)"
                let newName = components.joined(separator: "-")
                if components[2] == "1]" {
                  return .continue(newName)
                }
                guard let tensor = (store.read(newName).map { Tensor<FloatType>(from: $0).toCPU() })
                else { return .continue(name) }
                let shape = tensor.shape
                if lastIndex == 0 {
                  return .final(tensor[0..<shape[0], 0..<64, 0..<shape[2], 0..<shape[3]].copied())
                } else {
                  return .final(
                    tensor[
                      0..<shape[0], (64 + (lastIndex - 1) * 32)..<(64 + lastIndex * 32),
                      0..<shape[2], 0..<shape[3]
                    ].copied())
                }
              } else if components[1].hasPrefix("rdb_conv5") {
                guard
                  let lastIndex =
                    (components[1].components(separatedBy: "_").last.flatMap { Int($0) })
                else { return .continue(name) }
                components.remove(at: 1)
                components[1] = "\(5 + index * 5)"
                let newName = components.joined(separator: "-")
                if components[2] == "1]" {
                  return .continue(newName)
                }
                guard let tensor = (store.read(newName).map { Tensor<FloatType>(from: $0).toCPU() })
                else { return .continue(name) }
                let shape = tensor.shape
                if lastIndex == 0 {
                  return .final(tensor[0..<shape[0], 0..<64, 0..<shape[2], 0..<shape[3]].copied())
                } else {
                  return .final(
                    tensor[
                      0..<shape[0], (64 + (lastIndex - 1) * 32)..<(64 + lastIndex * 32),
                      0..<shape[2], 0..<shape[3]
                    ].copied())
                }
              } else {
                return .continue(name)
              }
            }
            components.remove(at: 1)
            components[1] = "\(legacyName)"
            return .continue(components.joined(separator: "-"))
          }
        switch nativeScaleFactor {
        case .x4:
          if numberOfBlocks == 6 {
            if store.read(like: "__realesrgan_x4plus_6b__[t-0-0]") != nil {
              store.read("realesrgan_x4plus_6b", model: rrdbnet, reader: reader)
            } else {  // It is not in legacy form, read directly.
              store.read("realesrgan_x4plus_6b", model: rrdbnet)
            }
          } else {
            if store.read(like: "__realesrgan_x4plus__[t-0-0]") != nil {
              store.read("realesrgan_x4plus", model: rrdbnet, reader: reader)
            } else {  // It is not in legacy form, read directly.
              store.read("realesrgan_x4plus", model: rrdbnet)
            }
          }
        case .x2:
          if store.read(like: "__realesrgan_x2plus__[t-0-0]") != nil {
            store.read("realesrgan_x2plus", model: rrdbnet, reader: reader)
          } else {  // It is not in legacy form, read directly.
            store.read("realesrgan_x2plus", model: rrdbnet)
          }
        }
      }
    }
    let upscaleFactor: Int
    switch forcedScaleFactor {
    case .x2:
      switch nativeScaleFactor {
      case .x2:
        upscaleFactor = 4
      case .x4:
        upscaleFactor = 2
      }
    case .x4:
      switch nativeScaleFactor {
      case .x2:
        upscaleFactor = 8
      case .x4:
        upscaleFactor = 4
      }
    }
    guard max(width, height) > tileSize + 2 * overlapping else {
      guard batchSize > 1 else {
        var result = rrdbnet(inputs: input)[0].as(of: FloatType.self)
        if upscaleFactor != 4 {
          result = Upsample(
            .bilinear, widthScale: Float(upscaleFactor) / 4, heightScale: Float(upscaleFactor) / 4)(
              result)
        }
        if isNHWCPreferred {
          return (
            (result * 2 - 1)
              .reshaped(.NHWC(1, height * upscaleFactor, width * upscaleFactor, 3)).copied()
              .toCPU(),
            rrdbnet
          )
        } else {
          return (
            (result * 2 - 1).permuted(0, 2, 3, 1)
              .reshaped(.NHWC(1, height * upscaleFactor, width * upscaleFactor, 3)).copied()
              .toCPU(),
            rrdbnet
          )
        }
      }
      var output = graph.variable(
        .CPU, .NHWC(batchSize, height * upscaleFactor, width * upscaleFactor, 3), of: FloatType.self
      )
      for i in 0..<batchSize {
        let z = input[i..<(i + 1), 0..<inputShape[1], 0..<inputShape[2], 0..<inputShape[3]].copied()
        var result = rrdbnet(inputs: z)[0].as(of: FloatType.self)
        if upscaleFactor != 4 {
          result = Upsample(
            .bilinear, widthScale: Float(upscaleFactor) / 4, heightScale: Float(upscaleFactor) / 4)(
              result)
        }
        if isNHWCPreferred {
          output[i..<(i + 1), 0..<(height * upscaleFactor), 0..<(width * upscaleFactor), 0..<3] =
            (result * 2 - 1)
            .reshaped(.NHWC(1, height * upscaleFactor, width * upscaleFactor, 3)).copied().toCPU()
        } else {
          output[i..<(i + 1), 0..<(height * upscaleFactor), 0..<(width * upscaleFactor), 0..<3] =
            (result * 2 - 1).permuted(0, 2, 3, 1)
            .reshaped(.NHWC(1, height * upscaleFactor, width * upscaleFactor, 3)).copied().toCPU()
        }
      }
      return (output, rrdbnet)
    }
    var output = graph.variable(
      .CPU, .NHWC(batchSize, height * upscaleFactor, width * upscaleFactor, 3), of: FloatType.self)
    if width > tileSize + 2 * overlapping && height > tileSize + 2 * overlapping {
      for i in 0..<batchSize {
        for y in 0..<yTiles {
          let yOfs = y * tileSize
          let inputStartY = yOfs
          let inputEndY = min(yOfs + tileSize, height)
          var inputStartYPad = inputStartY - overlapping
          var inputEndYPad = inputEndY + overlapping
          if inputStartYPad < 0 {
            inputStartYPad = 0
            inputEndYPad = tileSize + overlapping * 2
            precondition(inputEndYPad <= height)
          } else if inputEndYPad > height {
            inputEndYPad = height
            inputStartYPad = height - tileSize - overlapping * 2
            precondition(inputStartYPad >= 0)
          }
          let outputStartY = inputStartY * upscaleFactor
          let outputEndY = inputEndY * upscaleFactor
          let outputStartYTile = (inputStartY - inputStartYPad) * upscaleFactor
          let outputEndYTile = outputStartYTile + (inputEndY - inputStartY) * upscaleFactor
          for x in 0..<xTiles {
            let xOfs = x * tileSize
            let inputStartX = xOfs
            let inputEndX = min(xOfs + tileSize, width)
            var inputStartXPad = inputStartX - overlapping
            var inputEndXPad = inputEndX + overlapping
            if inputStartXPad < 0 {
              inputStartXPad = 0
              inputEndXPad = tileSize + overlapping * 2
              precondition(inputEndXPad <= width)
            } else if inputEndXPad > width {
              inputEndXPad = width
              inputStartXPad = width - tileSize - overlapping * 2
              precondition(inputStartXPad >= 0)
            }
            let outputStartX = inputStartX * upscaleFactor
            let outputEndX = inputEndX * upscaleFactor
            let outputStartXTile = (inputStartX - inputStartXPad) * upscaleFactor
            let outputEndXTile = outputStartXTile + (inputEndX - inputStartX) * upscaleFactor
            let z: DynamicGraph.Tensor<FloatType>
            if isNHWCPreferred {
              z = input[
                i..<(i + 1), inputStartYPad..<inputEndYPad,
                inputStartXPad..<inputEndXPad, 0..<inputShape[3]
              ].copied()
            } else {
              z = input[
                i..<(i + 1), 0..<inputShape[1],
                inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad
              ].copied()
            }
            var outputTile = rrdbnet(inputs: z)[0].as(of: FloatType.self)
            if upscaleFactor != 4 {
              outputTile = Upsample(
                .bilinear, widthScale: Float(upscaleFactor) / 4,
                heightScale: Float(upscaleFactor) / 4)(outputTile)
            }
            outputTile = outputTile * 2 - 1
            if isNHWCPreferred {
              output[i..<(i + 1), outputStartY..<outputEndY, outputStartX..<outputEndX, 0..<3] =
                outputTile[
                  0..<1, outputStartYTile..<outputEndYTile, outputStartXTile..<outputEndXTile, 0..<3
                ].copied().toCPU()
            } else {
              output[i..<(i + 1), outputStartY..<outputEndY, outputStartX..<outputEndX, 0..<3] =
                outputTile[
                  0..<1, 0..<3, outputStartYTile..<outputEndYTile, outputStartXTile..<outputEndXTile
                ].copied().permuted(0, 2, 3, 1).copied().toCPU()
            }
          }
        }
      }
    } else if width > tileSize + 2 * overlapping {
      for i in 0..<batchSize {
        for x in 0..<xTiles {
          let xOfs = x * tileSize
          let inputStartX = xOfs
          let inputEndX = min(xOfs + tileSize, width)
          var inputStartXPad = inputStartX - overlapping
          var inputEndXPad = inputEndX + overlapping
          if inputStartXPad < 0 {
            inputStartXPad = 0
            inputEndXPad = tileSize + overlapping * 2
            precondition(inputEndXPad <= width)
          } else if inputEndXPad > width {
            inputEndXPad = width
            inputStartXPad = width - tileSize - overlapping * 2
            precondition(inputStartXPad >= 0)
          }
          let outputStartX = inputStartX * upscaleFactor
          let outputEndX = inputEndX * upscaleFactor
          let outputStartXTile = (inputStartX - inputStartXPad) * upscaleFactor
          let outputEndXTile = outputStartXTile + (inputEndX - inputStartX) * upscaleFactor
          let z: DynamicGraph.Tensor<FloatType>
          if isNHWCPreferred {
            z = input[i..<(i + 1), 0..<height, inputStartXPad..<inputEndXPad, 0..<inputShape[3]]
              .copied()
          } else {
            z = input[i..<(i + 1), 0..<inputShape[1], 0..<height, inputStartXPad..<inputEndXPad]
              .copied()
          }
          var outputTile = rrdbnet(inputs: z)[0].as(of: FloatType.self)
          if upscaleFactor != 4 {
            outputTile = Upsample(
              .bilinear, widthScale: Float(upscaleFactor) / 4, heightScale: Float(upscaleFactor) / 4
            )(outputTile)
          }
          outputTile = outputTile * 2 - 1
          if isNHWCPreferred {
            output[i..<(i + 1), 0..<(height * upscaleFactor), outputStartX..<outputEndX, 0..<3] =
              outputTile[
                0..<1, 0..<(height * upscaleFactor), outputStartXTile..<outputEndXTile, 0..<3
              ].copied().toCPU()
          } else {
            output[i..<(i + 1), 0..<(height * upscaleFactor), outputStartX..<outputEndX, 0..<3] =
              outputTile[
                0..<1, 0..<3, 0..<(height * upscaleFactor), outputStartXTile..<outputEndXTile
              ].copied().permuted(0, 2, 3, 1).copied().toCPU()
          }
        }
      }
    } else {
      precondition(height > tileSize + 2 * overlapping)
      for i in 0..<batchSize {
        for y in 0..<yTiles {
          let yOfs = y * tileSize
          let inputStartY = yOfs
          let inputEndY = min(yOfs + tileSize, height)
          var inputStartYPad = inputStartY - overlapping
          var inputEndYPad = inputEndY + overlapping
          if inputStartYPad < 0 {
            inputStartYPad = 0
            inputEndYPad = tileSize + overlapping * 2
            precondition(inputEndYPad <= height)
          } else if inputEndYPad > height {
            inputEndYPad = height
            inputStartYPad = height - tileSize - overlapping * 2
            precondition(inputStartYPad >= 0)
          }
          let outputStartY = inputStartY * upscaleFactor
          let outputEndY = inputEndY * upscaleFactor
          let outputStartYTile = (inputStartY - inputStartYPad) * upscaleFactor
          let outputEndYTile = outputStartYTile + (inputEndY - inputStartY) * upscaleFactor
          let z: DynamicGraph.Tensor<FloatType>
          if isNHWCPreferred {
            z = input[i..<(i + 1), inputStartYPad..<inputEndYPad, 0..<width, 0..<inputShape[3]]
              .copied()
          } else {
            z = input[i..<(i + 1), 0..<inputShape[1], inputStartYPad..<inputEndYPad, 0..<width]
              .copied()
          }
          var outputTile = rrdbnet(inputs: z)[0].as(of: FloatType.self)
          if upscaleFactor != 4 {
            outputTile = Upsample(
              .bilinear, widthScale: Float(upscaleFactor) / 4, heightScale: Float(upscaleFactor) / 4
            )(outputTile)
          }
          outputTile = outputTile * 2 - 1
          if isNHWCPreferred {
            output[i..<(i + 1), outputStartY..<outputEndY, 0..<(width * upscaleFactor), 0..<3] =
              outputTile[
                0..<1, outputStartYTile..<outputEndYTile, 0..<(width * upscaleFactor), 0..<3
              ].copied().toCPU()
          } else {
            output[i..<(i + 1), outputStartY..<outputEndY, 0..<(width * upscaleFactor), 0..<3] =
              outputTile[
                0..<1, 0..<3, outputStartYTile..<outputEndYTile, 0..<(width * upscaleFactor)
              ].copied().permuted(0, 2, 3, 1).copied().toCPU()
          }
        }
      }
    }
    return (output, rrdbnet)
  }
}
