import DiffusionMappings
import Fickling
import Foundation
import NNC
import ZIPFoundation

public enum InflateError: Error {
  case tensorNotFound
  case dataNoBaseAddress
  case interrupted
}

public struct Storage {
  var name: String
  var size: Int
  var dataType: DataType
  var BF16: Bool
  var FP8_E4M3: Bool
  var FP8_E5M2: Bool
}

public struct TensorDescriptor {
  public var storage: Storage
  public var storageOffset: Int
  public var shape: [Int]
  public var strides: [Int]
}

extension TensorDescriptor {
  public func inflate<T: TensorNumeric>(from archive: TensorArchive, of type: T.Type) throws
    -> Tensor<T>
  {
    return try archive.with(self) {
      Tensor<T>(from: $0).copied()
    }
  }
}

extension ModelWeightElement {
  public func write<FloatType: BinaryFloatingPoint & TensorNumeric>(
    graph: DynamicGraph, to store: DynamicGraph.Store, tensor: Tensor<FloatType>, format: Format,
    isDiagonalUp: Bool, isDiagonalDown: Bool, renamer: (String) -> String
  ) {
    if isBF16 && FloatType.self != BFloat16.self {
      internalWrite(
        graph: graph, to: store, tensor: Tensor<BFloat16>(from: tensor), format: format,
        isDiagonalUp: isDiagonalUp, isDiagonalDown: isDiagonalDown, renamer: renamer)
      return
    }
    internalWrite(
      graph: graph, to: store, tensor: tensor, format: format, isDiagonalUp: isDiagonalUp,
      isDiagonalDown: isDiagonalDown, renamer: renamer)
  }
  private func internalWrite<FloatType: TensorNumeric>(
    graph: DynamicGraph, to store: DynamicGraph.Store, tensor: Tensor<FloatType>, format: Format,
    isDiagonalUp: Bool, isDiagonalDown: Bool, renamer: (String) -> String
  ) {
    var tensor = tensor
    if scale != 1 || shift != 0 {
      // Apply simple affine transforms if needed.
      tensor = graph.withNoGrad {
        Tensor<FloatType>(
          from: (scale * graph.variable(Tensor<Float>(from: tensor).toCPU()) + shift).rawValue)
      }
    }
    func interleavedOutput(_ output: Tensor<FloatType>, index: Int) -> Tensor<FloatType> {
      guard interleavedIndices.contains(index), numberOfHeads > 0, headDimension > 0 else {
        return output
      }
      if let dimension = interleavedDimension, dimension != headDimension {
        precondition(dimension > 0 && dimension <= headDimension && dimension % 2 == 0)
        let rows = output.reshaped(
          format: output.format, shape: [numberOfHeads, headDimension, -1])
        let width = rows.shape[2]
        var result = rows.copied()
        result[0..<numberOfHeads, 0..<dimension, 0..<width] = graph.withNoGrad {
          graph.variable(
            rows[0..<numberOfHeads, 0..<dimension, 0..<width].copied().reshaped(
              format: output.format, shape: [numberOfHeads, 2, dimension / 2, width]
            ).toGPU()
          ).transposed(1, 2).reshaped(
            format: output.format, shape: [numberOfHeads, dimension, width]
          ).toCPU().rawValue
        }
        return result.reshaped(format: output.format, shape: output.shape)
      }
      return graph.withNoGrad {
        Tensor<FloatType>(
          from: graph.variable(
            output.reshaped(
              format: output.format, shape: [numberOfHeads, 2, headDimension / 2, -1]
            ).toGPU()
          ).transposed(1, 2).reshaped(
            format: output.format, shape: [numberOfHeads * headDimension, -1]
          ).toCPU().rawValue)
      }
    }
    switch format {
    case .O:
      if self.count > 1 {
        let squeezedDim = tensor.shape.compactMap { $0 == 1 ? nil : $0 }
        tensor = tensor.reshaped(format: tensor.format, shape: TensorShape(squeezedDim))
        let shape = tensor.shape
        switch self.format {
        case .O:
          let count = shape[0] / self.count
          if isDiagonalUp {
            let jCount = shape[1] / self.count
            if let offsets = offsets {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: interleavedOutput(
                    tensor[
                      (offsets[i])..<(i < offsets.count - 1 ? offsets[i + 1] : shape[0]),
                      (i * jCount)..<((i + 1) * jCount)
                    ].copied(), index: i))
              }
            } else {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: interleavedOutput(
                    tensor[
                      (i * count)..<((i + 1) * count), (i * jCount)..<((i + 1) * jCount)
                    ].copied(), index: i))
              }
            }
          } else {
            if let offsets = offsets {
              for (i, name) in self.enumerated() {
                if shape.count > 1 {
                  store.write(
                    renamer(name),
                    tensor: interleavedOutput(
                      tensor[
                        (offsets[i])..<(i < offsets.count - 1 ? offsets[i + 1] : shape[0]),
                        0..<shape[1]
                      ].copied(), index: i))
                } else {
                  store.write(
                    renamer(name),
                    tensor: interleavedOutput(
                      tensor[
                        (offsets[i])..<(i < offsets.count - 1 ? offsets[i + 1] : shape[0])
                      ].copied(), index: i))
                }
              }
            } else {
              for (i, name) in self.enumerated() {
                if shape.count > 1 {
                  store.write(
                    renamer(name),
                    tensor: interleavedOutput(
                      tensor[(i * count)..<((i + 1) * count), 0..<shape[1]].copied(),
                      index: i))
                } else {
                  store.write(
                    renamer(name),
                    tensor: interleavedOutput(
                      tensor[(i * count)..<((i + 1) * count)].copied(), index: i))
                }
              }
            }
          }
        case .I:
          if isDiagonalDown {
            let jCount = shape[1] / self.count
            for (i, name) in self.enumerated() {
              store.write(
                renamer(name),
                tensor: interleavedOutput(
                  tensor[0..<shape[0], (i * jCount)..<((i + 1) * jCount)].copied(), index: i))
            }
          } else {
            for (i, name) in self.enumerated() {
              store.write(renamer(name), tensor: interleavedOutput(tensor, index: i))
            }
          }
        }
      } else {
        for (i, name) in self.enumerated() {
          store.write(renamer(name), tensor: interleavedOutput(tensor, index: i))
        }
      }
    case .I:
      if self.count > 1 {
        let shape = tensor.shape
        switch self.format {
        case .I:
          if isDiagonalDown {
            let iCount = shape[0] / self.count
            if let offsets = offsets {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[
                    (i * iCount)..<((i + 1) * iCount),
                    offsets[i]..<(i < offsets.count - 1 ? offsets[i + 1] : shape[1])
                  ].copied())
              }
            } else {
              let count = shape[1] / self.count
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[(i * iCount)..<((i + 1) * iCount), (i * count)..<((i + 1) * count)]
                    .copied())
              }
            }
          } else {
            if let offsets = offsets {
              for (i, name) in self.enumerated() {
                if shape.count > 1 {
                  store.write(
                    renamer(name),
                    tensor: tensor[
                      0..<shape[0],
                      offsets[i]..<(i < offsets.count - 1 ? offsets[i + 1] : shape[1])
                    ].copied())
                } else {
                  store.write(
                    renamer(name),
                    tensor: tensor[
                      offsets[i]..<(i < offsets.count - 1 ? offsets[i + 1] : shape[0])
                    ].copied())
                }
              }
            } else {
              let count = shape.count > 1 ? shape[1] / self.count : shape[0] / self.count
              for (i, name) in self.enumerated() {
                if shape.count > 1 {
                  store.write(
                    renamer(name),
                    tensor: tensor[0..<shape[0], (i * count)..<((i + 1) * count)].copied())
                } else {
                  store.write(
                    renamer(name),
                    tensor: tensor[(i * count)..<((i + 1) * count)].copied())
                }
              }
            }
          }
        case .O:
          if isDiagonalUp {
            let count = shape[0] / self.count
            if shape.count == 2 {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[(i * count)..<((i + 1) * count), 0..<shape[1]].copied())
              }
            } else if shape.count == 4 {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[
                    (i * count)..<((i + 1) * count), 0..<shape[1], 0..<shape[2],
                    0..<shape[3]
                  ].copied())
              }
            } else {
              for (i, name) in self.enumerated() {
                store.write(
                  renamer(name),
                  tensor: tensor[(i * count)..<((i + 1) * count)].copied())
              }
            }
          } else {
            for name in self {
              store.write(renamer(name), tensor: tensor)
            }
          }
        }
      } else {
        for name in self {
          store.write(renamer(name), tensor: tensor)
        }
      }
    }
  }
}
