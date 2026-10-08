import Diffusion
import Foundation
import NNC
import XCTest

final class FirstStageTests: XCTestCase {
  // Parameter-free models exercise the public FirstStage path without checkpoints.
  private func stage(version: ModelVersion, tilePixels: Int, tiled: Bool, overlap: Int = 0)
    -> FirstStage<Float16>
  {
    let tiling = TiledConfiguration(
      isEnabled: tiled,
      tileSize: .init(width: tilePixels / 64, height: tilePixels / 64), tileOverlap: overlap)
    return FirstStage<Float16>(
      filePath: "", version: version,
      latentsScaling: (nil, nil, 1, nil, nil, nil),
      highPrecisionKeysAndValues: false, highPrecisionFallback: false,
      tiledDecoding: tiling, tiledDiffusion: tiling, externalOnDemand: false,
      usesFlashAttention: false, alternativeFilePath: nil, alternativeDecoderVersion: nil,
      deviceProperties: DeviceProperties(
        isFreadPreferred: true, memoryCapacity: .high, isNHWCPreferred: true,
        cacheUri: URL(fileURLWithPath: NSTemporaryDirectory()), isPartialOffloadPreferred: false))
  }

  func testHunyuanSpatialTilesPreserveShortClipDepth() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      for frames in [1, 2, 14, 15, 16] {
        let expectedDepth = min(frames, 15)
        let input = Input()
        let mask = Input()
        var inputDepths = [Int]()
        let checked = input.debug { tensors, _ in
          inputDepths.append(tensors[0]!.shape[0])
        }
        let pixels = Upsample(.nearest, widthScale: 8, heightScale: 8)(
          checked + ReduceMax(axis: [0, 1, 2, 3])(mask))
        let decoder = Model(
          [input, mask],
          [
            Pad(
              .replicate, begin: [0, 0, 0, 0], end: [(expectedDepth - 1) * 3, 0, 0, 0])(
                pixels)
          ])
        let latent = graph.variable(.GPU(0), .NHWC(frames, 64, 64, 3), of: Float16.self)
        latent.full(0.25)
        let result = stage(version: .hunyuanVideo, tilePixels: 256, tiled: true).decode(
          latent, decoder: decoder, cancellation: { _ in }
        ).0.rawValue.toCPU()
        graph.joined()
        XCTAssertFalse(inputDepths.isEmpty)
        XCTAssertTrue(inputDepths.allSatisfy { $0 == expectedDepth }, "\(frames): \(inputDepths)")
        XCTAssertEqual(Array(result.shape), [(frames - 1) * 4 + 1, 512, 512, 3])
        XCTAssertEqual(Float(result[0, 0, 0, 0]), 0.25, accuracy: 0.001)
        XCTAssertEqual(Float(result[result.shape[0] - 1, 511, 511, 2]), 0.25, accuracy: 0.001)
      }
    }
  }

  func testMiniMaxH3EncoderNormalizesPixelsBeforeTiling() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      var pixels = Tensor<Float16>(.CPU, .NHWC(1, 128, 128, 3))
      let values: [Float16] = [-1, 0, 1]
      for y in 0..<128 {
        for x in 0..<128 {
          for channel in 0..<3 {
            pixels[0, y, x, channel] = values[channel]
          }
        }
      }
      let mean: [Float] = [0.485, 0.456, 0.406]
      let std: [Float] = [0.229, 0.224, 0.225]
      for tiled in [false, true] {
        let input = Input()
        let pooled = AveragePool(filterSize: [16, 16], hint: Hint(stride: [16, 16]))(input)
        let encoder = Model([input], [Concat(axis: 3)(Array(repeating: pooled, count: 16))])
        let firstStage = stage(version: .minimaxH3, tilePixels: 64, tiled: tiled)
        // Reuse the encoder as well: normalization must happen on every encode call.
        for _ in 0..<2 {
          let result = firstStage.encode(
            graph.variable(pixels.toGPU(0)), encoder: encoder, cancellation: { _ in }
          ).0.rawValue.toCPU()
          XCTAssertEqual(Array(result.shape), [1, 8, 8, 48])
          guard Array(result.shape) == [1, 8, 8, 48] else { continue }
          for y in 0..<8 {
            for x in 0..<8 {
              for channel in 0..<48 {
                let c = channel % 3
                let expected = ((Float(values[c]) + 1) * 0.5 - mean[c]) / std[c]
                XCTAssertEqual(Float(result[0, y, x, channel]), expected, accuracy: 0.01)
              }
            }
          }
        }
      }
    }
  }

  func testMiniMaxH3TemporalEncodingPadsChunksAndDropsTrailingLatents() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      for tilePixels in [0, 64, 192] {
        let size = tilePixels == 192 ? 256 : 128
        let width = (tilePixels > 0 ? tilePixels : size) / 16
        let input = Input()
        var inputDepths = [Int]()
        let checked = input.debug { tensors, _ in
          inputDepths.append(tensors[0]!.shape[0])
        }
        let pooled = AveragePool(filterSize: [16, 16], hint: Hint(stride: [16, 16]))(checked)
        let sampled = Concat(axis: 0)(
          (0..<5).map {
            pooled.reshaped(
              .NHWC(1, width, width, 3), offset: [$0 * 4, 0, 0, 0],
              strides: [width * width * 3, width * 3, 3, 1]
            ).copied()
          })
        let encoder = Model([input], [Concat(axis: 3)(Array(repeating: sampled, count: 16))])
        let firstStage = stage(
          version: .minimaxH3, tilePixels: max(64, tilePixels), tiled: tilePixels > 0,
          overlap: tilePixels == 192 ? 1 : 0)
        for frames in [5, 22, 39, 56, 107] {
          var pixels = Tensor<Float16>(.CPU, .NHWC(frames, size, size, 3))
          pixels.withUnsafeMutableBytes {
            let values = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
            for t in 0..<frames {
              for y in 0..<size {
                for x in 0..<size {
                  for c in 0..<3 {
                    values[((t * size + y) * size + x) * 3 + c] = Float16(
                      Float(t) / Float(frames) + Float(y) / Float(size) * 0.1
                        + Float(x) / Float(size) * 0.05)
                  }
                }
              }
            }
          }
          let result = firstStage.encode(
            graph.variable(pixels.toGPU(0)), encoder: encoder, cancellation: { _ in }
          ).0.rawValue.toCPU()
          let latentFrames = ((frames + 16) / 17) * 5 - 3
          XCTAssertEqual(Array(result.shape), [latentFrames, size / 16, size / 16, 48])
          for t in 0..<latentFrames {
            let sourceFrame = min(t / 5 * 17 + t % 5 * 4, frames - 1)
            for y in 0..<(size / 16) {
              for x in 0..<(size / 16) {
                let pixel =
                  Float(sourceFrame) / Float(frames)
                  + (Float(y * 16) + 7.5) / Float(size) * 0.1
                  + (Float(x * 16) + 7.5) / Float(size) * 0.05
                let expected = ((pixel + 1) * 0.5 - 0.485) / 0.229
                XCTAssertEqual(Float(result[t, y, x, 0]), expected, accuracy: 0.01)
                XCTAssertEqual(Float(result[t, y, x, 45]), expected, accuracy: 0.01)
              }
            }
          }
        }
        graph.joined()
        XCTAssertFalse(inputDepths.isEmpty)
        XCTAssertTrue(inputDepths.allSatisfy { $0 == 17 }, "\(inputDepths)")
      }
    }
  }

  func testHunyuanTemporalEncodingUsesEachChunk() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      for frames in [57, 61, 81, 121] {
        let depth = min((frames - 1) / 4 + 1, 15)
        let input = Input()
        let mask = Input()
        let pooled = AveragePool(filterSize: [8, 8], hint: Hint(stride: [8, 8]))(input)
        let sampled = Concat(axis: 0)(
          (0..<depth).map {
            pooled.reshaped(
              .NHWC(1, 16, 16, 1), offset: [$0 * 4, 0, 0, 0],
              strides: [16 * 16 * 3, 16 * 3, 3, 1]
            ).copied()
          })
        let encoder = Model(
          [input, mask],
          [
            Concat(axis: 3)(Array(repeating: sampled, count: 32))
              + ReduceMax(axis: [0, 1, 2, 3])(mask)
          ])
        var pixels = Tensor<Float16>(.CPU, .NHWC(frames, 128, 128, 3))
        pixels.withUnsafeMutableBytes {
          let values = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
          for t in 0..<frames {
            for i in 0..<(128 * 128 * 3) {
              values[t * 128 * 128 * 3 + i] = Float16(t) / Float16(frames)
            }
          }
        }
        let result = stage(version: .hunyuanVideo, tilePixels: 128, tiled: false).encode(
          graph.variable(pixels.toGPU(0)), encoder: encoder, cancellation: { _ in }
        ).0.rawValue.toCPU()
        XCTAssertEqual(Array(result.shape), [(frames - 1) / 4 + 1, 16, 16, 32])
        for t in 0..<result.shape[0] {
          let expected = Float(t * 4) / Float(frames)
          XCTAssertEqual(Float(result[t, 0, 0, 0]), expected, accuracy: 0.002)
          XCTAssertEqual(Float(result[t, 15, 15, 31]), expected, accuracy: 0.002)
        }
      }
    }
  }
}
