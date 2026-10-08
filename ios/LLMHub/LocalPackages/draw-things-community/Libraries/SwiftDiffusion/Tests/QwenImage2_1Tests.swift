import Foundation
import NNC
import WeightsCache
import XCTest

@testable import Diffusion

final class QwenImage2_1Tests: XCTestCase {
  func testVAEPerformance() throws {
    guard let path = ProcessInfo.processInfo.environment["QWEN_IMAGE_2_1_VAE_BENCHMARK"] else {
      throw XCTSkip("Set QWEN_IMAGE_2_1_VAE_BENCHMARK to benchmark the converted VAE checkpoint")
    }
    let graph = DynamicGraph()
    graph.withNoGrad {
      for size in [512, 1024, 2048] {
        for decode in [false, true] {
          for flash in [false, true] {
            let input = graph.variable(
              .GPU(0),
              decode ? .NHWC(1, size / 16, size / 16, 64) : .NHWC(1, size, size, 4),
              of: Float16.self)
            input.full(0.25)
            print("VAE_BENCH_START size=\(size) decode=\(decode) flash=\(flash)")
            fflush(stdout)
            let buildStart = Date.timeIntervalSinceReferenceDate
            let model =
              decode
              ? QwenImage2_1Decoder(
                channels: [1152, 1152, 576, 288, 144], height: size / 16, width: size / 16,
                usesFlashAttention: flash)
              : QwenImage2_1Encoder(
                channels: [96, 192, 384, 768, 768], height: size, width: size,
                usesFlashAttention: flash)
            model.maxConcurrency = .limit(1)
            model.compile(inputs: input)
            let buildTime = Date.timeIntervalSinceReferenceDate - buildStart
            let loadStart = Date.timeIntervalSinceReferenceDate
            graph.openStore(
              path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)
            ) {
              try! $0.read(
                decode ? "decoder" : "encoder", model: model, strict: true,
                codec: [.jit, .externalData(.mmap)])
            }
            let loadTime = Date.timeIntervalSinceReferenceDate - loadStart
            var times = [Double]()
            var output: Tensor<Float16>?
            for _ in 0..<4 {
              let start = Date.timeIntervalSinceReferenceDate
              output = model(inputs: input)[0].as(of: Float16.self).rawValue.toCPU()
              times.append(Date.timeIntervalSinceReferenceDate - start)
            }
            var nonfinite = 0
            output!.withUnsafeBytes { bytes in
              for value in bytes.bindMemory(to: Float16.self) {
                if !value.isFinite { nonfinite += 1 }
              }
            }
            print(
              "VAE_BENCH size=\(size) decode=\(decode) flash=\(flash) build=\(buildTime) load=\(loadTime) first=\(times[0]) warm=\(Array(times.dropFirst())) nonfinite=\(nonfinite)"
            )
            fflush(stdout)
            XCTAssertEqual(nonfinite, 0)
          }
        }
      }
    }
  }

  func testTextEncoderWeightsCacheAndCFG() throws {
    guard let path = ProcessInfo.processInfo.environment["QWEN_IMAGE_2_1_TEXT_CHECKPOINT"] else {
      throw XCTSkip("Set QWEN_IMAGE_2_1_TEXT_CHECKPOINT to the converted Qwen3-VL checkpoint")
    }
    let graph = DynamicGraph()
    graph.withNoGrad {
      let cache = WeightsCache(maxTotalCacheSize: 32 * 1024 * 1024 * 1024, memorySubsystem: .UMA)
      let noCache = WeightsCache(maxTotalCacheSize: 0, memorySubsystem: .UMA)
      for hasImage in [false, true] {
        let prefix: [Int32] =
          [151_644, 8_948, 198, 8_948, 151_645, 198, 151_644, 872, 198]
          + (hasImage ? [151_652, 151_655, 151_653] : [])
        let uncond = prefix + [1_000, 151_645, 198]
        let cond = prefix + [2_000, 3_000, 4_000, 151_645, 198]
        let tokenLength = cond.count
        let tokens = graph.variable(
          Tensor<Int32>(
            uncond + Array(repeating: 151_643, count: tokenLength - uncond.count) + cond,
            kind: .CPU, format: .NHWC, shape: [2 * tokenLength]))
        let image = graph.variable(.GPU(0), .NHWC(1, 32, 32, 3), of: Float16.self)
        image.full(0.25)
        func encode(_ weightsCache: WeightsCache, cfg: Bool) -> [Tensor<Float16>] {
          let encoder = TextEncoder<Float16>(
            filePaths: [path], version: .qwenImage2_1, textEncoderVersion: nil,
            isCfgEnabled: cfg, usesFlashAttention: true, injectEmbeddings: false,
            externalOnDemand: false,
            deviceProperties: .init(
              isFreadPreferred: true, memoryCapacity: .high, isNHWCPreferred: true,
              cacheUri: URL(fileURLWithPath: NSTemporaryDirectory()),
              isPartialOffloadPreferred: false),
            weightsCache: weightsCache)
          var tokenLengthUncond = uncond.count
          var tokenLengthCond = cond.count
          let result = encoder.encode(
            tokenLengthUncond: &tokenLengthUncond, tokenLengthCond: &tokenLengthCond,
            tokens: [tokens], positions: [], mask: [], injectedEmbeddings: [],
            images: hasImage ? [image] : [], lengthsOfUncond: [uncond.count],
            lengthsOfCond: [cond.count], injectedTextEmbeddings: [], modifier: .kontext,
            textModels: [])
          let expansion = hasImage ? 1023 : 0
          XCTAssertEqual(tokenLengthUncond, uncond.count + expansion - 6)
          XCTAssertEqual(tokenLengthCond, cond.count + expansion - 6)
          XCTAssertEqual(Array(result.0[0].shape), [cfg ? 2 : 1, tokenLengthCond, 4096])
          return result.0.map { $0.rawValue.toCPU() }
        }
        let reference = encode(noCache, cfg: true)
        for cfg in [true, true, false] {
          let actual = encode(cache, cfg: cfg)
          XCTAssertNotNil(cache[path])
          if hasImage {
            XCTAssertNotNil(cache["\(path):[vision_model]"])
          }
          for index in 0..<actual.count {
            let expected: Tensor<Float16>
            if index == 0 && !cfg {
              expected = reference[0][1..<2, 0..<reference[0].shape[1], 0..<4096].copied()
            } else {
              expected = reference[index]
            }
            XCTAssertEqual(Array(actual[index].shape), Array(expected.shape))
            var squaredError: Double = 0
            var squaredSignal: Double = 0
            var nonfinite = 0
            actual[index].withUnsafeBytes { actualBytes in
              expected.withUnsafeBytes { expectedBytes in
                for (a, b) in zip(
                  actualBytes.bindMemory(to: Float16.self),
                  expectedBytes.bindMemory(to: Float16.self))
                {
                  if !a.isFinite || !b.isFinite { nonfinite += 1 }
                  squaredError += pow(Double(a) - Double(b), 2)
                  squaredSignal += pow(Double(b), 2)
                }
              }
            }
            XCTAssertEqual(nonfinite, 0)
            XCTAssertLessThan(sqrt(squaredError / max(squaredSignal, 1e-12)), 1e-4)
          }
        }
      }
    }
  }

  func testDecoderFloat16Range() throws {
    guard let path = ProcessInfo.processInfo.environment["QWEN_IMAGE_2_1_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set QWEN_IMAGE_2_1_VAE_CHECKPOINT to the converted f16 VAE checkpoint")
    }
    let graph = DynamicGraph()
    graph.withNoGrad {
      let latent: Tensor<Float16>
      if let latentPath = ProcessInfo.processInfo.environment["QWEN_IMAGE_2_1_VAE_LATENT"] {
        latent = try! graph.openStore(latentPath, flags: .readOnly) {
          Tensor<Float16>(from: $0.read("latent")!)
        }.get()
      } else {
        // Cover the range of unscaled latents from the failing reference-edit smoke test.
        var values = Tensor<Float16>(.CPU, .NHWC(1, 8, 12, 64))
        for y in 0..<8 {
          for x in 0..<12 {
            for c in 0..<64 {
              values[0, y, x, c] = Float16(20 * sin(Float((y * 12 + x) * 64 + c) * 0.03))
            }
          }
        }
        latent = values
      }
      let height = latent.shape[1]
      let width = latent.shape[2]
      for batch in 0..<latent.shape[0] {
        let input = graph.variable(latent.toGPU(0))[
          batch..<(batch + 1), 0..<height, 0..<width, 0..<64
        ].copied()
        var reference: Tensor<Float>?
        for (useFloat16, flash) in [(false, false), (true, false), (true, true)] {
          let model = QwenImage2_1Decoder(
            channels: [1152, 1152, 576, 288, 144], height: height, width: width,
            usesFlashAttention: flash)
          let x: DynamicGraph.AnyTensor =
            useFloat16 ? input : DynamicGraph.Tensor<Float>(from: input)
          model.maxConcurrency = .limit(1)
          model.compile(inputs: x)
          graph.openStore(
            path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)
          ) {
            try! $0.read("decoder", model: model, strict: true, codec: [.externalData(.mmap)])
          }
          let result = model(inputs: x)[0]
          let output =
            useFloat16
            ? Tensor<Float>(from: result.as(of: Float16.self).rawValue.toCPU())
            : result.as(of: Float.self).rawValue.toCPU()
          if let reference = reference {
            XCTAssertEqual(Array(output.shape), Array(reference.shape))
            var squaredError: Double = 0
            var squaredSignal: Double = 0
            var maximum: Float = 0
            var nonfinite = 0
            output.withUnsafeBytes { outputBytes in
              reference.withUnsafeBytes { referenceBytes in
                for (a, b) in zip(
                  outputBytes.bindMemory(to: Float.self), referenceBytes.bindMemory(to: Float.self))
                {
                  if !a.isFinite || !b.isFinite { nonfinite += 1 }
                  maximum = max(maximum, abs(a - b))
                  squaredError += Double(a - b) * Double(a - b)
                  squaredSignal += Double(b) * Double(b)
                }
              }
            }
            let relative = sqrt(squaredError / max(squaredSignal, 1e-12))
            print(
              "Qwen2.1 VAE range batch=\(batch) flash=\(flash): nonfinite=\(nonfinite), relative RMS=\(relative), max=\(maximum)"
            )
            // This is an overflow regression; report numerical differences separately.
            XCTAssertEqual(nonfinite, 0)
          } else {
            reference = output
          }
        }
      }
    }
  }

  func testDecoderHighPrecisionFallback() throws {
    guard let path = ProcessInfo.processInfo.environment["QWEN_IMAGE_2_1_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set QWEN_IMAGE_2_1_VAE_CHECKPOINT to the converted f16 VAE checkpoint")
    }
    let graph = DynamicGraph()
    graph.withNoGrad {
      let tiling = TiledConfiguration(
        isEnabled: false, tileSize: .init(width: 4, height: 4), tileOverlap: 0)
      let firstStage = FirstStage<Float16>(
        filePath: path, version: .qwenImage2_1,
        latentsScaling: (nil, nil, 1, nil, nil, nil),
        highPrecisionKeysAndValues: false, highPrecisionFallback: true,
        tiledDecoding: tiling, tiledDiffusion: tiling, externalOnDemand: false,
        usesFlashAttention: false, alternativeFilePath: nil,
        alternativeDecoderVersion: .transparent,
        deviceProperties: .init(
          isFreadPreferred: true, memoryCapacity: .high, isNHWCPreferred: true,
          cacheUri: URL(fileURLWithPath: NSTemporaryDirectory()), isPartialOffloadPreferred: false))
      // Force the public retry path, which must compile the checkpoint decoder in FP32.
      let input = Input()
      let pooled = ReduceMean(axis: [3])(input)
      let pixels = Upsample(.nearest, widthScale: 16, heightScale: 16)(
        Concat(axis: 3)(pooled, pooled, pooled, pooled))
      let decoder = Model([input], [pixels * Float.nan])
      let latent = graph.variable(.GPU(0), .NHWC(1, 4, 6, 64), of: Float16.self)
      latent.full(0.1)
      let result = firstStage.decode(latent, decoder: decoder, cancellation: { _ in }).0.rawValue
        .toCPU()
      XCTAssertEqual(Array(result.shape), [1, 64, 96, 4])
      for y in 0..<64 {
        for x in 0..<96 {
          for channel in 0..<4 {
            XCTAssertTrue(Float(result[0, y, x, channel]).isFinite)
          }
        }
      }
    }
  }

  func testTiledDiffusionMatchesExplicitTiles() throws {
    guard let path = ProcessInfo.processInfo.environment["QWEN_IMAGE_2_1_CHECKPOINT"] else {
      throw XCTSkip("Set QWEN_IMAGE_2_1_CHECKPOINT to the converted f16 DiT checkpoint")
    }
    let graph = DynamicGraph()
    graph.withNoGrad {
      // Two samples in each CFG branch, with unequal text lengths and one reference image.
      let batchSize = 4
      let height = 6
      let width = 10
      let tileHeight = 4
      let tileWidth = 8
      let prefixLength = 9
      var image = Tensor<Float16>(.CPU, .NHWC(batchSize, height, width, 64))
      var rotary = Tensor<Float>(.CPU, .NHWC(batchSize, height * width, 1, 128))
      for batch in 0..<batchSize {
        let textLength = batch < 2 ? 3 : 5
        let positions = QwenImage2_1RotaryEmbedding([
          .init(image: false, length: textLength, sourceOffset: 0),
          .init(image: true, length: 4, sourceOffset: 0, height: 2, width: 2),
          .init(image: true, length: height * width, sourceOffset: 4, height: height, width: width),
        ])
        for y in 0..<height {
          for x in 0..<width {
            for channel in 0..<64 {
              image[batch, y, x, channel] = Float16(
                sin(Float(((batch * height + y) * width + x) * 64 + channel) * 0.03))
            }
            for channel in 0..<128 {
              rotary[batch, y * width + x, 0, channel] =
                positions[0, textLength + 4 + y * width + x, 0, channel]
            }
          }
        }
      }
      let input = graph.variable(image.toGPU(0))
      var conditions: [DynamicGraph.AnyTensor] = [graph.variable(rotary.toGPU(0))]
      for value: Float in [1, 0.1, 1, 0.1, 1] {
        let modulation = graph.variable(.GPU(0), .WC(1, 4096), of: Float16.self)
        modulation.full(value)
        conditions.append(modulation)
      }
      for i in 0..<64 {
        let kv = graph.variable(
          .GPU(0), .NHWC(batchSize, prefixLength, 32, 128), of: Float16.self)
        kv.full(Float(i + 1) * 0.001)
        conditions.append(kv)
      }
      let tiled = TiledConfiguration(
        isEnabled: true, tileSize: .init(width: 2, height: 1), tileOverlap: 0)
      var unet = UNetFromNNC<Float16>()
      XCTAssertTrue(
        unet.compileModel(
          filePath: path, externalOnDemand: false,
          deviceProperties: .init(
            isFreadPreferred: false, memoryCapacity: .high, isNHWCPreferred: true,
            cacheUri: URL(fileURLWithPath: NSTemporaryDirectory()), isPartialOffloadPreferred: false
          ),
          version: .qwenImage2_1, modifier: .none,
          qkNorm: false, dualAttentionLayers: [], upcastAttention: false, usesFlashAttention: .sdpa,
          usesSolAttention: false, solAttentionStart: 0, solAttentionTau: 0,
          injectControlsAndAdapters: .init(
            injectControls: false, injectT2IAdapters: false, injectAttentionKV: false,
            injectIPAdapterLengths: [], injectControlModels: []),
          lora: [], isQuantizedModel: false, canRunLoRASeparately: false,
          inputs: input, nil, conditions, tokenLengthUncond: 3, tokenLengthCond: 5,
          isCfgEnabled: true, extraProjection: nil,
          injectedControlsAndAdapters: .init(
            injectedControls: [], injectedT2IAdapters: [], injectedIPAdapters: [],
            injectedAttentionKVs: []),
          referenceImageCount: 1, referenceAudioCount: 0, tiledDiffusion: tiled,
          teaCache: .init(
            coefficients: (0, 0, 0, 0, 0), steps: 0...0, threshold: 0, maxSkipSteps: 0),
          causalInference: (0, 0), isBF16: false, activationQkScaling: [:],
          activationProjScaling: [:], activationFfnProjUpScaling: [:], activationFfnScaling: [:],
          weightsCache: .init(maxTotalCacheSize: 0, memorySubsystem: .UMA)))
      func evaluate(
        _ input: DynamicGraph.Tensor<Float16>, _ conditions: [DynamicGraph.AnyTensor],
        tiledDiffusion: TiledConfiguration
      ) -> Tensor<Float16> {
        var controlNets = [Model?]()
        return unet(
          timestep: (0.9, 0.3), audioShiftRatio: 1, inputs: input, nil, conditions,
          extraProjection: nil,
          injectedControlsAndAdapters: { _, _, _, _, _, _, _ in ([], [], []) },
          injectedIPAdapters: [], referenceImageCount: 1, referenceAudioCount: 0, step: 0,
          tokenLengthUncond: 3, tokenLengthCond: 5, isCfgEnabled: true,
          tiledDiffusion: tiledDiffusion, controlNets: &controlNets
        ).rawValue.toCPU()
      }
      let actual = evaluate(input, conditions, tiledDiffusion: tiled)
      XCTAssertEqual(Array(actual.shape), [batchSize, height, width, 64])
      var tiles = [Tensor<Float16>]()
      // The right/bottom tiles are anchored to the canvas edge and overlap earlier tiles.
      for y in [0, height - tileHeight] {
        for x in [0, width - tileWidth] {
          var tileRotary = Tensor<Float>(
            .CPU, .NHWC(batchSize, tileHeight * tileWidth, 1, 128))
          for batch in 0..<batchSize {
            for row in 0..<tileHeight {
              for column in 0..<tileWidth {
                for channel in 0..<128 {
                  tileRotary[batch, row * tileWidth + column, 0, channel] =
                    rotary[batch, (y + row) * width + x + column, 0, channel]
                }
              }
            }
          }
          let tile = input[
            0..<batchSize, y..<(y + tileHeight), x..<(x + tileWidth), 0..<64
          ].copied()
          tiles.append(
            evaluate(
              tile, [graph.variable(tileRotary.toGPU(0))] + conditions.dropFirst(),
              tiledDiffusion: .init(isEnabled: false, tileSize: tiled.tileSize, tileOverlap: 0)))
        }
      }
      let xWeights = unet.xTileWeightsAndIndexes!
      let yWeights = unet.yTileWeightsAndIndexes!
      var maximum: Float = 0
      for batch in 0..<batchSize {
        for y in 0..<height {
          for x in 0..<width {
            for channel in 0..<64 {
              var expected: Float16 = 0
              for row in yWeights[y] {
                for column in xWeights[x] {
                  expected +=
                    Float16(row.weight * column.weight)
                    * tiles[row.index * 2 + column.index][batch, row.offset, column.offset, channel]
                }
              }
              let value = Float(actual[batch, y, x, channel])
              XCTAssertTrue(value.isFinite)
              maximum = max(maximum, abs(value - Float(expected)))
            }
          }
        }
      }
      print("Qwen2.1 tiled diffusion versus explicit tiles: max=\(maximum)")
      XCTAssertLessThan(maximum, 0.001)
    }
  }

  func testNHWCVAEParity() throws {
    guard let path = ProcessInfo.processInfo.environment["QWEN_IMAGE_2_1_VAE_CHECKPOINT"] else {
      throw XCTSkip("Set QWEN_IMAGE_2_1_VAE_CHECKPOINT to the converted f16 VAE checkpoint")
    }
    for (encoder, useFloat16, flash) in [
      (true, false, false), (false, false, false), (true, true, false), (false, true, false),
      (true, false, true), (false, false, true), (true, true, true), (false, true, true),
    ] {
      let graph = DynamicGraph()
      graph.withNoGrad {
        let height = encoder ? 32 : 2
        let width = encoder ? 48 : 3
        let channels = encoder ? 4 : 64
        var input = Tensor<Float>(.CPU, .NHWC(1, height, width, channels))
        for y in 0..<height {
          for x in 0..<width {
            for c in 0..<channels {
              input[0, y, x, c] = sin(Float(y * width * channels + x * channels + c) * 0.03)
            }
          }
        }
        let model =
          encoder
          ? QwenImage2_1Encoder(
            channels: [96, 192, 384, 768, 768], height: height, width: width,
            usesFlashAttention: flash)
          : QwenImage2_1Decoder(
            channels: [1152, 1152, 576, 288, 144], height: height, width: width,
            usesFlashAttention: flash)
        let baseline = QwenImage2_1VAEReference(encoder: encoder, height: height, width: width).0
        let x: DynamicGraph.AnyTensor =
          useFloat16
          ? graph.variable(Tensor<Float16>(from: input).toGPU(0)) : graph.variable(input.toGPU(0))
        model.maxConcurrency = .limit(1)
        baseline.maxConcurrency = .limit(1)
        model.compile(inputs: x)
        baseline.compile(inputs: x)
        graph.openStore(
          path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)
        ) {
          try! $0.read(
            encoder ? "encoder" : "decoder", model: model, strict: true,
            codec: [.externalData(.mmap)])
          try! $0.read(
            encoder ? "encoder" : "decoder", model: baseline, strict: true,
            codec: [.externalData(.mmap)])
        }
        let actualOutput = model(inputs: x)[0]
        let expectedOutput = baseline(inputs: x)[0]
        let actual =
          useFloat16
          ? Tensor<Float>(from: actualOutput.as(of: Float16.self).rawValue.toCPU())
          : actualOutput.as(of: Float.self).rawValue.toCPU()
        let expected =
          useFloat16
          ? Tensor<Float>(from: expectedOutput.as(of: Float16.self).rawValue.toCPU())
          : expectedOutput.as(of: Float.self).rawValue.toCPU()
        XCTAssertEqual(Array(actual.shape), Array(expected.shape))
        var squaredError: Double = 0
        var squaredSignal: Double = 0
        var maximum: Float = 0
        for y in 0..<actual.shape[1] {
          for x in 0..<actual.shape[2] {
            for c in 0..<actual.shape[3] {
              let a = actual[0, y, x, c]
              let b = expected[0, y, x, c]
              XCTAssertTrue(a.isFinite)
              maximum = max(maximum, abs(a - b))
              squaredError += Double((a - b) * (a - b))
              squaredSignal += Double(b * b)
            }
          }
        }
        let relative = sqrt(squaredError / max(squaredSignal, 1e-12))
        print(
          "Qwen2.1 VAE parity encoder=\(encoder) fp16=\(useFloat16) flash=\(flash): relative RMS=\(relative), max=\(maximum)"
        )
        XCTAssertLessThan(relative, useFloat16 ? 0.01 : 0.001)
        XCTAssertLessThan(maximum, useFloat16 ? 0.1 : 0.01)
      }
    }
  }

  func testCachedPrefixMatchesFullSequence() throws {
    guard let path = ProcessInfo.processInfo.environment["QWEN_IMAGE_2_1_CHECKPOINT"] else {
      throw XCTSkip("Set QWEN_IMAGE_2_1_CHECKPOINT to the converted f16 DiT checkpoint")
    }
    for flashAttention in [FlashAttentionLevel.none, .scale1, .scaleMerged, .quantized] {
      let flash = flashAttention != .none
      for references in [false, true] {
        let graph = DynamicGraph()
        graph.withNoGrad {
          let textLength = 8
          let referenceLength = references ? 8 : 0
          let prefixLength = textLength + referenceLength
          let targetLength = 4
          let count = prefixLength + targetLength
          let batchSize = 2
          // Adjacent image blocks must not see one another's future keys.
          let prefix: [QwenImage2_1Segment] =
            references
            ? [
              .init(image: false, length: 3, sourceOffset: 0),
              .init(image: true, length: 4, sourceOffset: 0, height: 2, width: 2),
              .init(image: true, length: 4, sourceOffset: 4, height: 2, width: 2),
              .init(image: false, length: 5, sourceOffset: 3),
            ] : [.init(image: false, length: textLength, sourceOffset: 0)]
          let target = QwenImage2_1Segment(
            image: true, length: targetLength,
            sourceOffset: referenceLength, height: 2, width: 2)
          let segments = prefix + [target]
          var indices = [Int32]()
          for segment in segments {
            let offset = segment.sourceOffset + (segment.image ? textLength : 0)
            indices += (offset..<(offset + segment.length)).map(Int32.init)
          }
          var context = Tensor<Float16>(.CPU, .HWC(batchSize, textLength, 4096))
          var reference = Tensor<Float16>(.CPU, .WC(max(1, referenceLength), 64))
          var image = Tensor<Float16>(.CPU, .NHWC(batchSize, 2, 2, 64))
          for batch in 0..<batchSize {
            for row in 0..<textLength {
              for col in 0..<4096 {
                context[batch, row, col] = Float16(
                  sin(Float(1 + batch * 13 + row * 4096 + col) * 0.013))
              }
            }
            for y in 0..<2 {
              for x in 0..<2 {
                for col in 0..<64 {
                  image[batch, y, x, col] = Float16(
                    cos(Float(batch * 256 + y * 128 + x * 64 + col) * 0.07))
                }
              }
            }
          }
          for row in 0..<max(1, referenceLength) {
            for col in 0..<64 {
              reference[row, col] = Float16(sin(Float(row * 64 + col) * 0.03))
            }
          }
          var rotary = Tensor<Float>(.CPU, .NHWC(batchSize, count, 1, 128))
          var prefixRotary = Tensor<Float>(.CPU, .NHWC(batchSize, prefixLength, 1, 128))
          var targetRotary = Tensor<Float>(.CPU, .NHWC(batchSize, targetLength, 1, 128))
          var mask = Tensor<Float16>(.CPU, .NHWC(batchSize, 1, count, count))
          let layout = QwenImage2_1AttentionMask(segments)
          for batch in 0..<batchSize {
            let padding = batch == 0 ? 2 : 0
            var actual = prefix
            let last = actual.count - 1
            actual[last] = .init(
              image: false, length: actual[last].length - padding,
              sourceOffset: actual[last].sourceOffset)
            let positions = QwenImage2_1RotaryEmbedding(actual + [target])
            for row in 0..<count {
              let positionRow =
                row >= prefixLength ? row - padding : min(row, prefixLength - padding - 1)
              for col in 0..<128 {
                let value = positions[0, positionRow, 0, col]
                rotary[batch, row, 0, col] = value
                if row < prefixLength {
                  prefixRotary[batch, row, 0, col] = value
                } else {
                  targetRotary[batch, row - prefixLength, 0, col] = value
                }
              }
              for col in 0..<count {
                let padded = col >= prefixLength - padding && col < prefixLength
                mask[batch, 0, row, col] =
                  padded || !layout[0, 0, row, col].isFinite ? -Float16.greatestFiniteMagnitude : 0

              }
            }
          }
          let timesteps: [Float] = [0.9, 0.3]
          var timeTable = Tensor<Float16>(.CPU, .WC(3, 256))
          for (i, t) in (timesteps + [0]).enumerated() {
            let value = timeEmbedding(
              timestep: t * 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
            for col in 0..<256 { timeTable[i, col] = Float16(value[0, col]) }
          }
          let contextInput = graph.variable(context.toGPU(0))
          let imageInput = graph.variable(image.toGPU(0))
          let referenceInput = graph.variable(reference.toGPU(0))
          let (fixedMapper, fixed) = QwenImage2_1Fixed(
            Float16.self, batchSize: batchSize,
            textLength: textLength, referenceLength: referenceLength, timesteps: timesteps.count,
            channels: 4096, layers: 32, segments: prefix.map { ($0.length, $0.image) },
            usesFlashAttention: flashAttention)
          let fixedInputs: [DynamicGraph.AnyTensor] =
            [
              contextInput, graph.variable(timeTable.toGPU(0)),
              graph.variable(prefixRotary.toGPU(0)),
            ]
            + (flash
              ? []
              : [graph.variable(Tensor<Float16>(from: QwenImage2_1AttentionMask(prefix)).toGPU(0))])
            + (references ? [referenceInput] : [])
          fixed.compile(inputs: fixedInputs)
          graph.openStore(
            path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)
          ) {
            try! $0.read("dit", model: fixed, strict: true, codec: [.externalData(.mmap)])
          }
          let cached = fixed(inputs: fixedInputs[0], Array(fixedInputs.dropFirst()))
          XCTAssertEqual(cached.count, 69)
          let denoisers = (0..<batchSize).map { batch in
            QwenImage2_1(
              Float16.self, batchSize: 1, height: 2, width: 2,
              prefixLength: prefixLength - (batch == 0 ? 2 : 0),
              channels: 4096, layers: 32, usesFlashAttention: flashAttention)
          }
          let (baselineMapper, baseline) = QwenImage2_1Unsplit(
            Float16.self, batchSize: batchSize,
            height: 2, width: 2, textLength: textLength, referenceLength: referenceLength,
            usesFlashAttention: flash)
          for (step, t) in timesteps.enumerated() {
            let inputs: [[DynamicGraph.AnyTensor]] = (0..<batchSize).map { batch in
              let length = prefixLength - (batch == 0 ? 2 : 0)
              let image = imageInput[batch..<(batch + 1), 0..<2, 0..<2, 0..<64].copied()
              let rot = graph.variable(targetRotary.toGPU(0))[
                batch..<(batch + 1), 0..<targetLength, 0..<1, 0..<128
              ].copied()
              let modulation: [DynamicGraph.AnyTensor] = cached.prefix(5).map {
                $0.as(of: Float16.self)[step..<(step + 1), 0..<4096].copied()
              }
              let kvs: [DynamicGraph.AnyTensor] = cached.dropFirst(5).map {
                $0.as(of: Float16.self)[batch..<(batch + 1), 0..<length, 0..<32, 0..<128].copied()
              }
              return [image, rot] + modulation + kvs
            }
            let baselineInputs: [DynamicGraph.AnyTensor] =
              [
                imageInput,
                graph.variable(Tensor<Float16>(from: QwenImage2_1UnsplitTimeEmbedding(t)).toGPU(0)),
                contextInput, graph.variable(rotary.toGPU(0)), graph.variable(mask.toGPU(0)),
                graph.variable(Tensor<Int32>(indices, .CPU, .C(count)).toGPU(0)),
              ]
              + (references ? [referenceInput] : [])
            if step == 0 {
              for batch in 0..<batchSize { denoisers[batch].1.compile(inputs: inputs[batch]) }
              baseline.compile(inputs: baselineInputs)
              let expectedMapping = baselineMapper(.diffusers)
              let fixedMapping = fixedMapper(.diffusers)
              let denoiserMapping = denoisers[0].0(.diffusers)
              XCTAssertEqual(
                Set(fixedMapping.keys).union(denoiserMapping.keys), Set(expectedMapping.keys))
              for mapping in [fixedMapping, denoiserMapping] {
                for (key, names) in mapping {
                  XCTAssertEqual(Array(names), Array(expectedMapping[key]!), key)
                }
              }
              graph.openStore(
                path, flags: .readOnly, externalStore: TensorData.externalStore(filePath: path)
              ) {
                for (_, denoiser) in denoisers {
                  try! $0.read("dit", model: denoiser, strict: true, codec: [.externalData(.mmap)])
                }
                try! $0.read("dit", model: baseline, strict: true, codec: [.externalData(.mmap)])
              }
            }
            let results = (0..<batchSize).map { batch in
              denoisers[batch].1(inputs: inputs[batch][0], Array(inputs[batch].dropFirst()))[0]
                .as(of: Float16.self).rawValue.toCPU()
            }
            let expected = baseline(inputs: baselineInputs[0], Array(baselineInputs.dropFirst()))[0]
              .as(of: Float16.self).rawValue.toCPU()
            var squaredError: Double = 0
            var squaredSignal: Double = 0
            var maximum: Float = 0
            var finite = true
            for batch in 0..<batchSize {
              for y in 0..<2 {
                for x in 0..<2 {
                  for col in 0..<64 {
                    let a = Float(results[batch][0, y, x, col])
                    let b = Float(expected[batch, y, x, col])
                    finite = finite && a.isFinite && b.isFinite
                    maximum = max(maximum, abs(a - b))
                    squaredError += Double((a - b) * (a - b))
                    squaredSignal += Double(b * b)
                  }
                }
              }
            }
            XCTAssertTrue(finite)
            let relative = sqrt(squaredError / max(squaredSignal, 1e-12))
            print(
              "Qwen2.1 parity flash=\(flashAttention) refs=\(references) t=\(t): relative RMS=\(relative), max=\(maximum)"
            )
            // INT8 attention uses a wider error budget than FP16 attention.
            XCTAssertLessThan(relative, flashAttention == .quantized ? 0.1 : 0.01)
            XCTAssertLessThan(maximum, flashAttention == .quantized ? 1 : 0.1)
          }
        }
      }
    }
  }
}
