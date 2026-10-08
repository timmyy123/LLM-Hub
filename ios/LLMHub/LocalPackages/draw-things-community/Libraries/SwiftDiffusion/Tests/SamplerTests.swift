import Diffusion
import NNC
import XCTest

final class SamplerTests: XCTestCase {
  func testWeightsCacheSuffix() {
    for version: ModelVersion in [
      .minimaxH3, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .flux2,
      .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .longcatVideoAvatar1_5, .v1,
    ] {
      for threshold: Float in [0, 0.04, .nan] {
        let teaCache = TeaCacheConfiguration(
          coefficients: (0, 0, 0, 1, 0), steps: 5...20, threshold: threshold, maxSkipSteps: 3)
        let sampler: any Sampler<Float, UNetFromNNC<Float>> =
          EulerASampler<Float, UNetFromNNC<Float>, Denoiser.LinearDiscretization>(
            filePath: "model.ckpt", modifier: .none, version: version, audioShiftRatio: 1,
            qkNorm: false, dualAttentionLayers: [], distilledGuidanceLayers: 0,
            activationQkScaling: [:], activationProjScaling: [:], activationFfnProjUpScaling: [:],
            activationFfnScaling: [:], usesFlashAttention: .none, upcastAttention: false,
            externalOnDemand: false, injectControls: false, injectT2IAdapters: false,
            injectAttentionKV: false, injectIPAdapterLengths: [], lora: [],
            classifierFreeGuidance: false, isGuidanceEmbedEnabled: false, isQuantizedModel: false,
            canRunLoRASeparately: false,
            deviceProperties: .init(
              isFreadPreferred: true, memoryCapacity: .high, isNHWCPreferred: true,
              cacheUri: URL(fileURLWithPath: NSTemporaryDirectory()),
              isPartialOffloadPreferred: false),
            conditioning: .timestep,
            tiledDiffusion: .init(
              isEnabled: false, tileSize: .init(width: 0, height: 0), tileOverlap: 0),
            teaCache: teaCache, causalInference: (0, 0),
            cfgZeroStar: .init(isEnabled: false, zeroInitSteps: 0), isBF16: false,
            discretization: .init(.rf(.init()), objective: .u(conditionScale: 1_000)),
            weightsCache: .init(maxTotalCacheSize: 0, memorySubsystem: .UMA))
        let expected = version == .minimaxH3 && threshold > 0 ? ":[teacache]" : ""
        XCTAssertEqual(sampler.suffix, expected)
      }
    }
  }

  func testH3IndependentAudioShift() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let height = 2 + MiniMaxH3AudioHeight(videoLatentFrames: 1, latentWidth: 2)
      let sample = graph.variable(.GPU(0), .NHWC(1, height, 2, 24), of: Float.self)
      let noise = graph.variable(like: sample)
      sample.full(2)
      noise.full(6)
      for shifts: (video: Double, audio: Double) in [
        (12, 3), (6, 3), (3, 3), (3, 6), (12, 0.1), (0.1, 12),
      ] {
        let ratio = Float(shifts.audio / shifts.video)
        let sampler = EulerASampler<Float, UNetFromNNC<Float>, Denoiser.LinearDiscretization>(
          filePath: "", modifier: .none, version: .minimaxH3, audioShiftRatio: ratio,
          qkNorm: false, dualAttentionLayers: [], distilledGuidanceLayers: 0,
          activationQkScaling: [:], activationProjScaling: [:], activationFfnProjUpScaling: [:],
          activationFfnScaling: [:], usesFlashAttention: .none, upcastAttention: false,
          externalOnDemand: false, injectControls: false, injectT2IAdapters: false,
          injectAttentionKV: false, injectIPAdapterLengths: [], lora: [],
          classifierFreeGuidance: false, isGuidanceEmbedEnabled: false, isQuantizedModel: false,
          canRunLoRASeparately: false,
          deviceProperties: .init(
            isFreadPreferred: true, memoryCapacity: .high, isNHWCPreferred: true,
            cacheUri: URL(fileURLWithPath: NSTemporaryDirectory()), isPartialOffloadPreferred: false
          ),
          conditioning: .timestep,
          tiledDiffusion: .init(
            isEnabled: false, tileSize: .init(width: 0, height: 0), tileOverlap: 0),
          teaCache: .init(
            coefficients: (0, 0, 0, 0, 0), steps: 0...0, threshold: 0, maxSkipSteps: 0),
          causalInference: (0, 0), cfgZeroStar: .init(isEnabled: false, zeroInitSteps: 0),
          isBF16: false,
          discretization: .init(.rf(.init()), objective: .u(conditionScale: 1_000)),
          weightsCache: .init(maxTotalCacheSize: 0, memorySubsystem: .UMA))
        for base: Double in [0, 0.125, 0.5, 0.875, 1] {
          let videoSigma = Float(shifts.video * base / (1 + (shifts.video - 1) * base))
          let audioSigma = Float(shifts.audio * base / (1 + (shifts.audio - 1) * base))
          let added = sampler.sampleAdd(
            sample, noise: noise, scale: (1 - videoSigma, videoSigma), version: .minimaxH3
          ).rawValue.toCPU()
          XCTAssertEqual(added[0, 0, 0, 0], 2 + 4 * videoSigma, accuracy: 1e-5)
          XCTAssertEqual(added[0, height - 1, 0, 0], 2 + 4 * audioSigma, accuracy: 1e-5)
          var calls = 0
          let combined = sampler.sampleScaledAdd(
            sample, noise, sigma: (videoSigma, 0), version: .minimaxH3
          ) { sigma in
            XCTAssertEqual(sigma.now, calls == 0 ? videoSigma : audioSigma, accuracy: 2e-6)
            XCTAssertEqual(sigma.next, 0)
            calls += 1
            return (1 - sigma.now, sigma.now)
          }.rawValue.toCPU()
          XCTAssertEqual(calls, 2)
          XCTAssertEqual(combined[0, height - 1, 0, 0], added[0, height - 1, 0, 0])
        }
      }
    }
  }

  func testOrdinaryModelsPreserveWeightedAddAndBroadcasting() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let sample = graph.variable(.GPU(0), .NHWC(1, 3, 4, 4), of: Float.self)
      let noise = graph.variable(.GPU(0), .NHWC(2, 3, 4, 4), of: Float.self)
      sample.full(2)
      noise.full(-3)
      let expected = Functional.add(left: sample, right: noise, leftScalar: 0.7, rightScalar: 0.4)
        .rawValue.toCPU()
      let added = sampleAdd(
        sample, noise: noise, scale: (sample: 0.7, noise: 0.4), version: .v1, audioShiftRatio: 1
      )
      .rawValue.toCPU()
      var calls = 0
      let combined = sampleScaledAdd(
        sample, noise, sigma: (now: 0.8, next: 0.2), version: .v1, audioShiftRatio: 1
      ) {
        sigma in
        calls += 1
        XCTAssertEqual(sigma.now, 0.8)
        XCTAssertEqual(sigma.next, 0.2)
        return (0.7, 0.4)
      }.rawValue.toCPU()
      XCTAssertEqual(calls, 1)
      XCTAssertEqual(Array(added.shape), [2, 3, 4, 4])
      for frame in 0..<2 {
        for y in 0..<3 {
          for x in 0..<4 {
            for c in 0..<4 {
              XCTAssertEqual(added[frame, y, x, c], expected[frame, y, x, c])
              XCTAssertEqual(combined[frame, y, x, c], expected[frame, y, x, c])
            }
          }
        }
      }
    }
  }

  func testH3CleanSampleNoiseAndEndpoints() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      for frames in [1, 2, 7, 12] {
        let audioHeight = MiniMaxH3AudioHeight(videoLatentFrames: frames, latentWidth: 4)
        let height = 3 + audioHeight
        let sample = graph.variable(.GPU(0), .NHWC(frames, height, 4, 24), of: Float16.self)
        let noise = graph.variable(like: sample)
        sample.full(2)
        noise.full(6)
        for sigma: Float in [0, 0.2, 0.6, 1] {
          let result = sampleAdd(
            sample, noise: noise, scale: (sample: 1 - sigma, noise: sigma), version: .minimaxH3,
            audioShiftRatio: 0.25
          ).rawValue.toCPU()
          let audioSigma = sigma / (4 - 3 * sigma)
          XCTAssertEqual(Array(result.shape), [frames, height, 4, 24])
          for frame in 0..<frames {
            for y in 0..<height {
              for x in 0..<4 {
                for c in 0..<24 {
                  let expected: Float = 2 + 4 * (y < 3 ? sigma : audioSigma)
                  XCTAssertEqual(Float(result[frame, y, x, c]), expected, accuracy: 0.008)
                }
              }
            }
          }
        }
        XCTAssertEqual(sample.rawValue.toCPU()[frames - 1, height - 1, 3, 23], 2)
        XCTAssertEqual(noise.rawValue.toCPU()[frames - 1, height - 1, 3, 23], 6)
      }
    }
  }

  func testH3CallbackMapsBothSigmasAndDoesNotModifyOperands() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let height = 3 + MiniMaxH3AudioHeight(videoLatentFrames: 7, latentWidth: 4)
      let left = graph.variable(.GPU(0), .NHWC(7, height, 4, 24), of: Float.self)
      let right = graph.variable(like: left)
      left.full(2)
      right.full(6)
      var sigmas = [(now: Float, next: Float)]()
      let result = sampleScaledAdd(
        left, right, sigma: (now: 0.8, next: 0.3), version: .minimaxH3, audioShiftRatio: 0.25
      ) {
        sigma in
        sigmas.append(sigma)
        return (1 + 2 * sigma.now, 3 * sigma.next - sigma.now)
      }.rawValue.toCPU()
      XCTAssertEqual(sigmas.count, 2)
      XCTAssertEqual(sigmas[0].now, 0.8)
      XCTAssertEqual(sigmas[0].next, 0.3)
      XCTAssertEqual(sigmas[1].now, 0.8 / (4 - 3 * 0.8), accuracy: 1e-6)
      XCTAssertEqual(sigmas[1].next, 0.3 / (4 - 3 * 0.3), accuracy: 1e-6)
      for frame in 0..<7 {
        for y in 0..<height {
          let sigma = sigmas[y < 3 ? 0 : 1]
          let expected = 2 * (1 + 2 * sigma.now) + 6 * (3 * sigma.next - sigma.now)
          XCTAssertEqual(result[frame, y, 3, 23], expected, accuracy: 1e-5)
        }
      }
      let aliased = sampleScaledAdd(
        left, left, sigma: (now: 0.6, next: 0), version: .minimaxH3, audioShiftRatio: 0.25
      ) {
        sigma in
        (sigma.now, 0)
      }.rawValue.toCPU()
      XCTAssertEqual(aliased[6, 0, 3, 23], 1.2, accuracy: 1e-6)
      XCTAssertEqual(aliased[6, height - 1, 3, 23], 2 * 0.6 / (4 - 3 * 0.6), accuracy: 1e-6)
      XCTAssertEqual(left.rawValue.toCPU()[6, height - 1, 3, 23], 2)
      XCTAssertEqual(right.rawValue.toCPU()[6, height - 1, 3, 23], 6)
    }
  }

  func testH3MultistepCleanPredictionAndLogSNR() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let height = 3 + MiniMaxH3AudioHeight(videoLatentFrames: 7, latentWidth: 4)
      let clean = graph.variable(.GPU(0), .NHWC(7, height, 4, 24), of: Float.self)
      let oldClean = graph.variable(like: clean)
      let noise = graph.variable(like: clean)
      clean.full(0.4)
      oldClean.full(0.2)
      noise.full(-0.9)
      for history: (previous: Float, now: Float, next: Float) in [
        (1, 0.8, 0.4), (0.93, 0.71, 0.2), (0.71, 0.2, 0.01), (0.2, 0.01, 0), (0.01, 0, 0),
      ] {
        let step = (now: history.now, next: history.next)
        let sample = sampleAdd(
          clean, noise: noise, scale: (sample: 1 - step.now, noise: step.now), version: .minimaxH3,
          audioShiftRatio: 0.25)
        let rawVelocity = noise - clean
        let velocity = sampleScaledAdd(
          rawVelocity, rawVelocity, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) {
          sigma in
          let delta = step.now - step.next
          return (delta > 0 ? (sigma.now - sigma.next) / delta : 0, 0)
        }
        let denoised = sampleScaledAdd(
          sample, velocity, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) {
          sigma in
          let delta = sigma.now - sigma.next
          let velocityScale = delta > 0 ? (step.now - step.next) / delta : 0
          return (1, -sigma.now * velocityScale)
        }
        let hLast =
          log(Double(history.previous) / (1 - Double(history.previous)))
          - log(Double(step.now) / (1 - Double(step.now)))
        let h =
          log(Double(step.now) / (1 - Double(step.now)))
          - log(Double(step.next) / (1 - Double(step.next)))
        let r = step.next > 0 && history.previous < 1 ? h / hLast / 2 : 0
        let combined = Functional.add(
          left: denoised, right: oldClean, leftScalar: Float(1 + r), rightScalar: Float(-r))
        let result = sampleScaledAdd(
          sample, combined, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) { sigma in
          let w = sigma.now > 0 ? Double(sigma.next) / Double(sigma.now) : 0
          return (Float(w), Float(1 - w))
        }.rawValue.toCPU()
        let denoisedCPU = denoised.rawValue.toCPU()
        for y in [0, height - 1] {
          let mapped = [history.previous, step.now, step.next].map { value -> Double in
            let s = Double(value)
            return y == 0 ? s : s / (4 - 3 * s)
          }
          let (previous, now, next) = (mapped[0], mapped[1], mapped[2])
          if previous < 1 && next > 0 {
            let audioHLast = log(previous / (1 - previous)) - log(now / (1 - now))
            let audioH = log(now / (1 - now)) - log(next / (1 - next))
            XCTAssertEqual(audioH / audioHLast / 2, r, accuracy: 1e-12)
          }
          let w = now > 0 ? next / now : 0
          let expected = w * ((1 - now) * 0.4 - now * 0.9) + (1 - w) * (0.4 + r * 0.2)
          XCTAssertEqual(denoisedCPU[6, y, 3, 23], 0.4, accuracy: 2e-6)
          XCTAssertEqual(Double(result[6, y, 3, 23]), expected, accuracy: 2e-6)
          XCTAssertTrue(result[6, y, 3, 23].isFinite)
        }
      }
    }
  }

  func testH3UniPCPredictorAndCorrectorCoefficients() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let height = 3 + MiniMaxH3AudioHeight(videoLatentFrames: 1, latentWidth: 4)
      let x = graph.variable(.GPU(0), .NHWC(1, height, 4, 24), of: Float.self)
      let m0 = graph.variable(like: x)
      let mi = graph.variable(like: x)
      let mt = graph.variable(like: x)
      x.full(-0.5)
      m0.full(0.4)
      mi.full(0.2)
      mt.full(0.6)
      for schedule in [[1.0, 0.8, 0.4], [0.91, 0.5, 0.05]] {
        let lambdas = schedule.map { log(1 - $0) - log($0) }
        let h = lambdas[2] - lambdas[1]
        let rk = (lambdas[0] - lambdas[1]) / h
        let hh = -h
        let bh = exp(hh) - 1
        let phi = bh / hh - 1
        let b0 = phi / bh
        let b1 = (phi / hh - 0.5) * 2 / bh
        let c0 = (b0 - b1) / (1 - rk)
        let c1 = b0 - c0
        let d1 = (mi - m0) * Float(1 / rk)
        let dc = Functional.add(
          left: d1, right: mt - m0, leftScalar: Float(c0), rightScalar: Float(c1))
        let step = (now: Float(schedule[1]), next: Float(schedule[2]))
        let base = sampleScaledAdd(x, m0, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25) {
          sigma in
          let w = Double(sigma.next) / Double(sigma.now)
          return (Float(w), Float(1 - w))
        }
        let predictor = sampleScaledAdd(
          base, d1, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) { sigma in
          (1, Float((1 - Double(sigma.next) / Double(sigma.now)) / 2))
        }.rawValue.toCPU()
        let corrector = sampleScaledAdd(
          base, dc, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) { sigma in
          (1, Float(1 - Double(sigma.next) / Double(sigma.now)))
        }.rawValue.toCPU()
        for y in [0, height - 1] {
          let s = schedule.map { y == 0 ? $0 : $0 / (4 - 3 * $0) }
          let lambda = s.map { log(1 - $0) - log($0) }
          let h = lambda[2] - lambda[1]
          let rk = (lambda[0] - lambda[1]) / h
          let bh = exp(-h) - 1
          let phi = bh / -h - 1
          let b0 = phi / bh
          let b1 = (phi / -h - 0.5) * 2 / bh
          let c0 = (b0 - b1) / (1 - rk)
          let c1 = b0 - c0
          let expectedBase = s[2] / s[1] * -0.5 - (1 - s[2]) * bh * 0.4
          let expectedP = expectedBase - (1 - s[2]) * bh * 0.5 * (0.2 - 0.4) / rk
          let expectedC =
            expectedBase - (1 - s[2]) * bh * (c0 * (0.2 - 0.4) / rk + c1 * (0.6 - 0.4))
          XCTAssertEqual(Double(predictor[0, y, 3, 23]), expectedP, accuracy: 2e-6)
          XCTAssertEqual(Double(corrector[0, y, 3, 23]), expectedC, accuracy: 2e-6)
        }
      }
    }
  }

  func testH3TCDNoiseMarginalAndZeroGamma() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let height = 3 + MiniMaxH3AudioHeight(videoLatentFrames: 7, latentWidth: 4)
      let clean = graph.variable(.GPU(0), .NHWC(7, height, 4, 24), of: Float.self)
      let oldNoise = graph.variable(like: clean)
      let freshNoise = graph.variable(like: clean)
      clean.full(0.4)
      oldNoise.full(-0.9)
      freshNoise.full(1.1)
      for step: (now: Float, next: Float) in [(0.8, 0.4), (0.1, 0), (0, 0)] {
        let sample = sampleAdd(
          clean, noise: oldNoise, scale: (sample: 1 - step.now, noise: step.now),
          version: .minimaxH3, audioShiftRatio: 0.25)
        let rawVelocity = oldNoise - clean
        let velocity = sampleScaledAdd(
          rawVelocity, rawVelocity, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) {
          sigma in
          let delta = step.now - step.next
          return (delta > 0 ? (sigma.now - sigma.next) / delta : 0, 0)
        }
        let denoised = sampleScaledAdd(
          sample, velocity, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) {
          sigma in
          let delta = sigma.now - sigma.next
          let velocityScale = delta > 0 ? (step.now - step.next) / delta : 0
          return (1, -sigma.now * velocityScale)
        }
        for gamma: Float in [0, 0.3, 1] {
          let sigmaS = (1 - gamma) * step.next
          let predicted = sampleScaledAdd(
            sample, denoised, sigma: (now: step.now, next: sigmaS), version: .minimaxH3,
            audioShiftRatio: 0.25
          ) { sigma in
            let w = sigma.now > 0 ? Double(sigma.next) / Double(sigma.now) : 0
            return (Float(w), Float(1 - w))
          }
          let noiseSigma = (now: step.next, next: sigmaS)
          let adjusted = sampleScaledAdd(
            predicted, denoised, sigma: noiseSigma, version: .minimaxH3, audioShiftRatio: 0.25
          ) { sigma in
            (1, sigma.next - sigma.now)
          }
          let result = sampleScaledAdd(
            adjusted, freshNoise, sigma: noiseSigma, version: .minimaxH3, audioShiftRatio: 0.25
          ) { sigma in
            let now = Double(sigma.now)
            let next = Double(sigma.next)
            return (1, Float(max(0, now * now - next * next).squareRoot()))
          }.rawValue.toCPU()
          let ddim = Functional.add(
            left: sample, right: velocity, leftScalar: 1, rightScalar: step.next - step.now
          ).rawValue.toCPU()
          for y in [0, height - 1] {
            let t = Double(step.next)
            let s = Double(sigmaS)
            let next = y == 0 ? t : t / (4 - 3 * t)
            let intermediate = y == 0 ? s : s / (4 - 3 * s)
            let std = max(0, next * next - intermediate * intermediate).squareRoot()
            let expected = (1 - next) * 0.4 - intermediate * 0.9 + std * 1.1
            XCTAssertEqual(Double(result[6, y, 3, 23]), expected, accuracy: 2e-6)
            if gamma == 0 {
              XCTAssertEqual(result[6, y, 3, 23], ddim[6, y, 3, 23], accuracy: 2e-6)
            }
            XCTAssertTrue(result[6, y, 3, 23].isFinite)
          }
        }
      }
    }
  }

  func testH3SDEMidpointAndBrownianIntervals() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let height = 3 + MiniMaxH3AudioHeight(videoLatentFrames: 1, latentWidth: 4)
      let clean = graph.variable(.GPU(0), .NHWC(1, height, 4, 24), of: Float.self)
      let clean2 = graph.variable(like: clean)
      let oldNoise = graph.variable(like: clean)
      clean.full(0.4)
      clean2.full(0.8)
      oldNoise.full(-0.9)
      let basis = (0..<3).map { channel in
        var cpu = Tensor<Float>(.CPU, .NHWC(1, height, 4, 24))
        for y in 0..<height {
          for x in 0..<4 {
            for c in 0..<24 {
              cpu[0, y, x, c] = c % 3 == channel ? 1 : 0
            }
          }
        }
        return graph.variable(cpu.toGPU(0))
      }
      for step: (now: Float, next: Float) in [(1, 0.9), (0.8, 0.4), (0.5, 0.01), (1, 0.99)] {
        let midpoint = Float((Double(step.now) * Double(step.next)).squareRoot())
        let first = (now: step.now, next: midpoint)
        let second = (now: midpoint, next: step.next)
        let sample = sampleAdd(
          clean, noise: oldNoise, scale: (sample: 1 - step.now, noise: step.now),
          version: .minimaxH3, audioShiftRatio: 0.25)
        let brownian = sampleScaledAdd(
          basis[0], basis[0], sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) {
          sigma in (sigma.now.squareRoot(), 0)
        }
        let leftW = sampleScaledAdd(
          brownian, basis[1], sigma: first, version: .minimaxH3, audioShiftRatio: 0.25
        ) {
          sigma in
          let ratio = Double(sigma.next) / Double(sigma.now)
          return (
            Float(ratio), Float(((Double(sigma.now) - Double(sigma.next)) * ratio).squareRoot())
          )
        }
        let rightW = brownian - leftW
        var x2 = sampleScaledAdd(
          sample, clean, sigma: first, version: .minimaxH3, audioShiftRatio: 0.25
        ) { sigma in
          let ratio = Double(sigma.next) / Double(sigma.now)
          return (Float(ratio * ratio), Float(1 - ratio * ratio + Double(sigma.next) * (ratio - 1)))
        }
        x2 = sampleScaledAdd(x2, rightW, sigma: first, version: .minimaxH3, audioShiftRatio: 0.25) {
          sigma in
          (
            1,
            Float(
              Double(sigma.next) / Double(sigma.now)
                * (Double(sigma.now) + Double(sigma.next)).squareRoot())
          )
        }
        // Emulate a UNet whose midpoint clean prediction differs from the first one,
        // including the existing secant conversion to video-sigma velocity units.
        let velocity = sampleScaledAdd(
          x2, clean2, sigma: second, version: .minimaxH3, audioShiftRatio: 0.25
        ) { sigma in
          let scale = (sigma.now - sigma.next) / (second.now - second.next) / sigma.now
          return (scale, -scale)
        }
        let denoised2 = sampleScaledAdd(
          x2, velocity, sigma: second, version: .minimaxH3, audioShiftRatio: 0.25
        ) { sigma in
          let velocityScale = (second.now - second.next) / (sigma.now - sigma.next)
          return (1, -sigma.now * velocityScale)
        }
        let derivative = sampleScaledAdd(
          denoised2, clean, sigma: first, version: .minimaxH3, audioShiftRatio: 0.25
        ) {
          sigma in
          let hMid = log(Double(sigma.now) / Double(sigma.next))
          return (Float(1 / hMid), Float(-1 / hMid))
        }
        let denoisedD = sampleScaledAdd(
          clean, derivative, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) {
          sigma in
          (1, Float(log(Double(sigma.now) / Double(sigma.next)) / 2))
        }
        var result = sampleScaledAdd(
          sample, denoisedD, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) { sigma in
          let ratio = Double(sigma.next) / Double(sigma.now)
          return (Float(ratio * ratio), Float(1 - ratio * ratio + Double(sigma.next) * (ratio - 1)))
        }
        let leftW2 = sampleScaledAdd(
          leftW, basis[2], sigma: second, version: .minimaxH3, audioShiftRatio: 0.25
        ) { sigma in
          let ratio = Double(sigma.next) / Double(sigma.now)
          return (
            Float(ratio), Float(((Double(sigma.now) - Double(sigma.next)) * ratio).squareRoot())
          )
        }
        let rightW2 = leftW - leftW2 + rightW
        result = sampleScaledAdd(
          result, rightW2, sigma: step, version: .minimaxH3, audioShiftRatio: 0.25
        ) { sigma in
          (
            1,
            Float(
              Double(sigma.next) / Double(sigma.now)
                * (Double(sigma.now) + Double(sigma.next)).squareRoot())
          )
        }
        let x2CPU = x2.rawValue.toCPU()
        let resultCPU = result.rawValue.toCPU()
        let leftCPU = leftW2.rawValue.toCPU()
        let rightCPU = rightW2.rawValue.toCPU()
        let denoisedCPU = denoised2.rawValue.toCPU()
        for y in [0, height - 1] {
          // Scalar oracle uses the original sigmaUp / sigmaDown equations.
          let mapped = [step.now, midpoint, step.next].map { value -> Double in
            let s = Double(value)
            return y == 0 ? s : s / (4 - 3 * s)
          }
          let (now, mid, next) = (mapped[0], mapped[1], mapped[2])
          let up1 = min(mid, (mid * mid * (now * now - mid * mid) / (now * now)).squareRoot())
          let down1 = (mid * mid - up1 * up1).squareRoot()
          let up2 = min(next, (next * next * (now * now - next * next) / (now * now)).squareRoot())
          let down2 = (next * next - up2 * up2).squareRoot()
          let d = 0.4 + log(now / next) / (2 * log(now / mid)) * (0.8 - 0.4)
          let sample = (1 - now) * 0.4 - now * 0.9
          var leftVariance: Double = 0
          var rightVariance: Double = 0
          var covariance: Double = 0
          for c in 0..<3 {
            let w = c == 0 ? now.squareRoot() : 0
            let left = (mid / now) * w + (c == 1 ? ((now - mid) * mid / now).squareRoot() : 0)
            let left2 =
              (next / mid) * left + (c == 2 ? ((mid - next) * next / mid).squareRoot() : 0)
            let expectedMid =
              (down1 / now) * sample + (1 - down1 / now + down1 - mid) * 0.4
              + up1 / (now - mid).squareRoot() * (w - left)
            let expected =
              (down2 / now) * sample + (1 - down2 / now + down2 - next) * d
              + up2 / (now - next).squareRoot() * (w - left2)
            XCTAssertEqual(Double(x2CPU[0, y, 0, c]), expectedMid, accuracy: 2e-5)
            XCTAssertEqual(Double(resultCPU[0, y, 0, c]), expected, accuracy: 2e-5)
            XCTAssertEqual(denoisedCPU[0, y, 0, c], 0.8, accuracy: 2e-5)
            let l = Double(leftCPU[0, y, 0, c])
            let r = Double(rightCPU[0, y, 0, c])
            leftVariance += l * l
            rightVariance += r * r
            covariance += l * r
          }
          XCTAssertEqual(leftVariance, next, accuracy: 2e-6)
          XCTAssertEqual(rightVariance, now - next, accuracy: 2e-6)
          XCTAssertEqual(covariance, 0, accuracy: 2e-6)
        }
      }
    }
  }

  func testH3AncestralUpdateWithScaledVelocity() {
    let graph = DynamicGraph()
    graph.withNoGrad {
      let height = 3 + MiniMaxH3AudioHeight(videoLatentFrames: 7, latentWidth: 4)
      let sigmaUp = { (sigma: (now: Float, next: Float)) -> Float in
        guard sigma.now > 0 else { return 0 }
        return min(
          sigma.next,
          (sigma.next * sigma.next
            * (sigma.now * sigma.now - sigma.next * sigma.next) / (sigma.now * sigma.now))
            .squareRoot())
      }
      for stepSigma: (now: Float, next: Float) in [
        (1, 0.9), (0.8, 0.4), (0.5, 0.01), (0.1, 0), (0, 0),
      ] {
        let audioSigma = (
          now: stepSigma.now / (4 - 3 * stepSigma.now),
          next: stepSigma.next / (4 - 3 * stepSigma.next)
        )
        let videoDelta = stepSigma.now - stepSigma.next
        let audioVelocityScale =
          videoDelta > 0 ? (audioSigma.now - audioSigma.next) / videoDelta : 0
        let clean: Float = 0.4
        let oldNoise: Float = -0.9
        let newNoise: Float = 1.1
        var sampleCPU = Tensor<Float>(.CPU, .NHWC(7, height, 4, 24))
        var velocityCPU = Tensor<Float>(.CPU, .NHWC(7, height, 4, 24))
        for frame in 0..<7 {
          for y in 0..<height {
            for x in 0..<4 {
              for c in 0..<24 {
                let sigma = y < 3 ? stepSigma.now : audioSigma.now
                sampleCPU[frame, y, x, c] = (1 - sigma) * clean + sigma * oldNoise
                velocityCPU[frame, y, x, c] = (oldNoise - clean) * (y < 3 ? 1 : audioVelocityScale)
              }
            }
          }
        }
        let sample = graph.variable(sampleCPU.toGPU(0))
        let velocity = graph.variable(velocityCPU.toGPU(0))
        let noise = graph.variable(like: sample)
        noise.full(newNoise)
        let denoised = sampleScaledAdd(
          sample, velocity, sigma: stepSigma, version: .minimaxH3, audioShiftRatio: 0.25
        ) {
          sigma in
          let delta = sigma.now - sigma.next
          let velocityScale = delta > 0 ? videoDelta / delta : 0
          return (1, -sigma.now * velocityScale)
        }.rawValue.toCPU()
        let drift = sampleScaledAdd(
          sample, velocity, sigma: stepSigma, version: .minimaxH3, audioShiftRatio: 0.25
        ) {
          sigma in
          let delta = sigma.now - sigma.next
          let velocityScale = delta > 0 ? videoDelta / delta : 0
          let up = sigmaUp(sigma)
          let down = (sigma.next * sigma.next - up * up).squareRoot()
          return (
            1 - sigma.next + down,
            (down - sigma.now * down - sigma.now * (1 - sigma.next)) * velocityScale
          )
        }
        let result = sampleScaledAdd(
          drift, noise, sigma: stepSigma, version: .minimaxH3, audioShiftRatio: 0.25
        ) { sigma in
          (1, sigmaUp(sigma))
        }.rawValue.toCPU()
        for frame in 0..<7 {
          for y in 0..<height {
            let sigma = y < 3 ? stepSigma : audioSigma
            let up = sigmaUp(sigma)
            let down = (sigma.next * sigma.next - up * up).squareRoot()
            let expected = (1 - sigma.next) * clean + down * oldNoise + up * newNoise
            XCTAssertEqual(denoised[frame, y, 3, 23], clean, accuracy: 1e-5)
            XCTAssertEqual(result[frame, y, 3, 23], expected, accuracy: 1e-5)
            XCTAssertTrue(result[frame, y, 3, 23].isFinite)
          }
        }
      }
    }
  }
}
