import DataModels
import Foundation

public enum GenerationEstimator {

  public static func estimateTotalDuration(
    _ estimation: GenerationEstimation, signposts: Set<ImageGeneratorSignpost>
  ) -> TimeInterval {
    var estimatedTotalDuration: TimeInterval = 0
    for signpost in signposts {
      switch signpost {
      case .textEncoded:
        estimatedTotalDuration += TimeInterval(estimation.textEncoded)
      case .imageEncoded:
        estimatedTotalDuration += TimeInterval(estimation.imageEncoded)
      case .controlsGenerated:
        continue
      case .sampling(let steps):
        estimatedTotalDuration += TimeInterval(estimation.samplingStep) * TimeInterval(steps)
      case .imageDecoded:
        estimatedTotalDuration += TimeInterval(estimation.imageDecoded)
      case .secondPassImageEncoded:
        estimatedTotalDuration += TimeInterval(estimation.secondPassImageEncoded)
      case .secondPassSampling(let steps):
        estimatedTotalDuration +=
          TimeInterval(estimation.secondPassSamplingStep) * TimeInterval(steps)
      case .secondPassImageDecoded:
        estimatedTotalDuration += TimeInterval(estimation.secondPassImageDecoded)
      case .faceRestored:
        estimatedTotalDuration += TimeInterval(estimation.faceRestored)
      case .imageUpscaled:
        estimatedTotalDuration += TimeInterval(estimation.imageUpscaled)
      }
    }
    return estimatedTotalDuration
  }

  public static func correct(
    duration: TimeInterval,
    estimation: inout GenerationEstimation, signpost: ImageGeneratorSignpost,
    signposts: Set<ImageGeneratorSignpost>
  ) {
    switch signpost {
    case .textEncoded:
      estimation.textEncoded = Float(duration)
    case .imageEncoded:
      estimation.imageEncoded = Float(duration) - estimation.textEncoded
    case .controlsGenerated:
      break
    case .sampling(let step):
      if step > 0 {
        estimation.samplingStep =
          (Float(duration) - estimation.textEncoded
            - (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)) / Float(step)
      }
    case .imageDecoded:
      var timePassed: Float = 0
      for signpost in signposts {
        switch signpost {
        case .textEncoded:
          timePassed += estimation.textEncoded
        case .imageEncoded:
          timePassed += estimation.imageEncoded
        case .sampling(let steps):
          timePassed += estimation.samplingStep * Float(steps)
        default:
          break
        }
      }
      estimation.imageDecoded = Float(duration) - timePassed
    case .secondPassImageEncoded:
      var timePassed: Float = 0
      for signpost in signposts {
        switch signpost {
        case .textEncoded:
          timePassed += estimation.textEncoded
        case .imageEncoded:
          timePassed += estimation.imageEncoded
        case .sampling(let steps):
          timePassed += estimation.samplingStep * Float(steps)
        case .imageDecoded:
          timePassed += estimation.imageDecoded
        default:
          break
        }
      }
      estimation.secondPassImageEncoded = Float(duration) - timePassed
    case .secondPassSampling(let step):
      var timePassed: Float = 0
      for signpost in signposts {
        switch signpost {
        case .textEncoded:
          timePassed += estimation.textEncoded
        case .imageEncoded:
          timePassed += estimation.imageEncoded
        case .sampling(let steps):
          timePassed += estimation.samplingStep * Float(steps)
        case .imageDecoded:
          timePassed += estimation.imageDecoded
        case .secondPassImageEncoded:
          timePassed += estimation.secondPassImageEncoded
        default:
          break
        }
      }
      if step > 0 {
        estimation.secondPassSamplingStep = (Float(duration) - timePassed) / Float(step)
      }
    case .secondPassImageDecoded:
      var timePassed: Float = 0
      for signpost in signposts {
        switch signpost {
        case .textEncoded:
          timePassed += estimation.textEncoded
        case .imageEncoded:
          timePassed += estimation.imageEncoded
        case .sampling(let steps):
          timePassed += estimation.samplingStep * Float(steps)
        case .imageDecoded:
          timePassed += estimation.imageDecoded
        case .secondPassImageEncoded:
          timePassed += estimation.secondPassImageEncoded
        case .secondPassSampling(let steps):
          timePassed += estimation.secondPassSamplingStep * Float(steps)
        default:
          break
        }
      }
      estimation.secondPassImageDecoded = Float(duration) - timePassed
    case .faceRestored:
      guard signposts.contains(.imageUpscaled) else { return }
      var timePassed: Float = 0
      for signpost in signposts {
        switch signpost {
        case .textEncoded:
          timePassed += estimation.textEncoded
        case .imageEncoded:
          timePassed += estimation.imageEncoded
        case .sampling(let steps):
          timePassed += estimation.samplingStep * Float(steps)
        case .imageDecoded:
          timePassed += estimation.imageDecoded
        case .secondPassImageEncoded:
          timePassed += estimation.secondPassImageEncoded
        case .secondPassSampling(let steps):
          timePassed += estimation.secondPassSamplingStep * Float(steps)
        case .secondPassImageDecoded:
          timePassed += estimation.secondPassImageDecoded
        default:
          break
        }
      }
      estimation.faceRestored = Float(duration) - timePassed
    case .imageUpscaled:
      guard signposts.contains(.imageUpscaled) else { return }
      var timePassed: Float = 0
      for signpost in signposts {
        switch signpost {
        case .textEncoded:
          timePassed += estimation.textEncoded
        case .imageEncoded:
          timePassed += estimation.imageEncoded
        case .sampling(let steps):
          timePassed += estimation.samplingStep * Float(steps)
        case .imageDecoded:
          timePassed += estimation.imageDecoded
        case .secondPassImageEncoded:
          timePassed += estimation.secondPassImageEncoded
        case .secondPassSampling(let steps):
          timePassed += estimation.secondPassSamplingStep * Float(steps)
        case .secondPassImageDecoded:
          timePassed += estimation.secondPassImageDecoded
        case .faceRestored:
          timePassed += estimation.faceRestored
        default:
          break
        }
      }
      estimation.imageUpscaled = Float(duration) - timePassed
    }
  }

  // This is a value from 0 to 1, estimating progress made so far.
  public static func estimateUpToDateDuration(
    from estimation: GenerationEstimation, signpost: ImageGeneratorSignpost,
    signposts: Set<ImageGeneratorSignpost>
  ) -> (updatedDurationThisFar: TimeInterval, updatedEstimatedTotalDuration: TimeInterval) {
    var updatedEstimatedTotalDuration: TimeInterval = 0
    for signpost in signposts {
      switch signpost {
      case .textEncoded:
        updatedEstimatedTotalDuration += TimeInterval(estimation.textEncoded)
      case .imageEncoded:
        updatedEstimatedTotalDuration += TimeInterval(estimation.imageEncoded)
      case .controlsGenerated:
        continue
      case .sampling(let steps):
        updatedEstimatedTotalDuration += TimeInterval(estimation.samplingStep) * TimeInterval(steps)
      case .imageDecoded:
        updatedEstimatedTotalDuration += TimeInterval(estimation.imageDecoded)
      case .secondPassImageEncoded:
        updatedEstimatedTotalDuration += TimeInterval(estimation.secondPassImageEncoded)
      case .secondPassSampling(let steps):
        updatedEstimatedTotalDuration +=
          TimeInterval(estimation.secondPassSamplingStep) * TimeInterval(steps)
      case .secondPassImageDecoded:
        updatedEstimatedTotalDuration += TimeInterval(estimation.secondPassImageDecoded)
      case .faceRestored:
        updatedEstimatedTotalDuration += TimeInterval(estimation.faceRestored)
      case .imageUpscaled:
        updatedEstimatedTotalDuration += TimeInterval(estimation.imageUpscaled)
      }
    }
    let updatedDurationThisFar: TimeInterval
    switch signpost {
    case .textEncoded:
      updatedDurationThisFar = TimeInterval(estimation.textEncoded)
    case .imageEncoded:
      updatedDurationThisFar = TimeInterval(estimation.textEncoded + estimation.imageEncoded)
    case .controlsGenerated:
      updatedDurationThisFar = TimeInterval(estimation.textEncoded + estimation.imageEncoded)
    case .sampling(let step):
      updatedDurationThisFar = TimeInterval(
        estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
          + estimation.samplingStep * Float(step))
    case .imageDecoded:
      let firstStageSampling: Float = signposts.reduce(0) { _, signpost in
        if case .sampling(let steps) = signpost {
          return estimation.samplingStep * Float(steps)
        }
        return 0
      }
      if signposts.contains(.secondPassImageDecoded) || signposts.contains(.faceRestored)
        || signposts.contains(.imageUpscaled)
      {
        updatedDurationThisFar = TimeInterval(
          estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
            + firstStageSampling + estimation.imageDecoded)
      } else {
        updatedDurationThisFar = updatedEstimatedTotalDuration
      }
    case .secondPassImageEncoded:
      let firstStageSampling: Float = signposts.reduce(0) { _, signpost in
        if case .sampling(let steps) = signpost {
          return estimation.samplingStep * Float(steps)
        }
        return 0
      }
      updatedDurationThisFar = TimeInterval(
        estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
          + firstStageSampling + estimation.imageDecoded + estimation.secondPassImageEncoded)
    case .secondPassSampling(let step):
      let firstStageSampling: Float = signposts.reduce(0) { _, signpost in
        if case .sampling(let steps) = signpost {
          return estimation.samplingStep * Float(steps)
        }
        return 0
      }
      updatedDurationThisFar = TimeInterval(
        estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
          + firstStageSampling + estimation.imageDecoded + estimation.secondPassImageEncoded
          + estimation.secondPassSamplingStep * Float(step))
    case .secondPassImageDecoded:
      let firstStageSampling: Float = signposts.reduce(0) { _, signpost in
        if case .sampling(let steps) = signpost {
          return estimation.samplingStep * Float(steps)
        }
        return 0
      }
      if signposts.contains(.faceRestored) || signposts.contains(.imageUpscaled) {
        let secondPassSampling: Float = signposts.reduce(0) { _, signpost in
          if case .secondPassSampling(let steps) = signpost {
            return estimation.secondPassSamplingStep * Float(steps)
          }
          return 0
        }
        updatedDurationThisFar = TimeInterval(
          estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
            + firstStageSampling + estimation.imageDecoded + estimation.secondPassImageEncoded
            + secondPassSampling + estimation.secondPassImageDecoded)
      } else {
        updatedDurationThisFar = updatedEstimatedTotalDuration
      }
    case .faceRestored:
      let firstStageSampling: Float = signposts.reduce(0) { _, signpost in
        if case .sampling(let steps) = signpost {
          return estimation.samplingStep * Float(steps)
        }
        return 0
      }
      let secondPassSampling: Float = signposts.reduce(0) { _, signpost in
        if case .secondPassSampling(let steps) = signpost {
          return estimation.secondPassSamplingStep * Float(steps)
        }
        return 0
      }
      updatedDurationThisFar = TimeInterval(
        estimation.textEncoded + (signposts.contains(.imageEncoded) ? estimation.imageEncoded : 0)
          + firstStageSampling + estimation.imageDecoded + estimation.secondPassImageEncoded
          + secondPassSampling + estimation.secondPassImageDecoded + estimation.faceRestored)
    case .imageUpscaled:
      updatedDurationThisFar = updatedEstimatedTotalDuration
    }
    return (updatedDurationThisFar, updatedEstimatedTotalDuration)
  }
}

public final class ProgressBarPrinter {
  private var hasPrintedBefore = false
  private let output: UnsafeMutablePointer<FILE>

  public init(output: UnsafeMutablePointer<FILE> = stdout) {
    self.output = output
  }

  private func terminalWidth() -> Int? {
    var winsize: winsize = winsize()
    let result = ioctl(fileno(output), UInt(TIOCGWINSZ), &winsize)
    return result == 0 ? Int(winsize.ws_col) : nil
  }

  public func update(progress: Float, label: String, detail: String?) {
    if !hasPrintedBefore {
      fputs("\n", output)
      hasPrintedBefore = true
    }
    let percent = Int(round(progress * 100))
    let template =
      if let detail, !detail.isEmpty {
        "\(label) \(detail) [] XXX%"
      } else {
        "\(label) [] XXX%"
      }
    // Note that in Xcode, terminalWidth() returns 0
    let terminalWidth = max(terminalWidth() ?? 0, template.count + 1)  // Minimum bar size of 1

    let barWidth = terminalWidth - template.count
    let filledWidth = Int(round(Float(barWidth) * progress))
    let filledPart = String(repeating: "█", count: filledWidth)
    let emptyPart = String(repeating: " ", count: barWidth - filledWidth)
    let bar = "[\(filledPart)\(emptyPart)]"

    var string = template
    string = string.replacingOccurrences(of: "[]", with: bar)
    let numberString = String(percent).padding(toLength: 3, withPad: " ", startingAt: 0)
    string = string.replacingOccurrences(of: "XXX", with: numberString)
    let lineClearString = "\u{1B}[1A\u{1B}[K"
    fputs("\(lineClearString)\(string)\n", output)
    fflush(output)
  }
}
