import Foundation
import NNC

#if canImport(AVFoundation)
  import AVFoundation
#endif

public enum AudioInputError: Swift.Error, LocalizedError {
  case unsupportedPlatform
  case cannotOpenAudio(String)
  case cannotReadAudio(String)
  case cannotConvertAudio
  case invalidSampleRate(Int)

  public var errorDescription: String? {
    switch self {
    case .unsupportedPlatform:
      return "Audio file conversion requires AVFoundation and is unavailable on this platform."
    case .cannotOpenAudio(let path):
      return "Cannot open audio file at \(path)."
    case .cannotReadAudio(let path):
      return "Cannot read audio file at \(path)."
    case .cannotConvertAudio:
      return "Cannot convert audio to PCM."
    case .invalidSampleRate(let sampleRate):
      return "Audio sample rate must be positive, but received \(sampleRate) Hz."
    }
  }
}

public struct AudioInput {
  /// PCM in [channels, samples] order, preserving mono or stereo at `sampleRate` Hz.
  public let waveform: Tensor<Float>
  public let sampleRate: Int

  public init(waveform: Tensor<Float>, sampleRate: Int) {
    precondition(waveform.kind == .CPU && waveform.shape.count == 2)
    precondition((waveform.shape[0] == 1 || waveform.shape[0] == 2) && waveform.shape[1] > 0)
    precondition(sampleRate > 0)
    self.waveform = waveform
    self.sampleRate = sampleRate
  }

  public init(samples: [Float], sampleRate: Int) {
    self.init(waveform: Tensor<Float>(samples, .CPU, .NC(1, samples.count)), sampleRate: sampleRate)
  }

  public func videoFrameCount(framesPerSecond: Int) -> Int {
    precondition(framesPerSecond > 0)
    return max(1, (waveform.shape[1] * framesPerSecond + sampleRate - 1) / sampleRate)
  }

  public func waveformTensor(videoFrames: Int, framesPerSecond: Int) -> Tensor<Float> {
    precondition(videoFrames > 0)
    precondition(framesPerSecond > 0)
    let targetSamples = (videoFrames * sampleRate + framesPerSecond - 1) / framesPerSecond
    let copiedSamples = min(targetSamples, waveform.shape[1])
    var tensor = Tensor<Float>(
      Array(repeating: 0, count: 2 * targetSamples), .CPU, .NC(2, targetSamples))
    for channel in 0..<2 {
      let sourceChannel = min(channel, waveform.shape[0] - 1)
      tensor[channel..<(channel + 1), 0..<copiedSamples] =
        waveform[sourceChannel..<(sourceChannel + 1), 0..<copiedSamples]
    }
    return tensor
  }

  public init(contentsOf path: String, sampleRate: Int) throws {
    guard sampleRate > 0 else {
      throw AudioInputError.invalidSampleRate(sampleRate)
    }
    #if canImport(AVFoundation)
      let url = URL(fileURLWithPath: path)
      guard let file = try? AVAudioFile(forReading: url) else {
        throw AudioInputError.cannotOpenAudio(path)
      }
      let channelCount = min(file.processingFormat.channelCount, 2)
      guard
        let outputFormat = AVAudioFormat(
          commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: channelCount,
          interleaved: false),
        let converter = AVAudioConverter(from: file.processingFormat, to: outputFormat)
      else {
        throw AudioInputError.cannotConvertAudio
      }
      let inputCapacity: AVAudioFrameCount = 65_536
      var samples = Array(repeating: [Float](), count: Int(channelCount))
      var inputEnded = false
      var readError: AudioInputError? = nil
      let ratio = Double(sampleRate) / file.processingFormat.sampleRate
      conversionLoop: while true {
        guard
          let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount((Double(inputCapacity) * ratio).rounded(.up) + 16))
        else {
          throw AudioInputError.cannotConvertAudio
        }
        var conversionError: NSError? = nil
        let status = converter.convert(to: outputBuffer, error: &conversionError) {
          packetCount, statusPointer in
          guard !inputEnded else {
            statusPointer.pointee = .endOfStream
            return nil
          }
          let remainingFrames = file.length - file.framePosition
          guard remainingFrames > 0 else {
            statusPointer.pointee = .endOfStream
            inputEnded = true
            return nil
          }
          let remainingCapacity = AVAudioFrameCount(
            min(remainingFrames, AVAudioFramePosition(inputCapacity)))
          guard
            let inputBuffer = AVAudioPCMBuffer(
              pcmFormat: file.processingFormat,
              frameCapacity: max(1, min(packetCount, remainingCapacity)))
          else {
            statusPointer.pointee = .endOfStream
            inputEnded = true
            readError = .cannotReadAudio(path)
            return nil
          }
          do {
            try file.read(into: inputBuffer)
          } catch {
            statusPointer.pointee = .endOfStream
            inputEnded = true
            readError = .cannotReadAudio(path)
            return nil
          }
          if inputBuffer.frameLength == 0 {
            statusPointer.pointee = .endOfStream
            inputEnded = true
            return nil
          }
          statusPointer.pointee = .haveData
          return inputBuffer
        }
        if let channelData = outputBuffer.floatChannelData, outputBuffer.frameLength > 0 {
          for channel in 0..<Int(channelCount) {
            samples[channel].append(
              contentsOf: UnsafeBufferPointer(
                start: channelData[channel], count: Int(outputBuffer.frameLength)))
          }
        }
        if let readError {
          throw readError
        }
        if let conversionError {
          throw conversionError
        }
        switch status {
        case .haveData, .inputRanDry:
          continue
        case .endOfStream:
          break conversionLoop
        case .error:
          throw AudioInputError.cannotConvertAudio
        @unknown default:
          throw AudioInputError.cannotConvertAudio
        }
      }
      guard let sampleCount = samples.first?.count, sampleCount > 0 else {
        throw AudioInputError.cannotConvertAudio
      }
      self.init(
        waveform: Tensor<Float>(samples.flatMap { $0 }, .CPU, .NC(Int(channelCount), sampleCount)),
        sampleRate: sampleRate)
    #else
      throw AudioInputError.unsupportedPlatform
    #endif
  }
}
