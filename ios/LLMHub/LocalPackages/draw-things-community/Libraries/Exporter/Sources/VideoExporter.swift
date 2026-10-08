import AVFoundation
import AudioToolbox
import CoreMedia
import Diffusion
import NNC
import UIKit

public protocol VideoExporterDelegate: AnyObject {
  func didProcessedFrameCount(processedFrameCount: Int)
}

public final class VideoExporter {
  private struct PendingAudioSegment {
    let tensor: Tensor<Float>
    let timing: Double
  }

  public enum Format: Int {
    case proRes4444 = 0
    case proRes422HQ
    case h264
    case hevc
  }

  public weak var delegate: VideoExporterDelegate? = nil
  private var pendingFrames = [CVPixelBuffer]()
  private var pendingAudioSegments = [PendingAudioSegment]()
  private lazy var videoExportQueue = DispatchQueue(
    label: "com.draw-things.video-exporter", qos: .userInteractive)
  public private(set) var isExportingInProgress = false  // main queue
  private var frameAppendingFinished = false  // videoExportQueue
  private var audioAppendingFinished = false  // videoExportQueue
  private var pixelBufferPool: CVPixelBufferPool? = nil
  private var exportingImageSize: CGSize? = nil
  private var processedFrameCount = 0
  private var aborted = false
  private var exportAudioSampleRate: Double? = nil

  public func abortVideoExporting() {
    videoExportQueue.async {
      self.aborted = true
    }
  }
  private let format: Format
  public init(format: Format) {
    self.format = format
  }

  public func startVideoExporting(
    frameDuration: CMTime, outputFileURL: URL, imageSize: CGSize,
    audioSampleRate: Double? = nil, completion: @escaping (Bool) -> Void
  ) {
    self.isExportingInProgress = true
    self.exportingImageSize = imageSize

    videoExportQueue.async {
      self.pendingFrames.removeAll()
      self.pendingAudioSegments.removeAll()
      self.frameAppendingFinished = false
      self.audioAppendingFinished = false
      self.aborted = false
      self.exportAudioSampleRate = audioSampleRate
      self.processedFrameCount = 0
      self.initialVideoExportingOnExporterQueue(
        frameDuration: frameDuration, outputFileURL: outputFileURL, imageSize: imageSize
      ) { finish in
        DispatchQueue.main.async {
          self.isExportingInProgress = false
          self.exportingImageSize = nil
          self.pixelBufferPool = nil
          self.exportAudioSampleRate = nil

          completion(finish)
        }
      }
    }
  }

  public func markFrameAppendingEnd() {
    videoExportQueue.async {
      self.frameAppendingFinished = true
      self.audioAppendingFinished = true
    }
  }

  /// Appends a stereo waveform tensor shaped [2, frames] with normalized samples in [-1, 1].
  /// The tensor format matches TensorAudioPlayer.play(_:timing:), but `timing` is the start time
  /// on the exported video timeline.
  public func appendAudio(audio: Tensor<Float>, timing: Double) {
    precondition(audio.kind == .CPU)
    precondition(audio.shape.count == 2 && audio.shape[0] == 2)
    precondition(audio.shape[1] > 0)
    precondition(timing >= 0)
    videoExportQueue.async {
      guard !self.aborted else { return }
      if let lastTiming = self.pendingAudioSegments.last?.timing {
        precondition(timing >= lastTiming)
      }
      self.pendingAudioSegments.append(PendingAudioSegment(tensor: audio, timing: timing))
    }
  }

  public func appendFrame(
    frame: Tensor<FloatType>, completion: @escaping () -> Void
  ) {
    videoExportQueue.async {
      guard !self.aborted else { return }
      self.processedFrameCount += 1  // for the original frame
      self.delegate?.didProcessedFrameCount(processedFrameCount: self.processedFrameCount)
      self.appendFrameToExport(frame: frame)
      completion()
    }
  }

  private func appendFrameToExport(frame: Tensor<FloatType>) {
    guard let exportingImageSize = exportingImageSize, !aborted,
      let pixelBufferPool = pixelBufferPool
    else {
      return
    }
    let imageHeight = Int(frame.shape[1])
    let imageWidth = Int(frame.shape[2])
    // resize frame size to fit what we want.
    guard imageHeight == Int(exportingImageSize.height), imageWidth == Int(exportingImageSize.width)
    else {
      let graph = DynamicGraph()
      graph.withNoGrad {
        let widthScale = Float(exportingImageSize.width) / Float(imageWidth)
        let heightScale = Float(exportingImageSize.height) / Float(imageHeight)
        let newFrame = Upsample(.bilinear, widthScale: widthScale, heightScale: heightScale)(
          graph.variable(frame.toGPU(0))
        ).rawValue.toCPU()
        if let buffer = newPixelBufferFrom(
          imageTensor: newFrame, pixelBufferPool: pixelBufferPool)
        {
          pendingFrames.append(buffer)
        }
      }
      return
    }
    if let buffer = newPixelBufferFrom(
      imageTensor: frame, pixelBufferPool: pixelBufferPool)
    {
      pendingFrames.append(buffer)
    }
  }

  private func initialVideoExportingOnExporterQueue(
    frameDuration: CMTime, outputFileURL: URL, imageSize: CGSize,
    completion: @escaping (Bool) -> Void
  ) {
    dispatchPrecondition(condition: .onQueue(videoExportQueue))

    let imageHeight = imageSize.height
    let imageWidth = imageSize.width

    let videoSettings: [String: Any]
    switch format {
    case .proRes4444:
      videoSettings = [
        AVVideoCodecKey: AVVideoCodecType.proRes4444.rawValue,
        AVVideoWidthKey: NSNumber(value: Float(imageWidth)),
        AVVideoHeightKey: NSNumber(value: Float(imageHeight)),
      ]
    case .proRes422HQ:
      videoSettings = [
        AVVideoCodecKey: AVVideoCodecType.proRes422HQ.rawValue,
        AVVideoWidthKey: NSNumber(value: Float(imageWidth)),
        AVVideoHeightKey: NSNumber(value: Float(imageHeight)),
      ]
    case .h264:
      videoSettings = [
        AVVideoCodecKey: AVVideoCodecType.h264.rawValue,
        AVVideoWidthKey: NSNumber(value: Float(imageWidth)),
        AVVideoHeightKey: NSNumber(value: Float(imageHeight)),
        AVVideoCompressionPropertiesKey: [
          AVVideoAverageBitRateKey: max(9_500_000, Int((imageWidth * imageHeight * 5).rounded())),  // 9.5 Mbps, or 5-bit per pixel, whichever is higher.
          AVVideoProfileLevelKey: AVVideoProfileLevelH264High41,
          AVVideoMaxKeyFrameIntervalKey: 30,
          AVVideoAllowFrameReorderingKey: true,
        ],
      ]
    case .hevc:
      videoSettings = [
        AVVideoCodecKey: AVVideoCodecType.hevc.rawValue,
        AVVideoWidthKey: NSNumber(value: Float(imageWidth)),
        AVVideoHeightKey: NSNumber(value: Float(imageHeight)),
        AVVideoCompressionPropertiesKey: [
          AVVideoAverageBitRateKey: max(7_500_000, Int((imageWidth * imageHeight * 4).rounded())),  // 7.5 Mbps, or 4-bit per pixel, whichever is higher.
          AVVideoMaxKeyFrameIntervalKey: 30,
          AVVideoAllowFrameReorderingKey: true,
        ],
      ]
    }

    let pixelBufferAttributes = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ]

    let videoWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
    videoWriterInput.expectsMediaDataInRealTime = false
    let audioSampleRate = exportAudioSampleRate
    let audioWriterInput = audioSampleRate.map {
      let audioSettings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: $0,
        AVNumberOfChannelsKey: 2,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsNonInterleaved: false,
      ]
      return AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
    }
    audioWriterInput?.expectsMediaDataInRealTime = false
    let pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: videoWriterInput, sourcePixelBufferAttributes: pixelBufferAttributes)

    guard let videoWriter = try? AVAssetWriter(outputURL: outputFileURL, fileType: .mov) else {
      completion(false)
      return
    }
    guard videoWriter.canAdd(videoWriterInput) else {
      completion(false)
      return
    }
    if let audioWriterInput = audioWriterInput, !videoWriter.canAdd(audioWriterInput) {
      completion(false)
      return
    }
    videoWriter.add(videoWriterInput)
    if let audioWriterInput = audioWriterInput {
      videoWriter.add(audioWriterInput)
    }
    videoWriter.startWriting()
    videoWriter.startSession(atSourceTime: .zero)
    pixelBufferPool = pixelBufferAdaptor.pixelBufferPool
    guard pixelBufferPool != nil else {
      completion(false)
      return
    }

    var completionCalled = false
    var videoInputFinished = false
    var audioInputFinished = (audioWriterInput == nil)
    var audioSegmentIndex = 0
    var audioNextSampleIndex = 0
    var time = CMTime.zero
    let audioChunkFrameCount = 1024

    func cancelAndComplete() {
      guard !completionCalled else { return }
      completionCalled = true
      videoWriter.cancelWriting()
      completion(false)
    }

    func finishWritingIfReady() {
      guard !completionCalled, videoInputFinished, audioInputFinished else { return }
      completionCalled = true
      videoWriter.finishWriting {
        completion(videoWriter.status == .completed)
      }
    }

    videoWriterInput.requestMediaDataWhenReady(on: videoExportQueue) { [weak self] in
      guard let self = self else { return }
      while videoWriterInput.isReadyForMoreMediaData {
        guard !completionCalled else { return }
        if let buffer = self.pendingFrames.first {
          guard pixelBufferAdaptor.append(buffer, withPresentationTime: time) else {
            cancelAndComplete()
            return
          }
          time = CMTimeAdd(time, frameDuration)
          self.pendingFrames.removeFirst()
        } else {
          if self.pendingFrames.isEmpty, self.frameAppendingFinished {
            if !videoInputFinished {
              videoWriterInput.markAsFinished()
              videoInputFinished = true
            }
            finishWritingIfReady()
            break
          } else if self.aborted {
            cancelAndComplete()
            return
          } else {
            break
          }
        }
      }
      Thread.sleep(forTimeInterval: 0.01)
    }

    audioWriterInput?.requestMediaDataWhenReady(on: videoExportQueue) { [weak self] in
      guard let self = self else { return }
      guard let audioWriterInput = audioWriterInput, let audioSampleRate = audioSampleRate else {
        return
      }
      while audioWriterInput.isReadyForMoreMediaData {
        guard !completionCalled else { return }
        if self.aborted {
          cancelAndComplete()
          return
        }
        if audioSegmentIndex < self.pendingAudioSegments.count {
          let pendingAudioSegment = self.pendingAudioSegments[audioSegmentIndex]
          let totalSampleCount = Int(pendingAudioSegment.tensor.shape[1])
          if audioNextSampleIndex < totalSampleCount {
            let chunkSampleCount = min(
              audioChunkFrameCount, totalSampleCount - audioNextSampleIndex)
            let segmentPresentationSampleOffset = max(
              0, Int((pendingAudioSegment.timing * audioSampleRate).rounded(.down)))
            guard
              let sampleBuffer = self.newAudioSampleBufferFrom(
                audioTensor: pendingAudioSegment.tensor, tensorSampleOffset: audioNextSampleIndex,
                presentationSampleOffset: segmentPresentationSampleOffset + audioNextSampleIndex,
                sampleCount: chunkSampleCount, sampleRate: audioSampleRate)
            else {
              cancelAndComplete()
              return
            }
            guard audioWriterInput.append(sampleBuffer) else {
              cancelAndComplete()
              return
            }
            audioNextSampleIndex += chunkSampleCount
          } else {
            audioSegmentIndex += 1
            audioNextSampleIndex = 0
          }
        } else if self.audioAppendingFinished {
          if !audioInputFinished {
            audioWriterInput.markAsFinished()
            audioInputFinished = true
          }
          finishWritingIfReady()
          break
        } else {
          break
        }
      }
      Thread.sleep(forTimeInterval: 0.01)
    }
  }

  private func newAudioSampleBufferFrom(
    audioTensor: Tensor<Float>, tensorSampleOffset: Int, presentationSampleOffset: Int,
    sampleCount: Int,
    sampleRate: Double
  ) -> CMSampleBuffer? {
    guard sampleCount > 0 else { return nil }

    let channelCount: Int = 2
    let byteCount = sampleCount * channelCount * MemoryLayout<Float32>.size
    var interleaved = [Float32](repeating: 0, count: sampleCount * channelCount)
    for i in 0..<sampleCount {
      interleaved[i * 2] = max(-1, min(1, audioTensor[0, tensorSampleOffset + i]))
      interleaved[i * 2 + 1] = max(-1, min(1, audioTensor[1, tensorSampleOffset + i]))
    }

    var blockBuffer: CMBlockBuffer?
    let blockBufferStatus = CMBlockBufferCreateWithMemoryBlock(
      allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: byteCount,
      blockAllocator: kCFAllocatorDefault, customBlockSource: nil, offsetToData: 0,
      dataLength: byteCount, flags: 0, blockBufferOut: &blockBuffer)
    guard blockBufferStatus == kCMBlockBufferNoErr, let blockBuffer = blockBuffer else {
      return nil
    }

    let replaceStatus = interleaved.withUnsafeBytes { rawBuffer in
      CMBlockBufferReplaceDataBytes(
        with: rawBuffer.baseAddress!, blockBuffer: blockBuffer, offsetIntoDestination: 0,
        dataLength: byteCount)
    }
    guard replaceStatus == kCMBlockBufferNoErr else { return nil }

    let bytesPerFrame = UInt32(channelCount * MemoryLayout<Float32>.size)
    var asbd = AudioStreamBasicDescription(
      mSampleRate: sampleRate,
      mFormatID: kAudioFormatLinearPCM,
      mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
      mBytesPerPacket: bytesPerFrame,
      mFramesPerPacket: 1,
      mBytesPerFrame: bytesPerFrame,
      mChannelsPerFrame: 2,
      mBitsPerChannel: 32,
      mReserved: 0)

    var audioFormatDescription: CMAudioFormatDescription?
    let formatStatus = CMAudioFormatDescriptionCreate(
      allocator: kCFAllocatorDefault, asbd: &asbd, layoutSize: 0, layout: nil,
      magicCookieSize: 0, magicCookie: nil, extensions: nil,
      formatDescriptionOut: &audioFormatDescription)
    guard formatStatus == noErr, let audioFormatDescription = audioFormatDescription else {
      return nil
    }

    let presentationTime = CMTime(
      value: CMTimeValue(presentationSampleOffset),
      timescale: CMTimeScale(max(1, Int(sampleRate.rounded()))))
    var sampleBuffer: CMSampleBuffer?
    let sampleBufferStatus = CMAudioSampleBufferCreateWithPacketDescriptions(
      allocator: kCFAllocatorDefault, dataBuffer: blockBuffer, dataReady: true,
      makeDataReadyCallback: nil, refcon: nil, formatDescription: audioFormatDescription,
      sampleCount: sampleCount, presentationTimeStamp: presentationTime, packetDescriptions: nil,
      sampleBufferOut: &sampleBuffer)
    guard sampleBufferStatus == noErr else { return nil }
    return sampleBuffer
  }

  private func newPixelBufferFrom(
    imageTensor: Tensor<FloatType>, pixelBufferPool: CVPixelBufferPool
  )
    -> CVPixelBuffer?
  {
    var newPixelBuffer: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &newPixelBuffer)
    guard let pixelBuffer = newPixelBuffer else { return nil }
    CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    guard let pixelBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      return nil
    }
    let imageHeight = imageTensor.shape[1]
    let imageWidth = imageTensor.shape[2]
    imageTensor.withUnsafeBytes {
      guard let fp16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      let pixelBaseAddressTyped = pixelBaseAddress.assumingMemoryBound(to: UInt8.self)

      for i in 0..<imageHeight * imageWidth {
        // BGRA
        pixelBaseAddressTyped[i * 4] = UInt8(min(max(Int((fp16[i * 3 + 2] + 1) * 127.5), 0), 255))
        pixelBaseAddressTyped[i * 4 + 1] = UInt8(
          min(max(Int((fp16[i * 3 + 1] + 1) * 127.5), 0), 255))
        pixelBaseAddressTyped[i * 4 + 2] = UInt8(min(max(Int((fp16[i * 3] + 1) * 127.5), 0), 255))
        pixelBaseAddressTyped[i * 4 + 3] = 255
      }
    }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    return pixelBuffer
  }
}

extension VideoExporter {
  public static func frameDuration(for frameRate: Double) -> CMTime {
    guard frameRate > 0 else {
      return CMTimeMake(value: 1, timescale: 30)
    }

    // Handle common NTSC rates precisely if possible
    if abs(frameRate - (30000.0 / 1001.0)) < 0.0001 {  // ~29.97
      return CMTimeMake(value: 1001, timescale: 30000)
    }
    if abs(frameRate - (24000.0 / 1001.0)) < 0.0001 {  // ~23.976
      return CMTimeMake(value: 1001, timescale: 24000)
    }
    if abs(frameRate - (60000.0 / 1001.0)) < 0.0001 {  // ~59.94
      return CMTimeMake(value: 1001, timescale: 60000)
    }

    if abs(frameRate - frameRate.rounded()) < 1e-16 {
      return CMTimeMake(value: 1, timescale: Int32(frameRate.rounded()))
    }

    // General approach: find suitable value/timescale
    // Start with a reasonable multiplier
    let preferredTimescale: Int32 = 60000  // Common high timescale for precision
    let value = Int64(round(Double(preferredTimescale) / frameRate))

    if value > 0 {
      let calculatedFrameRate = Double(preferredTimescale) / Double(value)
      // Add a check to see if the calculated rate is close enough
      if abs(calculatedFrameRate - frameRate) / frameRate < 0.001 {  // Within 0.1% error
        return CMTimeMake(value: value, timescale: preferredTimescale)
      }
    }

    // Fallback: Use a simpler multiplier approach if the preferred timescale didn't work well
    // Or if you need to handle rates with many decimal places accurately.
    // Example: Use 10000 as multiplier
    let multiplier: Double = 10000.0
    let valueApprox: Int64 = Int64(multiplier)
    let timescaleApprox: Int32 = Int32(round(frameRate * multiplier))

    guard timescaleApprox > 0 else {
      return CMTimeMake(value: 1, timescale: 30)
    }
    return CMTimeMake(value: valueApprox, timescale: timescaleApprox)
  }
}
