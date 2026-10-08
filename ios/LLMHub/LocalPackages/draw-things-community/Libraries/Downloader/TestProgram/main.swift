import Darwin
import Downloader
import Foundation

private struct Configuration {
  var url = URL(string: "https://static.libnnc.org/hidream_i1_dev_q8p.ckpt")!
  var output: URL?
  var cancelAfterMiB: Double = 256
  var stopAfterMiB: Double = 0
  var backend: ResumableDownloader.Backend = .automatic
}

private func parseConfiguration() throws -> Configuration {
  var configuration = Configuration()
  var index = 1
  while index < CommandLine.arguments.count {
    let argument = CommandLine.arguments[index]
    switch argument {
    case "--url":
      index += 1
      guard index < CommandLine.arguments.count,
        let url = URL(string: CommandLine.arguments[index])
      else {
        throw CLIError.missingValue("--url")
      }
      configuration.url = url
    case "--output":
      index += 1
      guard index < CommandLine.arguments.count else {
        throw CLIError.missingValue("--output")
      }
      configuration.output = URL(fileURLWithPath: CommandLine.arguments[index])
    case "--cancel-after-mib":
      index += 1
      guard index < CommandLine.arguments.count,
        let cancelAfterMiB = Double(CommandLine.arguments[index])
      else {
        throw CLIError.missingValue("--cancel-after-mib")
      }
      configuration.cancelAfterMiB = cancelAfterMiB
    case "--stop-after-mib":
      index += 1
      guard index < CommandLine.arguments.count,
        let stopAfterMiB = Double(CommandLine.arguments[index])
      else {
        throw CLIError.missingValue("--stop-after-mib")
      }
      configuration.stopAfterMiB = stopAfterMiB
    case "--download-task":
      configuration.backend = .downloadTask
    default:
      throw CLIError.unknownArgument(argument)
    }
    index += 1
  }
  return configuration
}

private enum CLIError: Error, LocalizedError {
  case missingValue(String)
  case unknownArgument(String)

  var errorDescription: String? {
    switch self {
    case .missingValue(let flag):
      return "Missing \(flag) value"
    case .unknownArgument(let argument):
      return "Unknown argument \(argument)"
    }
  }
}

private func defaultOutput(for remoteURL: URL) -> URL {
  let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
    "DownloaderTestCLI", isDirectory: true)
  try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  return directory.appendingPathComponent(remoteURL.lastPathComponent)
}

private func progressLine(
  bytesWritten: Int64, expectedBytes: Int64, startTime: Date
) -> String {
  let elapsed = max(Date().timeIntervalSince(startTime), 0.001)
  let mib = Double(bytesWritten) / 1_048_576
  let expectedMiB = Double(expectedBytes) / 1_048_576
  let speed = mib / elapsed
  let percent = expectedBytes > 0 ? Double(bytesWritten) / Double(expectedBytes) * 100 : 0
  return String(
    format: "%.1f / %.1f MiB (%.1f%%) at %.1f MiB/s", mib, expectedMiB, percent, speed)
}

private func run(configuration: Configuration) throws {
  let output = configuration.output ?? defaultOutput(for: configuration.url)
  try FileManager.default.createDirectory(
    at: output.deletingLastPathComponent(), withIntermediateDirectories: true)

  let semaphore = DispatchSemaphore(value: 0)
  let cancelThreshold = Int64(configuration.cancelAfterMiB * 1_048_576)
  let stopThreshold = Int64(configuration.stopAfterMiB * 1_048_576)
  let startTime = Date()

  var cancelled = false
  var stopped = false
  var phase = 1
  var currentDownloader: ResumableDownloader? = nil
  var finalError: Error? = nil
  var finalBytesWritten: Int64 = 0
  var finalExpectedBytes: Int64 = 0

  func startDownload() {
    let downloader = ResumableDownloader(
      remoteUrl: configuration.url,
      localUrl: output,
      sha256: nil,
      backend: configuration.backend)
    currentDownloader = downloader
    downloader.resume { bytesWritten, totalBytesExpectedToWrite, isComplete, error in
      if stopped {
        return
      }
      if let error {
        finalError = error
        semaphore.signal()
        return
      }
      finalBytesWritten = bytesWritten
      finalExpectedBytes = totalBytesExpectedToWrite
      print(
        progressLine(
          bytesWritten: bytesWritten, expectedBytes: totalBytesExpectedToWrite, startTime: startTime
        ))
      if stopThreshold > 0, bytesWritten >= stopThreshold {
        stopped = true
        downloader.cancel()
        semaphore.signal()
        return
      }
      if !cancelled, phase == 1, cancelThreshold > 0, bytesWritten >= cancelThreshold {
        cancelled = true
        phase = 2
        print("Cancelling at \(bytesWritten) bytes to exercise resume...")
        downloader.cancel()
        Thread.sleep(forTimeInterval: 0.5)
        startDownload()
        return
      }
      if isComplete {
        semaphore.signal()
      }
    }
  }

  startDownload()
  semaphore.wait()
  currentDownloader?.cancel()

  if let finalError {
    throw finalError
  }
  let elapsed = max(Date().timeIntervalSince(startTime), 0.001)
  let averageMiBPerSecond = Double(finalBytesWritten) / 1_048_576 / elapsed
  if stopThreshold > 0 {
    print(
      "Stopped at \(Double(finalBytesWritten) / 1_048_576) MiB / \(Double(finalExpectedBytes) / 1_048_576) MiB with average \(averageMiBPerSecond) MiB/s."
    )
    return
  }
  let finalSize =
    ((try? FileManager.default.attributesOfItem(atPath: output.path)[.size] as? NSNumber) ?? nil)?
    .int64Value ?? 0
  print("Completed \(output.path) (\(Double(finalSize) / 1_048_576) MiB).")
}

do {
  try run(configuration: try parseConfiguration())
} catch {
  fputs("Downloader test failed: \(error)\n", stderr)
  exit(1)
}
