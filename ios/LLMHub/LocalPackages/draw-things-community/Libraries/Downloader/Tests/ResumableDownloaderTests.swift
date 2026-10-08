import CryptoKit
import Foundation
import Network
import XCTest

@testable import Downloader

final class ResumableDownloaderTests: XCTestCase {
  private var temporaryDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    temporaryDirectory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
      "ResumableDownloaderTests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: temporaryDirectory, withIntermediateDirectories: true, attributes: nil)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: temporaryDirectory)
    temporaryDirectory = nil
    try super.tearDownWithError()
  }

  func testSegmentedDownloadUsesMultipleConnections() throws {
    let payload = makePayload(size: 24 * 1024 * 1024)
    let server = try LocalRangeServer(payload: payload, chunkDelay: 0.003)
    defer {
      server.stop()
    }

    let destination = temporaryDirectory.appendingPathComponent("multi-connection.bin")
    let downloader = ResumableDownloader(
      remoteUrl: server.url, localUrl: destination,
      sha256: Data(SHA256.hash(data: payload)).hexString)

    let completion = expectation(description: "download-complete")
    var reportedError: Error? = nil

    downloader.resume { _, _, isComplete, error in
      if let error {
        reportedError = error
        completion.fulfill()
        return
      }
      if isComplete {
        completion.fulfill()
      }
    }

    wait(for: [completion], timeout: 60)
    XCTAssertNil(reportedError)
    XCTAssertEqual(try Data(contentsOf: destination), payload)
    XCTAssertGreaterThan(server.maxConcurrentRangeRequests, 1)
  }

  func testCancelResumeRevalidatesCompletedBlocks() throws {
    let payload = makePayload(size: 16 * 1024 * 1024)
    let server = try LocalRangeServer(payload: payload, chunkDelay: 0.005)
    defer {
      server.stop()
    }

    let destination = temporaryDirectory.appendingPathComponent("cancel-resume.bin")
    let partialURL = destination.appendingPathExtension("partial")
    let downloader = ResumableDownloader(
      remoteUrl: server.url, localUrl: destination,
      sha256: Data(SHA256.hash(data: payload)).hexString)

    let cancelled = expectation(description: "download-cancelled")
    cancelled.assertForOverFulfill = false

    let cancelLock = NSLock()
    var didCancel = false
    downloader.resume { totalBytesWritten, _, _, error in
      XCTAssertNil(error)
      cancelLock.lock()
      defer {
        cancelLock.unlock()
      }
      guard !didCancel, totalBytesWritten >= 4 * 1024 * 1024 else { return }
      didCancel = true
      downloader.cancel()
      cancelled.fulfill()
    }

    wait(for: [cancelled], timeout: 30)
    Thread.sleep(forTimeInterval: 0.5)

    XCTAssertTrue(FileManager.default.fileExists(atPath: partialURL.path))

    var partial = try Data(contentsOf: partialURL)
    partial.withUnsafeMutableBytes { rawBuffer in
      guard let baseAddress = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
        return
      }
      baseAddress[0] ^= 0xff
    }
    try partial.write(to: partialURL, options: .atomic)

    let resumed = ResumableDownloader(
      remoteUrl: server.url, localUrl: destination,
      sha256: Data(SHA256.hash(data: payload)).hexString)
    let completion = expectation(description: "resume-complete")
    var completionError: Error? = nil

    resumed.resume { _, _, isComplete, error in
      if let error {
        completionError = error
        completion.fulfill()
        return
      }
      if isComplete {
        completion.fulfill()
      }
    }

    wait(for: [completion], timeout: 60)
    XCTAssertNil(completionError)
    XCTAssertEqual(try Data(contentsOf: destination), payload)
    XCTAssertGreaterThanOrEqual(server.rangeRequestCount(forStartOffset: 0), 2)
    XCTAssertLessThan(server.totalRangeBytesServed, Int64(payload.count * 2))
  }

  func testURLSessionDownloadTaskCancelBeforeTaskCreationDoesNotStartDownload() throws {
    let payload = makePayload(size: 8 * 1024 * 1024)
    let server = try LocalRangeServer(payload: payload, chunkDelay: 0.01)
    defer {
      server.stop()
    }

    let destination = temporaryDirectory.appendingPathComponent("download-task-cancel.bin")
    let downloader = ResumableDownloader(
      remoteUrl: server.url,
      localUrl: destination,
      sha256: nil,
      backend: .downloadTask)
    downloader.resume { _, _, _, _ in }
    downloader.cancel()
    Thread.sleep(forTimeInterval: 0.5)

    XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
  }

  func testResumeDataBytesReceivedParsesPropertyList() throws {
    let data = try PropertyListSerialization.data(
      fromPropertyList: ["NSURLSessionResumeBytesReceived": 2_097_152],
      format: .binary,
      options: 0)
    XCTAssertEqual(resumeDataBytesReceived(data), 2_097_152)
  }

  func testProbeReportsExpectedLengthAndRangeSupport() throws {
    let payload = makePayload(size: 4 * 1024 * 1024)
    let server = try LocalRangeServer(payload: payload, chunkDelay: 0)
    defer {
      server.stop()
    }

    let probed = expectation(description: "download-probed")
    var probeResult: Result<URLDownloadProbe, Error>? = nil

    ResumableDownloader.probe(remoteUrl: server.url) { result in
      probeResult = result
      probed.fulfill()
    }

    wait(for: [probed], timeout: 30)
    let probe = try XCTUnwrap(probeResult?.get())
    XCTAssertEqual(probe.expectedLength, Int64(payload.count))
    XCTAssertTrue(probe.supportsSegmentedDownloads)
  }

  func testSegmentedResumeReportsRestoredProgressImmediately() throws {
    let payload = makePayload(size: 16 * 1024 * 1024)
    let server = try LocalRangeServer(payload: payload, chunkDelay: 0.005)
    defer {
      server.stop()
    }

    let destination = temporaryDirectory.appendingPathComponent("segmented-progress.bin")
    let downloader = ResumableDownloader(
      remoteUrl: server.url,
      localUrl: destination,
      sha256: Data(SHA256.hash(data: payload)).hexString)

    let cancelled = expectation(description: "segmented-cancelled")
    cancelled.assertForOverFulfill = false

    var cancelledBytes: Int64 = 0
    let cancelLock = NSLock()
    var didCancel = false
    downloader.resume { totalBytesWritten, _, _, error in
      XCTAssertNil(error)
      cancelLock.lock()
      defer {
        cancelLock.unlock()
      }
      guard !didCancel, totalBytesWritten >= 10 * 1024 * 1024 else { return }
      didCancel = true
      cancelledBytes = totalBytesWritten
      downloader.cancel()
      cancelled.fulfill()
    }

    wait(for: [cancelled], timeout: 30)
    Thread.sleep(forTimeInterval: 0.5)

    let resumed = ResumableDownloader(
      remoteUrl: server.url,
      localUrl: destination,
      sha256: Data(SHA256.hash(data: payload)).hexString)
    let restoredProgress = expectation(description: "segmented-restored-progress")
    restoredProgress.assertForOverFulfill = false

    var firstResumedProgress: Int64? = nil
    resumed.resume { totalBytesWritten, totalBytesExpectedToWrite, _, error in
      XCTAssertNil(error)
      guard firstResumedProgress == nil else { return }
      firstResumedProgress = totalBytesWritten
      XCTAssertEqual(totalBytesExpectedToWrite, Int64(payload.count))
      resumed.cancel()
      restoredProgress.fulfill()
    }

    wait(for: [restoredProgress], timeout: 30)
    XCTAssertNotNil(firstResumedProgress)
    XCTAssertGreaterThan(firstResumedProgress ?? 0, 0)
    XCTAssertLessThanOrEqual(firstResumedProgress ?? 0, cancelledBytes)
  }

  func testSegmentedDownloadReportsInFlightByteProgress() throws {
    let payload = makePayload(size: 8 * 1024 * 1024)
    let server = try LocalRangeServer(payload: payload, chunkDelay: 0.003)
    defer {
      server.stop()
    }

    let destination = temporaryDirectory.appendingPathComponent("segmented-inflight-progress.bin")
    let downloader = ResumableDownloader(
      remoteUrl: server.url,
      localUrl: destination,
      sha256: Data(SHA256.hash(data: payload)).hexString)

    let completion = expectation(description: "segmented-inflight-progress-complete")
    let progressLock = NSLock()
    var progressValues: [Int64] = []
    var reportedError: Error? = nil

    downloader.resume { totalBytesWritten, totalBytesExpectedToWrite, isComplete, error in
      XCTAssertEqual(totalBytesExpectedToWrite, Int64(payload.count))
      if let error {
        reportedError = error
        completion.fulfill()
        return
      }
      progressLock.lock()
      progressValues.append(totalBytesWritten)
      progressLock.unlock()
      if isComplete {
        completion.fulfill()
      }
    }

    wait(for: [completion], timeout: 60)
    XCTAssertNil(reportedError)
    XCTAssertEqual(try Data(contentsOf: destination), payload)

    progressLock.lock()
    let sawSubBlockProgress = progressValues.contains {
      $0 > 0 && $0 < Int64(payload.count) && $0 % Int64(1024 * 1024) != 0
    }
    progressLock.unlock()
    XCTAssertTrue(sawSubBlockProgress)
  }

  func testSegmentedDownloadRetriesTransientRangeFailure() throws {
    let payload = makePayload(size: 12 * 1024 * 1024)
    let server = try LocalRangeServer(
      payload: payload, chunkDelay: 0.003, transientRangeFailuresToInject: 1)
    defer {
      server.stop()
    }

    let destination = temporaryDirectory.appendingPathComponent("segmented-transient-retry.bin")
    let downloader = ResumableDownloader(
      remoteUrl: server.url,
      localUrl: destination,
      sha256: Data(SHA256.hash(data: payload)).hexString)

    let completion = expectation(description: "segmented-transient-retry-complete")
    var reportedError: Error? = nil

    downloader.resume { _, _, isComplete, error in
      if let error {
        reportedError = error
        completion.fulfill()
        return
      }
      if isComplete {
        completion.fulfill()
      }
    }

    wait(for: [completion], timeout: 60)
    XCTAssertNil(reportedError)
    XCTAssertEqual(try Data(contentsOf: destination), payload)
    let failedOffsets = server.transientFailureStartOffsets()
    XCTAssertEqual(failedOffsets.count, 1)
    XCTAssertGreaterThanOrEqual(server.rangeRequestCount(forStartOffset: failedOffsets[0]), 2)
  }

  func testSegmentedDownloadRetriesIncompleteRangeBody() throws {
    let payload = makePayload(size: 12 * 1024 * 1024)
    let server = try LocalRangeServer(
      payload: payload, chunkDelay: 0.003, shortRangeResponsesToInject: 1)
    defer {
      server.stop()
    }

    let destination = temporaryDirectory.appendingPathComponent("segmented-short-body-retry.bin")
    let downloader = ResumableDownloader(
      remoteUrl: server.url,
      localUrl: destination,
      sha256: Data(SHA256.hash(data: payload)).hexString)

    let completion = expectation(description: "segmented-short-body-retry-complete")
    var reportedError: Error? = nil

    downloader.resume { _, _, isComplete, error in
      if let error {
        reportedError = error
        completion.fulfill()
        return
      }
      if isComplete {
        completion.fulfill()
      }
    }

    wait(for: [completion], timeout: 60)
    XCTAssertNil(reportedError)
    XCTAssertEqual(try Data(contentsOf: destination), payload)
    let shortBodyOffsets = server.shortResponseStartOffsets()
    XCTAssertEqual(shortBodyOffsets.count, 1)
    XCTAssertGreaterThanOrEqual(server.rangeRequestCount(forStartOffset: shortBodyOffsets[0]), 2)
  }

  func testSegmentedDownloadRetriesTransientHTTPRangeStatus() throws {
    let payload = makePayload(size: 12 * 1024 * 1024)
    let server = try LocalRangeServer(
      payload: payload, chunkDelay: 0.003, retryableStatusResponsesToInject: 1)
    defer {
      server.stop()
    }

    let destination = temporaryDirectory.appendingPathComponent("segmented-http-status-retry.bin")
    let downloader = ResumableDownloader(
      remoteUrl: server.url,
      localUrl: destination,
      sha256: Data(SHA256.hash(data: payload)).hexString)

    let completion = expectation(description: "segmented-http-status-retry-complete")
    var reportedError: Error? = nil

    downloader.resume { _, _, isComplete, error in
      if let error {
        reportedError = error
        completion.fulfill()
        return
      }
      if isComplete {
        completion.fulfill()
      }
    }

    wait(for: [completion], timeout: 60)
    XCTAssertNil(reportedError)
    XCTAssertEqual(try Data(contentsOf: destination), payload)
    let retryableStatusOffsets = server.retryableStatusStartOffsets()
    XCTAssertEqual(retryableStatusOffsets.count, 1)
    XCTAssertGreaterThanOrEqual(
      server.rangeRequestCount(forStartOffset: retryableStatusOffsets[0]), 2)
  }

}

private func makePayload(size: Int) -> Data {
  Data((0..<size).map { UInt8($0 % 251) })
}

private final class LocalRangeServer {
  private(set) var url: URL!

  private let payload: Data
  private let chunkDelay: TimeInterval
  private let queue = DispatchQueue(label: "LocalRangeServer", attributes: .concurrent)
  private let listener: NWListener
  private let stateLock = NSLock()
  private let etag = "\"resumable-downloader-test\""
  private let lastModified = "Sun, 29 Mar 2026 00:00:00 GMT"
  private var transientRangeFailuresToInject: Int
  private var shortRangeResponsesToInject: Int
  private var retryableStatusResponsesToInject: Int

  private var rangeRequests: [ClosedRange<Int64>] = []
  private var transientFailureOffsets: [Int64] = []
  private var shortResponseOffsets: [Int64] = []
  private var retryableStatusOffsets: [Int64] = []
  private var activeRangeRequests = 0
  private(set) var maxConcurrentRangeRequests = 0
  private(set) var totalRangeBytesServed: Int64 = 0

  init(
    payload: Data, chunkDelay: TimeInterval, transientRangeFailuresToInject: Int = 0,
    shortRangeResponsesToInject: Int = 0, retryableStatusResponsesToInject: Int = 0
  ) throws {
    self.payload = payload
    self.chunkDelay = chunkDelay
    self.transientRangeFailuresToInject = transientRangeFailuresToInject
    self.shortRangeResponsesToInject = shortRangeResponsesToInject
    self.retryableStatusResponsesToInject = retryableStatusResponsesToInject
    listener = try NWListener(using: .tcp, on: .any)

    let ready = DispatchSemaphore(value: 0)
    listener.stateUpdateHandler = { state in
      if case .ready = state {
        ready.signal()
      }
    }
    listener.newConnectionHandler = { [weak self] connection in
      self?.handle(connection)
    }
    listener.start(queue: queue)
    _ = ready.wait(timeout: .now() + 5)
    guard let port = listener.port else {
      throw XCTSkip("Failed to acquire local listener port")
    }
    url = URL(string: "http://127.0.0.1:\(port.rawValue)/payload.bin")!
  }

  func stop() {
    listener.cancel()
  }

  func rangeRequestCount(forStartOffset startOffset: Int64) -> Int {
    stateLock.lock()
    defer {
      stateLock.unlock()
    }
    return rangeRequests.filter { $0.lowerBound == startOffset }.count
  }

  func transientFailureStartOffsets() -> [Int64] {
    stateLock.lock()
    defer {
      stateLock.unlock()
    }
    return transientFailureOffsets
  }

  func shortResponseStartOffsets() -> [Int64] {
    stateLock.lock()
    defer {
      stateLock.unlock()
    }
    return shortResponseOffsets
  }

  func retryableStatusStartOffsets() -> [Int64] {
    stateLock.lock()
    defer {
      stateLock.unlock()
    }
    return retryableStatusOffsets
  }

  private func handle(_ connection: NWConnection) {
    connection.start(queue: queue)
    receiveRequest(on: connection, buffer: Data())
  }

  private func receiveRequest(on connection: NWConnection, buffer: Data) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) {
      [weak self] content, _, isComplete, _ in
      guard let self else { return }
      var buffer = buffer
      if let content {
        buffer.append(content)
      }
      if let headerRange = buffer.range(of: Data("\r\n\r\n".utf8)) {
        self.respond(to: buffer.subdata(in: 0..<headerRange.upperBound), on: connection)
        return
      }
      if isComplete {
        connection.cancel()
        return
      }
      self.receiveRequest(on: connection, buffer: buffer)
    }
  }

  private func respond(to requestData: Data, on connection: NWConnection) {
    guard let request = String(data: requestData, encoding: .utf8) else {
      connection.cancel()
      return
    }
    let lines = request.components(separatedBy: "\r\n")
    guard let requestLine = lines.first, !requestLine.isEmpty else {
      connection.cancel()
      return
    }

    let components = requestLine.split(separator: " ")
    guard components.count >= 2 else {
      connection.cancel()
      return
    }

    let method = String(components[0])
    let headers = Dictionary(
      uniqueKeysWithValues: lines.dropFirst().compactMap { line -> (String, String)? in
        guard let separator = line.firstIndex(of: ":") else { return nil }
        let key = line[..<separator].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let value = line[line.index(after: separator)...].trimmingCharacters(
          in: .whitespacesAndNewlines)
        return (key, value)
      })

    let fullRange = 0...Int64(payload.count - 1)
    let hasRangeHeader = headers["range"] != nil
    let range = headers["range"].flatMap { parseRange($0, totalLength: payload.count) } ?? fullRange
    let isRangeResponse = hasRangeHeader && range != fullRange
    var body = payload.subdata(in: Int(range.lowerBound)..<(Int(range.upperBound) + 1))
    var responseRange = range
    var shouldInjectShortResponse = false
    var shouldInjectRetryableStatus = false
    if isRangeResponse {
      stateLock.lock()
      if shortRangeResponsesToInject > 0 {
        shortRangeResponsesToInject -= 1
        shortResponseOffsets.append(range.lowerBound)
        shouldInjectShortResponse = true
      } else if retryableStatusResponsesToInject > 0 {
        retryableStatusResponsesToInject -= 1
        retryableStatusOffsets.append(range.lowerBound)
        shouldInjectRetryableStatus = true
      }
      stateLock.unlock()
    }
    if shouldInjectShortResponse {
      let shortLength = max(1, body.count / 2)
      body = body.subdata(in: 0..<shortLength)
      responseRange = range.lowerBound...(range.lowerBound + Int64(shortLength) - 1)
    } else if shouldInjectRetryableStatus {
      body = Data()
    }

    var responseHeaders: String
    if shouldInjectRetryableStatus {
      responseHeaders = "HTTP/1.1 503 Service Unavailable\r\n"
    } else {
      responseHeaders =
        isRangeResponse ? "HTTP/1.1 206 Partial Content\r\n" : "HTTP/1.1 200 OK\r\n"
    }
    responseHeaders += "Accept-Ranges: bytes\r\n"
    responseHeaders += "Content-Length: \(body.count)\r\n"
    responseHeaders += "ETag: \(etag)\r\n"
    responseHeaders += "Last-Modified: \(lastModified)\r\n"
    if isRangeResponse && !shouldInjectRetryableStatus {
      responseHeaders +=
        "Content-Range: bytes \(responseRange.lowerBound)-\(responseRange.upperBound)/\(payload.count)\r\n"
    }
    responseHeaders += "\r\n"

    if isRangeResponse {
      stateLock.lock()
      rangeRequests.append(range)
      activeRangeRequests += 1
      maxConcurrentRangeRequests = max(maxConcurrentRangeRequests, activeRangeRequests)
      totalRangeBytesServed += Int64(body.count)
      stateLock.unlock()
    }

    let headerData = Data(responseHeaders.utf8)
    let shouldInjectTransientFailure: Bool
    if isRangeResponse {
      stateLock.lock()
      if transientRangeFailuresToInject > 0 {
        transientRangeFailuresToInject -= 1
        transientFailureOffsets.append(range.lowerBound)
        shouldInjectTransientFailure = true
      } else {
        shouldInjectTransientFailure = false
      }
      stateLock.unlock()
    } else {
      shouldInjectTransientFailure = false
    }
    connection.send(
      content: headerData,
      completion: .contentProcessed { [weak self] error in
        guard let self else { return }
        if error != nil {
          self.finish(connection: connection, isRangeResponse: isRangeResponse)
          return
        }
        if shouldInjectTransientFailure {
          self.finish(connection: connection, isRangeResponse: isRangeResponse)
          return
        }
        if shouldInjectRetryableStatus {
          self.finish(connection: connection, isRangeResponse: isRangeResponse)
          return
        }
        if method == "HEAD" {
          self.finish(connection: connection, isRangeResponse: isRangeResponse)
          return
        }
        self.sendBody(body, on: connection, offset: 0, isRangeResponse: isRangeResponse)
      })
  }

  private func sendBody(
    _ body: Data, on connection: NWConnection, offset: Int, isRangeResponse: Bool
  ) {
    guard offset < body.count else {
      finish(connection: connection, isRangeResponse: isRangeResponse)
      return
    }
    let end = min(offset + 64 * 1024, body.count)
    let chunk = body.subdata(in: offset..<end)
    connection.send(
      content: chunk,
      completion: .contentProcessed { [weak self] error in
        guard let self else { return }
        if error != nil {
          self.finish(connection: connection, isRangeResponse: isRangeResponse)
          return
        }
        if self.chunkDelay > 0 {
          Thread.sleep(forTimeInterval: self.chunkDelay)
        }
        self.sendBody(body, on: connection, offset: end, isRangeResponse: isRangeResponse)
      })
  }

  private func finish(connection: NWConnection, isRangeResponse: Bool) {
    if isRangeResponse {
      stateLock.lock()
      activeRangeRequests -= 1
      stateLock.unlock()
    }
    connection.cancel()
  }

  private func parseRange(_ value: String, totalLength: Int) -> ClosedRange<Int64>? {
    guard value.hasPrefix("bytes=") else { return nil }
    let rangeComponents = value.dropFirst("bytes=".count).split(separator: "-", maxSplits: 1)
    guard rangeComponents.count == 2,
      let lowerBound = Int64(rangeComponents[0]),
      let upperBound = Int64(rangeComponents[1]),
      lowerBound >= 0,
      upperBound >= lowerBound,
      upperBound < Int64(totalLength)
    else {
      return nil
    }
    return lowerBound...upperBound
  }
}
