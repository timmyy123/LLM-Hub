import Foundation

final class SegmentedResumableDownloaderBackend: NSObject, DownloadBackend {
  private static let retryDelayNanoseconds: UInt64 = 2_000_000_000

  private let remoteUrl: URL
  private let localUrl: URL
  private let sha256: String?
  private let probe: URLDownloadProbe
  private let handler: ResumableDownloader.ProgressHandler
  private let partUrl: URL
  private let manifestUrl: URL
  private let lock = NSLock()
  private let connectionCount: Int
  private lazy var delegateQueue: OperationQueue = {
    let queue = OperationQueue()
    queue.maxConcurrentOperationCount = 1
    queue.name = "org.drawthings.segmented-downloader"
    return queue
  }()
  private lazy var session: URLSession = {
    let configuration = URLSessionConfiguration.default
    configuration.httpMaximumConnectionsPerHost = connectionCount
    configuration.waitsForConnectivity = true
    return URLSession(configuration: configuration, delegate: self, delegateQueue: delegateQueue)
  }()

  private var file: RandomAccessFile? = nil
  private var manifest: SegmentedDownloadManifest? = nil
  private var claimedBlocks: [Bool] = []
  private var activeTasks: [Int: URLSessionDataTask] = [:]
  private var activeDownloads: [Int: ActiveBlockDownload] = [:]
  private var verifiedBytes: Int64 = 0
  private var isCancelling = false
  private var isFinalizing = false
  private var isScheduling = false
  private var didFinish = false

  init(
    remoteUrl: URL,
    localUrl: URL,
    sha256: String?,
    probe: URLDownloadProbe,
    handler: @escaping ResumableDownloader.ProgressHandler
  ) {
    self.remoteUrl = remoteUrl
    self.localUrl = localUrl
    self.sha256 = sha256
    self.probe = probe
    self.handler = handler
    self.partUrl = localUrl.appendingPathExtension("partial")
    self.manifestUrl = localUrl.appendingPathExtension("partial.map")
    let blockSize = Self.defaultBlockSize(for: probe.expectedLength)
    let blockCount = Int((probe.expectedLength + blockSize - 1) / blockSize)
    self.connectionCount = max(1, min(8, blockCount))
  }

  func start() {
    do {
      try FileManager.default.createDirectory(
        at: localUrl.deletingLastPathComponent(), withIntermediateDirectories: true)
      let preparedState = try prepareState()
      lock.lock()
      guard !isCancelling, !didFinish else {
        lock.unlock()
        return
      }
      file = preparedState.file
      manifest = preparedState.manifest
      claimedBlocks = Array(repeating: false, count: preparedState.manifest.completedBlocks.count)
      verifiedBytes = preparedState.verifiedBytes
      let shouldFinalize = verifiedBytes == probe.expectedLength
      lock.unlock()

      if shouldFinalize {
        completeDownload()
      } else {
        if preparedState.verifiedBytes > 0 {
          handler(preparedState.verifiedBytes, probe.expectedLength, false, nil)
        }
        scheduleMoreTasks()
      }
    } catch {
      fail(ResumableDownloader.DownloadError.local(error))
    }
  }

  func cancel() {
    let tasks: [URLSessionDataTask]
    lock.lock()
    guard !isCancelling, !didFinish else {
      lock.unlock()
      return
    }
    isCancelling = true
    isScheduling = false
    tasks = Array(activeTasks.values)
    activeTasks.removeAll()
    activeDownloads.removeAll()
    lock.unlock()

    for task in tasks {
      task.cancel()
    }
    session.invalidateAndCancel()
  }
}

extension SegmentedResumableDownloaderBackend {
  private final class ActiveBlockDownload {
    let assignment: DownloadBlockAssignment
    var data: Data
    var receivedBytes: Int64

    init(assignment: DownloadBlockAssignment) {
      self.assignment = assignment
      self.data = Data()
      self.data.reserveCapacity(assignment.length)
      self.receivedBytes = 0
    }
  }

  fileprivate static func defaultBlockSize(for expectedLength: Int64) -> Int64 {
    switch expectedLength {
    case ..<(128 * 1024 * 1024):
      return 1 * 1024 * 1024
    case ..<(1024 * 1024 * 1024):
      return 2 * 1024 * 1024
    case ..<(8 * 1024 * 1024 * 1024):
      return 4 * 1024 * 1024
    default:
      return 8 * 1024 * 1024
    }
  }

  fileprivate func prepareState() throws -> (
    file: RandomAccessFile, manifest: SegmentedDownloadManifest, verifiedBytes: Int64
  ) {
    let manifest = try loadOrCreateManifest()
    let file = try RandomAccessFile(url: partUrl, length: manifest.expectedLength)
    excludeFromBackup(partUrl)
    excludeFromBackup(manifestUrl)

    var validatedManifest = manifest
    var verifiedBytes: Int64 = 0
    for index in validatedManifest.completedBlocks.indices
    where validatedManifest.completedBlocks[index] {
      guard let digest = validatedManifest.blockDigests[index] else {
        validatedManifest.completedBlocks[index] = false
        continue
      }
      let assignment = blockAssignment(index: index, blockSize: validatedManifest.blockSize)
      let blockDigest = try sha256Digest(
        forFileAt: partUrl, offset: assignment.offset, length: Int64(assignment.length))
      if blockDigest == digest {
        verifiedBytes += Int64(assignment.length)
      } else {
        validatedManifest.completedBlocks[index] = false
        validatedManifest.blockDigests[index] = nil
      }
    }

    if validatedManifest != manifest {
      try saveManifest(validatedManifest)
    }
    return (file, validatedManifest, verifiedBytes)
  }

  fileprivate func loadOrCreateManifest() throws -> SegmentedDownloadManifest {
    let decoder = PropertyListDecoder()
    if FileManager.default.fileExists(atPath: manifestUrl.path),
      let data = try? Data(contentsOf: manifestUrl),
      let manifest = try? decoder.decode(SegmentedDownloadManifest.self, from: data),
      manifestMatchesProbe(manifest)
    {
      if fileSize(at: partUrl) == manifest.expectedLength {
        return manifest
      }
    }

    try? FileManager.default.removeItem(at: manifestUrl)
    try? FileManager.default.removeItem(at: partUrl)

    let blockSize = Int(Self.defaultBlockSize(for: probe.expectedLength))
    let blockCount = Int((probe.expectedLength + Int64(blockSize) - 1) / Int64(blockSize))
    let manifest = SegmentedDownloadManifest(
      version: SegmentedDownloadManifest.currentVersion,
      remoteURL: remoteUrl.absoluteString,
      expectedLength: probe.expectedLength,
      blockSize: blockSize,
      etag: probe.etag,
      lastModified: probe.lastModified,
      completedBlocks: Array(repeating: false, count: blockCount),
      blockDigests: Array(repeating: nil, count: blockCount))
    try saveManifest(manifest)
    return manifest
  }

  fileprivate func manifestMatchesProbe(_ manifest: SegmentedDownloadManifest) -> Bool {
    guard manifest.version == SegmentedDownloadManifest.currentVersion else { return false }
    guard manifest.remoteURL == remoteUrl.absoluteString else { return false }
    guard manifest.expectedLength == probe.expectedLength else { return false }
    guard manifest.completedBlocks.count == manifest.blockDigests.count else { return false }
    if let etag = probe.etag, let manifestETag = manifest.etag, etag != manifestETag {
      return false
    }
    if let lastModified = probe.lastModified,
      let manifestLastModified = manifest.lastModified,
      lastModified != manifestLastModified
    {
      return false
    }
    return true
  }

  fileprivate func saveManifest(_ manifest: SegmentedDownloadManifest) throws {
    let encoder = PropertyListEncoder()
    encoder.outputFormat = .binary
    let data = try encoder.encode(manifest)
    try data.write(to: manifestUrl, options: .atomic)
  }

  fileprivate func blockAssignment(index: Int, blockSize: Int) -> DownloadBlockAssignment {
    let offset = Int64(index * blockSize)
    let remaining = Int(probe.expectedLength - offset)
    return DownloadBlockAssignment(index: index, offset: offset, length: min(blockSize, remaining))
  }

  fileprivate func nextAssignment() -> DownloadBlockAssignment? {
    guard let manifest else { return nil }
    for index in manifest.completedBlocks.indices
    where !manifest.completedBlocks[index] && !claimedBlocks[index] {
      claimedBlocks[index] = true
      return blockAssignment(index: index, blockSize: manifest.blockSize)
    }
    return nil
  }

  fileprivate func displayProgressLocked() -> Int64 {
    let activeBytes = activeDownloads.values.reduce(Int64(0)) { partialResult, download in
      partialResult + download.receivedBytes
    }
    return min(probe.expectedLength, verifiedBytes + activeBytes)
  }

  fileprivate func scheduleMoreTasks() {
    lock.lock()
    guard !isScheduling, !isCancelling, !isFinalizing, !didFinish else {
      lock.unlock()
      return
    }
    isScheduling = true
    lock.unlock()

    while true {
      let assignment: DownloadBlockAssignment
      lock.lock()
      guard !isCancelling, !isFinalizing, !didFinish, activeTasks.count < connectionCount,
        let nextAssignment = nextAssignment()
      else {
        let shouldFinalize =
          !isCancelling && !isFinalizing && !didFinish && activeTasks.isEmpty
          && (manifest?.completedBlocks.allSatisfy { $0 } ?? false)
        if shouldFinalize {
          isFinalizing = true
        }
        isScheduling = false
        lock.unlock()
        if shouldFinalize {
          completeDownload()
        }
        return
      }
      assignment = nextAssignment
      lock.unlock()

      var request = URLRequest(url: remoteUrl)
      request.setValue(
        "bytes=\(assignment.offset)-\(assignment.offset + Int64(assignment.length) - 1)",
        forHTTPHeaderField: "Range")
      if let etag = probe.etag {
        request.setValue(etag, forHTTPHeaderField: "If-Range")
      } else if let lastModified = probe.lastModified {
        request.setValue(lastModified, forHTTPHeaderField: "If-Range")
      }

      let task = session.dataTask(with: request)

      lock.lock()
      guard !isCancelling, !isFinalizing, !didFinish else {
        claimedBlocks[assignment.index] = false
        isScheduling = false
        lock.unlock()
        task.cancel()
        return
      }
      activeTasks[assignment.index] = task
      activeDownloads[task.taskIdentifier] = ActiveBlockDownload(assignment: assignment)
      lock.unlock()
      task.resume()
    }
  }

  fileprivate func handleCompletion(for task: URLSessionTask, error: Error?) {
    if let error = error as? NSError {
      handleFailure(for: task, error: error)
      return
    }

    let activeFile: RandomAccessFile
    let assignment: DownloadBlockAssignment
    let data: Data
    lock.lock()
    guard let activeDownload = activeDownloads[task.taskIdentifier] else {
      lock.unlock()
      return
    }
    assignment = activeDownload.assignment
    data = activeDownload.data
    guard !isCancelling, !isFinalizing, !didFinish else {
      activeDownloads[task.taskIdentifier] = nil
      activeTasks[assignment.index] = nil
      claimedBlocks[assignment.index] = false
      lock.unlock()
      return
    }
    guard let file else {
      activeDownloads[task.taskIdentifier] = nil
      activeTasks[assignment.index] = nil
      claimedBlocks[assignment.index] = false
      lock.unlock()
      fail(ResumableDownloader.DownloadError.unexpected)
      return
    }
    activeFile = file
    lock.unlock()

    guard data.count == assignment.length else {
      retryBlock(taskIdentifier: task.taskIdentifier, assignment: assignment)
      return
    }

    do {
      try activeFile.write(data, at: assignment.offset)
      let digest = sha256Digest(data)

      var manifestToSave: SegmentedDownloadManifest
      lock.lock()
      guard !isCancelling, !isFinalizing, !didFinish else {
        activeDownloads[task.taskIdentifier] = nil
        activeTasks[assignment.index] = nil
        claimedBlocks[assignment.index] = false
        lock.unlock()
        return
      }
      guard var manifest else {
        activeDownloads[task.taskIdentifier] = nil
        activeTasks[assignment.index] = nil
        claimedBlocks[assignment.index] = false
        lock.unlock()
        fail(ResumableDownloader.DownloadError.unexpected)
        return
      }
      manifest.completedBlocks[assignment.index] = true
      manifest.blockDigests[assignment.index] = digest
      manifestToSave = manifest
      do {
        try saveManifest(manifestToSave)
      } catch {
        lock.unlock()
        throw error
      }
      activeDownloads[task.taskIdentifier] = nil
      activeTasks[assignment.index] = nil
      claimedBlocks[assignment.index] = false
      self.manifest = manifestToSave
      verifiedBytes += Int64(assignment.length)
      let progress = displayProgressLocked()
      let shouldFinalize = activeTasks.isEmpty && manifestToSave.completedBlocks.allSatisfy { $0 }
      if shouldFinalize {
        isFinalizing = true
      }
      lock.unlock()

      if shouldFinalize {
        completeDownload()
      } else {
        handler(progress, probe.expectedLength, false, nil)
        scheduleMoreTasks()
      }
    } catch {
      fail(ResumableDownloader.DownloadError.local(error))
    }
  }

  fileprivate func handleFailure(for task: URLSessionTask, error: NSError) {
    if error.domain == NSURLErrorDomain, error.code == URLError.cancelled.rawValue {
      lock.lock()
      let assignment = activeDownloads[task.taskIdentifier]?.assignment
      let shouldRetry = assignment != nil && !isCancelling && !isFinalizing && !didFinish
      if !shouldRetry, let assignment {
        activeDownloads[task.taskIdentifier] = nil
        activeTasks[assignment.index] = nil
        claimedBlocks[assignment.index] = false
      }
      lock.unlock()
      if shouldRetry, let assignment {
        retryBlock(taskIdentifier: task.taskIdentifier, assignment: assignment)
      }
      return
    }

    if isTransientDownloadError(error) {
      lock.lock()
      let assignment = activeDownloads[task.taskIdentifier]?.assignment
      lock.unlock()
      if let assignment {
        retryBlock(taskIdentifier: task.taskIdentifier, assignment: assignment)
      }
      return
    }

    fail(ResumableDownloader.DownloadError.local(error))
  }

  fileprivate func retryBlock(taskIdentifier: Int, assignment: DownloadBlockAssignment) {
    lock.lock()
    let wasActive =
      activeDownloads[taskIdentifier] != nil || activeTasks[assignment.index] != nil
    activeDownloads[taskIdentifier] = nil
    activeTasks[assignment.index] = nil
    claimedBlocks[assignment.index] = false
    let shouldRetry = wasActive && !isCancelling && !isFinalizing && !didFinish
    lock.unlock()
    guard shouldRetry else { return }
    Task { [weak self] in
      try? await Task.sleep(nanoseconds: Self.retryDelayNanoseconds)
      self?.scheduleMoreTasks()
    }
  }

  fileprivate func completeDownload() {
    do {
      if let sha256 {
        let digest = try sha256Hex(forFileAt: partUrl)
        guard digest == sha256 else {
          try? FileManager.default.removeItem(at: partUrl)
          try? FileManager.default.removeItem(at: manifestUrl)
          fail(ResumableDownloader.DownloadError.fileMismatch)
          return
        }
      }
      try? FileManager.default.removeItem(at: localUrl)
      try FileManager.default.moveItem(at: partUrl, to: localUrl)
      try? FileManager.default.removeItem(at: manifestUrl)
      excludeFromBackup(localUrl)
      let totalBytes = fileSize(at: localUrl)
      lock.lock()
      guard !didFinish else {
        lock.unlock()
        return
      }
      didFinish = true
      isScheduling = false
      file = nil
      lock.unlock()
      session.finishTasksAndInvalidate()
      handler(totalBytes, totalBytes, true, nil)
    } catch {
      fail(ResumableDownloader.DownloadError.local(error))
    }
  }

  fileprivate func fail(_ error: Error) {
    let tasks: [URLSessionDataTask]
    let progress: Int64
    lock.lock()
    guard !didFinish else {
      lock.unlock()
      return
    }
    didFinish = true
    isFinalizing = true
    isScheduling = false
    progress = verifiedBytes
    tasks = Array(activeTasks.values)
    activeTasks.removeAll()
    activeDownloads.removeAll()
    file = nil
    lock.unlock()

    for task in tasks {
      task.cancel()
    }
    session.invalidateAndCancel()
    handler(progress, probe.expectedLength, false, error)
  }
}

extension SegmentedResumableDownloaderBackend: URLSessionDataDelegate {
  func urlSession(_ session: URLSession, taskIsWaitingForConnectivity task: URLSessionTask) {
    lock.lock()
    let assignment = activeDownloads[task.taskIdentifier]?.assignment
    let shouldRetry = assignment != nil && !isCancelling && !isFinalizing && !didFinish
    lock.unlock()

    guard shouldRetry, let assignment else { return }
    retryBlock(taskIdentifier: task.taskIdentifier, assignment: assignment)
    task.cancel()
  }

  func urlSession(
    _ session: URLSession,
    dataTask: URLSessionDataTask,
    didReceive response: URLResponse,
    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
  ) {
    let shouldAccept: Bool
    let assignment: DownloadBlockAssignment?
    lock.lock()
    assignment = activeDownloads[dataTask.taskIdentifier]?.assignment
    shouldAccept =
      !isCancelling && !isFinalizing && !didFinish
      && assignment != nil
    lock.unlock()
    guard shouldAccept else {
      completionHandler(.cancel)
      return
    }

    guard let httpResponse = response as? HTTPURLResponse else {
      completionHandler(.cancel)
      fail(ResumableDownloader.DownloadError.unexpected)
      return
    }
    guard httpResponse.statusCode == 206 else {
      completionHandler(.cancel)
      if let assignment, isTransientHTTPStatusCode(httpResponse.statusCode) {
        retryBlock(taskIdentifier: dataTask.taskIdentifier, assignment: assignment)
        return
      }
      if httpResponse.statusCode >= 400 {
        fail(ResumableDownloader.DownloadError.server(httpResponse.statusCode))
      } else {
        fail(ResumableDownloader.DownloadError.unexpected)
      }
      return
    }

    completionHandler(.allow)
  }

  func urlSession(
    _ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data
  ) {
    let progress: Int64
    lock.lock()
    guard !isCancelling, !isFinalizing, !didFinish,
      let download = activeDownloads[dataTask.taskIdentifier]
    else {
      lock.unlock()
      return
    }
    download.data.append(data)
    download.receivedBytes += Int64(data.count)
    guard download.receivedBytes <= Int64(download.assignment.length) else {
      lock.unlock()
      fail(ResumableDownloader.DownloadError.unexpected)
      return
    }
    progress = displayProgressLocked()
    lock.unlock()

    handler(progress, probe.expectedLength, false, nil)
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    handleCompletion(for: task, error: error)
  }
}
