import Foundation

final class URLSessionDownloadTaskResumableDownloaderBackend: NSObject, DownloadBackend {
  private let remoteUrl: URL
  private let localUrl: URL
  private let sha256: String?
  private let expectedLength: Int64?
  private let handler: ResumableDownloader.ProgressHandler
  private let lock = NSLock()
  private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
  private var task: URLSessionDownloadTask? = nil
  private var resumedBytesReceived: Int64 = 0
  private var isCancelled = false
  private var didFinish = false

  init(
    remoteUrl: URL,
    localUrl: URL,
    sha256: String?,
    expectedLength: Int64?,
    handler: @escaping ResumableDownloader.ProgressHandler
  ) {
    self.remoteUrl = remoteUrl
    self.localUrl = localUrl
    self.sha256 = sha256
    self.expectedLength = expectedLength
    self.handler = handler
  }

  func start() {
    do {
      try FileManager.default.createDirectory(
        at: localUrl.deletingLastPathComponent(), withIntermediateDirectories: true)
    } catch {
      handler(0, 0, false, ResumableDownloader.DownloadError.local(error))
      return
    }
    lock.lock()
    let shouldStart = !isCancelled && !didFinish
    lock.unlock()
    guard shouldStart else {
      session.invalidateAndCancel()
      return
    }
    resumeTask()
  }

  func cancel() {
    let task: URLSessionDownloadTask?
    lock.lock()
    guard !didFinish else {
      lock.unlock()
      return
    }
    isCancelled = true
    task = self.task
    self.task = nil
    lock.unlock()

    guard let task else {
      session.invalidateAndCancel()
      return
    }
    task.cancel { [weak self] resumeData in
      guard let self else { return }
      if let resumeData {
        try? self.writeResumeData(resumeData)
      }
      self.session.invalidateAndCancel()
    }
  }

  private func resumeTask() {
    let partUrl = localUrl.appendingPathExtension("part")
    let resumeData =
      FileManager.default.fileExists(atPath: partUrl.path) ? try? Data(contentsOf: partUrl) : nil
    let resumedBytesReceived = resumeData.flatMap(resumeDataBytesReceived) ?? 0
    let task: URLSessionDownloadTask
    if let resumeData {
      task = session.downloadTask(withResumeData: resumeData)
    } else {
      task = session.downloadTask(with: URLRequest(url: remoteUrl))
    }
    lock.lock()
    guard !isCancelled, !didFinish else {
      lock.unlock()
      task.cancel()
      session.invalidateAndCancel()
      return
    }
    self.resumedBytesReceived = resumedBytesReceived
    self.task = task
    lock.unlock()
    task.resume()
  }

  private func writeResumeData(_ resumeData: Data) throws {
    let partUrl = localUrl.appendingPathExtension("part")
    try resumeData.write(to: partUrl, options: .atomic)
  }

  private func fail(_ error: Error) {
    lock.lock()
    guard !didFinish, !isCancelled else {
      lock.unlock()
      return
    }
    didFinish = true
    task = nil
    lock.unlock()
    session.invalidateAndCancel()
    handler(0, 0, false, error)
  }
}

extension URLSessionDownloadTaskResumableDownloaderBackend: URLSessionDownloadDelegate {
  func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didResumeAtOffset fileOffset: Int64,
    expectedTotalBytes: Int64
  ) {
    lock.lock()
    let isCancelled = self.isCancelled
    let didFinish = self.didFinish
    resumedBytesReceived = fileOffset
    lock.unlock()
    guard !isCancelled, !didFinish else { return }
    let totalBytesExpectedToWrite =
      expectedLength ?? max(expectedTotalBytes, fileOffset)
    if fileOffset > 0 {
      handler(fileOffset, totalBytesExpectedToWrite, false, nil)
    }
  }

  func urlSession(
    _ session: URLSession, downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) {
    lock.lock()
    guard !isCancelled, !didFinish else {
      lock.unlock()
      session.invalidateAndCancel()
      return
    }
    didFinish = true
    task = nil
    lock.unlock()

    do {
      if let sha256 {
        let digest = try sha256Hex(forFileAt: location)
        guard digest == sha256 else {
          handler(
            fileSize(at: location), fileSize(at: location), false,
            ResumableDownloader.DownloadError.fileMismatch)
          return
        }
      }
      let partUrl = localUrl.appendingPathExtension("part")
      try? FileManager.default.removeItem(at: partUrl)
      try? FileManager.default.removeItem(at: localUrl)
      try FileManager.default.moveItem(at: location, to: localUrl)
      excludeFromBackup(localUrl)
      let totalBytes = fileSize(at: localUrl)
      session.finishTasksAndInvalidate()
      handler(totalBytes, totalBytes, true, nil)
    } catch {
      session.invalidateAndCancel()
      handler(0, 0, false, ResumableDownloader.DownloadError.local(error))
    }
  }

  func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64
  ) {
    lock.lock()
    let resumedBytesReceived = self.resumedBytesReceived
    let isCancelled = self.isCancelled
    let didFinish = self.didFinish
    lock.unlock()
    guard !isCancelled, !didFinish else { return }
    let normalizedTotalBytesWritten =
      totalBytesWritten >= resumedBytesReceived
      ? totalBytesWritten
      : resumedBytesReceived + totalBytesWritten
    let normalizedTotalBytesExpectedToWrite =
      expectedLength ?? max(totalBytesExpectedToWrite, normalizedTotalBytesWritten)
    handler(normalizedTotalBytesWritten, normalizedTotalBytesExpectedToWrite, false, nil)
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    guard let error else { return }
    lock.lock()
    self.task = nil
    let isCancelled = self.isCancelled
    let didFinish = self.didFinish
    lock.unlock()
    if isCancelled || didFinish {
      return
    }

    let nsError = error as NSError
    if nsError.domain == NSURLErrorDomain, nsError.code == URLError.cancelled.rawValue {
      return
    }

    var resumeDataWriteError: Error? = nil
    if let resumeData = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
      do {
        try writeResumeData(resumeData)
      } catch {
        resumeDataWriteError = error
      }
    }

    if let resumeDataWriteError {
      fail(ResumableDownloader.DownloadError.local(resumeDataWriteError))
      return
    }

    if isTransientDownloadError(nsError) {
      Task { [weak self] in
        try? await Task.sleep(nanoseconds: 200_000_000)
        self?.resumeTask()
      }
      return
    }

    if let httpResponse = task.response as? HTTPURLResponse, httpResponse.statusCode >= 400 {
      fail(ResumableDownloader.DownloadError.server(httpResponse.statusCode))
    } else {
      fail(ResumableDownloader.DownloadError.local(error))
    }
  }
}
