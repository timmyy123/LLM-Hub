import CryptoKit
import Darwin
import Foundation

public struct URLDownloadProbe {
  public let expectedLength: Int64
  public let etag: String?
  public let lastModified: String?
  public let supportsSegmentedDownloads: Bool

  public init?(httpResponse: HTTPURLResponse) {
    guard (200..<300).contains(httpResponse.statusCode) else { return nil }
    let expectedLength =
      httpResponse.expectedContentLength >= 0
      ? httpResponse.expectedContentLength
      : Int64(httpResponse.value(forHTTPHeaderField: "Content-Length") ?? "") ?? -1
    guard expectedLength > 0 else { return nil }
    let acceptsRanges =
      httpResponse.value(forHTTPHeaderField: "Accept-Ranges")?
      .lowercased()
      .contains("bytes") == true
    self.expectedLength = expectedLength
    self.etag = httpResponse.value(forHTTPHeaderField: "ETag")
    self.lastModified = httpResponse.value(forHTTPHeaderField: "Last-Modified")
    self.supportsSegmentedDownloads = acceptsRanges
  }
}

struct SegmentedDownloadManifest: Codable, Equatable {
  static let currentVersion = 1

  var version: Int
  var remoteURL: String
  var expectedLength: Int64
  var blockSize: Int
  var etag: String?
  var lastModified: String?
  var completedBlocks: [Bool]
  var blockDigests: [Data?]
}

struct DownloadBlockAssignment {
  let index: Int
  let offset: Int64
  let length: Int
}

final class RandomAccessFile {
  private let fd: Int32

  init(url: URL, length: Int64) throws {
    fd = open(url.path, O_RDWR | O_CREAT, 0o644)
    guard fd >= 0 else { throw posixError(errno) }
    guard ftruncate(fd, off_t(length)) == 0 else {
      let error = posixError(errno)
      close(fd)
      throw error
    }
  }

  deinit {
    close(fd)
  }

  func write(_ data: Data, at offset: Int64) throws {
    var written = 0
    try data.withUnsafeBytes { rawBuffer in
      guard let baseAddress = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
        return
      }
      while written < data.count {
        let count = pwrite(
          fd, baseAddress + written, data.count - written, off_t(offset + Int64(written)))
        guard count >= 0 else { throw posixError(errno) }
        written += count
      }
    }
  }

}

func posixError(_ code: Int32) -> Error {
  if let posixCode = POSIXErrorCode(rawValue: code) {
    return POSIXError(posixCode)
  }
  return ResumableDownloader.DownloadError.unexpected
}

func isTransientDownloadError(_ error: NSError) -> Bool {
  if error.domain == NSURLErrorDomain {
    let code = URLError.Code(rawValue: error.code)
    switch code {
    case .timedOut, .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed,
      .networkConnectionLost, .notConnectedToInternet:
      return true
    default:
      break
    }
  }
  if error.domain == NSPOSIXErrorDomain, error.code == 54 || error.code == 32 {
    return true
  }
  if error.domain == kCFErrorDomainCFNetwork as String,
    let underlyingError = error.userInfo[NSUnderlyingErrorKey] as? NSError,
    underlyingError.domain == NSPOSIXErrorDomain,
    underlyingError.code == 54 || underlyingError.code == 32
  {
    return true
  }
  return false
}

func isTransientHTTPStatusCode(_ statusCode: Int) -> Bool {
  switch statusCode {
  case 408, 429, 500, 502, 503, 504:
    return true
  default:
    return false
  }
}

func excludeFromBackup(_ url: URL) {
  var url = url
  var resourceValues = URLResourceValues()
  resourceValues.isExcludedFromBackup = true
  try? url.setResourceValues(resourceValues)
}

func sha256Digest(_ data: Data) -> Data {
  Data(SHA256.hash(data: data))
}

func sha256Digest(forFileAt url: URL, offset: Int64 = 0, length: Int64? = nil) throws -> Data {
  guard offset >= 0 else {
    throw ResumableDownloader.DownloadError.unexpected
  }
  if let length, length < 0 {
    throw ResumableDownloader.DownloadError.unexpected
  }
  let handle = try FileHandle(forReadingFrom: url)
  defer {
    try? handle.close()
  }
  if offset > 0 {
    try handle.seek(toOffset: UInt64(offset))
  }
  var remaining = length
  var hasher = SHA256()
  while remaining == nil || remaining! > 0 {
    var reachedEndOfFile = false
    try autoreleasepool {
      let readLength = remaining.map { Int(min(Int64(1024 * 1024), $0)) } ?? 1024 * 1024
      let data = try handle.read(upToCount: readLength) ?? Data()
      if data.isEmpty {
        guard remaining == nil else {
          throw ResumableDownloader.DownloadError.unexpected
        }
        reachedEndOfFile = true
        return
      }
      hasher.update(data: data)
      if let currentRemaining = remaining {
        remaining = currentRemaining - Int64(data.count)
      }
    }
    if reachedEndOfFile {
      break
    }
  }
  return Data(hasher.finalize())
}

func sha256Hex(forFileAt url: URL) throws -> String {
  try sha256Digest(forFileAt: url).hexString
}

func fileSize(at url: URL) -> Int64 {
  let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
  let size = attributes?[.size] as? NSNumber
  return size?.int64Value ?? 0
}

func resumeDataBytesReceived(_ resumeData: Data) -> Int64? {
  guard
    let propertyList = try? PropertyListSerialization.propertyList(
      from: resumeData, options: [], format: nil),
    let dictionary = propertyList as? [String: Any]
  else {
    return nil
  }
  return int64Value(dictionary["NSURLSessionResumeBytesReceived"])
}

private func int64Value(_ value: Any?) -> Int64? {
  switch value {
  case let value as NSNumber:
    return value.int64Value
  case let value as NSString:
    return value.longLongValue
  case let value as String:
    return Int64(value)
  case let value as Int:
    return Int64(value)
  case let value as Int64:
    return value
  default:
    return nil
  }
}

extension Data {
  var hexString: String {
    map { String(format: "%02x", $0) }.joined()
  }
}
