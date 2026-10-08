import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Recoverable transport failures shared by the browser-backed search providers.
enum WebSearchBrowserFallback {
  static func shouldUseBrowser(after error: Error) -> Bool {
    switch error {
    case WebSearchError.searchBlocked, WebSearchError.unexpectedSearchResponse,
      WebSearchError.bodyDecodingFailed:
      return true
    case WebSearchError.httpStatus(let status, _, _, _, _):
      return status == 403 || status == 408 || status == 429 || (500..<600).contains(status)
    case let error as URLError:
      return error.code == .timedOut || error.code == .networkConnectionLost
    default:
      return false
    }
  }

}
