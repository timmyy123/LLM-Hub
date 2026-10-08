/// Account capabilities supplied by an embedding app. Commands do not own account
/// state or presentation; implementations marshal calls onto their owning thread.
public protocol ToolAccountProvider: AnyObject {
  /// A read-only lookup when `prepareIfNeeded` is false. Otherwise the host may
  /// sign in and provision a credential before completing the request.
  func resolveDrawThingsCredential(
    prepareIfNeeded: Bool, completion: @escaping (Result<String?, Error>) -> Void)

  /// Requests top-up for the credential used by this command, not another account.
  func drawThingsInsufficientFunds(apiKey: String)
}
