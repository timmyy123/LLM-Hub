#if canImport(WebKit)
  /// Both providers share a queue so verification sheets cannot compete for presentation.
  /// All access is on the main thread.
  protocol WebKitSearchRequest: AnyObject {
    func start(onFinish: @escaping () -> Void)
  }

  final class WebKitSearchSession {
    static let shared = WebKitSearchSession()
    private var pending = [WebKitSearchRequest]()
    private var active: WebKitSearchRequest?

    func enqueue(_ request: WebKitSearchRequest) {
      pending.append(request)
      startNext()
    }

    private func startNext() {
      guard active == nil, !pending.isEmpty else { return }
      let request = pending.removeFirst()
      active = request
      request.start { [self] in
        active = nil
        startNext()
      }
    }
  }
#endif
