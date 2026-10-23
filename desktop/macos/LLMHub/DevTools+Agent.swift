#if DEBUG
import Foundation

extension DevTools {
    /// `--agenttest …`: an end-to-end Agent run on this Mac, results in the log.
    static func agentTest(_ args: [String]) async {}
}
#endif
