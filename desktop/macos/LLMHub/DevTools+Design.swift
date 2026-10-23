#if DEBUG
import Foundation

extension DevTools {
    /// `--designtest …`: an end-to-end Design run on this Mac, results in the log.
    static func designTest(_ args: [String]) async {}
}
#endif
