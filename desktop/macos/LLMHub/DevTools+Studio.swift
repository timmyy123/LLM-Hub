#if DEBUG
import Foundation

extension DevTools {
    /// `--studiotest …`: an end-to-end Studio run on this Mac, results in the log.
    static func studioTest(_ args: [String]) async {}
}
#endif
