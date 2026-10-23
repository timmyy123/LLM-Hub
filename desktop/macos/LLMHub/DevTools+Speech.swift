#if DEBUG
import Foundation

extension DevTools {
    /// `--speechtest …`: an end-to-end Speech run on this Mac, results in the log.
    static func speechTest(_ args: [String]) async {}
}
#endif
