#if canImport(UIKit) && !os(visionOS)
  import UIKit

  extension UIScreen {
    public var isAllScreen: Bool {
      // If the ratio is larger than 16:9, it is all screen (somewhere around 21:10).
      return bounds.height / bounds.width > 1.8
    }
  }
#endif
