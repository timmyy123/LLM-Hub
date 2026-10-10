import Foundation
#if canImport(AppKit) && !canImport(UIKit)
import AppKit

public typealias UIImage = NSImage

extension NSImage {
    public convenience init(cgImage: CGImage) {
        self.init(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }

    public var cgImage: CGImage? {
        cgImage(forProposedRect: nil, context: nil, hints: nil)
    }

    public func jpegData(compressionQuality: CGFloat) -> Data? {
        guard let tiffRepresentation,
              let bitmapImage = NSBitmapImageRep(data: tiffRepresentation) else {
            return nil
        }
        return bitmapImage.representation(using: .jpeg, properties: [.compressionFactor: compressionQuality])
    }

    public func pngData() -> Data? {
        guard let tiffRepresentation,
              let bitmapImage = NSBitmapImageRep(data: tiffRepresentation) else {
            return nil
        }
        return bitmapImage.representation(using: .png, properties: [:])
    }
}

public final class UIApplication: @unchecked Sendable {
    public static let shared = UIApplication()
    private init() {}

    @discardableResult
    public func open(_ url: URL) -> Bool {
        NSWorkspace.shared.open(url)
    }

    public func canOpenURL(_ url: URL) -> Bool {
        return true
    }
}

public final class UIPasteboard: @unchecked Sendable {
    public static let general = UIPasteboard()
    private init() {}

    public var string: String? {
        get {
            NSPasteboard.general.string(forType: .string)
        }
        set {
            NSPasteboard.general.clearContents()
            if let newValue {
                NSPasteboard.general.setString(newValue, forType: .string)
            }
        }
    }
}
#endif
