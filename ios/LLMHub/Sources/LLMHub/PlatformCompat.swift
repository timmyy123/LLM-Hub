//
//  PlatformCompat.swift
//  LLMHub
//
//  Native macOS bridge for the UIKit APIs the app relies on. On iOS this file
//  compiles to nothing; on macOS it maps the same names onto AppKit so the
//  shared code paths run natively without per-call-site #if blocks.
//

#if os(macOS)
import AppKit
import Photos
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Core type aliases

typealias UIImage = NSImage
typealias UIColor = NSColor
typealias UIFont = NSFont

// MARK: - UIImage API surface on NSImage

extension NSImage {
    /// Mirrors `UIImage(cgImage:)` — size is the pixel size of the bitmap.
    convenience init(cgImage: CGImage) {
        self.init(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }

    /// Mirrors `UIImage(cgImage:scale:orientation:)`; macOS images are always upright.
    convenience init(cgImage: CGImage, scale: CGFloat, orientation: UIImage.Orientation) {
        self.init(cgImage: cgImage)
    }

    enum Orientation: Int {
        case up, down, left, right, upMirrored, downMirrored, leftMirrored, rightMirrored
    }

    var imageOrientation: Orientation { .up }

    /// Pixel-to-point scale; AppKit images built from bitmaps are 1:1.
    var scale: CGFloat { 1 }

    /// Mirrors `UIImage.cgImage`.
    var cgImage: CGImage? {
        var rect = CGRect(origin: .zero, size: size)
        return cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }

    /// Mirrors `UIImage.jpegData(compressionQuality:)`.
    func jpegData(compressionQuality: CGFloat) -> Data? {
        guard let rep = bitmapRep else { return nil }
        return rep.representation(using: .jpeg, properties: [.compressionFactor: compressionQuality])
    }

    /// Mirrors `UIImage.pngData()`.
    func pngData() -> Data? {
        bitmapRep?.representation(using: .png, properties: [:])
    }

    /// Mirrors `UIImage.preparingThumbnail(of:)`.
    func preparingThumbnail(of targetSize: CGSize) -> NSImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    private var bitmapRep: NSBitmapImageRep? {
        guard let cg = cgImage else { return nil }
        return NSBitmapImageRep(cgImage: cg)
    }
}

extension Image {
    /// Mirrors `Image(uiImage:)`.
    init(uiImage: NSImage) {
        self.init(nsImage: uiImage)
    }
}

extension Color {
    /// Mirrors `Color(uiColor:)`.
    init(uiColor: NSColor) {
        self.init(nsColor: uiColor)
    }
}

extension NSFont {
    /// Mirrors `UIFont.italicSystemFont(ofSize:)`.
    static func italicSystemFont(ofSize size: CGFloat) -> NSFont {
        let base = NSFont.systemFont(ofSize: size)
        let descriptor = base.fontDescriptor.withSymbolicTraits(.italic)
        return NSFont(descriptor: descriptor, size: size) ?? base
    }
}

// MARK: - Graphics renderer

final class UIGraphicsImageRendererFormat {
    var scale: CGFloat = 1
    var opaque: Bool = false

    static func `default`() -> UIGraphicsImageRendererFormat { UIGraphicsImageRendererFormat() }
    static func preferred() -> UIGraphicsImageRendererFormat { UIGraphicsImageRendererFormat() }
}

final class UIGraphicsImageRendererContext {
    let cgContext: CGContext
    let format: UIGraphicsImageRendererFormat

    init(cgContext: CGContext, format: UIGraphicsImageRendererFormat) {
        self.cgContext = cgContext
        self.format = format
    }

    func fill(_ rect: CGRect) {
        cgContext.fill(rect)
    }
}

/// AppKit implementation of `UIGraphicsImageRenderer` that renders into an
/// 8-bit sRGB bitmap and makes it the current NSGraphicsContext so
/// `NSImage.draw(in:)`, `NSColor.setFill()` and string drawing work as on iOS.
final class UIGraphicsImageRenderer {
    let size: CGSize
    let format: UIGraphicsImageRendererFormat

    init(size: CGSize, format: UIGraphicsImageRendererFormat = .default()) {
        self.size = size
        self.format = format
    }

    func image(actions: (UIGraphicsImageRendererContext) -> Void) -> NSImage {
        let scale = max(format.scale, 1)
        let pixelWidth = max(Int((size.width * scale).rounded()), 1)
        let pixelHeight = max(Int((size.height * scale).rounded()), 1)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(
                data: nil,
                width: pixelWidth,
                height: pixelHeight,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else {
            return NSImage(size: size)
        }
        ctx.scaleBy(x: scale, y: scale)

        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)
        actions(UIGraphicsImageRendererContext(cgContext: ctx, format: format))
        NSGraphicsContext.restoreGraphicsState()

        guard let cg = ctx.makeImage() else { return NSImage(size: size) }
        return NSImage(cgImage: cg, size: size)
    }

    func jpegData(withCompressionQuality quality: CGFloat, actions: (UIGraphicsImageRendererContext) -> Void) -> Data {
        image(actions: actions).jpegData(compressionQuality: quality) ?? Data()
    }

    func pngData(actions: (UIGraphicsImageRendererContext) -> Void) -> Data {
        image(actions: actions).pngData() ?? Data()
    }
}

// MARK: - Pasteboard

/// AppKit-backed `UIPasteboard.general`.
final class UIPasteboard: @unchecked Sendable {
    static let general = UIPasteboard()

    private var board: NSPasteboard { .general }

    var string: String? {
        get { board.string(forType: .string) }
        set {
            board.clearContents()
            if let newValue { board.setString(newValue, forType: .string) }
        }
    }

    var hasStrings: Bool { board.string(forType: .string) != nil }

    var image: NSImage? {
        get { NSImage(pasteboard: board) }
        set {
            board.clearContents()
            if let newValue { board.writeObjects([newValue]) }
        }
    }

    var hasImages: Bool { NSImage.canInit(with: board) }

    var url: URL? {
        get { board.readObjects(forClasses: [NSURL.self])?.first as? URL }
        set {
            board.clearContents()
            if let newValue { board.writeObjects([newValue as NSURL]) }
        }
    }
}

// MARK: - Application

/// AppKit-backed subset of `UIApplication` used by the app (URL opening,
/// Settings deep links, keyboard dismissal).
@MainActor
final class UIApplication {
    static let shared = UIApplication()

    struct OpenExternalURLOptionsKey: Hashable, RawRepresentable, Sendable {
        let rawValue: String
        init(rawValue: String) { self.rawValue = rawValue }
    }

    /// Opens the app's Privacy & Security pane in System Settings.
    nonisolated static let openSettingsURLString = "x-apple.systempreferences:com.apple.preference.security?Privacy"

    func canOpenURL(_ url: URL) -> Bool {
        NSWorkspace.shared.urlForApplication(toOpen: url) != nil
    }

    func open(
        _ url: URL,
        options: [OpenExternalURLOptionsKey: Any] = [:],
        completionHandler: (@MainActor @Sendable (Bool) -> Void)? = nil
    ) {
        let success = NSWorkspace.shared.open(url)
        completionHandler?(success)
    }

    @discardableResult
    func open(_ url: URL, options: [OpenExternalURLOptionsKey: Any] = [:]) async -> Bool {
        NSWorkspace.shared.open(url)
    }

    /// Mirrors `sendAction(#selector(UIResponder.resignFirstResponder), ...)`
    /// used to dismiss the keyboard: ends editing in the key window.
    @discardableResult
    func sendAction(_ action: Selector, to target: Any?, from sender: Any?, for event: Any?) -> Bool {
        if action == #selector(NSResponder.resignFirstResponder) {
            NSApp.keyWindow?.makeFirstResponder(nil)
            return true
        }
        return NSApp.sendAction(action, to: target, from: sender)
    }

    var isIdleTimerDisabled: Bool {
        get { idleActivity != nil }
        set {
            if newValue, idleActivity == nil {
                idleActivity = ProcessInfo.processInfo.beginActivity(
                    options: [.idleDisplaySleepDisabled, .userInitiated],
                    reason: "LLM Hub long-running generation"
                )
            } else if !newValue, let activity = idleActivity {
                ProcessInfo.processInfo.endActivity(activity)
                idleActivity = nil
            }
        }
    }

    private var idleActivity: NSObjectProtocol?
}

typealias UIResponder = NSResponder

// MARK: - Screen

/// AppKit-backed `UIScreen.main` (bounds of the screen hosting the key window).
@MainActor
final class UIScreen {
    static let main = UIScreen()

    var bounds: CGRect {
        let screen = NSApp.keyWindow?.screen ?? NSScreen.main
        return screen?.visibleFrame ?? CGRect(x: 0, y: 0, width: 1440, height: 900)
    }

    var scale: CGFloat {
        (NSApp.keyWindow?.screen ?? NSScreen.main)?.backingScaleFactor ?? 2
    }
}

// MARK: - Saving to Photos

/// Calls a UIKit-style `-(void)x:(id)item didFinishSavingWithError:(NSError *)e contextInfo:(void *)c`
/// completion selector, mirroring what UIKit does after a save.
private func invokeSaveCompletion(
    target: Any?, selector: Selector?, item: AnyObject, error: Error?, contextInfo: UnsafeMutableRawPointer?
) {
    guard let object = target as? NSObject, let selector, object.responds(to: selector) else { return }
    typealias Completion = @convention(c) (AnyObject, Selector, AnyObject, NSError?, UnsafeMutableRawPointer?) -> Void
    let implementation = unsafeBitCast(object.method(for: selector), to: Completion.self)
    implementation(object, selector, item, error as NSError?, contextInfo)
}

/// PhotoKit implementation of `UIImageWriteToSavedPhotosAlbum`.
func UIImageWriteToSavedPhotosAlbum(
    _ image: NSImage, _ completionTarget: Any?, _ completionSelector: Selector?, _ contextInfo: UnsafeMutableRawPointer?
) {
    nonisolated(unsafe) let target = completionTarget
    nonisolated(unsafe) let context = contextInfo
    nonisolated(unsafe) let savedImage = image
    let data = image.pngData()
    PHPhotoLibrary.requestAuthorization(for: .addOnly) { _ in
        PHPhotoLibrary.shared().performChanges({
            guard let data else { return }
            PHAssetCreationRequest.forAsset().addResource(with: .photo, data: data, options: nil)
        }) { _, error in
            DispatchQueue.main.async {
                invokeSaveCompletion(target: target, selector: completionSelector, item: savedImage, error: error, contextInfo: context)
            }
        }
    }
}

/// PhotoKit implementation of `UISaveVideoAtPathToSavedPhotosAlbum`.
func UISaveVideoAtPathToSavedPhotosAlbum(
    _ videoPath: String, _ completionTarget: Any?, _ completionSelector: Selector?, _ contextInfo: UnsafeMutableRawPointer?
) {
    nonisolated(unsafe) let target = completionTarget
    nonisolated(unsafe) let context = contextInfo
    let url = URL(fileURLWithPath: videoPath)
    PHPhotoLibrary.requestAuthorization(for: .addOnly) { _ in
        PHPhotoLibrary.shared().performChanges({
            PHAssetCreationRequest.forAsset().addResource(with: .video, fileURL: url, options: nil)
        }) { _, error in
            DispatchQueue.main.async {
                invokeSaveCompletion(target: target, selector: completionSelector, item: videoPath as NSString, error: error, contextInfo: context)
            }
        }
    }
}

// MARK: - SwiftUI modifiers that are iOS-only

/// Stand-ins for iOS text-input traits; macOS has a hardware keyboard and no
/// auto-capitalization, so these keep the shared view code unchanged.
enum TextInputAutocapitalizationCompat {
    case never, words, sentences, characters
}

enum KeyboardTypeCompat {
    case `default`, asciiCapable, numbersAndPunctuation, URL, numberPad, phonePad, emailAddress, decimalPad, webSearch
}

extension View {
    func textInputAutocapitalization(_ autocapitalization: TextInputAutocapitalizationCompat?) -> some View {
        self
    }

    func keyboardType(_ type: KeyboardTypeCompat) -> some View {
        self
    }
}

/// Stand-in for `NavigationBarItem.TitleDisplayMode` on macOS, where window
/// titles are always shown inline in the toolbar.
enum NavigationBarTitleDisplayModeCompat {
    case automatic, inline, large
}

extension View {
    func navigationBarTitleDisplayMode(_ mode: NavigationBarTitleDisplayModeCompat) -> some View {
        self
    }

    /// macOS has no full-screen cover presentation; a sheet is the native equivalent.
    func fullScreenCover<Content: View>(
        isPresented: Binding<Bool>,
        onDismiss: (() -> Void)? = nil,
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        sheet(isPresented: isPresented, onDismiss: onDismiss) {
            content().apolloMacSheetSizing()
        }
    }

    func fullScreenCover<Item: Identifiable, Content: View>(
        item: Binding<Item?>,
        onDismiss: (() -> Void)? = nil,
        @ViewBuilder content: @escaping (Item) -> Content
    ) -> some View {
        sheet(item: item, onDismiss: onDismiss) { value in
            content(value).apolloMacSheetSizing()
        }
    }
}

/// Mirrors the iOS `.automatic` button style: the label is drawn as designed
/// (no macOS push-button bezel), dimmed while pressed or disabled.
struct ApolloAutomaticButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .contentShape(Rectangle())
            .opacity(configuration.isPressed ? 0.55 : (isEnabled ? 1 : 0.4))
    }
}

extension ToolbarItemPlacement {
    static var navigationBarLeading: ToolbarItemPlacement { .navigation }
    static var navigationBarTrailing: ToolbarItemPlacement { .primaryAction }
    static var topBarLeading: ToolbarItemPlacement { .navigation }
    static var topBarTrailing: ToolbarItemPlacement { .primaryAction }
}

extension ToolbarPlacement {
    /// The window toolbar plays the navigation bar's role on macOS.
    static var navigationBar: ToolbarPlacement { .windowToolbar }
}

extension ListStyle where Self == InsetListStyle {
    static var insetGrouped: InsetListStyle { .inset }
}
#endif
