import SwiftUI
import UIKit

enum ApolloPalette {
    static let accent = Color(hex: "8AAE9F")
    static let accentStrong = Color(hex: "A7D4C3")
    static let accentMuted = Color(hex: "5C7682")
    static let accentSoft = Color(hex: "6F8E85")
    static let warning = Color(hex: "E0A96D")
    static let destructive = Color(hex: "E96A63")
}

enum ApolloLayout {
    /// Max width of the centered message + composer column in chat-style screens on
    /// wide layouts (iPad, iPhone Duo unfolded, landscape). Includes the 16pt side padding.
    static let chatColumnMaxWidth: CGFloat = 760
}

struct ApolloLiquidBackground: View {
    var body: some View {
        GeometryReader { geo in
            let insets = geo.safeAreaInsets
            let full = CGSize(width: geo.size.width + insets.leading + insets.trailing,
                              height: geo.size.height + insets.top + insets.bottom)
            // The gradient covers the whole screen and the glows centre on the safe area, as before.
            let shift = CGSize(width: (insets.leading - insets.trailing) / 2, height: (insets.top - insets.bottom) / 2)
            ApolloStaticRender(key: "liquid|\(shift.width),\(shift.height)", padding: 0) {
                Self.layers(shift: shift)
            }
            .frame(width: full.width, height: full.height)
            .position(x: geo.size.width / 2 - shift.width, y: geo.size.height / 2 - shift.height)
        }
    }

    private static func layers(shift: CGSize) -> some View {
        ZStack {
            LinearGradient(
                colors: [Color.black, Color(hex: "0b0f19"), Color(hex: "131a2a")],
                startPoint: .top,
                endPoint: .bottom
            )

            Circle()
                .fill(ApolloPalette.accentSoft.opacity(0.06))
                .frame(width: 260, height: 260)
                .blur(radius: 80)
                .offset(x: 130 + shift.width, y: -260 + shift.height)

            Circle()
                .fill(ApolloPalette.accent.opacity(0.18))
                .frame(width: 220, height: 220)
                .blur(radius: 90)
                .offset(x: -140 + shift.width, y: -220 + shift.height)

            Circle()
                .fill(ApolloPalette.accentMuted.opacity(0.16))
                .frame(width: 300, height: 300)
                .blur(radius: 110)
                .offset(x: 160 + shift.width, y: 260 + shift.height)
        }
    }
}

/// Draws `content` once into an image and shows the image. Live blur, shadow and material
/// effects are redrawn by the GPU when the app returns to the foreground; with the iOS 27.1
/// SDK that showed as cards appearing in pieces on a warm open. An image is only composited.
/// Layout still comes from `content` (kept in place at zero opacity), so sizing is unchanged.
struct ApolloStaticRender<Content: View>: View {
    var key: String
    var padding: CGFloat = 24
    @ViewBuilder var content: Content

    @Environment(\.displayScale) private var displayScale
    @Environment(\.locale) private var locale
    @Environment(\.layoutDirection) private var layoutDirection
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @State private var size: CGSize = .zero
    @State private var rendered: UIImage?

    private static var cache: NSCache<NSString, UIImage> {
        ApolloStaticRenderCache.shared
    }

    private var renderKey: String {
        "\(key)|\(size.width)x\(size.height)|\(displayScale)|\(locale.identifier)|\(layoutDirection)|\(dynamicTypeSize)"
    }

    var body: some View {
        let image = Self.cache.object(forKey: renderKey as NSString) ?? rendered
        content
            .opacity(image == nil ? 1 : 0)
            .overlay {
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .frame(width: size.width + padding * 2, height: size.height + padding * 2)
                        .allowsHitTesting(false)
                }
            }
            // The live content is at opacity 0, and SwiftUI skips hit testing for invisible
            // views: keep the whole frame tappable so Buttons around this still receive taps.
            .contentShape(Rectangle())
            .onGeometryChange(for: CGSize.self) { $0.size } action: { size = $0 }
            .task(id: renderKey) { render() }
    }

    private func render() {
        let renderKey = renderKey
        guard size.width > 0, size.height > 0 else { return }
        if let cached = Self.cache.object(forKey: renderKey as NSString) {
            rendered = cached
            return
        }
        let renderer = ImageRenderer(content:
            content
                .frame(width: size.width, height: size.height)
                .padding(padding)
                .environment(\.locale, locale)
                .environment(\.layoutDirection, layoutDirection)
                .environment(\.colorScheme, colorScheme)
                .environment(\.dynamicTypeSize, dynamicTypeSize)
        )
        renderer.scale = displayScale
        renderer.isOpaque = false
        guard let image = renderer.uiImage else { return }
        Self.cache.setObject(image, forKey: renderKey as NSString)
        rendered = image
    }
}

@MainActor
enum ApolloStaticRenderCache {
    static let shared = NSCache<NSString, UIImage>()
}

struct ApolloIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(.white.opacity(configuration.isPressed ? 0.75 : 0.95))
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
    }
}

private struct ApolloScreenBackgroundModifier: ViewModifier {
    func body(content: Content) -> some View {
        ZStack {
            ApolloLiquidBackground()
            content
        }
    }
}

private struct ApolloSheetModifier<SheetContent: View>: ViewModifier {
    @Binding var isPresented: Bool
    @State private var coversPresenter = false
    let sheetContent: () -> SheetContent

    func body(content: Content) -> some View {
        content
            .overlay {
                if coversPresenter {
                    Color.black.ignoresSafeArea().allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
            }
            .sheet(isPresented: $isPresented, onDismiss: { coversPresenter = false }) {
                sheetContent()
                    .presentationBackground(Color.black)
                    .onAppear {
                        // Wait until the sheet's presentation has started. Covering the
                        // presenter before this point produces a full-screen black flash.
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                            if isPresented { coversPresenter = true }
                        }
                    }
            }
            .onChange(of: isPresented) { _, presented in
                if !presented { coversPresenter = false }
            }
    }
}

private struct ApolloItemSheetModifier<Item: Identifiable, SheetContent: View>: ViewModifier {
    @Binding var item: Item?
    @State private var coversPresenter = false
    let sheetContent: (Item) -> SheetContent

    func body(content: Content) -> some View {
        content
            .overlay {
                if coversPresenter {
                    Color.black.ignoresSafeArea().allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
            }
            .sheet(item: $item, onDismiss: { coversPresenter = false }) { value in
                sheetContent(value)
                    .presentationBackground(Color.black)
                    .onAppear {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                            if item != nil { coversPresenter = true }
                        }
                    }
            }
            .onChange(of: item?.id) { _, id in
                if id == nil { coversPresenter = false }
            }
    }
}

extension View {
    /// Keep the presenting screen out of the exposed area above a modal sheet.
    /// A sheet preserves the presenter's lifecycle, including active model sessions.
    func apolloSheet<SheetContent: View>(
        isPresented: Binding<Bool>,
        @ViewBuilder content: @escaping () -> SheetContent
    ) -> some View {
        modifier(ApolloSheetModifier(isPresented: isPresented, sheetContent: content))
    }

    func apolloSheet<Item: Identifiable, SheetContent: View>(
        item: Binding<Item?>,
        @ViewBuilder content: @escaping (Item) -> SheetContent
    ) -> some View {
        modifier(ApolloItemSheetModifier(item: item, sheetContent: content))
    }

    /// The navigation bar and its scroll views share the screen's background.
    /// This affects the full-width backdrop, while retaining native button styling.
    func apolloNavigationBackground() -> some View {
        toolbarBackground(.hidden, for: .navigationBar)
            .apolloTopScrollEdgeFade()
    }

    /// Use the native soft transition beneath top bars and the status area.
    /// Select soft explicitly instead of relying on the system automatic style.
    @ViewBuilder
    func apolloTopScrollEdgeFade() -> some View {
        if #available(iOS 26.0, *) {
            scrollEdgeEffectHidden(false, for: .top)
                .scrollEdgeEffectStyle(.soft, for: .top)
        } else {
            self
        }
    }

    @ViewBuilder
    func apolloTopScrollEdgeHidden() -> some View {
        if #available(iOS 26.0, *) {
            scrollEdgeEffectHidden(true, for: .top)
        } else {
            self
        }
    }

    func apolloScreenBackground() -> some View {
        modifier(ApolloScreenBackgroundModifier())
    }

    func enableSwipeBack() -> some View {
        background(ApolloSwipeBackEnabler())
    }

    /// Applies `transform` only when `value` is non-nil.
    /// Used to conditionally set `.environment(\.layoutDirection, ...)` so we
    /// never override the system layout direction when the user has chosen
    /// "System Default" — preventing SwiftUI from mirroring text glyphs.
    @ViewBuilder
    func ifLet<T>(_ value: T?, transform: (Self, T) -> some View) -> some View {
        if let value {
            transform(self, value)
        } else {
            self
        }
    }
}

// MARK: - Hinge-aligned split

extension EnvironmentValues {
    /// Full window width, including horizontal safe areas. Split layouts use its
    /// midpoint, which is where the hinge sits on iPhone Duo.
    @Entry var apolloWindowWidth: CGFloat? = nil
    /// Full window height, including vertical safe areas. On iPhone Duo held unfolded
    /// in portrait, its midpoint is where the hinge sits.
    @Entry var apolloWindowHeight: CGFloat? = nil
}

extension View {
    /// Publishes the window width to descendants. Apply once at the root of a window.
    func apolloProvidesWindowWidth() -> some View {
        modifier(ApolloWindowWidthReader())
    }
}

private struct ApolloWindowWidthReader: ViewModifier {
    @State private var windowSize: CGSize?

    func body(content: Content) -> some View {
        content
            .onGeometryChange(for: CGSize.self) { proxy in
                let insets = proxy.safeAreaInsets
                return CGSize(
                    width: proxy.size.width + insets.leading + insets.trailing,
                    height: proxy.size.height + insets.top + insets.bottom
                )
            } action: { windowSize = $0 }
            .environment(\.apolloWindowWidth, windowSize?.width)
            .environment(\.apolloWindowHeight, windowSize?.height)
    }
}

@MainActor
enum ApolloDevice {
    /// Whether this device has a hinge (iPhone Duo). Set by `apolloDetectsHinge`.
    static var hasHinge = false
}

extension View {
    /// Sets `hasHinge` to true on hardware with a hinge (iPhone Duo), in any posture
    /// or orientation. Stays false elsewhere and before iOS 27.1.
    @ViewBuilder
    func apolloDetectsHinge(_ hasHinge: Binding<Bool>) -> some View {
        if #available(iOS 27.1, *) {
            background(ApolloHingeDetector(hasHinge: hasHinge, isUnfolded: nil))
        } else {
            self
        }
    }

    /// Sets `isUnfolded` to true while a hinged device (iPhone Duo) is open, so the
    /// inner display spans the hinge. False when folded, elsewhere, and before iOS 27.1.
    @ViewBuilder
    func apolloDetectsUnfoldedHinge(_ isUnfolded: Binding<Bool>) -> some View {
        if #available(iOS 27.1, *) {
            background(ApolloHingeDetector(hasHinge: nil, isUnfolded: isUnfolded))
        } else {
            self
        }
    }
}

@available(iOS 27.1, *)
private struct ApolloHingeDetector: UIViewRepresentable {
    var hasHinge: Binding<Bool>?
    var isUnfolded: Binding<Bool>?

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.isUserInteractionEnabled = false
        // The update's `hinge` is nil when the hierarchy provides no hinge updates.
        view.addInteraction(UIHingeInteraction { [weak view] _, update in
            let detected = update.hinge != nil
            if let hasHinge, hasHinge.wrappedValue != detected { hasHinge.wrappedValue = detected }
            if let isUnfolded {
                let unfolded = update.hinge.map { $0.status == .partiallyOpen || $0.status == .fullyOpen } ?? false
                if isUnfolded.wrappedValue != unfolded { isUnfolded.wrappedValue = unfolded }
            }
            if ApolloDevice.hasHinge != detected {
                ApolloDevice.hasHinge = detected
                // The app delegate allows extra orientations on a hinged device.
                view?.window?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
            }
        })
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}

/// Passes whether the system shows its vertical bar (iPhone Duo unfolded) to `content`.
/// Read it on the screen itself: the value isn't resolved above the NavigationStack.
struct ApolloVerticalBarReader<Content: View>: View {
    @ViewBuilder var content: (_ hasVerticalBar: Bool) -> Content

    var body: some View {
        if #available(iOS 27.1, *) {
            ApolloVerticalBarEdgeReader(content: content)
        } else {
            content(false)
        }
    }
}

@available(iOS 27.1, *)
private struct ApolloVerticalBarEdgeReader<Content: View>: View {
    // nil where the system never places a vertical bar.
    @Environment(\.toolbarVerticalEdge) private var toolbarVerticalEdge
    let content: (Bool) -> Content

    var body: some View {
        content(toolbarVerticalEdge != nil)
    }
}

extension View {
    /// Keeps `hasVerticalBar` in sync with whether the system shows its vertical bar
    /// (iPhone Duo unfolded). Stays false before iOS 27.1.
    @ViewBuilder
    func apolloReadsVerticalBar(_ hasVerticalBar: Binding<Bool>) -> some View {
        if #available(iOS 27.1, *) {
            modifier(ApolloVerticalBarBinding(hasVerticalBar: hasVerticalBar))
        } else {
            self
        }
    }
}

@available(iOS 27.1, *)
private struct ApolloVerticalBarBinding: ViewModifier {
    @Environment(\.toolbarVerticalEdge) private var toolbarVerticalEdge
    @Binding var hasVerticalBar: Bool

    func body(content: Content) -> some View {
        content.onChange(of: toolbarVerticalEdge != nil, initial: true) { _, isPresent in
            hasVerticalBar = isPresent
        }
    }
}

/// Two side-by-side panels whose gap is centered on the window's midpoint, so the
/// divider lands on the iPhone Duo hinge even when the system's vertical bar insets
/// one edge. Falls back to an even split when the midpoint leaves too little room
/// for either panel (for example in a narrow multitasking window).
struct ApolloHingeSplit<Leading: View, Trailing: View>: View {
    var alignment: VerticalAlignment = .center
    var spacing: CGFloat = 16
    @ViewBuilder var leading: Leading
    @ViewBuilder var trailing: Trailing

    @Environment(\.apolloWindowWidth) private var windowWidth
    @Environment(\.layoutDirection) private var layoutDirection
    @State private var globalFrame: CGRect = .zero

    private static var minimumPanelWidth: CGFloat { 200 }

    private var leadingWidth: CGFloat? {
        guard let windowWidth, globalFrame.width > 0 else { return nil }
        let hinge = windowWidth / 2
        let distanceToHinge = layoutDirection == .leftToRight
            ? hinge - globalFrame.minX
            : globalFrame.maxX - hinge
        let width = distanceToHinge - spacing / 2
        guard width >= Self.minimumPanelWidth,
              globalFrame.width - spacing - width >= Self.minimumPanelWidth else { return nil }
        return width
    }

    var body: some View {
        HStack(alignment: alignment, spacing: spacing) {
            // One modifier chain in both cases keeps the panel's identity (and its
            // state, like text focus) stable when switching between split modes.
            leading
                .frame(width: leadingWidth)
                .frame(maxWidth: leadingWidth == nil ? .infinity : nil)
            trailing
                .frame(maxWidth: .infinity)
        }
        .onGeometryChange(for: CGRect.self) { $0.frame(in: .global) } action: { globalFrame = $0 }
    }
}

/// A feature screen's input, output, and footer (error text and action button).
///
/// - Landscape: input and footer in the leading column, output in the trailing
///   column, divided at the iPhone Duo hinge. With `columnsOnlyWhenShort`, only on
///   compact-height screens (iPhone, folded iPhone Duo), where a stack would push
///   the output off screen; taller screens keep the stack.
/// - Portrait on an unfolded iPhone Duo (unless `columnsOnlyWhenShort`): input above
///   the hinge, output and footer below it.
/// - Otherwise: all three stacked.
///
/// The sections carry their own 16pt horizontal padding, which forms the gap.
struct ApolloInputOutputLayout<Input: View, Output: View, Footer: View>: View {
    var spacing: CGFloat = 12
    var columnsOnlyWhenShort = false
    @ViewBuilder var input: Input
    @ViewBuilder var output: Output
    @ViewBuilder var footer: Footer

    @Environment(\.verticalSizeClass) private var verticalSizeClass
    @State private var isLandscape = false
    @State private var isUnfolded = false

    private var usesColumns: Bool {
        isLandscape && (!columnsOnlyWhenShort || verticalSizeClass == .compact)
    }

    var body: some View {
        Group {
            if usesColumns {
                ApolloHingeSplit(alignment: .top, spacing: 0) {
                    VStack(spacing: spacing) {
                        input
                        footer
                    }
                } trailing: {
                    output
                }
            } else if !isLandscape && isUnfolded && !columnsOnlyWhenShort {
                ApolloHingeVSplit(spacing: spacing) {
                    input
                } bottom: {
                    VStack(spacing: spacing) {
                        output
                        footer
                    }
                }
            } else {
                VStack(spacing: spacing) {
                    input
                    output
                    footer
                }
            }
        }
        .onGeometryChange(for: Bool.self) { $0.size.width > $0.size.height } action: { isLandscape = $0 }
        .apolloDetectsUnfoldedHinge($isUnfolded)
    }
}

/// Empty stand-in for a generator's output area before anything is generated.
struct ApolloOutputPlaceholder: View {
    let systemImage: String

    var body: some View {
        RoundedRectangle(cornerRadius: 16)
            .fill(.ultraThinMaterial)
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.14), lineWidth: 1)
            )
            .overlay {
                Image(systemName: systemImage)
                    .font(.system(size: 40, weight: .regular))
                    .foregroundStyle(.white.opacity(0.3))
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityHidden(true)
    }
}

/// Two stacked panels whose gap is centered on the window's vertical midpoint, so
/// the divider lands on the hinge of an iPhone Duo held unfolded in portrait. Falls
/// back to an even split when the midpoint leaves too little room for either panel.
struct ApolloHingeVSplit<Top: View, Bottom: View>: View {
    var spacing: CGFloat = 12
    @ViewBuilder var top: Top
    @ViewBuilder var bottom: Bottom

    @Environment(\.apolloWindowHeight) private var windowHeight
    @State private var globalFrame: CGRect = .zero

    private static var minimumPanelHeight: CGFloat { 160 }

    private var topHeight: CGFloat? {
        guard let windowHeight, globalFrame.height > 0 else { return nil }
        let height = windowHeight / 2 - globalFrame.minY - spacing / 2
        guard height >= Self.minimumPanelHeight,
              globalFrame.height - spacing - height >= Self.minimumPanelHeight else { return nil }
        return height
    }

    var body: some View {
        VStack(spacing: spacing) {
            // Same modifier chain in both modes keeps the panels' identity stable.
            top
                .frame(height: topHeight, alignment: .top)
                .frame(maxHeight: topHeight == nil ? .infinity : nil, alignment: .top)
            bottom
                .frame(maxHeight: .infinity, alignment: .top)
        }
        .onGeometryChange(for: CGRect.self) { $0.frame(in: .global) } action: { globalFrame = $0 }
    }
}

private struct ApolloSwipeBackEnabler: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> UIViewController {
        let controller = UIViewController()
        controller.view.backgroundColor = .clear
        return controller
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        DispatchQueue.main.async {
            guard let navigationController = uiViewController.navigationController else { return }
            navigationController.interactivePopGestureRecognizer?.isEnabled = true
            navigationController.interactivePopGestureRecognizer?.delegate = nil
        }
    }
}
