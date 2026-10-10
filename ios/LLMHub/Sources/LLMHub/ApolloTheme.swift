import SwiftUI
#if canImport(UIKit)
import UIKit
#else
import AppKit
#endif

enum ApolloPalette {
    static let accent = Color(hex: "8AAE9F")
    static let accentStrong = Color(hex: "A7D4C3")
    static let accentMuted = Color(hex: "5C7682")
    static let accentSoft = Color(hex: "6F8E85")
    static let warning = Color(hex: "E0A96D")
    static let destructive = Color(hex: "E96A63")
}

struct ApolloLiquidBackground: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color.black, Color(hex: "0b0f19"), Color(hex: "131a2a")],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            Circle()
                .fill(ApolloPalette.accentSoft.opacity(0.06))
                .frame(width: 260, height: 260)
                .blur(radius: 80)
                .offset(x: 130, y: -260)

            Circle()
                .fill(ApolloPalette.accent.opacity(0.18))
                .frame(width: 220, height: 220)
                .blur(radius: 90)
                .offset(x: -140, y: -220)

            Circle()
                .fill(ApolloPalette.accentMuted.opacity(0.16))
                .frame(width: 300, height: 300)
                .blur(radius: 110)
                .offset(x: 160, y: 260)
        }
    }
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

/// iOS sheets cover the screen, so the presenter is blacked out behind them to
/// avoid flashes. macOS sheets are window-modal panels; keep the window visible.
#if os(macOS)
private let apolloSheetCoversPresenter = false
#else
private let apolloSheetCoversPresenter = true
#endif

private struct ApolloSheetModifier<SheetContent: View>: ViewModifier {
    @Binding var isPresented: Bool
    @State private var coversPresenter = false
    let sheetContent: () -> SheetContent

    func body(content: Content) -> some View {
        content
            .overlay {
                if coversPresenter && apolloSheetCoversPresenter {
                    Color.black.ignoresSafeArea().allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
            }
            .sheet(isPresented: $isPresented, onDismiss: { coversPresenter = false }) {
                sheetContent()
                    .presentationBackground(Color.black)
                    .apolloMacSheetSizing()
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
                if coversPresenter && apolloSheetCoversPresenter {
                    Color.black.ignoresSafeArea().allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
            }
            .sheet(item: $item, onDismiss: { coversPresenter = false }) { value in
                sheetContent(value)
                    .presentationBackground(Color.black)
                    .apolloMacSheetSizing()
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
        if #available(iOS 26.0, macOS 26.0, *) {
            scrollEdgeEffectHidden(false, for: .top)
                .scrollEdgeEffectStyle(.soft, for: .top)
        } else {
            self
        }
    }

    @ViewBuilder
    func apolloTopScrollEdgeHidden() -> some View {
        if #available(iOS 26.0, macOS 26.0, *) {
            scrollEdgeEffectHidden(true, for: .top)
        } else {
            self
        }
    }

    /// macOS sizes sheets to their ideal size, which collapses scroll-based
    /// content. Give sheets a desktop-sized frame and Escape-to-close.
    @ViewBuilder
    func apolloMacSheetSizing() -> some View {
        #if os(macOS)
        modifier(ApolloMacSheetModifier())
        #else
        self
        #endif
    }

    /// Toolbar items keep the platform's native toolbar button appearance even
    /// when the content uses a custom default button style (macOS).
    @ViewBuilder
    func apolloToolbarControl() -> some View {
        #if os(macOS)
        buttonStyle(.automatic)
        #else
        self
        #endif
    }

    /// A `Picker` inside a `Menu` becomes a nested submenu on macOS (one click
    /// opens a blank item, a second opens the list). Inline style lists the
    /// options directly in the menu, matching iOS.
    @ViewBuilder
    func apolloMenuEmbeddedPicker() -> some View {
        #if os(macOS)
        pickerStyle(.inline)
        #else
        self
        #endif
    }

    /// Renders a `Menu` as just its custom label on macOS (no push-button bezel
    /// or extra disclosure chevron), as it appears on iOS.
    @ViewBuilder
    func apolloPlainMenu() -> some View {
        #if os(macOS)
        menuStyle(.button)
            .buttonStyle(ApolloAutomaticButtonStyle())
            .menuIndicator(.hidden)
        #else
        self
        #endif
    }

    func apolloScreenBackground() -> some View {
        modifier(ApolloScreenBackgroundModifier())
    }

    func enableSwipeBack() -> some View {
        #if os(macOS)
        // macOS navigation stacks already support the trackpad back-swipe natively.
        self
        #else
        background(ApolloSwipeBackEnabler())
        #endif
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

#if !os(macOS)
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
#endif

#if os(macOS)
private struct ApolloMacSheetModifier: ViewModifier {
    @Environment(\.dismiss) private var dismiss

    func body(content: Content) -> some View {
        content
            .frame(minWidth: 560, idealWidth: 680, minHeight: 600, idealHeight: 780)
            // Sheets are separate windows on macOS and don't pick up the
            // app-root control styles, so apply the iOS-matching ones here too.
            .tint(ApolloPalette.accentStrong)
            .textFieldStyle(.plain)
            .buttonStyle(ApolloAutomaticButtonStyle())
            .toggleStyle(.switch)
            .formStyle(.grouped)
            .onExitCommand { dismiss() }
    }
}
#endif

/// A stepped slider. On iOS this is the standard stepped `Slider`. macOS draws a
/// tick mark for every step (which turns into a solid dashed line for large
/// ranges), so there it uses a continuous slider that snaps values to `step`.
struct ApolloSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    var onEditingChanged: (Bool) -> Void = { _ in }

    init(
        value: Binding<Double>,
        in range: ClosedRange<Double>,
        step: Double,
        onEditingChanged: @escaping (Bool) -> Void = { _ in }
    ) {
        self._value = value
        self.range = range
        self.step = step
        self.onEditingChanged = onEditingChanged
    }

    var body: some View {
        #if os(macOS)
        Slider(value: snappedValue, in: range, onEditingChanged: onEditingChanged)
        #else
        Slider(value: $value, in: range, step: step, onEditingChanged: onEditingChanged)
        #endif
    }

    private var snappedValue: Binding<Double> {
        Binding(
            get: { value },
            set: { newValue in
                guard step > 0 else { value = newValue; return }
                let snapped = range.lowerBound + ((newValue - range.lowerBound) / step).rounded() * step
                value = min(max(snapped, range.lowerBound), range.upperBound)
            }
        )
    }
}

extension ToolbarContent {
    /// The app draws its own capsule for this item; on macOS 26+ hide the shared
    /// toolbar glass so it isn't layered under a second glass background.
    @ToolbarContentBuilder
    func apolloHideSharedToolbarBackground() -> some ToolbarContent {
        #if os(macOS)
        if #available(macOS 26.0, *) {
            sharedBackgroundVisibility(.hidden)
        } else {
            self
        }
        #else
        self
        #endif
    }
}

extension ToolbarItemPlacement {
    /// Leading action inside a sheet. macOS sheets don't show navigation-bar
    /// placements, only cancellation/confirmation actions, so map to those there.
    static var apolloSheetLeading: ToolbarItemPlacement {
        #if os(macOS)
        .cancellationAction
        #else
        .navigationBarLeading
        #endif
    }

    /// Trailing (done/save) action inside a sheet; see `apolloSheetLeading`.
    static var apolloSheetTrailing: ToolbarItemPlacement {
        #if os(macOS)
        .confirmationAction
        #else
        .navigationBarTrailing
        #endif
    }
}
