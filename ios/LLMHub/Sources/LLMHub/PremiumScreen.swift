import SwiftUI

// MARK: - PremiumScreen

struct PremiumScreen: View {
    @EnvironmentObject var settings: AppSettings
    @StateObject private var purchases = PurchaseManager.shared
    @Environment(\.dismiss) private var dismiss

    @State private var isRestoring = false
    @State private var restoreMessage: String? = nil
    @State private var showRestoreAlert = false
    @State private var purchaseError: String? = nil
    @State private var showPurchaseErrorAlert = false

    // Pulse animation for crown
    @State private var crownPulse: Bool = false

    var body: some View {
        ZStack {
            // Background gradient matching Android (deep purple → dark navy)
            LinearGradient(
                colors: [Color(hex: "1A0533"), Color(hex: "0D1B4B")],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            // Subtle particle glow
            Circle()
                .fill(Color(hex: "7C4DFF").opacity(0.18))
                .frame(width: 300, height: 300)
                .blur(radius: 80)
                .offset(x: -60, y: -120)
                .allowsHitTesting(false)

            Circle()
                .fill(Color(hex: "FFD700").opacity(0.09))
                .frame(width: 250, height: 250)
                .blur(radius: 70)
                .offset(x: 80, y: 200)
                .allowsHitTesting(false)

            ScrollView {
                VStack(spacing: 0) {

                    // ── Close button row ──────────────────────────────────
                    HStack {
                        Spacer()
                        Button { dismiss() } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 26))
                                .foregroundStyle(.white.opacity(0.55))
                        }
                        .buttonStyle(.plain)
                        .padding(.trailing, 20)
                        .padding(.top, 16)
                    }

                    // ── Crown icon ────────────────────────────────────────
                    ZStack {
                        Circle()
                            .fill(
                                RadialGradient(
                                    colors: [Color(hex: "FFD700"), Color(hex: "FFA500")],
                                    center: .center,
                                    startRadius: 0,
                                    endRadius: 48
                                )
                            )
                            .frame(width: 96, height: 96)
                            .scaleEffect(crownPulse ? 1.10 : 1.0)
                            .animation(
                                .easeInOut(duration: 0.9).repeatForever(autoreverses: true),
                                value: crownPulse
                            )
                            .shadow(color: Color(hex: "FFD700").opacity(0.55), radius: 24, x: 0, y: 8)

                        Image(systemName: "crown.fill")
                            .font(.system(size: 44, weight: .bold))
                            .foregroundStyle(.white)
                    }
                    .padding(.top, 8)
                    .onAppear { crownPulse = true }

                    Spacer().frame(height: 20)

                    if purchases.isPremium {
                        // ── Already premium state ─────────────────────────
                        premiumActiveContent
                    } else {
                        // ── Upgrade flow ──────────────────────────────────
                        upgradeContent
                    }

                    Spacer().frame(height: 24)
                }
                .padding(.horizontal, 24)
            }
        }
        .task {
            // Load price if not yet fetched
            if purchases.product == nil {
                await purchases.loadProduct()
            }
        }
        .alert(settings.localized("premium_restore_success"), isPresented: Binding(
            get: { restoreMessage == settings.localized("premium_restore_success") },
            set: { if !$0 { restoreMessage = nil } }
        )) {
            Button(settings.localized("ok")) { restoreMessage = nil }
        }
        .alert(settings.localized("premium_restore_nothing"), isPresented: Binding(
            get: { restoreMessage == settings.localized("premium_restore_nothing") },
            set: { if !$0 { restoreMessage = nil } }
        )) {
            Button(settings.localized("ok")) { restoreMessage = nil }
        }
        .alert(purchaseError ?? "", isPresented: $showPurchaseErrorAlert) {
            Button(settings.localized("ok")) { purchaseError = nil }
        }
    }

    // MARK: - Already Premium

    @ViewBuilder
    private var premiumActiveContent: some View {
        VStack(spacing: 12) {
            Text(settings.localized("premium_active_title"))
                .font(.system(size: 28, weight: .heavy))
                .foregroundStyle(Color(hex: "FFD700"))
                .multilineTextAlignment(.center)

            Text(settings.localized("premium_active_subtitle"))
                .font(.body)
                .foregroundStyle(.white.opacity(0.82))
                .multilineTextAlignment(.center)

            Spacer().frame(height: 24)

            Button { dismiss() } label: {
                Text(settings.localized("close"))
                    .font(.headline)
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 15)
                    .background(Color.white.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.white.opacity(0.25), lineWidth: 1)
                    )
            }
        }
    }

    // MARK: - Upgrade Flow

    @ViewBuilder
    private var upgradeContent: some View {
        VStack(spacing: 0) {

            // Title + subtitle
            Text(settings.localized("premium_title"))
                .font(.system(size: 28, weight: .heavy))
                .foregroundStyle(Color(hex: "FFD700"))
                .multilineTextAlignment(.center)

            Spacer().frame(height: 6)

            Text(settings.localized("premium_subtitle"))
                .font(.body)
                .foregroundStyle(.white.opacity(0.72))
                .multilineTextAlignment(.center)

            Spacer().frame(height: 28)

            // Feature list card — iOS: only remove ads + import external models
            VStack(spacing: 0) {
                PremiumFeatureRow(icon: "nosign",                   tint: Color(hex: "FF7043"), textKey: "premium_feature_no_ads")
                PremiumFeatureRow(icon: "square.and.arrow.up.fill", tint: Color(hex: "FFC107"), textKey: "premium_feature_import_models")
                PremiumFeatureRow(icon: "sparkles",                 tint: Color(hex: "00BCD4"), textKey: "premium_feature_future", isLast: true)
            }
            .padding(20)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(Color.white.opacity(0.14), lineWidth: 1)
            )

            Spacer().frame(height: 24)

            // Purchase button — price comes 100% from App Store, no hardcoded fallback
            Button {
                Task {
                    let success = await purchases.purchase()
                    if !success && !purchases.isPremium {
                        purchaseError = "Purchase could not be completed. Please try again."
                        showPurchaseErrorAlert = true
                    }
                }
            } label: {
                HStack(spacing: 10) {
                    if purchases.isPurchasing {
                        ProgressView()
                            .tint(Color(hex: "1A0533"))
                    } else {
                        Image(systemName: "crown.fill")
                            .font(.system(size: 16, weight: .bold))

                        if let price = purchases.formattedPrice {
                            // Real price from App Store for user's region
                            Text("\(settings.localized("premium_button_unlock"))  —  \(price)")
                                .font(.system(size: 16, weight: .bold))
                        } else {
                            // Price still loading — show spinner, button disabled
                            ProgressView()
                                .tint(Color(hex: "1A0533"))
                        }
                    }
                }
                .foregroundStyle(Color(hex: "1A0533"))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(
                    LinearGradient(
                        colors: purchases.formattedPrice == nil
                            ? [Color.gray.opacity(0.5), Color.gray.opacity(0.4)]
                            : [Color(hex: "FFD700"), Color(hex: "FFA500")],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .shadow(color: Color(hex: "FFD700").opacity(0.38), radius: 14, x: 0, y: 6)
            }
            // Disabled until real price is fetched from App Store
            .disabled(purchases.isPurchasing || purchases.isLoadingProduct || purchases.formattedPrice == nil)

            Spacer().frame(height: 12)

            // Restore button
            Button {
                Task {
                    isRestoring = true
                    let found = await purchases.restorePurchases()
                    isRestoring = false
                    restoreMessage = found
                        ? settings.localized("premium_restore_success")
                        : settings.localized("premium_restore_nothing")
                }
            } label: {
                HStack(spacing: 8) {
                    if isRestoring {
                        ProgressView()
                            .tint(.white)
                            .scaleEffect(0.85)
                    } else {
                        Image(systemName: "arrow.counterclockwise")
                    }
                    Text(settings.localized("premium_button_restore"))
                        .font(.system(size: 15, weight: .medium))
                }
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.white.opacity(0.30), lineWidth: 1)
                )
            }
            .disabled(isRestoring || purchases.isPurchasing)

            Spacer().frame(height: 4)

            // Maybe later
            Button { dismiss() } label: {
                Text(settings.localized("premium_button_later"))
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.42))
                    .padding(.vertical, 8)
            }

            Spacer().frame(height: 12)

            // Payment note — iOS-specific wording
            Text(settings.localized("premium_payment_note_ios"))
                .font(.system(size: 11))
                .foregroundStyle(.white.opacity(0.38))
                .multilineTextAlignment(.center)
        }
    }
}

// MARK: - Premium Feature Row

private struct PremiumFeatureRow: View {
    @EnvironmentObject var settings: AppSettings

    let icon: String
    let tint: Color
    let textKey: String
    var isLast: Bool = false

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 24)

            Text(settings.localized(textKey))
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.white.opacity(0.90))

            Spacer()
        }
        .padding(.bottom, isLast ? 0 : 14)
    }
}

// MARK: - Premium Locked Overlay
/// Wrap any view with this to show an upgrade prompt when the user isn't premium.
struct PremiumLockedOverlay: View {
    @EnvironmentObject var settings: AppSettings
    @StateObject private var purchases = PurchaseManager.shared
    @Binding var showPremium: Bool

    var body: some View {
        if !purchases.isPremium {
            Button { showPremium = true } label: {
                HStack(spacing: 8) {
                    Image(systemName: "crown.fill")
                        .font(.caption)
                        .foregroundStyle(Color(hex: "FFD700"))
                    Text(settings.localized("premium_tap_to_unlock"))
                        .font(.caption.bold())
                        .foregroundStyle(.white)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
                .clipShape(Capsule())
                .overlay(
                    Capsule()
                        .stroke(Color(hex: "FFD700").opacity(0.55), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)
        }
    }
}
