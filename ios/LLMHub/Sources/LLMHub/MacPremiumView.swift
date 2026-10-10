//
//  MacPremiumView.swift
//  LLMHub
//
//  Native macOS Premium sheet. Same purchase/restore flow, alerts, feature
//  list and localized strings as `PremiumScreen` on iOS.
//

#if os(macOS)
import SwiftUI

struct MacPremiumView: View {
    @EnvironmentObject var settings: AppSettings
    @StateObject private var purchases = PurchaseManager.shared
    @Environment(\.dismiss) private var dismiss

    @State private var isRestoring = false
    @State private var restoreMessage: String? = nil
    @State private var purchaseError: String? = nil
    @State private var showPurchaseErrorAlert = false
    @State private var crownPulse: Bool = false

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                crown
                    .padding(.top, 28)

                if purchases.isPremium {
                    premiumActiveContent
                } else {
                    upgradeContent
                }
            }
            .frame(maxWidth: 480)
            .padding(.horizontal, 32)
            .padding(.bottom, 28)
            .frame(maxWidth: .infinity)
        }
        .frame(minWidth: 520, idealWidth: 560, minHeight: 600, idealHeight: 720)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button(settings.localized("close")) { dismiss() }
            }
        }
        .task {
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

    private var crown: some View {
        ZStack {
            Circle()
                .fill(
                    RadialGradient(
                        colors: [Color(hex: "FFD700"), Color(hex: "FFA500")],
                        center: .center,
                        startRadius: 0,
                        endRadius: 40
                    )
                )
                .frame(width: 80, height: 80)
                .scaleEffect(crownPulse ? 1.08 : 1.0)
                .animation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true), value: crownPulse)
                .shadow(color: Color(hex: "FFD700").opacity(0.45), radius: 18, x: 0, y: 6)

            Image(systemName: "crown.fill")
                .font(.system(size: 36, weight: .bold))
                .foregroundStyle(.white)
        }
        .onAppear { crownPulse = true }
    }

    // MARK: - Already Premium

    private var premiumActiveContent: some View {
        VStack(spacing: 12) {
            Text(settings.localized("premium_active_title"))
                .font(.largeTitle.weight(.heavy))
                .foregroundStyle(Color(hex: "FFD700"))
                .multilineTextAlignment(.center)

            Text(settings.localized("premium_active_subtitle"))
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Button(settings.localized("close")) { dismiss() }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .padding(.top, 16)
        }
    }

    // MARK: - Upgrade Flow

    private var upgradeContent: some View {
        VStack(spacing: 16) {
            VStack(spacing: 6) {
                Text(settings.localized("premium_title"))
                    .font(.largeTitle.weight(.heavy))
                    .foregroundStyle(Color(hex: "FFD700"))
                    .multilineTextAlignment(.center)

                Text(settings.localized("premium_subtitle"))
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            GroupBox {
                VStack(alignment: .leading, spacing: 12) {
                    featureRow("square.and.arrow.up.fill", "FFC107", settings.localized("premium_feature_import_models"))
                    featureRow("cpu.fill", "A78BFA", settings.localized("premium_feature_agent"))
                    featureRow("waveform.circle.fill", "89D3F7", settings.localized("premium_feature_vibevoice"))
                    featureRow("chevron.left.slash.chevron.right", "A8BCFF", settings.localized("premium_feature_vibecoder"))
                    featureRow("paintpalette.fill", "9CC3FF", settings.localized("premium_feature_image_generator"))
                    featureRow("video.fill", "FF99C8", settings.localized("premium_feature_video_generator"))
                    featureRow(
                        "music.note", "FF9A9E",
                        "\(settings.localized("feature_music_generator")) – \(settings.localized("feature_music_generator_desc"))"
                    )
                    featureRow("sparkles", "00BCD4", settings.localized("premium_feature_future"))
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            VStack(spacing: 10) {
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
                    HStack(spacing: 8) {
                        if purchases.isPurchasing {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "crown.fill")
                            if let price = purchases.formattedPrice {
                                Text("\(settings.localized("premium_button_unlock"))  —  \(price)")
                                    .fontWeight(.bold)
                            } else {
                                ProgressView().controlSize(.small)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(Color(hex: "FFB300"))
                .foregroundStyle(.black)
                .controlSize(.extraLarge)
                .keyboardShortcut(.defaultAction)
                // Disabled until real price is fetched from App Store
                .disabled(purchases.isPurchasing || purchases.isLoadingProduct || purchases.formattedPrice == nil)

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
                    HStack(spacing: 6) {
                        if isRestoring {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "arrow.counterclockwise")
                        }
                        Text(settings.localized("premium_button_restore"))
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .disabled(isRestoring || purchases.isPurchasing)

                Button(settings.localized("premium_button_later")) { dismiss() }
                    .buttonStyle(.link)
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
            }

            Text(settings.localized("premium_payment_note_ios"))
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
    }

    private func featureRow(_ icon: String, _ tintHex: String, _ text: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Color(hex: tintHex))
                .frame(width: 22)
            Text(text)
                .font(.callout)
            Spacer(minLength: 0)
        }
    }
}
#endif
