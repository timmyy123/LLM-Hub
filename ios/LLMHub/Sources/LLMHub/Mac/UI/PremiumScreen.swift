import SwiftUI

public struct PremiumScreen: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var purchases = PurchaseManager.shared

    public init() {}

    public var body: some View {
        VStack(spacing: 0) {
            // Header Bar
            HStack {
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 20))
                        .foregroundColor(.white.opacity(0.6))
                }
                .buttonStyle(.plain)
            }
            .padding()

            ScrollView {
                VStack(spacing: 24) {
                    // Badge & Crown
                    ZStack {
                        Circle()
                            .fill(Color(hex: "FFD700").opacity(0.18))
                            .frame(width: 80, height: 80)
                            .blur(radius: 12)

                        Circle()
                            .fill(
                                LinearGradient(
                                    colors: [Color(hex: "FFE259"), Color(hex: "FFA751")],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 64, height: 64)
                            .shadow(color: Color(hex: "FFA751").opacity(0.5), radius: 10, x: 0, y: 5)

                        Image(systemName: "crown.fill")
                            .font(.system(size: 30))
                            .foregroundColor(.black)
                    }

                    VStack(spacing: 6) {
                        Text("LLM Hub Pro for macOS")
                            .font(.system(size: 24, weight: .bold))
                            .foregroundColor(.white)

                        Text("Unlock the full power of on-device AI and desktop automation")
                            .font(.system(size: 13))
                            .foregroundColor(.white.opacity(0.7))
                    }

                    // Features List
                    VStack(alignment: .leading, spacing: 14) {
                        featureRow(icon: "bolt.fill", title: "Unlimited Offline Inference", desc: "No usage caps, no token limits, and zero cloud dependency.")
                        featureRow(icon: "cpu.fill", title: "Full Autonomous AI Agent", desc: "Execute multi-step goals with native macOS terminal integration.")
                        featureRow(icon: "wand.and.stars", title: "4× Super-Resolution & Video", desc: "High-resolution AI upscaling and video generation.")
                        featureRow(icon: "laptopcomputer", title: "Vibe Coder Live Workspace", desc: "Full live HTML/JS interactive execution and instant export.")
                    }
                    .padding(20)
                    .background(Color(hex: "101626"))
                    .cornerRadius(14)
                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(ApolloPalette.borderGlass, lineWidth: 1))

                    // Unlock Button
                    VStack(spacing: 12) {
                        Button {
                            Task {
                                await purchases.purchase()
                                if purchases.isPremium {
                                    dismiss()
                                }
                            }
                        } label: {
                            HStack {
                                Spacer()
                                Text(purchases.isPremium ? "Premium Active" : "Unlock Lifetime Access")
                                    .font(.system(size: 14, weight: .bold))
                                Spacer()
                            }
                            .padding(.vertical, 12)
                        }
                        .buttonStyle(ApolloPrimaryButtonStyle())
                        .disabled(purchases.isPremium)

                        Button("Restore Purchases") {
                            Task {
                                await purchases.restorePurchases()
                            }
                        }
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.6))
                        .buttonStyle(.plain)
                    }
                    .padding(.top, 10)
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 32)
            }
        }
        .frame(width: 480, height: 600)
        .background(Color(hex: "0a0e18"))
    }

    private func featureRow(icon: String, title: String, desc: String) -> some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 16))
                .foregroundColor(ApolloPalette.accentStrong)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white)
                Text(desc)
                    .font(.system(size: 12))
                    .foregroundColor(.white.opacity(0.6))
            }
        }
    }
}
