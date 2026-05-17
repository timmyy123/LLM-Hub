import Foundation
@preconcurrency import GoogleMobileAds
import SwiftUI
import UIKit
import UserMessagingPlatform

// Swift 6: GADInterstitialAd is an ObjC reference type passed in the load() callback.
// The callback is guaranteed to run on the main thread, but Swift 6's region isolation
// still rejects sending the non-Sendable type into @MainActor scope.
// @unchecked Sendable is safe here — it's a ref-counted ObjC object.
extension GADInterstitialAd: @unchecked Sendable {}

// MARK: - AdConfig (reads keys injected via xcconfig → Info.plist)
/// Keys flow: Secrets.*.xcconfig → Info.plist user-defined → Bundle.main.infoDictionary
/// Zero production keys are hard-coded in source.
enum AdConfig {
    static func value(forKey key: String, testFallback: String) -> String {
        (Bundle.main.infoDictionary?[key] as? String)
            .flatMap { $0.isEmpty ? nil : $0 }
            ?? testFallback
    }

    /// AdMob App ID (tilde format). Also injected into Info.plist as GADApplicationIdentifier.
    static var appID: String {
        value(forKey: "ADMOB_APP_ID", testFallback: "ca-app-pub-3940256099942544~3347511713")
    }

    static var bannerID: String {
        value(forKey: "ADMOB_BANNER_ID", testFallback: "ca-app-pub-3940256099942544/2934735716")
    }

    static var interstitialID: String {
        value(forKey: "ADMOB_INTERSTITIAL_ID", testFallback: "ca-app-pub-3940256099942544/4411468910")
    }
}

// MARK: - AdMob SDK Initializer
enum AdMobSDK {
    static func initialize() {
        GADMobileAds.sharedInstance().start(completionHandler: nil)
        NSLog("[AdMobSDK] GoogleMobileAds initialized — App ID: \(AdConfig.appID)")
    }
}

// MARK: - UMP Consent Manager (iOS equivalent of Android ConsentManager.kt)
/// Wraps the Google User Messaging Platform SDK for GDPR/CCPA consent.
///
/// Lifecycle:
///   1. `requestConsentUpdate()` is called once from `LLMHubApp.init()`
///   2. If the user is in the EEA/UK the consent form appears automatically.
///   3. The Settings "Privacy & Ads" row calls `showPrivacyOptionsForm()` so users
///      can revoke or change their consent at any time.
@MainActor
final class ConsentManager: ObservableObject {
    static let shared = ConsentManager()

    /// True when consent has been obtained or is not required in this region.
    @Published private(set) var isConsentGathered: Bool = false

    /// True when the "Privacy & Ads" entry point should be visible (GDPR regions).
    @Published private(set) var isPrivacyOptionsRequired: Bool = false

    private init() {
        // Reflect current persisted consent status immediately on launch
        let status = UMPConsentInformation.sharedInstance.consentStatus
        isConsentGathered = (status == .obtained || status == .notRequired)
        updatePrivacyOptionsRequired()
    }

    // MARK: - On-Launch Request

    /// Request a consent info update and show the form if required.
    /// Safe to call on every launch — UMP throttles duplicate requests internally.
    /// - Parameter debugGeography: Pass `true` to force the EEA consent dialog in
    ///   non-EU regions (for QA). Must be `false` in production builds.
    func requestConsentUpdate(debugGeography: Bool = false) {
        let params = UMPRequestParameters()
        params.tagForUnderAgeOfConsent = false

        if debugGeography {
            let debugSettings = UMPDebugSettings()
            debugSettings.geography = .EEA
            params.debugSettings = debugSettings
            NSLog("[ConsentManager] DEBUG: forcing EEA geography")
        }

        UMPConsentInformation.sharedInstance.requestConsentInfoUpdate(with: params) { [weak self] error in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let error {
                    NSLog("[ConsentManager] Info update failed: \(error)")
                    self.isConsentGathered = true
                    return
                }
                self.updatePrivacyOptionsRequired()
                // UMP 2.x removed isConsentFormAvailable — use formStatus instead
                if UMPConsentInformation.sharedInstance.formStatus == .available {
                    self.loadAndShowFormIfRequired()
                } else {
                    let status = UMPConsentInformation.sharedInstance.consentStatus
                    self.isConsentGathered = (status == .obtained || status == .notRequired)
                }
            }
        }
    }

    // MARK: - Load & Show Form (called internally after info update)

    private func loadAndShowFormIfRequired() {
        guard let rootVC = rootViewController() else {
            NSLog("[ConsentManager] No root view controller — cannot show consent form")
            isConsentGathered = true
            return
        }
        UMPConsentForm.loadAndPresentIfRequired(from: rootVC) { [weak self] error in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let error {
                    NSLog("[ConsentManager] Form error: \(error)")
                }
                let status = UMPConsentInformation.sharedInstance.consentStatus
                self.isConsentGathered = (status == .obtained || status == .notRequired)
                self.updatePrivacyOptionsRequired()
                NSLog("[ConsentManager] Consent gathered: \(self.isConsentGathered), status: \(status.rawValue)")
            }
        }
    }

    // MARK: - Manual Re-open (Settings "Privacy & Ads" row)

    /// Shows the privacy options form so the user can change/revoke consent.
    /// Only presents a form when `isPrivacyOptionsRequired == true` (EEA/GDPR regions).
    /// Silently no-ops for users where consent is not required.
    func showPrivacyOptionsForm() {
        guard let rootVC = rootViewController() else { return }
        UMPConsentForm.presentPrivacyOptionsForm(from: rootVC) { [weak self] error in
            Task { @MainActor [weak self] in
                if let error {
                    NSLog("[ConsentManager] Privacy options form error: \(error)")
                }
                self?.updatePrivacyOptionsRequired()
            }
        }
    }

    // MARK: - Reset (for QA / testing)

    func resetConsentForTesting() {
        UMPConsentInformation.sharedInstance.reset()
        isConsentGathered = false
        isPrivacyOptionsRequired = false
        NSLog("[ConsentManager] Consent reset for testing")
    }

    // MARK: - Private

    private func updatePrivacyOptionsRequired() {
        isPrivacyOptionsRequired =
            UMPConsentInformation.sharedInstance.privacyOptionsRequirementStatus == .required
    }

    private func rootViewController() -> UIViewController? {
        UIApplication.shared.connectedScenes
            .compactMap { ($0 as? UIWindowScene)?.keyWindow?.rootViewController }
            .first
    }
}

// MARK: - Banner Ad View (UIViewRepresentable)
/// Renders a GADAdSizeBanner (320×50) adaptive banner.
/// Automatically hidden for premium users via `BannerAdContainer`.
struct BannerAdView: UIViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> GADBannerView {
        let bannerView = GADBannerView(adSize: GADAdSizeBanner)
        bannerView.adUnitID = AdConfig.bannerID
        bannerView.delegate = context.coordinator
        bannerView.rootViewController = UIApplication.shared.connectedScenes
            .compactMap { ($0 as? UIWindowScene)?.keyWindow?.rootViewController }
            .first
        bannerView.load(GADRequest())
        return bannerView
    }

    func updateUIView(_ uiView: GADBannerView, context: Context) {}

    final class Coordinator: NSObject, GADBannerViewDelegate {
        func bannerView(_ bannerView: GADBannerView, didFailToReceiveAdWithError error: Error) {
            NSLog("[Banner] Failed to load: \(error.localizedDescription)")
        }
        func bannerViewDidReceiveAd(_ bannerView: GADBannerView) {
            NSLog("[Banner] Ad loaded")
        }
    }
}

// MARK: - Banner Ad SwiftUI Container
/// Drop this into any screen. Automatically hides itself for premium users.
struct BannerAdContainer: View {
    @ObservedObject private var purchases = PurchaseManager.shared

    var body: some View {
        if !purchases.isPremium {
            BannerAdView()
                .frame(height: 50)
                .frame(maxWidth: .infinity)
        }
    }
}

// MARK: - Interstitial Ad Manager
/// Fires a full-screen interstitial every `showEveryN` calls to `onEvent()`.
/// Skipped entirely for premium users. Automatically reloads after each show.
@MainActor
final class InterstitialAdManager: NSObject, ObservableObject {
    static let shared = InterstitialAdManager()

    private let showEveryN: Int
    private var eventCount = 0
    private var interstitialAd: GADInterstitialAd?
    /// Set to true after each interstitial dismiss — screens observe this to prompt premium upgrade.
    @Published var showPremiumAfterAd: Bool = false

    private override init() {
        showEveryN = 4
        super.init()
        loadAd()
    }

    // MARK: - Load

    private func loadAd() {
        let request = GADRequest()
        GADInterstitialAd.load(
            withAdUnitID: AdConfig.interstitialID,
            request: request
        ) { [weak self] ad, error in
            Task { @MainActor [weak self] in
                if let error {
                    NSLog("[Interstitial] Load failed: \(error.localizedDescription)")
                    return
                }
                self?.interstitialAd = ad
                self?.interstitialAd?.fullScreenContentDelegate = self
                NSLog("[Interstitial] Ad loaded")
            }
        }
    }

    // MARK: - Trigger

    /// Call this on meaningful user actions (e.g. starting a new chat).
    /// The ad fires every `showEveryN` events, only for non-premium users.
    func onEvent() {
        guard !PurchaseManager.shared.isPremium else { return }
        eventCount += 1
        guard eventCount % showEveryN == 0 else { return }

        guard let rootVC = UIApplication.shared.connectedScenes
            .compactMap({ ($0 as? UIWindowScene)?.keyWindow?.rootViewController })
            .first else { return }

        guard let ad = interstitialAd else {
            NSLog("[Interstitial] No ad ready, reloading")
            loadAd()
            return
        }
        ad.present(fromRootViewController: rootVC)
    }
}

// MARK: - GADFullScreenContentDelegate (Interstitial lifecycle)
extension InterstitialAdManager: GADFullScreenContentDelegate {
    nonisolated func adDidDismissFullScreenContent(_ ad: GADFullScreenPresentingAd) {
        Task { @MainActor in
            self.interstitialAd = nil
            self.loadAd() // Pre-load next ad immediately
            // Prompt free users to go premium after every interstitial
            if !PurchaseManager.shared.isPremium {
                self.showPremiumAfterAd = true
            }
        }
    }

    nonisolated func ad(_ ad: GADFullScreenPresentingAd, didFailToPresentFullScreenContentWithError error: Error) {
        NSLog("[Interstitial] Failed to present: \(error.localizedDescription)")
        Task { @MainActor in
            self.interstitialAd = nil
            self.loadAd()
        }
    }
}
