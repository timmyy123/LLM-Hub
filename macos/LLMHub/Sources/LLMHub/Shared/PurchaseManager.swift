import Foundation
import StoreKit
import SwiftUI

// MARK: - PurchaseManager
/// One-time IAP manager using StoreKit 2.
/// Product ID must match exactly what you configure in App Store Connect.
@MainActor
final class PurchaseManager: ObservableObject {
    static let shared = PurchaseManager()

    // ── Change this to your actual App Store Connect product ID ──
    static let premiumProductID = "com.llmhub.premium.lifetime"
    // ────────────────────────────────────────────────────────────

    @Published private(set) var isPremium: Bool = false
    @Published private(set) var product: Product? = nil
    @Published private(set) var isLoadingProduct = false
    @Published private(set) var isPurchasing = false

    /// Formatted price string from the App Store for the user's region.
    var formattedPrice: String? {
        guard let product else { return nil }
        return product.displayPrice
    }

    private var transactionUpdateTask: Task<Void, Never>? = nil

    private init() {
        // Restore persisted premium state or bypass if enabled in xcconfig
        let bypassStr = Bundle.main.infoDictionary?["BYPASS_PREMIUM_UNLOCK"] as? String
        if bypassStr == "YES" {
            isPremium = true
            NSLog("[PurchaseManager] Premium automatically unlocked via BYPASS_PREMIUM_UNLOCK = YES")
        } else {
            isPremium = UserDefaults.standard.bool(forKey: "premium_purchased")
        }

        // Listen for StoreKit 2 transaction updates (e.g. family sharing, deferred purchases)
        transactionUpdateTask = Task.detached(priority: .background) { [weak self] in
            for await update in StoreKit.Transaction.updates {
                await self?.handle(verificationResult: update)
            }
        }

        // Load product & restore purchases on launch
        Task {
            await loadProduct()
            await restoreFromStoreKit()
        }
    }

    deinit {
        transactionUpdateTask?.cancel()
    }

    // MARK: - Product Loading

    func loadProduct() async {
        guard product == nil else { return }
        isLoadingProduct = true
        defer { isLoadingProduct = false }

        do {
            let products = try await Product.products(for: [PurchaseManager.premiumProductID])
            product = products.first
            NSLog("[PurchaseManager] Product loaded: \(product?.displayName ?? "nil") — \(product?.displayPrice ?? "nil")")
        } catch {
            NSLog("[PurchaseManager] Failed to load product: \(error)")
        }
    }

    // MARK: - Purchase

    func purchase() async -> Bool {
        guard let product else {
            NSLog("[PurchaseManager] No product loaded yet")
            // Try loading again
            await loadProduct()
            guard product != nil else { return false }
            return await purchase()
        }

        isPurchasing = true
        defer { isPurchasing = false }

        do {
            let result = try await product.purchase()
            switch result {
            case .success(let verification):
                await handle(verificationResult: verification)
                return isPremium
            case .userCancelled:
                NSLog("[PurchaseManager] User cancelled purchase")
                return false
            case .pending:
                NSLog("[PurchaseManager] Purchase pending (Ask to Buy / parental approval)")
                return false
            @unknown default:
                return false
            }
        } catch {
            NSLog("[PurchaseManager] Purchase error: \(error)")
            return false
        }
    }

    // MARK: - Restore

    /// Full restore — hits App Store servers.
    /// Call this from the "Restore Purchase" button.
    func restorePurchases() async -> Bool {
        do {
            try await AppStore.sync()
        } catch {
            NSLog("[PurchaseManager] AppStore.sync() failed: \(error)")
        }
        await restoreFromStoreKit()
        return isPremium
    }

    // MARK: - Private Helpers

    private func restoreFromStoreKit() async {
        for await result in StoreKit.Transaction.currentEntitlements {
            await handle(verificationResult: result)
        }
    }

    private func handle(verificationResult: VerificationResult<StoreKit.Transaction>) async {
        switch verificationResult {
        case .unverified(_, let error):
            NSLog("[PurchaseManager] Unverified transaction: \(error)")
        case .verified(let transaction):
            if transaction.productID == PurchaseManager.premiumProductID {
                switch transaction.revocationDate {
                case .none:
                    // Valid purchase
                    setPremium(true)
                case .some:
                    // Revoked (refund / family sharing removed)
                    NSLog("[PurchaseManager] Transaction revoked: \(transaction.productID)")
                    // Note: we do NOT downgrade automatically for lifetime IAPs per Apple's guideline.
                    // If you want strict revocation, uncomment: setPremium(false)
                }
            }
            await transaction.finish()
        }
    }

    private func setPremium(_ value: Bool) {
        isPremium = value
        UserDefaults.standard.set(value, forKey: "premium_purchased")
        NSLog("[PurchaseManager] isPremium = \(value)")
    }
}
