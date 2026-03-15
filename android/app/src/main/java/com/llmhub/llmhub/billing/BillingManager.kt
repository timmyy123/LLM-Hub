package com.llmhub.llmhub.billing

import android.app.Activity
import android.content.Context
import android.util.Log
import com.android.billingclient.api.*
import com.llmhub.llmhub.BuildConfig
import com.llmhub.llmhub.data.ThemePreferences
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

class BillingManager(private val context: Context) {

    companion object {
        const val PRODUCT_PREMIUM_LIFETIME = "premium_lifetime"
        private const val TAG = "BillingManager"
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val prefs = ThemePreferences(context)

    private val _isPremium = MutableStateFlow(false)
    val isPremium: StateFlow<Boolean> = _isPremium.asStateFlow()

    // Formatted price string fetched from Play Store (e.g. "$4.99", "€4,49")
    private val _productPrice = MutableStateFlow<String?>(null)
    val productPrice: StateFlow<String?> = _productPrice.asStateFlow()

    // Non-null once product details have been queried successfully
    private var productDetails: ProductDetails? = null

    private val purchasesUpdatedListener = PurchasesUpdatedListener { billingResult, purchases ->
        when (billingResult.responseCode) {
            BillingClient.BillingResponseCode.OK -> {
                purchases?.forEach { purchase ->
                    scope.launch { handlePurchase(purchase) }
                }
            }
            BillingClient.BillingResponseCode.USER_CANCELED -> {
                Log.d(TAG, "User canceled purchase")
            }
            else -> {
                Log.w(TAG, "Purchase update error: ${billingResult.debugMessage}")
            }
        }
    }

    private val billingClient = BillingClient.newBuilder(context)
        .setListener(purchasesUpdatedListener)
        .enablePendingPurchases(
            PendingPurchasesParams.newBuilder().enableOneTimeProducts().build()
        )
        .build()

    init {
        if (BuildConfig.DEBUG_PREMIUM) {
            // Debug override: bypass Play Billing entirely
            _isPremium.value = true
            _productPrice.value = "$0.00"
            Log.d(TAG, "DEBUG_PREMIUM=true — all premium features unlocked")
        } else {
            // Restore cached premium state immediately (fast path, no network)
            scope.launch {
                prefs.isPremium.collect { stored ->
                    _isPremium.value = stored
                }
            }
            connectAndQuery()
        }
    }

    private fun connectAndQuery() {
        billingClient.startConnection(object : BillingClientStateListener {
            override fun onBillingSetupFinished(result: BillingResult) {
                if (result.responseCode == BillingClient.BillingResponseCode.OK) {
                    scope.launch {
                        queryProductDetails()
                        restorePurchasesInternal()
                    }
                } else {
                    Log.w(TAG, "Billing setup failed: ${result.debugMessage}")
                }
            }

            override fun onBillingServiceDisconnected() {
                Log.w(TAG, "Billing service disconnected, retrying…")
                scope.launch {
                    delay(5_000)
                    if (!billingClient.isReady) connectAndQuery()
                }
            }
        })
    }

    private suspend fun queryProductDetails() {
        val params = QueryProductDetailsParams.newBuilder()
            .setProductList(
                listOf(
                    QueryProductDetailsParams.Product.newBuilder()
                        .setProductId(PRODUCT_PREMIUM_LIFETIME)
                        .setProductType(BillingClient.ProductType.INAPP)
                        .build()
                )
            )
            .build()

        val result = billingClient.queryProductDetails(params)
        if (result.billingResult.responseCode == BillingClient.BillingResponseCode.OK) {
            productDetails = result.productDetailsList?.firstOrNull()
            _productPrice.value = productDetails?.oneTimePurchaseOfferDetails?.formattedPrice
            Log.d(TAG, "Product price: ${_productPrice.value}")
        } else {
            Log.w(TAG, "queryProductDetails failed: ${result.billingResult.debugMessage}")
        }
    }

    /**
     * Silently restore from local Play Store cache — fast, used on startup.
     * May miss purchases on a brand-new device that hasn't synced yet.
     */
    suspend fun restorePurchases() {
        if (!billingClient.isReady) {
            Log.d(TAG, "Billing not ready, skipping restore")
            return
        }
        restorePurchasesInternal()
    }

    /**
     * Full server-side restore — always hits Google's servers.
     * Use this for the manual "Restore Purchase" button so new-device users are covered.
     * Returns true if premium was found and activated.
     */
    suspend fun restorePurchasesFromServer(): Boolean {
        if (!billingClient.isReady) {
            // Try to reconnect and wait briefly
            connectAndQuery()
            delay(3_000)
            if (!billingClient.isReady) {
                Log.w(TAG, "Billing not ready for server restore")
                return false
            }
        }

        // First try the local cache (fast)
        restorePurchasesInternal()
        if (_isPremium.value) return true

        // Fall back to server history query — works on new devices with empty local cache
        val historyParams = QueryPurchaseHistoryParams.newBuilder()
            .setProductType(BillingClient.ProductType.INAPP)
            .build()

        val historyResult = billingClient.queryPurchaseHistory(historyParams)
        if (historyResult.billingResult.responseCode == BillingClient.BillingResponseCode.OK) {
            val foundInHistory = historyResult.purchaseHistoryRecordList?.any { record ->
                record.products.contains(PRODUCT_PREMIUM_LIFETIME)
            } ?: false

            if (foundInHistory) {
                Log.d(TAG, "Premium found in purchase history — activating")
                setPremium(true)
                return true
            }
        } else {
            Log.w(TAG, "queryPurchaseHistory failed: ${historyResult.billingResult.debugMessage}")
        }

        return _isPremium.value
    }

    private suspend fun restorePurchasesInternal() {
        val params = QueryPurchasesParams.newBuilder()
            .setProductType(BillingClient.ProductType.INAPP)
            .build()

        val result = billingClient.queryPurchasesAsync(params)
        if (result.billingResult.responseCode == BillingClient.BillingResponseCode.OK) {
            val ownsPremium = result.purchasesList.any { purchase ->
                purchase.products.contains(PRODUCT_PREMIUM_LIFETIME) &&
                    purchase.purchaseState == Purchase.PurchaseState.PURCHASED
            }
            if (ownsPremium) {
                setPremium(true)
                // Acknowledge any unacknowledged purchases to complete the transaction
                result.purchasesList
                    .filter {
                        it.products.contains(PRODUCT_PREMIUM_LIFETIME) &&
                            it.purchaseState == Purchase.PurchaseState.PURCHASED &&
                            !it.isAcknowledged
                    }
                    .forEach { acknowledgeIfNeeded(it) }
            }
        } else {
            Log.w(TAG, "queryPurchases failed: ${result.billingResult.debugMessage}")
        }
    }

    /**
     * Launch the Google Play purchase flow.
     * Must be called from a composable or function that has access to the current Activity.
     */
    fun launchPurchaseFlow(activity: Activity) {
        val details = productDetails
        if (details == null) {
            // Re-query and try again on next user attempt
            scope.launch { queryProductDetails() }
            Log.w(TAG, "Product details not yet loaded")
            return
        }

        val productDetailsParams = BillingFlowParams.ProductDetailsParams.newBuilder()
            .setProductDetails(details)
            .build()

        val billingFlowParams = BillingFlowParams.newBuilder()
            .setProductDetailsParamsList(listOf(productDetailsParams))
            .build()

        billingClient.launchBillingFlow(activity, billingFlowParams)
    }

    private suspend fun handlePurchase(purchase: Purchase) {
        if (purchase.purchaseState == Purchase.PurchaseState.PURCHASED &&
            purchase.products.contains(PRODUCT_PREMIUM_LIFETIME)
        ) {
            setPremium(true)
            acknowledgeIfNeeded(purchase)
        }
    }

    private suspend fun acknowledgeIfNeeded(purchase: Purchase) {
        if (!purchase.isAcknowledged) {
            val params = AcknowledgePurchaseParams.newBuilder()
                .setPurchaseToken(purchase.purchaseToken)
                .build()
            val result = billingClient.acknowledgePurchase(params)
            if (result.responseCode == BillingClient.BillingResponseCode.OK) {
                Log.d(TAG, "Purchase acknowledged")
            } else {
                Log.w(TAG, "Acknowledge failed: ${result.debugMessage}")
            }
        }
    }

    private suspend fun setPremium(premium: Boolean) {
        _isPremium.value = premium
        prefs.setIsPremium(premium)
    }

    fun destroy() {
        scope.cancel()
        billingClient.endConnection()
    }
}
