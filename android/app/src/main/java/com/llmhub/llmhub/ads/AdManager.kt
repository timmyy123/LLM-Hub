package com.llmhub.llmhub.ads

import android.app.Activity
import android.content.Context
import android.util.Log
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.google.android.gms.ads.*
import com.google.android.gms.ads.interstitial.InterstitialAd
import com.google.android.gms.ads.interstitial.InterstitialAdLoadCallback
import com.google.android.gms.ads.rewarded.RewardedAd
import com.google.android.gms.ads.rewarded.RewardedAdLoadCallback
import com.llmhub.llmhub.BuildConfig

/**
 * Call once from Application.onCreate() to initialise the AdMob SDK.
 */
object AdManager {
    fun initialize(context: Context) {
        MobileAds.initialize(context)
    }
}

/**
 * A 320×50 banner ad that loads once when first composed.
 * Only wire this into the UI after confirming the user is not premium.
 */
@Composable
fun BannerAd(modifier: Modifier = Modifier) {
    AndroidView(
        modifier = modifier,
        factory = { context ->
            AdView(context).apply {
                setAdSize(AdSize.BANNER)
                adUnitId = BuildConfig.ADMOB_BANNER_ID
                loadAd(AdRequest.Builder().build())
            }
        }
        // No update block — ad loads once; avoid wasting impressions on recompositions
    )
}

/**
 * Manages one interstitial slot.
 * Show the ad every [showEveryN] new-chat events.
 */
class InterstitialAdManager(
    private val context: Context,
    private val showEveryN: Int = 4
) {
    private var interstitialAd: InterstitialAd? = null
    private var eventCount = 0
    private val TAG = "InterstitialAdManager"

    init {
        loadAd()
    }

    fun loadAd() {
        InterstitialAd.load(
            context,
            BuildConfig.ADMOB_INTERSTITIAL_ID,
            AdRequest.Builder().build(),
            object : InterstitialAdLoadCallback() {
                override fun onAdLoaded(ad: InterstitialAd) {
                    interstitialAd = ad
                    Log.d(TAG, "Interstitial loaded")
                }

                override fun onAdFailedToLoad(error: LoadAdError) {
                    Log.d(TAG, "Interstitial load failed: ${error.message}")
                    interstitialAd = null
                }
            }
        )
    }

    /**
     * Call this every time the user starts a new chat session.
     * The ad will be shown on every [showEveryN]th call.
     */
    fun onNewChatStarted(activity: Activity) {
        eventCount++
        if (eventCount % showEveryN == 0 && interstitialAd != null) {
            interstitialAd?.fullScreenContentCallback = object : FullScreenContentCallback() {
                override fun onAdDismissedFullScreenContent() {
                    interstitialAd = null
                    loadAd() // pre-load the next one
                }

                override fun onAdFailedToShowFullScreenContent(error: AdError) {
                    interstitialAd = null
                    loadAd()
                }
            }
            interstitialAd?.show(activity)
        }
    }
}

/**
 * Shows a rewarded ad before granting the requested action.
 * If no ad is ready yet, [onGranted] is called immediately so the user is never blocked.
 * Pre-loads the next ad automatically after each show.
 */
class RewardedAdManager(private val context: Context) {
    private var rewardedAd: RewardedAd? = null
    private val TAG = "RewardedAdManager"

    init {
        loadAd()
    }

    fun loadAd() {
        RewardedAd.load(
            context,
            BuildConfig.ADMOB_REWARDED_ID,
            AdRequest.Builder().build(),
            object : RewardedAdLoadCallback() {
                override fun onAdLoaded(ad: RewardedAd) {
                    rewardedAd = ad
                    Log.d(TAG, "Rewarded ad loaded")
                }

                override fun onAdFailedToLoad(error: LoadAdError) {
                    Log.d(TAG, "Rewarded ad load failed: ${error.message}")
                    rewardedAd = null
                }
            }
        )
    }

    /**
     * Show a rewarded ad then call [onGranted] when the user earns the reward.
     * If no ad is available, [onGranted] is called immediately (never block the user).
     */
    fun showAdOrGrant(activity: Activity, onGranted: () -> Unit) {
        val ad = rewardedAd
        if (ad == null) {
            // No ad ready — grant immediately and try to load for next time
            onGranted()
            loadAd()
            return
        }
        ad.fullScreenContentCallback = object : FullScreenContentCallback() {
            override fun onAdDismissedFullScreenContent() {
                rewardedAd = null
                loadAd()
            }

            override fun onAdFailedToShowFullScreenContent(error: AdError) {
                rewardedAd = null
                loadAd()
                onGranted() // fallback — don't punish user for ad failure
            }
        }
        ad.show(activity) { /* RewardItem */ onGranted() }
    }
}
