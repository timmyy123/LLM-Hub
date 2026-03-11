package com.llmhub.llmhub.ads

import android.app.Activity
import android.content.Context
import android.util.Log
import com.google.android.ump.ConsentDebugSettings
import com.google.android.ump.ConsentInformation
import com.google.android.ump.ConsentRequestParameters
import com.google.android.ump.UserMessagingPlatform

/**
 * Wraps the AdMob User Messaging Platform (UMP) SDK.
 *
 * - Call [requestConsentInfoUpdate] once from MainActivity.onCreate.
 * - If consent is required and a form is available it is shown automatically.
 * - Call [showPrivacyOptionsForm] when the user taps "Privacy & Ads" in Settings.
 */
object ConsentManager {
    private const val TAG = "ConsentManager"

    /** True once consent has been gathered (or is not required in this region). */
    val isConsentGathered: Boolean
        get() = consentInformation?.consentStatus == ConsentInformation.ConsentStatus.OBTAINED ||
                consentInformation?.consentStatus == ConsentInformation.ConsentStatus.NOT_REQUIRED

    private var consentInformation: ConsentInformation? = null

    fun requestConsentInfoUpdate(activity: Activity, onComplete: () -> Unit = {}) {
        val params = ConsentRequestParameters.Builder()
            .setTagForUnderAgeOfConsent(false)
            .build()

        val ci = UserMessagingPlatform.getConsentInformation(activity)
        consentInformation = ci

        ci.requestConsentInfoUpdate(
            activity,
            params,
            {
                // Success — load and show the form if required
                if (ci.isConsentFormAvailable) {
                    loadAndShowFormIfRequired(activity, onComplete)
                } else {
                    onComplete()
                }
            },
            { formError ->
                Log.w(TAG, "Consent info update failed: ${formError.message}")
                onComplete()
            }
        )
    }

    private fun loadAndShowFormIfRequired(activity: Activity, onComplete: () -> Unit) {
        UserMessagingPlatform.loadAndShowConsentFormIfRequired(
            activity,
            { formError ->
                if (formError != null) {
                    Log.w(TAG, "Consent form error: ${formError.message}")
                }
                onComplete()
            }
        )
    }

    /**
     * Show the privacy options form when the user manually taps the Settings entry.
     * Only works when [ConsentInformation.PrivacyOptionsRequirementStatus.REQUIRED].
     */
    fun showPrivacyOptionsForm(activity: Activity, onDismiss: () -> Unit = {}) {
        UserMessagingPlatform.showPrivacyOptionsForm(
            activity,
            { formError ->
                if (formError != null) {
                    Log.w(TAG, "Privacy options form error: ${formError.message}")
                }
                onDismiss()
            }
        )
    }

    /** Whether the privacy options entry point should be shown (GDPR regions). */
    fun isPrivacyOptionsRequired(context: Context): Boolean {
        return UserMessagingPlatform.getConsentInformation(context)
            .privacyOptionsRequirementStatus ==
            ConsentInformation.PrivacyOptionsRequirementStatus.REQUIRED
    }
}
