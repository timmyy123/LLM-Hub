package com.llmhub.llmhub.utils

import android.util.Log
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager
import java.security.cert.X509Certificate

/**
 * Utility to configure tolerant SSL trust managers across Android HTTP & HTTPS connections.
 * Resolves 'java.security.cert.CertPathValidatorException: Trust anchor for certification path not found'
 * on devices missing updated Root CA trust anchors (e.g. Let's Encrypt / Hugging Face CDN certs).
 */
object SslUtils {
    private const val TAG = "SslUtils"

    val trustAllTrustManager: Array<TrustManager> = arrayOf(
        object : X509TrustManager {
            override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) {}
            override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {}
            override fun getAcceptedIssuers(): Array<X509Certificate> = arrayOf()
        }
    )

    val trustAllSslContext: SSLContext? by lazy {
        try {
            SSLContext.getInstance("TLS").apply {
                init(null, trustAllTrustManager, java.security.SecureRandom())
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create trust-all SSLContext: ${e.message}", e)
            null
        }
    }

    fun configureHttpsConnection(connection: HttpsURLConnection) {
        try {
            trustAllSslContext?.let {
                connection.sslSocketFactory = it.socketFactory
            }
            connection.hostnameVerifier = HostnameVerifier { _, _ -> true }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure HttpsURLConnection SSL: ${e.message}", e)
        }
    }

    fun configureOkHttpClient(builder: okhttp3.OkHttpClient.Builder): okhttp3.OkHttpClient.Builder {
        try {
            trustAllSslContext?.let {
                builder.sslSocketFactory(it.socketFactory, trustAllTrustManager[0] as X509TrustManager)
            }
            builder.hostnameVerifier { _, _ -> true }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure OkHttpClient SSL: ${e.message}", e)
        }
        return builder
    }

    fun enableGlobalTolerantSsl() {
        try {
            trustAllSslContext?.let {
                HttpsURLConnection.setDefaultSSLSocketFactory(it.socketFactory)
                HttpsURLConnection.setDefaultHostnameVerifier { _, _ -> true }
                Log.i(TAG, "Global tolerant SSL configured for HttpsURLConnection")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set default SSLSocketFactory: ${e.message}", e)
        }
    }
}
