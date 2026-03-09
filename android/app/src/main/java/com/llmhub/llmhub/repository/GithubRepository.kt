package com.llmhub.llmhub.repository

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import android.util.Log

object GithubRepository {
    private const val TAG = "GithubRepository"
    private const val REPO_URL = "https://api.github.com/repos/timmyy123/LLM-Hub"
    
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()
    
    suspend fun refreshStars(preferences: com.llmhub.llmhub.data.ThemePreferences) {
        withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url(REPO_URL)
                    .header("Accept", "application/vnd.github.v3+json")
                    .header("User-Agent", "LLM-Hub-Android-App")
                    .build()
                
                val response = client.newCall(request).execute()
                
                if (!response.isSuccessful) {
                    Log.e(TAG, "Failed to fetch GitHub data: ${response.code}")
                    return@withContext
                }
                
                val body = response.body?.string() ?: return@withContext
                val json = JSONObject(body)
                val stars = json.optInt("stargazers_count", -1)
                
                if (stars >= 0) {
                    preferences.setGithubStars(stars)
                    Log.d(TAG, "Updated GitHub stars to: $stars")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error fetching GitHub stars", e)
            }
        }
    }
}
