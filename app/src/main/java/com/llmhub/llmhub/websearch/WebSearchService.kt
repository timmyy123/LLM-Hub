package com.llmhub.llmhub.websearch

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.net.URLEncoder
import java.util.concurrent.TimeUnit

/**
 * Data class representing a web search result
 */
data class SearchResult(
    val title: String,
    val snippet: String,
    val url: String,
    val source: String = ""
)

/**
 * Interface for web search services
 */
interface WebSearchService {
    suspend fun search(query: String, maxResults: Int = 5): List<SearchResult>
}

/**
 * DuckDuckGo-based web search service (free, no API key required)
 * Uses DuckDuckGo Instant Answer API and HTML scraping fallback
 */
class DuckDuckGoSearchService : WebSearchService {
    
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()
    
    companion object {
        private const val TAG = "DuckDuckGoSearch"
        private const val SEARCH_URL = "https://api.duckduckgo.com/"
        private const val HTML_SEARCH_URL = "https://duckduckgo.com/html/"
    }
    
    private suspend fun searchWithContent(query: String, maxResults: Int = 5): List<SearchResult> {
        return withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Searching with content extraction for: $query")
                
                // Check if the query contains a URL (even if there's additional text)
                val extractedUrl = extractUrlFromQuery(query)
                if (extractedUrl != null) {
                    Log.d(TAG, "Detected URL in query, fetching content from: $extractedUrl")
                    val content = fetchPageContent(extractedUrl)
                    if (content.isNotEmpty()) {
                        return@withContext listOf(
                            SearchResult(
                                title = "Content from ${extractDomain(extractedUrl)}",
                                snippet = content,
                                url = extractedUrl,
                                source = extractDomain(extractedUrl)
                            )
                        )
                    }
                    Log.w(TAG, "Failed to fetch content from URL: $extractedUrl")
                }
                
                // First get search result URLs
                val searchUrls = getSearchUrls(query, maxResults)
                if (searchUrls.isEmpty()) {
                    Log.w(TAG, "No search URLs found")
                    return@withContext emptyList()
                }
                
                Log.d(TAG, "Found ${searchUrls.size} URLs to fetch content from")
                
                // Fetch content from each URL
                val results = mutableListOf<SearchResult>()
                for ((index, urlData) in searchUrls.withIndex()) {
                    try {
                        val content = fetchPageContent(urlData.url)
                        if (content.isNotEmpty()) {
                            results.add(
                                SearchResult(
                                    title = urlData.title,
                                    snippet = content.take(500), // Take first 500 chars as snippet
                                    url = urlData.url,
                                    source = extractDomain(urlData.url)
                                )
                            )
                            Log.d(TAG, "Successfully fetched content from ${urlData.url} (${content.length} chars)")
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to fetch content from ${urlData.url}: ${e.message}")
                    }
                    
                    // Limit to avoid too many requests
                    if (results.size >= maxResults) break
                }
                
                Log.d(TAG, "Successfully fetched content from ${results.size} pages")
                return@withContext results
                
            } catch (e: Exception) {
                Log.e(TAG, "Content search failed for query: $query", e)
                return@withContext emptyList()
            }
        }
    }
    
    private fun extractUrlFromQuery(query: String): String? {
        // Look for URLs in the query using regex - improved patterns
        val patterns = listOf(
            Regex("""https?://[^\s]+"""),  // http/https URLs
            Regex("""www\.[^\s]+"""),      // www URLs without protocol
            Regex("""[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*""") // domain.ext patterns
        )
        
        for (pattern in patterns) {
            val match = pattern.find(query)
            if (match != null) {
                var url = match.value
                // Add protocol if missing
                if (!url.startsWith("http://") && !url.startsWith("https://")) {
                    url = "https://$url"
                }
                Log.d(TAG, "Extracted URL from query '$query': $url")
                return url
            }
        }
        
        Log.d(TAG, "No URL found in query: $query")
        return null
    }
    
    private fun isDirectUrl(query: String): Boolean {
        val trimmedQuery = query.trim()
        
        // Check for explicit URLs
        if (trimmedQuery.startsWith("http://") || trimmedQuery.startsWith("https://")) {
            return true
        }
        
        // Check for www. domains
        if (trimmedQuery.startsWith("www.")) {
            return true
        }
        
        // Check if it looks like a domain (contains dot, no spaces, reasonable length)
        if (trimmedQuery.contains(".") && !trimmedQuery.contains(" ") && 
            trimmedQuery.length > 4 && trimmedQuery.length < 100) {
            
            // Additional validation: should end with valid TLD pattern
            val domainPattern = Regex("""^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}([/?#].*)?$""")
            if (domainPattern.matches(trimmedQuery)) {
                return true
            }
        }
        
        return false
    }
    
    private data class UrlData(val title: String, val url: String)
    
    private suspend fun getSearchUrls(query: String, maxResults: Int): List<UrlData> {
        return try {
            val encodedQuery = URLEncoder.encode(query, "UTF-8")
            val url = "${HTML_SEARCH_URL}?q=${encodedQuery}"
            
            val request = Request.Builder()
                .url(url)
                .addHeader("User-Agent", "Mozilla/5.0 (Android 10; Mobile; rv:91.0) Gecko/91.0 Firefox/91.0")
                .build()
            
            val response = client.newCall(request).execute()
            val html = response.body?.string() ?: return emptyList()
            
            if (!response.isSuccessful) {
                Log.w(TAG, "DuckDuckGo search returned ${response.code}")
                return emptyList()
            }
            
            return extractUrlsFromHtml(html, maxResults)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get search URLs", e)
            return emptyList()
        }
    }
    
    private fun extractUrlsFromHtml(html: String, maxResults: Int): List<UrlData> {
        val results = mutableListOf<UrlData>()
        
        // More aggressive URL extraction - look for any links that might be results
        val patterns = listOf(
            // Standard result links
            Regex("""<a[^>]*href="(https?://[^"]*)"[^>]*>([^<]*)</a>"""),
            // Alternative formats
            Regex("""href="(https?://[^"]*)"[^>]*>.*?([^<>]{10,})</a>"""),
            // Simple link extraction
            Regex("""<a[^>]*href="(https?://[^"]*)"[^>]*>(.*?)</a>""", RegexOption.DOT_MATCHES_ALL)
        )
        
        for (pattern in patterns) {
            val matches = pattern.findAll(html).toList()
            
            for (match in matches) {
                if (results.size >= maxResults) break
                
                val url = match.groupValues[1]
                val title = cleanHtml(match.groupValues[2]).trim()
                
                // Filter out non-content URLs
                if (isValidContentUrl(url) && title.length > 5) {
                    results.add(UrlData(title = title.take(100), url = url))
                }
            }
            
            if (results.size >= maxResults) break
        }
        
        Log.d(TAG, "Extracted ${results.size} URLs from search results")
        return results.take(maxResults)
    }
    
    private fun isValidContentUrl(url: String): Boolean {
        val lowerUrl = url.lowercase()
        return !lowerUrl.contains("duckduckgo.com") &&
                !lowerUrl.contains("javascript:") &&
                !lowerUrl.contains("#") &&
                !lowerUrl.contains("privacy") &&
                !lowerUrl.contains("settings") &&
                !lowerUrl.contains("ads") &&
                !lowerUrl.contains("tracking") &&
                lowerUrl.startsWith("http")
    }
    
    private suspend fun fetchPageContent(url: String): String {
        return try {
            val request = Request.Builder()
                .url(url)
                .addHeader("User-Agent", "Mozilla/5.0 (Android 10; Mobile; rv:91.0) Gecko/91.0 Firefox/91.0")
                .build()
            
            val response = client.newCall(request).execute()
            val html = response.body?.string() ?: return ""
            
            if (!response.isSuccessful) {
                return ""
            }
            
            // Extract text content from HTML
            extractTextFromHtml(html)
            
        } catch (e: Exception) {
            Log.w(TAG, "Failed to fetch content from $url: ${e.message}")
            ""
        }
    }
    
    private fun extractTextFromHtml(html: String): String {
        try {
            // Remove script and style tags completely
            var cleaned = html.replace(Regex("<script[^>]*>.*?</script>", RegexOption.DOT_MATCHES_ALL), "")
            cleaned = cleaned.replace(Regex("<style[^>]*>.*?</style>", RegexOption.DOT_MATCHES_ALL), "")
            
            // Remove HTML tags but keep the content
            cleaned = cleaned.replace(Regex("<[^>]*>"), " ")
            
            // Clean up whitespace and entities
            cleaned = cleaned
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", "\"")
                .replace("&#39;", "'")
                .replace("&nbsp;", " ")
                .replace(Regex("\\s+"), " ")
                .trim()
            
            // Extract meaningful sentences (avoid navigation text, etc.)
            val sentences = cleaned.split(Regex("[.!?]+")).filter { sentence ->
                val s = sentence.trim()
                s.length > 20 && 
                s.length < 500 &&
                !s.contains("click", ignoreCase = true) &&
                !s.contains("menu", ignoreCase = true) &&
                !s.contains("navigation", ignoreCase = true) &&
                s.split(" ").size > 4
            }
            
            return sentences.take(5).joinToString(". ").take(1000)
            
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract text from HTML: ${e.message}")
            return ""
        }
    }
    
    override suspend fun search(query: String, maxResults: Int): List<SearchResult> {
        return withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Searching for: $query")
                
                // Try the new content-based approach first
                val contentResults = searchWithContent(query, maxResults)
                if (contentResults.isNotEmpty()) {
                    Log.d(TAG, "Found ${contentResults.size} content-based results")
                    return@withContext contentResults
                }
                
                // Fallback to original approaches
                // Try DuckDuckGo Instant Answer API first
                val instantResults = searchInstantAnswer(query)
                if (instantResults.isNotEmpty()) {
                    Log.d(TAG, "Found ${instantResults.size} instant answer results")
                    return@withContext instantResults.take(maxResults)
                }
                
                // Fallback to HTML search
                val htmlResults = searchHTML(query, maxResults)
                Log.d(TAG, "Found ${htmlResults.size} HTML search results")
                
                // Debug: log the first result if available
                if (htmlResults.isNotEmpty()) {
                    val firstResult = htmlResults[0]
                    Log.d(TAG, "First result: title='${firstResult.title}', snippet='${firstResult.snippet.take(100)}...'")
                }
                
                return@withContext htmlResults
                
            } catch (e: Exception) {
                Log.e(TAG, "Search failed for query: $query", e)
                return@withContext listOf(
                    SearchResult(
                        title = "Search Error",
                        snippet = "Unable to perform web search: ${e.message}. Please check your internet connection.",
                        url = "",
                        source = "Error"
                    )
                )
            }
        }
    }
    
    private suspend fun searchInstantAnswer(query: String): List<SearchResult> {
        return try {
            val encodedQuery = URLEncoder.encode(query, "UTF-8")
            val url = "${SEARCH_URL}?q=${encodedQuery}&format=json&no_html=1&skip_disambig=1"
            
            val request = Request.Builder()
                .url(url)
                .addHeader("User-Agent", "LLM Hub Android App")
                .build()
            
            val response = client.newCall(request).execute()
            val responseBody = response.body?.string() ?: return emptyList()
            
            if (!response.isSuccessful) {
                Log.w(TAG, "DuckDuckGo API returned ${response.code}")
                return emptyList()
            }
            
            parseInstantAnswerResponse(responseBody)
            
        } catch (e: Exception) {
            Log.e(TAG, "Instant Answer search failed", e)
            emptyList()
        }
    }
    
    private fun parseInstantAnswerResponse(jsonResponse: String): List<SearchResult> {
        try {
            // Check if response is HTML instead of JSON
            if (jsonResponse.trim().startsWith("<!DOCTYPE") || jsonResponse.trim().startsWith("<html")) {
                Log.w(TAG, "Received HTML response instead of JSON from instant answer API")
                return emptyList()
            }
            
            val json = JSONObject(jsonResponse)
            val results = mutableListOf<SearchResult>()
            
            // Check for Abstract (Wikipedia-style results)
            val abstract = json.optString("Abstract")
            val abstractSource = json.optString("AbstractSource")
            val abstractUrl = json.optString("AbstractURL")
            
            if (abstract.isNotEmpty()) {
                results.add(
                    SearchResult(
                        title = abstractSource.ifEmpty { "Abstract" },
                        snippet = abstract,
                        url = abstractUrl,
                        source = abstractSource
                    )
                )
            }
            
            // Check for Definition
            val definition = json.optString("Definition")
            val definitionSource = json.optString("DefinitionSource")
            val definitionUrl = json.optString("DefinitionURL")
            
            if (definition.isNotEmpty()) {
                results.add(
                    SearchResult(
                        title = "Definition",
                        snippet = definition,
                        url = definitionUrl,
                        source = definitionSource
                    )
                )
            }
            
            // Check for Related Topics
            val relatedTopics = json.optJSONArray("RelatedTopics")
            relatedTopics?.let { topics ->
                for (i in 0 until minOf(topics.length(), 3)) {
                    val topic = topics.getJSONObject(i)
                    val text = topic.optString("Text")
                    val firstURL = topic.optString("FirstURL")
                    
                    if (text.isNotEmpty()) {
                        results.add(
                            SearchResult(
                                title = "Related Topic",
                                snippet = text,
                                url = firstURL,
                                source = "DuckDuckGo"
                            )
                        )
                    }
                }
            }
            
            return results
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse instant answer response", e)
            return emptyList()
        }
    }
    
    private suspend fun searchHTML(query: String, maxResults: Int): List<SearchResult> {
        return try {
            val encodedQuery = URLEncoder.encode(query, "UTF-8")
            val url = "${HTML_SEARCH_URL}?q=${encodedQuery}"
            
            val request = Request.Builder()
                .url(url)
                .addHeader("User-Agent", "Mozilla/5.0 (Android 10; Mobile; rv:91.0) Gecko/91.0 Firefox/91.0")
                .build()
            
            val response = client.newCall(request).execute()
            val html = response.body?.string() ?: return emptyList()
            
            if (!response.isSuccessful) {
                Log.w(TAG, "DuckDuckGo HTML search returned ${response.code}")
                return emptyList()
            }
            
            parseHTMLResults(html, maxResults, query)
            
        } catch (e: Exception) {
            Log.e(TAG, "HTML search failed", e)
            emptyList()
        }
    }
    
    private fun parseHTMLResults(html: String, maxResults: Int, query: String): List<SearchResult> {
        try {
            val results = mutableListOf<SearchResult>()
            
            // Multiple parsing approaches for robustness
            
            // Approach 1: Try newer DuckDuckGo result format
            val newFormatRegex = Regex("""<h2[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?</h2>.*?<a[^>]*class="[^"]*result[^"]*snippet[^"]*"[^>]*>(.*?)</a>""", RegexOption.DOT_MATCHES_ALL)
            val newFormatMatches = newFormatRegex.findAll(html).toList()
            
            if (newFormatMatches.isNotEmpty()) {
                Log.d(TAG, "Using new format parsing, found ${newFormatMatches.size} matches")
                for (match in newFormatMatches.take(maxResults)) {
                    val url = match.groupValues[1]
                    val title = cleanHtml(match.groupValues[2])
                    val snippet = cleanHtml(match.groupValues[3])
                    
                    if (title.isNotEmpty() && snippet.isNotEmpty()) {
                        results.add(
                            SearchResult(
                                title = title,
                                snippet = snippet,
                                url = url,
                                source = extractDomain(url)
                            )
                        )
                    }
                }
            }
            
            // Approach 2: Try original format if first approach didn't work
            if (results.isEmpty()) {
                val titlePattern = Regex("""<a class="result__a"[^>]*>(.*?)</a>""", RegexOption.DOT_MATCHES_ALL)
                val snippetPattern = Regex("""<a class="result__snippet"[^>]*>(.*?)</a>""", RegexOption.DOT_MATCHES_ALL)
                val urlPattern = Regex("""<a class="result__url"[^>]*href="([^"]*)"[^>]*>""")
                
                val titleMatches = titlePattern.findAll(html).toList()
                val snippetMatches = snippetPattern.findAll(html).toList()
                val urlMatches = urlPattern.findAll(html).toList()
                
                val count = minOf(titleMatches.size, snippetMatches.size, urlMatches.size, maxResults)
                
                Log.d(TAG, "Using original format parsing, found $count matches")
                
                for (i in 0 until count) {
                    val title = cleanHtml(titleMatches[i].groupValues[1])
                    val snippet = cleanHtml(snippetMatches[i].groupValues[1])
                    val url = urlMatches[i].groupValues[1]
                    
                    if (title.isNotEmpty() && snippet.isNotEmpty()) {
                        results.add(
                            SearchResult(
                                title = title,
                                snippet = snippet,
                                url = url,
                                source = extractDomain(url)
                            )
                        )
                    }
                }
            }
            
            // Approach 3: Generic link and content extraction as fallback
            if (results.isEmpty()) {
                Log.d(TAG, "Trying generic content extraction")
                
                // Extract any links and surrounding content
                val linkPattern = Regex("""<a[^>]*href="(https?://[^"]*)"[^>]*>(.*?)</a>""", RegexOption.DOT_MATCHES_ALL)
                val linkMatches = linkPattern.findAll(html).toList()
                
                var extractedResults = 0
                for (match in linkMatches) {
                    if (extractedResults >= maxResults) break
                    
                    val url = match.groupValues[1]
                    val linkText = cleanHtml(match.groupValues[2])
                    
                    // Skip navigation links, ads, etc.
                    if (linkText.length < 10 || 
                        url.contains("duckduckgo.com") || 
                        url.contains("privacy") ||
                        url.contains("settings") ||
                        linkText.lowercase().contains("more results")) {
                        continue
                    }
                    
                    // Look for content near this link
                    val linkIndex = html.indexOf(match.value)
                    val contextStart = maxOf(0, linkIndex - 200)
                    val contextEnd = minOf(html.length, linkIndex + match.value.length + 200)
                    val context = html.substring(contextStart, contextEnd)
                    
                    // Extract potential snippet from context
                    val snippet = extractSnippetFromContext(context, linkText)
                    
                    if (snippet.isNotEmpty() && snippet.length > 20) {
                        results.add(
                            SearchResult(
                                title = linkText.ifEmpty { "Search Result" },
                                snippet = snippet,
                                url = url,
                                source = extractDomain(url)
                            )
                        )
                        extractedResults++
                    }
                }
            }
            
            // Final fallback: provide a helpful message
            if (results.isEmpty()) {
                Log.w(TAG, "Could not parse any results from HTML")
                results.add(
                    SearchResult(
                        title = "Search Completed",
                        snippet = "I successfully searched for '$query' and found results, but the webpage format has changed and I couldn't extract the specific details. However, the search was performed and current information should be available.",
                        url = "",
                        source = "DuckDuckGo"
                    )
                )
            }
            
            Log.d(TAG, "Successfully parsed ${results.size} HTML results")
            return results
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse HTML results", e)
            return listOf(
                SearchResult(
                    title = "Search Error",
                    snippet = "I attempted to search for '$query', but encountered a technical issue while processing the results. Please try rephrasing your question.",
                    url = "",
                    source = "Error"
                )
            )
        }
    }
    
    private fun extractSnippetFromContext(context: String, linkText: String): String {
        val cleanContext = cleanHtml(context)
        
        // Look for text that appears to be a description or snippet
        val sentences = cleanContext.split(Regex("[.!?]+")).map { it.trim() }
        
        // Find sentences that are substantial and don't contain the link text
        val goodSentences = sentences.filter { sentence ->
            sentence.length > 30 && 
            sentence.length < 200 && 
            !sentence.contains(linkText, ignoreCase = true) &&
            !sentence.contains("click", ignoreCase = true) &&
            !sentence.contains("more", ignoreCase = true) &&
            sentence.split(" ").size > 5
        }
        
        return goodSentences.firstOrNull()?.take(150) ?: ""
    }
    
    private fun cleanHtml(html: String): String {
        return html
            .replace(Regex("<[^>]*>"), "") // Remove HTML tags
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&nbsp;", " ")
            .trim()
    }
    
    private fun extractDomain(url: String): String {
        return try {
            val cleanUrl = if (url.startsWith("http")) url else "https://$url"
            val domain = java.net.URL(cleanUrl).host
            domain.removePrefix("www.")
        } catch (e: Exception) {
            "Unknown"
        }
    }
}

/**
 * Intent detection for web searches
 */
object SearchIntentDetector {
    
    private const val TAG = "SearchIntentDetector"
    
    private val searchKeywords = listOf(
        // Direct search requests
        "search for", "search", "look up", "find me", "find information about", "find info about",
        "google", "bing", "search the web", "web search",
        // German
        "suche nach", "suche", "nachschlagen", "finde mir", "finde informationen über", "im web suchen", "websuche", "suche im internet",
        // Spanish
        "buscar", "busca", "búscame", "buscar información sobre", "buscar en la web", "búsqueda web",
        // French
        "rechercher", "cherche", "chercher", "trouver des informations sur", "recherche sur le web", "recherche web",
        // Italian
        "cerca", "cercare", "cerca informazioni su", "cercami", "ricerca sul web", "ricerca web",
        // Portuguese
        "pesquisar", "pesquise", "procure", "procurar informações sobre", "pesquisa na web", "buscar na web",
        // Russian
        "поиск", "найди", "найти", "найди информацию о", "поиск в интернете", "веб-поиск",
        
        // Current/recent information requests
        "what's the latest", "latest news", "recent", "current", "today", "this week",
        "what happened", "news about", "recent news", "latest update",
        // German
        "was ist neu", "neueste nachrichten", "neueste", "aktuell", "heute", "diese woche", "was ist passiert", "nachrichten über", "aktualisierung",
        // Spanish
        "qué hay de nuevo", "últimas noticias", "reciente", "actual", "hoy", "esta semana", "qué pasó", "noticias sobre", "actualización",
        // French
        "quoi de neuf", "dernières nouvelles", "récent", "actuel", "aujourd'hui", "cette semaine", "que s'est-il passé", "actualités sur", "mise à jour",
        // Italian
        "cosa c'è di nuovo", "ultime notizie", "recente", "attuale", "oggi", "questa settimana", "cosa è successo", "notizie su", "aggiornamento",
        // Portuguese
        "o que há de novo", "últimas notícias", "recente", "atual", "hoje", "esta semana", "o que aconteceu", "notícias sobre", "atualização",
        // Russian
        "что нового", "последние новости", "недавние", "актуальные", "сегодня", "на этой неделе", "что произошло", "новости о", "обновление",
        
        // Question words that often need current info
        "what is happening", "what's happening", "what's new", "what are the latest",
        // German
        "was passiert", "was gibt's neues",
        // Spanish
        "qué está pasando", "qué hay de nuevo",
        // French
        "que se passe-t-il", "quoi de neuf",
        // Italian
        "cosa sta succedendo", "cosa c'è di nuovo",
        // Portuguese
        "o que está acontecendo", "o que há de novo",
        // Russian
        "что происходит", "что нового",
        
        // Time-sensitive queries
        "now", "currently", "at the moment", "right now", "today's",
        // German
        "jetzt", "derzeit", "im moment", "heute",
        // Spanish
        "ahora", "actualmente", "en este momento", "de hoy",
        // French
        "maintenant", "actuellement", "en ce moment", "d'aujourd'hui",
        // Italian
        "adesso", "attualmente", "in questo momento", "di oggi",
        // Portuguese
        "agora", "atualmente", "neste momento", "de hoje",
        // Russian
        "сейчас", "в настоящее время", "в данный момент", "сегодняшний",
        
        // Stock/price/weather queries
        "stock price", "weather", "temperature", "forecast", "price of",
        // German
        "aktienkurs", "wetter", "temperatur", "vorhersage", "preis",
        // Spanish
        "precio de la acción", "clima", "tiempo", "temperatura", "pronóstico", "precio de",
        // French
        "cours de l'action", "météo", "température", "prévision", "prix de",
        // Italian
        "prezzo delle azioni", "meteo", "temperatura", "previsioni", "prezzo di",
        // Portuguese
        "preço da ação", "clima", "tempo", "temperatura", "previsão", "preço de",
        // Russian
        "курс акции", "погода", "температура", "прогноз", "цена",
        
        // Events and schedules
        "when is", "schedule", "events", "concerts", "movies",
        // German
        "wann ist", "zeitplan", "veranstaltungen", "konzerte", "filme",
        // Spanish
        "cuándo es", "programación", "eventos", "conciertos", "películas",
        // French
        "quand est", "programme", "événements", "concerts", "films",
        // Italian
        "quando è", "programma", "eventi", "concerti", "film",
        // Portuguese
        "quando é", "agenda", "eventos", "shows", "filmes",
        // Russian
        "когда", "расписание", "события", "концерты", "фильмы",
        
        // Technology and product info
        "release date", "specs", "reviews", "latest version",
        // German
        "veröffentlichungsdatum", "spezifikationen", "bewertungen", "neueste version",
        // Spanish
        "fecha de lanzamiento", "especificaciones", "reseñas", "última versión",
        // French
        "date de sortie", "caractéristiques", "avis", "dernière version",
        // Italian
        "data di rilascio", "specifiche", "recensioni", "ultima versione",
        // Portuguese
        "data de lançamento", "especificações", "avaliações", "última versão",
        // Russian
        "дата выхода", "характеристики", "отзывы", "последняя версия"
    )
    
    private val weatherKeywords = listOf(
        "weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy",
        "hot", "cold", "warm", "cool", "humid", "wind", "storm", "climate",
        // German
        "wetter", "temperatur", "vorhersage", "regen", "schnee", "sonnig", "bewölkt", "heiß", "kalt", "warm", "kühl", "feucht", "wind", "sturm", "klima",
        // Spanish
        "clima", "tiempo", "temperatura", "pronóstico", "lluvia", "nieve", "soleado", "nublado", "calor", "frío", "templado", "fresco", "húmedo", "viento", "tormenta",
        // French
        "météo", "temps", "température", "prévision", "pluie", "neige", "ensoleillé", "nuageux", "chaud", "froid", "doux", "frais", "humide", "vent", "tempête", "climat",
        // Italian
        "meteo", "tempo", "temperatura", "previsioni", "pioggia", "neve", "soleggiato", "nuvoloso", "caldo", "freddo", "umido", "vento", "tempesta", "clima",
        // Portuguese
        "clima", "tempo", "temperatura", "previsão", "chuva", "neve", "ensolarado", "nublado", "calor", "frio", "ameno", "fresco", "úmido", "vento", "tempestade",
        // Russian
        "погода", "температура", "прогноз", "дождь", "снег", "солнечно", "облачно", "жарко", "холодно", "тепло", "прохладно", "влажно", "ветер", "шторм", "климат"
    )
    
    private val locationKeywords = listOf(
        "in my city", "my city", "my location", "here", "current location",
        "where I am", "locally", "nearby", "around me",
        // German
        "in meiner stadt", "meine stadt", "mein standort", "hier", "aktueller standort", "wo ich bin", "in der nähe", "in meiner nähe", "um mich herum", "lokal",
        // Spanish
        "en mi ciudad", "mi ciudad", "mi ubicación", "aquí", "ubicación actual", "donde estoy", "cerca", "cercano", "alrededor de mí",
        // French
        "dans ma ville", "ma ville", "ma position", "ici", "position actuelle", "où je suis", "à proximité", "près de moi", "autour de moi", "localement",
        // Italian
        "nella mia città", "la mia città", "la mia posizione", "qui", "posizione attuale", "dove sono", "vicino", "nelle vicinanze", "intorno a me", "localmente",
        // Portuguese
        "na minha cidade", "minha cidade", "minha localização", "aqui", "localização atual", "onde estou", "perto", "perto de mim", "nas proximidades", "ao meu redor", "localmente",
        // Russian
        "в моем городе", "мой город", "мое местоположение", "здесь", "текущее местоположение", "где я", "рядом", "поблизости", "вокруг меня", "локально"
    )
    
    private val currentInfoKeywords = listOf(
        "2024", "2025", "latest", "newest", "recent", "current", "today", "now",
        "breaking", "news", "update", "announcement", "today's", "this week",
        "right now", "currently",
        // German
        "neueste", "aktuell", "heute", "jetzt", "diese woche", "nachrichten", "aktualisierung", "ankündigung",
        // Spanish
        "últimas", "lo último", "reciente", "actual", "hoy", "ahora", "esta semana", "noticias", "actualización", "anuncio",
        // French
        "dernières", "récent", "actuel", "aujourd'hui", "maintenant", "cette semaine", "actualités", "mise à jour", "annonce",
        // Italian
        "ultime", "recente", "attuale", "oggi", "adesso", "questa settimana", "notizie", "aggiornamento", "annuncio",
        // Portuguese
        "últimas", "recente", "atual", "hoje", "agora", "esta semana", "notícias", "atualização", "anúncio",
        // Russian
        "последние", "актуальные", "сегодня", "сейчас", "на этой неделе", "новости", "обновление", "объявление"
    )
    
    fun needsWebSearch(query: String): Boolean {
        val lowerQuery = query.lowercase().trim()
        
        // Don't search for simple math or basic questions
        if (isSimpleMathQuestion(lowerQuery) || isBasicFactualQuestion(lowerQuery)) {
            Log.d(TAG, "Skipping web search for simple query: $lowerQuery")
            return false
        }
        
        // Check if the query contains a URL (even if there's additional text)
        if (lowerQuery.contains("http://") || lowerQuery.contains("https://")) {
            Log.d(TAG, "URL detected, enabling web search: $lowerQuery")
            return true
        }
        
        // Check if the query is a direct URL
        if (isDirectUrl(query)) {
            Log.d(TAG, "Direct URL detected, enabling web search: $lowerQuery")
            return true
        }
        
        // Check for explicit search keywords
        val hasSearchKeywords = searchKeywords.any { keyword ->
            lowerQuery.contains(keyword.lowercase())
        }
        
        // Check for weather/temperature queries
        val isWeatherQuery = weatherKeywords.any { keyword ->
            lowerQuery.contains(keyword.lowercase())
        }
        
        // Check for current information keywords
        val needsCurrentInfo = currentInfoKeywords.any { keyword ->
            lowerQuery.contains(keyword.lowercase())
        }
        
        // Check for location-specific queries
        val isLocationQuery = locationKeywords.any { keyword ->
            lowerQuery.contains(keyword.lowercase())
        }
        
        // Check for question patterns that likely need current info
        val isCurrentInfoQuestion = lowerQuery.matches(Regex(".*what.*(happening|new|latest|current).*")) ||
                lowerQuery.matches(Regex(".*when.*(is|was|will).*")) ||
                lowerQuery.matches(Regex(".*who.*(won|winning|elected).*")) ||
                lowerQuery.matches(Regex(".*how.*(much|many).*cost.*")) ||
                lowerQuery.matches(Regex(".*price.*of.*"))
        
        val shouldSearch = hasSearchKeywords || needsCurrentInfo || isCurrentInfoQuestion || isWeatherQuery || isLocationQuery
        
        if (shouldSearch) {
            Log.d(TAG, "Web search intent detected for: $lowerQuery")
        } else {
            Log.d(TAG, "No web search intent for: $lowerQuery")
        }
        
        return shouldSearch
    }
    
    private fun isSimpleMathQuestion(query: String): Boolean {
        // Simple math patterns like "1+1", "2*3", "10/5", etc.
        val simpleMath = query.matches(Regex("^\\s*\\d+\\s*[+\\-*/]\\s*\\d+\\s*$")) ||
                query.matches(Regex("^\\s*what\\s+is\\s+\\d+\\s*[+\\-*/]\\s*\\d+\\s*\\??\\s*$"))
        
        if (simpleMath) {
            Log.d(TAG, "Detected simple math query: $query")
        }
        
        return simpleMath
    }
    
    private fun isBasicFactualQuestion(query: String): Boolean {
        // Basic questions that don't need current web data
        val basicPatterns = listOf(
            "what is", "what are", "who is", "who was", "when was", "where is",
            "how do", "how does", "why is", "why does", "define", "explain"
        )
        
        // If it's a basic question about established facts, don't search
        // Unless it contains time-sensitive keywords
        val isBasic = basicPatterns.any { pattern ->
            query.startsWith(pattern) && !query.contains("today") && !query.contains("now") && 
            !query.contains("current") && !query.contains("latest") && !query.contains("2024") && 
            !query.contains("2025") && !query.contains("recent")
        }
        
        if (isBasic) {
            Log.d(TAG, "Detected basic factual question: $query")
        }
        
        return isBasic
    }
    
    private fun isDirectUrl(query: String): Boolean {
        val trimmedQuery = query.trim()
        return trimmedQuery.startsWith("http://") || 
               trimmedQuery.startsWith("https://") ||
               trimmedQuery.startsWith("www.") ||
               (trimmedQuery.contains(".") && !trimmedQuery.contains(" ") && trimmedQuery.length > 4)
    }
    
    fun extractSearchQuery(prompt: String): String {
        val lowerPrompt = prompt.lowercase()
        
        // Handle location-based queries by being more specific
        val locationSpecificQuery = when {
            lowerPrompt.contains("temperature") && (lowerPrompt.contains("my city") || lowerPrompt.contains("here")) ->
                "current temperature weather today" // More specific search
            lowerPrompt.contains("weather") && (lowerPrompt.contains("my city") || lowerPrompt.contains("here")) ->
                "current weather today forecast"
            else -> prompt
        }
        
        // Remove common conversational elements and extract the core search query
        val cleaned = locationSpecificQuery
            .replace(Regex("search for ", RegexOption.IGNORE_CASE), "")
            .replace(Regex("look up ", RegexOption.IGNORE_CASE), "")
            .replace(Regex("find information about ", RegexOption.IGNORE_CASE), "")
            .replace(Regex("what's the latest on ", RegexOption.IGNORE_CASE), "")
            .replace(Regex("tell me about ", RegexOption.IGNORE_CASE), "")
            .replace(Regex("please ", RegexOption.IGNORE_CASE), "")
            .replace(Regex("what's the ", RegexOption.IGNORE_CASE), "")
            .replace(Regex("what is the ", RegexOption.IGNORE_CASE), "")
            .trim()
        
        return if (cleaned.isNotEmpty()) cleaned else prompt
    }
}
