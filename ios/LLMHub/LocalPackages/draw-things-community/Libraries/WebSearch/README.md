# Browser-backed web search

DuckDuckGo and Sogou offer an HTTP path with persistent WebKit fallback. The [Sogou study](#sogou-browser-fallback-and-quality-study--2026-09-08) includes its transport and relevance measurements.

## Brave Search API

`BraveSearch(apiKey:).search(query:options:completion:)` mirrors Kagi's callback-first
interface and calls `HttpTransport.data(for:completion:)` directly. Local Code
offers Brave in Search Provider settings, persists its key separately from Kagi,
and uses it for local and remote agents. The Local Code CLI accepts
`--web-search-provider brave` with `BRAVE_API_KEY`, or a custom variable named by
`--web-search-api-key-environment`.

Options mirror `KagiSearchOptions`: freshness, safe search, result/page limits,
timeout, and user agent. The implementation maps these to Brave's web API,
authenticates with `X-Subscription-Token`, requests undecorated web results, and
deduplicates URLs. Brave supports at most 20 results per page and ten pages; the
page size stays fixed across requests because `offset` counts pages. HTTP errors
are surfaced without retries. See the [Brave API reference](https://api-dashboard.search.brave.com/api-reference/web/search/get).

Run the optional live smoke test with `BRAVE_API_KEY` in the environment:

```sh
bazel test //Libraries/WebSearch:WebSearchTests --test_env=BRAVE_API_KEY \
  --test_filter=WebSearchTests.testBraveLiveSearchWhenAPIKeyIsProvided
```

A separate [native WKWebView feasibility study](Validation/2026-09-16-brave.md)
found that keyless Brave results and pagination work, but rapid sequential searches
can require interactive verification. This study does not add a browser provider
or change the API provider's missing-key behavior.

## DuckDuckGo browser fallback

`DuckDuckGoSearch()` first uses the HTML endpoint and falls back to `WebKitDuckDuckGoSearch` on Apple platforms when it receives a challenge, an unrecognized page, HTTP 403/408/429/5xx, or a transient timeout/connection loss. A successful HTTP search stays on that path. Offline, cancellation and TLS errors do not launch a browser. Pass `browserSearch: nil` to explicitly use HTTP only.

## Why this change

On the requesting Mac, the original HTTP implementation returned results for 4 of 10 benchmark queries. The other six returned DuckDuckGo's HTTP 202 duck-image challenge. The transport deliberately disables cookie storage; changing its user agent does not provide JavaScript execution or browser state.

Loading DuckDuckGo's regular site in WKWebView returned usable responses. The fallback uses the native WebKit user agent and `WKWebsiteDataStore.default()`, which persists cookies and website data. It reads the rendered DOM using semantic attributes rather than obfuscated CSS classes. [Apple documents the persistent data-store behavior](https://developer.apple.com/documentation/webkit/wkwebsitedatastore).

## Flow and limits

1. Attempt the existing HTML request. An empty parse is an error unless the document explicitly says there are no results.
2. On a recoverable failure, restart the original query in WebKit, preserving region, safe search, freshness, page limit and result limit. A failure on a later HTTP page also restarts the complete query in the browser.
3. Serialize browser searches on the main run loop. Each request owns its web view and shares the persistent website data store. Wait for a stable result count, normalize/deduplicate URLs, and use the site's “More results” button for pagination.
4. If the browser displays a challenge, Local Code presents that same live view. Completing verification resumes the search, including a single return to the original query if verification ends on a confirmation URL. Cancel and timeout dismiss the view and complete the request once. A CLI without a presenter returns `searchBlocked`.
5. If the page shell never produces results, allow one ordinary reload. No reload loop runs around a detected challenge. Main-frame navigation is limited to DuckDuckGo HTTPS pages; external result links and `!bang` redirects are not followed.

The HTTP timeout and browser timeout apply to their individual attempts; a browser reload or additional page can take another timeout interval. Human verification has a separate 120-second default timeout. Headless callers must keep the main run loop available to use WebKit, or explicitly select HTTP. Non-WebKit platforms retain HTTP-only behavior.

If an additional browser page produces no new unique results before its timeout, the search returns the results already collected. An explicit no-results page also preserves earlier results. A pending verification challenge still fails on timeout. Restarting the original query after verification resets pagination so later pages can be requested again.

The parser distinguishes genuine no-results pages from incomplete pages, including pages with related-search suggestions before the no-results message. Query matching decodes form-style `+` spaces while preserving literal `+` characters. Challenge wording inside result snippets or scripts does not by itself trigger verification.

## Alternatives considered

- **Cookie-enabled URLSession or the Lite endpoint:** potentially useful, but neither executes the regular site's JavaScript. DuckDuckGo describes HTML and Lite as non-JavaScript alternatives; another lightweight endpoint does not establish a reliable browser session. The working browser path avoids adding another scrape/retry chain. [DuckDuckGo documentation](https://safe.duckduckgo.com/duckduckgo-help-pages/features/non-javascript).
- **Model solving the image challenge:** adds vision-model dependencies and challenge-specific interaction logic. The browser fallback recovered these failures without solving a CAPTCHA. The implemented final step lets a user complete a persistent challenge directly in the provider's UI.
- **A supported search API:** the appropriate option when contractual reliability and sustained automated volume are required. Kagi and Brave are supported by this library and require an API key. Neither is silently selected, since this changes provider/account/billing requirements. [Kagi Search API](https://help.kagi.com/kagi/api/search.html), [Brave Search API](https://brave.com/search/api/).

## Validation on 2026-09-08

Measured responses on this computer:

| Path | Valid responses | Nonempty responses | Median latency | P95 latency |
| --- | ---: | ---: | ---: | ---: |
| HTTP baseline | 4/10 | 4/10 | 0.64 s | 0.72 s |
| Browser | 10/10 | 9/10 | 1.38 s | 1.69 s |
| Automatic, batch 1 | 10/10 | 9/10 | 1.46 s | 1.78 s |
| Automatic, batch 2 | 10/10 | 9/10 | 1.18 s | 1.76 s |
| Automatic, batch 3 | 10/10 | 9/10 | 1.47 s | 1.77 s |
| Automatic, final (document cache bypassed) | 10/10 | 9/10 | 1.46 s | 1.76 s |

The exact query `"Swift ArgumentParser" GitHub` displayed a genuine no-results page in the regular browser; it is counted separately from nonempty results. Latency percentiles exclude errors. Additional browser checks returned 25 results over three pages, and five results each for a literal-plus query, a Chinese query, and a month-filtered strict-safe-search query. Raw benchmark metrics are in [Validation/2026-09-08.json](Validation/2026-09-08.json).

These are repeated local observations, not a guarantee against future provider outages, markup changes, rate limits or challenges. The live tests did not require user verification; challenge resume, cancellation, timeout and serialized searches are covered by deterministic WKWebView test doubles.

The final verification passed 35 tests, skipped the optional live Kagi test without credentials, and built both WebSearchCLI and the affected Local Code iOS library.

Reproduce with:

```sh
bazel test //Libraries/WebSearch:WebSearchTests
bazel build //Apps:WebSearchCLI
bazel-bin/Apps/WebSearchCLI benchmark --transport http --pretty
bazel-bin/Apps/WebSearchCLI benchmark --transport browser --pretty
bazel-bin/Apps/WebSearchCLI benchmark --transport automatic --pretty
bazel-bin/Apps/WebSearchCLI search 'SwiftSoup GitHub' --transport browser --pages 3 --max-results 25
bazel build //Apps/LocalCode:LocalCodeLib --platforms=@build_bazel_apple_support//platforms:ios_arm64
```

The full `//Apps/LocalCode:LocalCode` bundle cannot link in this checkout because the NodeMobile and Python iOS archives are Git LFS pointers rather than binary archives. This is independent of the WebSearch code.

## Sogou browser fallback and quality study — 2026-09-08

`SogouSearch()` now uses the same selective HTTP-to-browser fallback policy through `WebKitSogouSearch`. Pass `browserSearch: nil` for HTTP only. `WebSearchCLI search` and `benchmark` accept `--provider sogou --transport automatic|http|browser`. Local Code supplies the existing verification presenter for Sogou too; the sheet title is now provider-neutral.

Both browser providers share a main-thread request queue, so their verification sessions cannot run concurrently. Each request owns its web view and uses `WKWebsiteDataStore.default()`. Sogou requests desktop content on iOS to match its result parser. Region and safe-search remain DuckDuckGo-only options; Sogou preserves its query, freshness filter, page limit, result limit and timeout.

Sogou uses separate result documents for pagination. The browser checks both query and page number, waits for stable result URLs, appends and deduplicates each page, and stops on duplicate-only or explicit empty pages. A stalled additional page returns already collected results after its timeout. A challenge remains an explicit failure until resolved; returning from verification to the original query resets pagination. The HTTP implementation also stops when a page adds no results.

Two live-provider details matter:

- The current verification page uses `seccodeForm` / `seccodeInput`, rather than the `captcha` wording the original detector expected. Unrecognized pages now fail explicitly; they cannot become successful empty searches. Genuine site-search empty responses use `.vrTips .icon_noRes`. Challenge wording inside result snippets or scripts is ignored.
- Sogou may redirect to an HTTP `/antispider/` URL. The browser loads its HTTPS equivalent and ignores only the cancelled navigation it replaced. Foundation normalizes the trailing slash, and WebKit can report the cancellation as its legacy policy-interruption error rather than `URLError.cancelled`. [Apple identifies that policy-interruption error](https://developer.apple.com/documentation/webkit/webkiterrorframeloadinterruptedbypolicychange?language=objc). Other navigation errors still fail the request. Main-frame navigation remains limited to `sogou.com` and `www.sogou.com`.

### Measurements

The study used the same ten built-in queries as the DuckDuckGo study, sequentially, with ten results and one page requested. Success requires parsed results or an explicit no-results message. Latencies below include successful queries only. No manual verification or result-page fetching was performed.

| Final batch | Valid responses | Nonempty responses | Mean result count | Median latency | P95 latency | Expected-URL MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HTTP | 10/10 | 7/10 | 6.2 | 0.77 s | 1.54 s | 0 |
| Browser | 10/10 | 7/10 | 6.2 | 1.65 s | 2.42 s | 0 |
| Automatic 1 | 10/10 | 7/10 | 6.2 | 0.75 s | 1.41 s | 0 |
| Automatic 2 | 10/10 | 7/10 | 6.2 | 0.82 s | 1.39 s | 0 |
| Automatic 3 | 10/10 | 7/10 | 6.2 | 0.80 s | 1.40 s | 0 |

The two `site:` queries and the `filetype:pdf` query produced explicit no-results messages. Each final batch returned 62 total results, with no empty titles and one empty snippet. None of the eight queries with an expected URL substring found that substring in its returned results. This uses the unchanged benchmark expectations as a relevance proxy, not a comprehensive relevance evaluation.

Provider availability varied during development. The original implementation reported 10/10 successes with zero results for every query, but it did not validate empty pages; an inspected response was a verification page. An earlier corrected HTTP batch returned 3/10 nonempty searches and seven explicit challenges. By the final comparison, HTTP itself returned 10/10 valid responses. These were sequential observations with evolving provider/session state, not a randomized experiment. The final automatic batches therefore do **not** demonstrate an improvement over HTTP, and browser-only was slower. Automatic still preserves the working HTTP path and provides browser recovery when HTTP fails.

Seven supplemental cases were run with each transport: Palace Museum official site, Swift Chinese tutorials, Sogou input method official site, a literal `C++` query, three-page pagination, a month filter, and a quoted unlikely query. All 21 calls returned results. Each pagination call returned 25 distinct URLs; literal-plus and month-filter requests returned ten results. Freshness of those results was not independently verified. No supplemental expected-URL check matched. The unlikely query returned eight unrelated results on all transports, so it must not be treated as a no-results control.

Rendered-DOM inspection of the Palace Museum query found ten `vrwrap` title blocks matching the returned results; the official-site miss in that sample was not caused by omitting another result-container class. Together with the technical corpus and unrelated nonsense-query results, this supports caution about navigational relevance on this machine even when transport success is high.

Raw metrics, supplemental result text and baseline observations are in [Validation/2026-09-08-sogou.json](Validation/2026-09-08-sogou.json). Challenge URL parameters are omitted. Earlier experimental browser runs with navigation-policy defects are excluded from the final measurements.

### Verification and reproduction

The deterministic suite covers option propagation, HTTP-only behavior, transient versus nonrecoverable failures, real challenge structure, explicit empty detection, separate-page deduplication, challenge restart, cancellation, timeout, cross-provider serialization, query identity, HTTPS upgrade, and unrelated policy failures. Live manual challenge completion and cookie reuse after solving one remain unverified; those lifecycle transitions are covered with WebKit test doubles.

```sh
bazel test //Libraries/WebSearch:WebSearchTests //Apps:WebSearchCLI
python3 Libraries/WebSearch/Validation/sogou-study.py --output /tmp/sogou-study.json
bazel build //Apps/LocalCode:LocalCodeLib --platforms=@build_bazel_apple_support//platforms:ios_arm64
```

The requesting Mac passed the WebSearch suite and built both WebSearchCLI and the Local Code iOS library. As with DuckDuckGo, a headless process needs the main run loop for WebKit; a CLI without a verification presenter reports `searchBlocked` when manual action is required.
