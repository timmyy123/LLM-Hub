import SwiftUI

// MARK: - Thinking Token Parsing

/// Sentinel constants — must match values emitted by the inference backend
/// (OnnxInferenceService / NexaInferenceService on Android; RunAnywhere on iOS).
private let kSentinelThink    = "\u{200B}\u{200B}THINK\u{200B}\u{200B}"
private let kSentinelEndThink = "\u{200B}\u{200B}ENDTHINK\u{200B}\u{200B}"
private let kRawOpenThink     = "<think>"
private let kRawCloseThink    = "</think>"

// Strip Harmony "analysis<|message|>" prefix that leaks into thinking text.
// <|channel|> is a non-rendering special token so the stream shows the short form.
private func stripHarmonyAnalysisPrefix(_ text: String) -> String {
    let shortPrefix = "analysis<|message|>"
    if text.hasPrefix(shortPrefix) { return String(text.dropFirst(shortPrefix.count)) }
    let longPrefix = "<|channel|>analysis<|message|>"
    if text.hasPrefix(longPrefix) { return String(text.dropFirst(longPrefix.count)) }
    return text
}

/// Returns `true` when `content` contains any recognised thinking marker
/// (even if the thinking text after the tag is still empty during streaming).
func contentHasThinkingMarkers(_ content: String) -> Bool {
    content.contains(kRawOpenThink) || content.contains(kSentinelThink)
        || content.contains(kRawCloseThink) || content.contains(kSentinelEndThink)
}

/// Split `content` into `(thinkingPart, answerPart)`.
/// Returns `("", content)` when no thinking markers are present.
func parseThinkingAndAnswer(_ content: String) -> (thinking: String, answer: String) {
    // 1) Sentinel-wrapped thinking (RunAnywhere / Nexa backend)
    if content.contains(kSentinelThink) {
        let afterThink = content.substringAfterFirst(kSentinelThink)
        if afterThink.contains(kSentinelEndThink) {
            let thinking = stripHarmonyAnalysisPrefix(afterThink.substringBeforeFirst(kSentinelEndThink))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let answer   = afterThink.substringAfterFirst(kSentinelEndThink)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (thinking, answer)
        }
        // Sentinel open but no close yet — still streaming thinking
        return (stripHarmonyAnalysisPrefix(afterThink).trimmingCharacters(in: .whitespacesAndNewlines), "")
    }
    // 2) Closing sentinel only: everything before the close marker is thinking,
    // everything after it is the visible answer.
    if content.contains(kSentinelEndThink) {
        let thinking = stripHarmonyAnalysisPrefix(content.substringBeforeFirst(kSentinelEndThink))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let answer = content.substringAfterFirst(kSentinelEndThink)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (thinking, answer)
    }
    // 3) Raw <think>…</think>
    if content.contains(kRawOpenThink) {
        let afterOpen = content.substringAfterFirst(kRawOpenThink)
        if afterOpen.contains(kRawCloseThink) {
            let thinking = afterOpen.substringBeforeFirst(kRawCloseThink)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let answer   = afterOpen.substringAfterFirst(kRawCloseThink)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (thinking, answer)
        }
        // <think> open but no </think> yet — still streaming
        return (afterOpen.trimmingCharacters(in: .whitespacesAndNewlines), "")
    }
    // 4) Only a closing tag (model emitted </think> without explicit <think>)
    if content.contains(kRawCloseThink) {
        let thinking = content.substringBeforeFirst(kRawCloseThink)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let answer = content.substringAfterFirst(kRawCloseThink)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (thinking, answer)
    }
    return ("", content)
}

/// Returns only the answer portion of the output, stripping any `<think>` block.
/// Returns an empty string while the model is still emitting thinking and no answer has arrived yet.
func getDisplayContentWithoutThinking(_ content: String) -> String {
    let result = parseThinkingAndAnswer(content)
    // If no thinking markers at all, the parser puts full content in .answer — return it.
    // If thinking markers present but no answer yet, return "".
    if contentHasThinkingMarkers(content) && result.answer.isEmpty {
        return ""
    }
    return result.answer
}

func supportsUnmarkedStreamingThinkingHeuristic(forModelNamed modelName: String?) -> Bool {
    guard let normalizedName = modelName?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
          !normalizedName.isEmpty else {
        return false
    }

    // Android gets GPT-OSS thinking via Harmony-aware backend formatting/parsing.
    // The current iOS RunAnywhere path in this app does not surface those boundaries,
    // so treating the raw stream as temporary reasoning creates a fake drawer that
    // later disappears. Disable the heuristic for GPT-OSS until real boundaries exist.
    if normalizedName.contains("gpt-oss") || normalizedName.contains("gpt_oss") {
        return false
    }

    // Apple Foundation Model provides thinking content via SDK result.thinkingContent;
    // the backend wraps it in sentinels directly. The fallback heuristic would incorrectly
    // show the answer words as thinking before the sentinels arrive.
    if normalizedName.contains("apple foundation model") {
        return false
    }

    return true
}

// MARK: - String helpers (file-private)

private extension String {
    func substringAfterFirst(_ needle: String) -> String {
        guard let range = self.range(of: needle) else { return self }
        return String(self[range.upperBound...])
    }
    func substringBeforeFirst(_ needle: String) -> String {
        guard let range = self.range(of: needle) else { return self }
        return String(self[..<range.lowerBound])
    }
}

// MARK: - ThinkingDrawerView

/// Expandable "Thinking" panel — mirrors Android's collapsible thinking drawer.
/// Shows while the model is streaming its reasoning; auto-collapses when the answer arrives.
struct ThinkingDrawerView: View {
    @EnvironmentObject var settings: AppSettings

    let thinking: String
    /// `true` while the model is still generating (streaming).
    let isGenerating: Bool
    /// `true` once an answer has been produced (thinking phase finished).
    let hasAnswer: Bool

    @State private var isExpanded: Bool = true

    private var showsBody: Bool {
        isExpanded && !thinking.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.spring(response: 0.28, dampingFraction: 0.9)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .frame(width: 16)
                        .rotationEffect(.degrees(isExpanded ? 0 : 0))

                    if !hasAnswer && isGenerating {
                        Text(settings.localized("ai_thinking"))
                            .font(.caption.weight(.semibold))
                    } else {
                        let tokenCount = max(1, thinking.count / 4)
                        Text(String(format: settings.localized("thinking_tokens"), tokenCount))
                            .font(.caption.weight(.semibold))
                    }

                    Spacer()
                }
                .foregroundStyle(.white.opacity(0.72))
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            VStack(spacing: 0) {
                if showsBody {
                    Rectangle()
                        .fill(Color.white.opacity(0.08))
                        .frame(height: 1)
                        .padding(.horizontal, 10)

                    ScrollViewReader { proxy in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 0) {
                                Text(thinking)
                                    .font(.caption)
                                    .foregroundStyle(.white.opacity(0.55))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 10)

                                Color.clear
                                    .frame(height: 1)
                                    .id("thinking-bottom")
                            }
                        }
                        .frame(maxHeight: 200)
                        .scrollIndicators(.hidden)
                        .transition(.opacity)
                        .task(id: thinking.count) {
                            guard isGenerating, isExpanded, !thinking.isEmpty else { return }
                            proxy.scrollTo("thinking-bottom", anchor: .bottom)
                        }
                    }
                }
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(.ultraThinMaterial)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .animation(.spring(response: 0.28, dampingFraction: 0.9), value: showsBody)
        .onChange(of: hasAnswer) { _, newValue in
            if newValue {
                // Answer arrived — collapse the drawer
                withAnimation(.spring(response: 0.32, dampingFraction: 0.92)) {
                    isExpanded = false
                }
            }
        }
    }
}

// MARK: - ThinkingAwareResultContent

/// Drop-in replacement for a plain output `Text` / `RenderMessageSegments` when the model may
/// produce thinking tokens.  Shows the thinking drawer above the answer text.
///
/// - `useChatRenderer`: when `true` delegates answer rendering to `RenderMessageSegments`
///   (rich code-block / math / table support used in chat bubbles).  When `false` uses a plain
///   `Text` view suitable for feature-screen output panels.
struct ThinkingAwareResultContent: View {
    @EnvironmentObject var settings: AppSettings

    let content: String
    /// `true` while the model is streaming this response.
    let isGenerating: Bool
    /// When `true`, treat pre-answer streaming text as reasoning for thinking-capable
    /// models even if the backend doesn't emit an opening marker until later.
    var preferThinkingWhileStreaming: Bool = false
    /// Use rich markdown rendering for the answer (chat bubbles).  Defaults to `false`.
    var useChatRenderer: Bool = false

    var body: some View {
        let parsed = parseThinkingAndAnswer(content)
        let fallbackStreamingThinking = preferThinkingWhileStreaming
            && isGenerating
            && !contentHasThinkingMarkers(content)
            && !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let thinking = fallbackStreamingThinking ? content : parsed.thinking
        let answer = fallbackStreamingThinking ? "" : parsed.answer
        let hasThinking = !thinking.isEmpty
        let hasAnswer   = !answer.isEmpty
        // Detect thinking mode even when parsed thinking text is still empty
        // (e.g. content is exactly "<think>" or "<think>\n" — parser returns ("",""))
        let isInThinkingMode = hasThinking || contentHasThinkingMarkers(content) || fallbackStreamingThinking

        VStack(alignment: .leading, spacing: 8) {
            if isInThinkingMode {
                ThinkingDrawerView(
                    thinking: thinking,
                    isGenerating: isGenerating,
                    hasAnswer: hasAnswer
                )
            }

            if hasAnswer {
                RenderMessageSegments(displayContent: answer)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else if !isInThinkingMode {
                // No thinking block at all — render content directly
                RenderMessageSegments(displayContent: content)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            // When isInThinkingMode && !hasAnswer: the drawer shows "AI is thinking…"
            // nothing else to render yet — no raw tags leak into the UI.
        }
    }
}
