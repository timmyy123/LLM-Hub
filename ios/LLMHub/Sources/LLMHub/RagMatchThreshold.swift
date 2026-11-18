import Foundation

/// User-set RAG matching cut-offs (global memory and documents in chats), one per number in the
/// matching rule. An unset value is the built-in default, so matching stays exactly as before
/// unless the user moves a slider: similarity > 0.60, or similarity > 0.45 with word overlap > 0.030.
enum RagMatchThreshold {
    enum Param: String, CaseIterable, Identifiable {
        case similarity = "rag_match_similarity"
        case similarityWithOverlap = "rag_match_similarity_with_overlap"
        case overlap = "rag_match_overlap"

        var id: String { rawValue }

        var defaultValue: Float {
            switch self {
            case .similarity: return 0.60
            case .similarityWithOverlap: return 0.45
            case .overlap: return 0.030
            }
        }

        var range: ClosedRange<Float> { self == .overlap ? 0...0.30 : 0...1 }

        /// Overlap values need three decimals, similarity two.
        var step: Float { self == .overlap ? 0.001 : 0.01 }

        var titleKey: String {
            switch self {
            case .similarity: return "rag_threshold_similarity_alone"
            case .similarityWithOverlap: return "rag_threshold_similarity_with_overlap"
            case .overlap: return "rag_threshold_overlap"
            }
        }

        func format(_ value: Float) -> String {
            String(format: self == .overlap ? "%.3f" : "%.2f", locale: Locale(identifier: "en_US_POSIX"), value)
        }
    }

    static func value(_ param: Param) -> Float {
        let defaults = UserDefaults.standard
        guard defaults.object(forKey: param.rawValue) != nil else { return param.defaultValue }
        return defaults.float(forKey: param.rawValue)
    }

    static func set(_ param: Param, _ value: Float) {
        UserDefaults.standard.set(min(max(value, param.range.lowerBound), param.range.upperBound), forKey: param.rawValue)
    }

    static func reset() {
        Param.allCases.forEach { UserDefaults.standard.removeObject(forKey: $0.rawValue) }
    }

    static func describe() -> String {
        Param.allCases.map { "\($0.rawValue)=\($0.format(value($0)))" }.joined(separator: ", ")
    }
}
