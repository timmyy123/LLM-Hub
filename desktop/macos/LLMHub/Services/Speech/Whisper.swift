import Foundation
import whisper

/// whisper.cpp in process (the official Apple build, Metal on Apple silicon). One context, used by one call at a time.
actor WhisperTranscriber {
    private var context: OpaquePointer?
    private var loadedPath: String?

    deinit {
        if let context { whisper_free(context) }
    }

    func load(_ modelPath: URL) throws {
        if loadedPath == modelPath.path, context != nil { return }
        if let context { whisper_free(context) }
        context = nil
        // Quiet: whisper.cpp logs every layer to stderr otherwise.
        whisper_log_set({ _, _, _ in }, nil)
        var params = whisper_context_default_params()
        params.use_gpu = true
        params.flash_attn = true
        guard let ctx = whisper_init_from_file_with_params(modelPath.path, params) else {
            throw NSError(domain: "LLMHub", code: 40, userInfo: [NSLocalizedDescriptionKey: "Whisper could not load \(modelPath.lastPathComponent)"])
        }
        context = ctx
        loadedPath = modelPath.path
    }

    func unload() {
        if let context { whisper_free(context) }
        context = nil
        loadedPath = nil
    }

    /// 16 kHz mono samples to text. `language` is a code like "en", or nil to detect it.
    func transcribe(_ samples: [Float], language: String?) throws -> String {
        guard let context else { throw NSError(domain: "LLMHub", code: 41, userInfo: [NSLocalizedDescriptionKey: "Whisper is not loaded"]) }
        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.n_threads = Int32(max(4, min(8, ProcessInfo.processInfo.activeProcessorCount - 2)))
        params.translate = false
        params.no_timestamps = true
        params.print_special = false
        params.print_progress = false
        params.print_realtime = false
        params.print_timestamps = false
        params.suppress_blank = true
        let code = language ?? "auto"
        let status = code.withCString { lang -> Int32 in
            params.language = lang
            return samples.withUnsafeBufferPointer { whisper_full(context, params, $0.baseAddress, Int32($0.count)) }
        }
        guard status == 0 else { throw NSError(domain: "LLMHub", code: 42, userInfo: [NSLocalizedDescriptionKey: "Whisper failed (\(status))"]) }
        var text = ""
        for i in 0..<whisper_full_n_segments(context) {
            if let segment = whisper_full_get_segment_text(context, i) { text += String(cString: segment) }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
