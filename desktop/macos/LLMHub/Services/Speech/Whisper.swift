import Foundation
import os
import whisper

/// whisper.cpp in process (the official Apple build, Metal on Apple silicon). One context, used by one call at a time.
actor WhisperTranscriber {
    private var context: OpaquePointer?
    private var loadedPath: String?
    /// The loaded context, also reachable from the quit handler: whisper.cpp's Metal device aborts in its exit-time
    /// destructor while a context is still alive ("LLM Hub quit unexpectedly" after any dictation).
    private static let live = OSAllocatedUnfairLock<(context: OpaquePointer?, busy: Bool)>(initialState: (nil, false))

    deinit {
        if let context { Self.release(context) }
    }

    /// Frees the context when the app quits (not mid-transcription, which still owns it).
    nonisolated static func freeOnQuit() {
        let context = live.withLock { state -> OpaquePointer? in
            guard !state.busy, let context = state.context else { return nil }
            state.context = nil
            return context
        }
        if let context { whisper_free(context) }
    }

    private static func release(_ context: OpaquePointer) {
        let owned = live.withLock { state -> Bool in
            guard state.context == context else { return false }
            state.context = nil
            return true
        }
        if owned { whisper_free(context) }
    }

    func load(_ modelPath: URL) throws {
        if loadedPath == modelPath.path, context != nil { return }
        if let context { Self.release(context) }
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
        Self.live.withLock { $0.context = ctx }
    }

    func unload() {
        if let context { Self.release(context) }
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
        Self.live.withLock { $0.busy = true }
        defer { Self.live.withLock { $0.busy = false } }
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
