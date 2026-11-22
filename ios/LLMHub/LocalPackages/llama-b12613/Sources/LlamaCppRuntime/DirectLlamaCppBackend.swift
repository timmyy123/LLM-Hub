import Foundation
import llama
import LLMHubBmoe

/// The app's GGUF runtime. RunAnywhere is not involved in model execution here.
public actor DirectLlamaCppBackend {
    public static let shared = DirectLlamaCppBackend()

    public struct Update: Sendable {
        public let text: String
        public let completionTokens: Int
        public let tokensPerSecond: Double
    }

    public enum RuntimeError: LocalizedError {
        case load(String)
        case inference(String)

        public var errorDescription: String? {
            switch self {
            case .load(let message), .inference(let message): return message
            }
        }
    }

    /// mtmd's media marker. A vision prompt that already contains it keeps the image at that spot
    /// (e.g. inside the user turn); otherwise the image goes before the prompt.
    public static let mediaMarker = String(cString: mtmd_default_marker())

    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vision: OpaquePointer?
    private var modelPath: String?
    private var projectorPath: String?
    private var configuredContext = 0
    private var configuredGpuLayers = 0
    /// Flash streaming session (MoE GGUF): the routed experts are read from storage as tokens need them, so a model
    /// larger than the device's memory runs on the CPU. `model` then holds only the vocabulary, for the chat template.
    private var flash: OpaquePointer?
    private var configuredFlash = false
    private var configuredGpuPrefill = false

    private init() {
        llama_backend_init()
    }

    /// With `flashStreaming`, `gpuPrefill` (the GPU backend selected) reads the prompt on the GPU when `gpuLayers` is 0.
    public func load(path: String, projector: String?, contextSize: Int, gpuLayers: Int, flashStreaming: Bool = false,
                     gpuPrefill: Bool = false) throws {
        if modelPath == path, projectorPath == projector,
           configuredContext == contextSize, configuredFlash == flashStreaming,
           configuredGpuLayers == gpuLayers, !flashStreaming || configuredGpuPrefill == gpuPrefill,
           model != nil, context != nil || flash != nil {
            return
        }
        unload()
        if flashStreaming {
            try loadFlash(path: path, projector: projector, contextSize: contextSize, gpuLayers: gpuLayers,
                          gpuPrefill: gpuPrefill)
            return
        }

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = Int32(max(0, gpuLayers))
        guard let loadedModel = path.withCString({ llama_model_load_from_file($0, modelParams) }) else {
            throw RuntimeError.load("llama.cpp could not load GGUF: \(path)")
        }

        var contextParams = llama_context_default_params()
        contextParams.n_ctx = UInt32(max(512, contextSize))
        contextParams.n_batch = UInt32(min(2048, max(512, contextSize)))
        contextParams.n_ubatch = UInt32(min(512, Int(contextParams.n_batch)))
        let threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        contextParams.n_threads = Int32(threads)
        contextParams.n_threads_batch = Int32(threads)
        guard let loadedContext = llama_init_from_model(loadedModel, contextParams) else {
            llama_model_free(loadedModel)
            throw RuntimeError.load("llama.cpp could not create a context for \(path)")
        }

        var loadedVision: OpaquePointer?
        if let projector {
            var params = mtmd_context_params_default()
            params.use_gpu = gpuLayers > 0
            params.n_threads = Int32(threads)
            loadedVision = projector.withCString { mtmd_init_from_file($0, loadedModel, params) }
            if loadedVision == nil {
                llama_free(loadedContext)
                llama_model_free(loadedModel)
                throw RuntimeError.load("llama.cpp could not load vision projector: \(projector)")
            }
        }

        model = loadedModel
        context = loadedContext
        vision = loadedVision
        modelPath = path
        projectorPath = projector
        configuredContext = contextSize
        configuredGpuLayers = gpuLayers
        configuredFlash = false
    }

    /// The routed experts stream from storage; `gpuLayers` dense layers go to Metal, or with none `gpuPrefill` reads
    /// the prompt there.
    private func loadFlash(path: String, projector: String?, contextSize: Int, gpuLayers: Int, gpuPrefill: Bool) throws {
        var vocabParams = llama_model_default_params()
        vocabParams.vocab_only = true
        vocabParams.n_gpu_layers = 0
        guard let vocabModel = path.withCString({ llama_model_load_from_file($0, vocabParams) }) else {
            throw RuntimeError.load("llama.cpp could not read GGUF: \(path)")
        }
        let threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        var error = [CChar](repeating: 0, count: 512)
        let session = path.withCString { modelPointer in
            (projector ?? "").withCString { projectorPointer in
                llmhub_bmoe_open(modelPointer, projector == nil ? nil : projectorPointer,
                                 Int32(max(512, contextSize)), Int32(threads), gpuPrefill ? 1 : 0, Int32(max(0, gpuLayers)), &error, error.count)
            }
        }
        guard let session else {
            llama_model_free(vocabModel)
            throw RuntimeError.load("Flash streaming could not load \(path): \(String(cString: error))")
        }
        model = vocabModel
        flash = session
        modelPath = path
        projectorPath = projector
        configuredContext = contextSize
        configuredGpuLayers = gpuLayers
        configuredGpuPrefill = gpuPrefill
        configuredFlash = true
    }

    public func unload() {
        if let flash { llmhub_bmoe_free(flash) }
        flash = nil
        configuredFlash = false
        if let vision { mtmd_free(vision) }
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        vision = nil
        context = nil
        model = nil
        modelPath = nil
        projectorPath = nil
    }

    /// Wraps a single-turn prompt in the loaded GGUF's embedded chat template.
    /// Returns nil when the model has no template or llama.cpp doesn't recognise it.
    public func applyChatTemplate(systemPrompt: String?, userPrompt: String) -> String? {
        guard let model, let template = llama_model_chat_template(model, nil) else { return nil }

        var turns: [(role: String, content: String)] = []
        if let systemPrompt, !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            turns.append(("system", systemPrompt))
        }
        turns.append(("user", userPrompt))

        let cStrings = turns.map { (strdup($0.role)!, strdup($0.content)!) }
        defer { cStrings.forEach { free($0.0); free($0.1) } }
        let messages = cStrings.map { llama_chat_message(role: $0.0, content: $0.1) }

        let estimate = turns.reduce(0) { $0 + $1.content.utf8.count } * 2 + 512
        var buffer = [CChar](repeating: 0, count: estimate)
        var length = llama_chat_apply_template(template, messages, messages.count, true, &buffer, Int32(buffer.count))
        if length > Int32(buffer.count) {
            buffer = [CChar](repeating: 0, count: Int(length) + 1)
            length = llama_chat_apply_template(template, messages, messages.count, true, &buffer, Int32(buffer.count))
        }
        guard length > 0 else { return nil }
        return String(decoding: buffer.prefix(Int(length)).map(UInt8.init(bitPattern:)), as: UTF8.self)
    }

    /// Emits cumulative UTF-8 text, matching LLMBackend's existing UI contract.
    public func stream(
        prompt: String,
        imageData: Data? = nil,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        stopSequences: [String]
    ) -> AsyncThrowingStream<Update, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try self.generate(
                        prompt: prompt, imageData: imageData, maxTokens: maxTokens,
                        temperature: temperature, topK: topK, topP: topP,
                        stopSequences: stopSequences, continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func generate(
        prompt: String,
        imageData: Data? = nil,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        stopSequences: [String],
        continuation: AsyncThrowingStream<Update, Error>.Continuation
    ) throws {
        if let flash {
            try generateFlash(
                session: flash, prompt: prompt, imageData: imageData, maxTokens: maxTokens,
                temperature: temperature, topK: topK, topP: topP,
                stopSequences: stopSequences, continuation: continuation
            )
            return
        }
        guard let model, let context, let vocab = llama_model_get_vocab(model) else {
            throw RuntimeError.inference("No GGUF model is loaded")
        }
        llama_memory_clear(llama_get_memory(context), false)
        var nextPosition: Int32 = 0
        if let imageData {
            guard let vision else { throw RuntimeError.inference("This GGUF has no loaded vision projector") }
            nextPosition = try prefillVision(prompt: prompt, imageData: imageData, context: context, vision: vision)
        } else {
            nextPosition = try prefillText(prompt: prompt, context: context, vocab: vocab)
        }

        let remaining = Int(llama_n_ctx(context)) - Int(nextPosition) - 1
        guard remaining > 0 else { throw RuntimeError.inference("Prompt exceeds the configured context window") }

        let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())!
        defer { llama_sampler_free(sampler) }
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(Int32(max(1, topK))))
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(max(0, min(1, topP)), 1))
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(max(0, temperature)))
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED))

        var batch = llama_batch_init(1, 0, 1)
        defer { llama_batch_free(batch) }
        var utf8Pending = Data()
        var output = ""
        let started = Date()
        var generated = 0

        for _ in 0..<min(maxTokens, remaining) {
            try Task.checkCancellation()
            let token = llama_sampler_sample(sampler, context, -1)
            if llama_vocab_is_eog(vocab, token) { break }
            llama_sampler_accept(sampler, token)
            generated += 1

            var piece = [CChar](repeating: 0, count: 256)
            var count = llama_token_to_piece(vocab, token, &piece, Int32(piece.count), 0, false)
            if count < 0 {
                piece = [CChar](repeating: 0, count: Int(-count))
                count = llama_token_to_piece(vocab, token, &piece, Int32(piece.count), 0, false)
            }
            if count > 0 {
                utf8Pending.append(contentsOf: piece.prefix(Int(count)).map(UInt8.init(bitPattern:)))
                if let decoded = String(data: utf8Pending, encoding: .utf8) {
                    output += decoded
                    utf8Pending.removeAll(keepingCapacity: true)
                }
            }

            if let stop = stopSequences.first(where: { !$0.isEmpty && output.hasSuffix($0) }) {
                output.removeLast(stop.count)
                break
            }
            let rate = Double(generated) / max(0.001, Date().timeIntervalSince(started))
            continuation.yield(Update(text: output, completionTokens: generated, tokensPerSecond: rate))

            batch.n_tokens = 1
            batch.token[0] = token
            batch.pos[0] = nextPosition
            batch.n_seq_id[0] = 1
            batch.seq_id[0]![0] = 0
            batch.logits[0] = 1
            guard llama_decode(context, batch) == 0 else {
                throw RuntimeError.inference("llama.cpp failed while decoding a token")
            }
            nextPosition += 1
        }
        if !utf8Pending.isEmpty { output += String(decoding: utf8Pending, as: UTF8.self) }
        continuation.yield(Update(
            text: output, completionTokens: generated,
            tokensPerSecond: Double(generated) / max(0.001, Date().timeIntervalSince(started))
        ))
    }

    private func prefillText(prompt: String, context: OpaquePointer, vocab: OpaquePointer) throws -> Int32 {
        let byteCount = prompt.utf8.count
        var tokens = [llama_token](repeating: 0, count: max(32, byteCount + 8))
        var count = prompt.withCString {
            llama_tokenize(vocab, $0, Int32(byteCount), &tokens, Int32(tokens.count), true, true)
        }
        if count < 0 {
            tokens = [llama_token](repeating: 0, count: Int(-count))
            count = prompt.withCString {
                llama_tokenize(vocab, $0, Int32(byteCount), &tokens, Int32(tokens.count), true, true)
            }
        }
        guard count > 0, Int(count) < Int(llama_n_ctx(context)) else {
            throw RuntimeError.inference("Prompt is empty or exceeds context")
        }
        var batch = llama_batch_init(512, 0, 1)
        defer { llama_batch_free(batch) }
        var position = 0
        while position < Int(count) {
            let batchSize = min(512, Int(count) - position)
            batch.n_tokens = Int32(batchSize)
            for index in 0..<batchSize {
                batch.token[index] = tokens[position + index]
                batch.pos[index] = Int32(position + index)
                batch.n_seq_id[index] = 1
                batch.seq_id[index]![0] = 0
                batch.logits[index] = position + index == Int(count) - 1 ? 1 : 0
            }
            guard llama_decode(context, batch) == 0 else {
                throw RuntimeError.inference("llama.cpp failed to prefill the prompt")
            }
            position += batchSize
        }
        return Int32(position)
    }

    private func prefillVision(prompt: String, imageData: Data, context: OpaquePointer, vision: OpaquePointer) throws -> Int32 {
        let image = imageData.withUnsafeBytes { raw in
            mtmd_helper_bitmap_init_from_buf(
                vision, raw.bindMemory(to: UInt8.self).baseAddress, imageData.count,
                false, mtmd_helper_init_opt_default()
            )
        }
        guard let bitmap = image.bitmap else {
            throw RuntimeError.inference("llama.cpp could not decode the image")
        }
        defer { mtmd_bitmap_free(bitmap) }
        let chunks = mtmd_input_chunks_init()!
        defer { mtmd_input_chunks_free(chunks) }
        let marker = String(cString: mtmd_get_marker(vision))
        let input = prompt.contains(marker) ? prompt : marker + "\n" + prompt
        let result = input.withCString { string in
            var text = mtmd_input_text(
                text: string, text_len: input.utf8.count,
                add_special: true, parse_special: true
            )
            var imagePointer: OpaquePointer? = bitmap
            return withUnsafePointer(to: &imagePointer) { ptr in
                mtmd_tokenize(vision, chunks, &text, ptr, 1)
            }
        }
        guard result == 0 else { throw RuntimeError.inference("llama.cpp could not tokenize the image prompt") }
        let positions = mtmd_helper_get_n_pos(chunks)
        guard positions < Int32(llama_n_ctx(context)) else {
            throw RuntimeError.inference("Image and prompt exceed the configured context window")
        }
        var nextPosition: llama_pos = 0
        guard mtmd_helper_eval_chunks(vision, context, chunks, 0, 0, 512, true, &nextPosition) == 0 else {
            throw RuntimeError.inference("llama.cpp failed to prefill the image")
        }
        return nextPosition
    }

    private func generateFlash(
        session: OpaquePointer,
        prompt: String,
        imageData: Data?,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        stopSequences: [String],
        continuation: AsyncThrowingStream<Update, Error>.Continuation
    ) throws {
        var input = prompt
        if imageData != nil {
            guard llmhub_bmoe_vision(session) != 0 else {
                throw RuntimeError.inference("This GGUF has no loaded vision projector")
            }
            let marker = String(cString: llmhub_bmoe_image_marker())
            if !input.contains(marker) { input = marker + "\n" + input }
        }
        let sink = FlashSink(stopSequences: stopSequences, continuation: continuation)
        let user = Unmanaged.passUnretained(sink).toOpaque()
        var generated: Int32 = 0
        var seconds = 0.0
        var error = [CChar](repeating: 0, count: 512)
        let onPiece: llmhub_bmoe_piece_fn = { piece, length, user in
            guard let piece, let user else { return 0 }
            let sink = Unmanaged<FlashSink>.fromOpaque(user).takeUnretainedValue()
            return sink.receive(UnsafeRawBufferPointer(start: piece, count: length)) ? 0 : 1
        }
        let mime = imageData.map { $0.starts(with: [0x89, 0x50, 0x4E, 0x47]) ? "image/png" : "image/jpeg" } ?? "image/jpeg"
        let status = input.withCString { promptPointer in
            mime.withCString { mimePointer in
                if let imageData {
                    return imageData.withUnsafeBytes { raw in
                        llmhub_bmoe_generate(
                            session, promptPointer, raw.bindMemory(to: UInt8.self).baseAddress, imageData.count, mimePointer,
                            Int32(max(1, maxTokens)), temperature, Int32(max(1, topK)), topP,
                            onPiece, user, &generated, &seconds, &error, error.count)
                    }
                }
                return llmhub_bmoe_generate(
                    session, promptPointer, nil, 0, mimePointer,
                    Int32(max(1, maxTokens)), temperature, Int32(max(1, topK)), topP,
                    onPiece, user, &generated, &seconds, &error, error.count)
            }
        }
        try Task.checkCancellation()
        guard status == 0 else {
            throw RuntimeError.inference("Flash streaming failed: \(String(cString: error))")
        }
        let rate = seconds > 0 ? Double(generated) / seconds : sink.rate
        continuation.yield(Update(text: sink.finish(), completionTokens: Int(max(generated, Int32(sink.generated))), tokensPerSecond: rate))
    }
}

/// Collects flash streaming pieces into the cumulative text the UI expects, applying the stop sequences.
private final class FlashSink {
    let stopSequences: [String]
    let continuation: AsyncThrowingStream<DirectLlamaCppBackend.Update, Error>.Continuation
    var output = ""
    var pending = Data()
    var generated = 0
    let started = Date()

    init(stopSequences: [String], continuation: AsyncThrowingStream<DirectLlamaCppBackend.Update, Error>.Continuation) {
        self.stopSequences = stopSequences
        self.continuation = continuation
    }

    var rate: Double { Double(generated) / max(0.001, Date().timeIntervalSince(started)) }

    /// Returns false to stop the generation.
    func receive(_ bytes: UnsafeRawBufferPointer) -> Bool {
        if Task.isCancelled { return false }
        generated += 1
        pending.append(contentsOf: bytes)
        if let decoded = String(data: pending, encoding: .utf8) {
            output += decoded
            pending.removeAll(keepingCapacity: true)
        }
        if let stop = stopSequences.first(where: { !$0.isEmpty && output.hasSuffix($0) }) {
            output.removeLast(stop.count)
            return false
        }
        continuation.yield(DirectLlamaCppBackend.Update(text: output, completionTokens: generated, tokensPerSecond: rate))
        return true
    }

    func finish() -> String {
        if !pending.isEmpty { output += String(decoding: pending, as: UTF8.self) }
        pending.removeAll()
        return output
    }
}
