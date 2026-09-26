import Foundation
import llama

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

    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vision: OpaquePointer?
    private var modelPath: String?
    private var projectorPath: String?
    private var configuredContext = 0
    private var configuredGpuLayers = 0

    private init() {
        llama_backend_init()
    }

    public func load(path: String, projector: String?, contextSize: Int, gpuLayers: Int) throws {
        if modelPath == path, projectorPath == projector,
           configuredContext == contextSize, configuredGpuLayers == gpuLayers,
           model != nil, context != nil {
            return
        }
        unload()

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
    }

    public func unload() {
        if let vision { mtmd_free(vision) }
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        vision = nil
        context = nil
        model = nil
        modelPath = nil
        projectorPath = nil
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
        let input = marker + "\n" + prompt
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
}
