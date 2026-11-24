// Why a request to the engine failed, as a kind a caller can act on.
//
// A failure's `error` string is prose for a person and free to change; every decision a front-end
// makes about a failure (keep the session or close it, offer a new chat, retry) reads the kind.
// The name of each kind is the stable spelling the CLI's line protocol and the server's API carry
// as `code` (docs/telemetry.md, docs/server-api.md).
//
// Pure policy header (no llama.cpp types).
#pragma once

namespace bmoe {

enum class RunError {
    None,
    ContextFull,      // the conversation leaves no room in n_ctx for an answer
    EmptyPrompt,      // nothing left to answer after tokenization
    TemplateRejected, // the model's chat template refused the message list
    NeedsChatMode,    // a message list sent to a session opened without the chat template
    InvalidRequest,   // the request itself is malformed (decide: no choices, colliding choices, off)
    NoVision,         // the request carries images and the session has no projector to read them
    Failed,           // the model did not load, or a decode or the expert stream failed mid-run
    InvalidOutput,    // the requested response contract failed; the turn was rolled back
};

// The stable name of a kind: "context_full", "empty_prompt", ...
const char * run_error_name(RunError e);

// True when the session can take the next request: a refused request left the context alone,
// or an invalid completed output was rolled back. A decode or I/O failure is not safe to reuse.
inline bool run_error_recoverable(RunError e) {
    return e != RunError::None && e != RunError::Failed;
}

} // namespace bmoe
