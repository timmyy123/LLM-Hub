// The engine's version, as a string.
//
// It exists so a benchmark file can name the engine that produced it. A committed CSV outlives the
// checkout that made it, and "which build was this?" is otherwise answerable only by the commit
// date next to the file — which is the date it was COPIED, not the date it was RUN. The run-parameter
// preamble records this next to the configuration, and `bmoe-cli --version` prints it.
//
// BMOE_VERSION is defined by the build from the CMake project version, so there is one source of
// truth. A build without that define (an unusual embedding) reports "unknown" rather than a number
// it cannot vouch for.
#pragma once

namespace bmoe {

#ifdef BMOE_VERSION
inline const char * version() {
    return BMOE_VERSION;
}
#else
inline const char * version() {
    return "unknown";
}
#endif

} // namespace bmoe
