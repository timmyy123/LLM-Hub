#include "MagentaAudioRing.h"
#include <atomic>
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

static_assert(ATOMIC_LLONG_LOCK_FREE == 2 && ATOMIC_LONG_LOCK_FREE == 2 && ATOMIC_BOOL_LOCK_FREE == 2,
              "Audio atomics must be lock-free");
struct MRTAudioRing {
    const size_t capacity;
    const size_t startup;
    std::vector<float> samples;
    alignas(64) std::atomic<uint64_t> read{0};
    alignas(64) std::atomic<uint64_t> write{0};
    std::atomic<uint64_t> underruns{0};
    std::atomic<bool> finished{false};
    std::atomic<bool> stopped{false};
    bool started = false; // Audio consumer only.
    MRTAudioRing(size_t c, size_t s) : capacity(c), startup(s), samples(c * 2) {}
};
MRTAudioRing *MRTAudioRingCreate(size_t capacity, size_t startup) {
    if (!capacity || startup > capacity || capacity > std::numeric_limits<size_t>::max() / (2 * sizeof(float))) return nullptr;
    try { return new MRTAudioRing(capacity, startup); } catch (...) { return nullptr; }
}
void MRTAudioRingDestroy(MRTAudioRing *ring) { delete ring; }
size_t MRTAudioRingWrite(MRTAudioRing *ring, const uint8_t *pcm, size_t frames) {
    if (ring->stopped.load(std::memory_order_acquire)) return 0;
    const auto write = ring->write.load(std::memory_order_relaxed);
    const auto read = ring->read.load(std::memory_order_acquire);
    const auto count = std::min(frames, ring->capacity - size_t(write - read));
    for (size_t i = 0; i < count; ++i) {
        const auto slot = ((write + i) % ring->capacity) * 2;
        for (size_t ch = 0; ch < 2; ++ch) {
            const auto offset = i * 4 + ch * 2;
            const uint16_t bits = uint16_t(pcm[offset]) | (uint16_t(pcm[offset + 1]) << 8);
            int16_t value;
            std::memcpy(&value, &bits, sizeof(value));
            ring->samples[slot + ch] = float(value) / 32768.0f;
        }
    }
    ring->write.store(write + count, std::memory_order_release);
    return count;
}
size_t MRTAudioRingRead(MRTAudioRing *ring, float *left, float *right, size_t frames) {
    std::fill_n(left, frames, 0.0f);
    std::fill_n(right, frames, 0.0f);
    if (ring->stopped.load(std::memory_order_acquire)) return 0;
    const auto read = ring->read.load(std::memory_order_relaxed);
    const bool finished = ring->finished.load(std::memory_order_acquire);
    const auto write = ring->write.load(std::memory_order_acquire);
    const size_t available = size_t(write - read);
    if (!ring->started) {
        if (available < ring->startup && !finished) return 0;
        ring->started = true;
    }
    const auto count = std::min(frames, available);
    for (size_t i = 0; i < count; ++i) {
        const auto slot = ((read + i) % ring->capacity) * 2;
        left[i] = ring->samples[slot];
        right[i] = ring->samples[slot + 1];
    }
    ring->read.store(read + count, std::memory_order_release);
    if (count < frames && !finished) ring->underruns.fetch_add(frames - count, std::memory_order_relaxed);
    return count;
}
size_t MRTAudioRingBuffered(const MRTAudioRing *ring) {
    // Read first: the producer can advance write concurrently with the consumer.
    const auto read = ring->read.load(std::memory_order_acquire);
    return size_t(ring->write.load(std::memory_order_acquire) - read);
}
uint64_t MRTAudioRingUnderruns(const MRTAudioRing *ring) { return ring->underruns.load(std::memory_order_relaxed); }
void MRTAudioRingFinish(MRTAudioRing *ring) { ring->finished.store(true, std::memory_order_release); }
void MRTAudioRingStop(MRTAudioRing *ring) { ring->stopped.store(true, std::memory_order_release); }
bool MRTAudioRingStopped(const MRTAudioRing *ring) { return ring->stopped.load(std::memory_order_acquire); }
