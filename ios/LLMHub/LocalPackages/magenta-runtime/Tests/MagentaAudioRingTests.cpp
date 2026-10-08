// Run from this package directory:
// clang++ -std=c++11 -O2 -pthread -I Sources/MagentaLiteRTBridge/include \
//   Sources/MagentaLiteRTBridge/MagentaAudioRing.cpp Tests/MagentaAudioRingTests.cpp -o /tmp/mrt-ring-test
// /tmp/mrt-ring-test
#include "MagentaAudioRing.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

static std::vector<uint8_t> pcm(size_t first, size_t count) {
    std::vector<uint8_t> result(count * 4);
    for (size_t i = 0; i < count; ++i) {
        const int16_t left = int16_t((first + i) % 30000 + 1);
        const int16_t right = -left;
        for (size_t ch = 0; ch < 2; ++ch) {
            const uint16_t bits = uint16_t(ch ? right : left);
            result[i * 4 + ch * 2] = uint8_t(bits);
            result[i * 4 + ch * 2 + 1] = uint8_t(bits >> 8);
        }
    }
    return result;
}
static void check(const std::vector<float>& left, const std::vector<float>& right, size_t first, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float expected = float((first + i) % 30000 + 1) / 32768.0f;
        assert(left[i] == expected && right[i] == -expected);
    }
}
int main() {
    assert(!MRTAudioRingCreate(0, 0));
    assert(!MRTAudioRingCreate(5, 6));
    {
        auto *ring = MRTAudioRingCreate(192000, 96000);
        auto initial = pcm(0, 96000);
        std::vector<float> left(960, 1), right(960, 1);
        assert(MRTAudioRingWrite(ring, initial.data(), 94080) == 94080);
        assert(MRTAudioRingRead(ring, left.data(), right.data(), 960) == 0);
        for (auto sample : left) assert(sample == 0);
        assert(MRTAudioRingUnderruns(ring) == 0); // Startup silence is deliberate.
        assert(MRTAudioRingWrite(ring, initial.data() + 94080 * 4, 1920) == 1920);
        size_t written = 96000, read = 0;
        // Three minutes of 300 ms generation bursts, rendered in 20 ms quanta.
        // Every sample must remain contiguous across ring wrap and burst boundaries.
        for (size_t step = 0; step < 9000; ++step) {
            if (step && step % 15 == 0) {
                auto burst = pcm(written, 14400);
                assert(MRTAudioRingWrite(ring, burst.data(), 14400) == 14400);
                written += 14400;
            }
            assert(MRTAudioRingRead(ring, left.data(), right.data(), 960) == 960);
            check(left, right, read, 960);
            read += 960;
            assert(MRTAudioRingBuffered(ring) <= 192000);
        }
        assert(MRTAudioRingUnderruns(ring) == 0);
        MRTAudioRingFinish(ring);
        while (read < written) {
            const size_t count = MRTAudioRingRead(ring, left.data(), right.data(), 960);
            assert(count > 0);
            check(left, right, read, count);
            read += count;
        }
        assert(MRTAudioRingBuffered(ring) == 0);
        assert(MRTAudioRingRead(ring, left.data(), right.data(), 960) == 0);
        assert(MRTAudioRingUnderruns(ring) == 0); // Completed audio isn't an underrun.
        MRTAudioRingDestroy(ring);
    }
    {
        auto *ring = MRTAudioRingCreate(8, 8);
        auto data = pcm(0, 4);
        std::vector<float> left(8), right(8);
        assert(MRTAudioRingWrite(ring, data.data(), 4) == 4);
        MRTAudioRingFinish(ring); // Short finite clip must not wait for startup threshold.
        assert(MRTAudioRingRead(ring, left.data(), right.data(), 8) == 4);
        check(left, right, 0, 4);
        for (size_t i = 4; i < 8; ++i) assert(left[i] == 0 && right[i] == 0);
        MRTAudioRingDestroy(ring);
    }
    {
        auto *ring = MRTAudioRingCreate(8, 0);
        auto data = pcm(0, 16);
        std::vector<float> left(8), right(8);
        assert(MRTAudioRingWrite(ring, data.data(), 16) == 8); // Bounded, never overwrites.
        assert(MRTAudioRingWrite(ring, data.data(), 1) == 0);
        assert(MRTAudioRingRead(ring, left.data(), right.data(), 8) == 8);
        assert(MRTAudioRingRead(ring, left.data(), right.data(), 8) == 0);
        assert(MRTAudioRingUnderruns(ring) == 8);
        assert(MRTAudioRingWrite(ring, data.data() + 8 * 4, 8) == 8);
        assert(MRTAudioRingRead(ring, left.data(), right.data(), 8) == 8);
        check(left, right, 8, 8); // Resume after real starvation.
        MRTAudioRingStop(ring);
        assert(MRTAudioRingStopped(ring));
        assert(MRTAudioRingWrite(ring, data.data(), 8) == 0);
        assert(MRTAudioRingRead(ring, left.data(), right.data(), 8) == 0);
        MRTAudioRingDestroy(ring);
    }
    {
        auto *ring = MRTAudioRingCreate(4096, 512);
        constexpr size_t total = 1000000;
        std::thread producer([&] {
            size_t written = 0;
            while (written < total) {
                const size_t count = std::min(size_t(409), total - written);
                auto block = pcm(written, count);
                written += MRTAudioRingWrite(ring, block.data(), count);
                std::this_thread::yield();
            }
            MRTAudioRingFinish(ring);
        });
        size_t consumed = 0;
        std::vector<float> left(127), right(127);
        while (consumed < total) {
            const size_t count = MRTAudioRingRead(ring, left.data(), right.data(), left.size());
            check(left, right, consumed, count);
            consumed += count;
            std::this_thread::yield();
        }
        producer.join();
        assert(MRTAudioRingBuffered(ring) == 0);
        MRTAudioRingDestroy(ring);
    }
    std::cout << "Passed continuous burst playback, startup, wraparound, bounded capacity, short clips, underrun recovery, stop, and concurrent sample integrity\n";
}
