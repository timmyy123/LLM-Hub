#ifndef MAGENTA_AUDIO_RING_H
#define MAGENTA_AUDIO_RING_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
// Single producer / single audio-render consumer. All storage is preallocated.
typedef struct MRTAudioRing MRTAudioRing;
MRTAudioRing *MRTAudioRingCreate(size_t capacity_frames, size_t startup_frames);
void MRTAudioRingDestroy(MRTAudioRing *ring);
size_t MRTAudioRingWrite(MRTAudioRing *ring, const uint8_t *pcm, size_t frames);
// Writes planar stereo Float32 and zero-fills missing samples. Never blocks.
size_t MRTAudioRingRead(MRTAudioRing *ring, float *left, float *right, size_t frames);
size_t MRTAudioRingBuffered(const MRTAudioRing *ring);
uint64_t MRTAudioRingUnderruns(const MRTAudioRing *ring);
void MRTAudioRingFinish(MRTAudioRing *ring);
void MRTAudioRingStop(MRTAudioRing *ring);
bool MRTAudioRingStopped(const MRTAudioRing *ring);
#ifdef __cplusplus
}
#endif
#endif
