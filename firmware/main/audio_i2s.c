// I²S mic capture + speaker playback.
//
// NOT IMPLEMENTED YET.
//
// TODO(i2s):
//   1. Configure I2S_NUM_0 as RX for the mic (INMP441: 24-bit, MSB justified,
//      we'll shift down to 16-bit).
//   2. Configure I2S_NUM_1 as TX for the speaker DAC (MAX98357A: 16-bit, LJ).
//   3. Expose:
//        int  audio_i2s_read_frame (int16_t *pcm, size_t samples);   // blocking
//        int  audio_i2s_write_frame(const int16_t *pcm, size_t samples);
//   4. 320-sample frames (20 ms @ 16 kHz) to line up with Opus.
//   5. Add a small ring buffer on the TX side so BLE jitter doesn't underrun.

#include "mimobot_ids.h"
