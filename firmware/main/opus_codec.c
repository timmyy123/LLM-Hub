// Opus encoder/decoder for 16 kHz mono, 20 ms frames.
//
// NOT IMPLEMENTED YET. Use the ESP-IDF managed component `opus` (Xiph libopus
// built for ESP32). Add to firmware/main/CMakeLists.txt REQUIRES list.
//
// TODO(opus):
//   1. opus_encoder_create(16000, 1, OPUS_APPLICATION_VOIP, &err)
//      opus_encoder_ctl(enc, OPUS_SET_BITRATE(24000));
//      opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC(1));
//      opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC(10));
//   2. opus_decoder_create(16000, 1, &err)
//   3. Expose:
//        int opus_encode_frame(const int16_t *pcm, uint8_t *out, size_t out_cap);
//        int opus_decode_frame(const uint8_t *in, size_t in_len, int16_t *pcm);
//   4. Output buffer ~80 bytes is plenty for 24 kbps / 20 ms.

#include "mimobot_ids.h"
