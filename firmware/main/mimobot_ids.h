// Canonical IDs shared with the phone apps.
// Must match docs/mimobot-protocol.md, android/.../MimoBotIds.kt, ios/.../MimoBotIds.swift.
#pragma once

// 128-bit UUIDs, little-endian byte order for NimBLE.
// Base: 6d696d6f-b07d-4e13-9e88-00000000____
//        ^^^^^^^^                          ^^^^ last 2 bytes vary per characteristic
// "6d 69 6d 6f" is ASCII "mimo".

// Service: ...0001
#define MIMO_SVC_UUID_BYTES \
    { 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x9e, \
      0x13, 0x4e, 0x7d, 0xb0, 0x6f, 0x6d, 0x69, 0x6d }

// Audio Up: ...0010
#define MIMO_AUDIO_UP_UUID_BYTES \
    { 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x9e, \
      0x13, 0x4e, 0x7d, 0xb0, 0x6f, 0x6d, 0x69, 0x6d }

// Audio Down: ...0011
#define MIMO_AUDIO_DOWN_UUID_BYTES \
    { 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x9e, \
      0x13, 0x4e, 0x7d, 0xb0, 0x6f, 0x6d, 0x69, 0x6d }

// Control: ...0012
#define MIMO_CONTROL_UUID_BYTES \
    { 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x9e, \
      0x13, 0x4e, 0x7d, 0xb0, 0x6f, 0x6d, 0x69, 0x6d }

// Audio format
#define MIMO_SAMPLE_RATE_HZ    16000
#define MIMO_FRAME_MS          20
#define MIMO_FRAME_SAMPLES     320   // SAMPLE_RATE_HZ * FRAME_MS / 1000
#define MIMO_CHANNELS          1
#define MIMO_OPUS_BITRATE      24000

#define MIMO_ADV_NAME_PREFIX   "mimobot-"
