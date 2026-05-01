# Mimo Bot ↔ Phone Protocol (v0)

Status: **draft** — implementation scaffolding only. No wire-format guarantee until v1.

This document is the contract between the ESP32-S3 firmware (`firmware/`) and
both mobile apps (`android/` and `ios/`). Whenever you change this file, both
sides must be updated in the same commit.

---

## Transport

v0 uses **BLE GATT** as the single link. Wi-Fi SoftAP is reserved for a future
"high-bandwidth mode" but is not implemented yet.

The device advertises as a BLE peripheral. The phone is the central.

- **Advertising name**: `mimobot-<hex4>` where `<hex4>` is the last 2 bytes of
  the MAC in hex.
- **Preferred connection interval**: 15 ms (for low audio latency).
- **MTU**: request 247 bytes (ATT MTU 244) on connect.

---

## GATT Service

Custom primary service.

| Role       | UUID                                     |
|------------|------------------------------------------|
| Service    | `6d696d6f-b07d-4e13-9e88-000000000001`   |
| Audio Up   | `6d696d6f-b07d-4e13-9e88-000000000010`   |
| Audio Down | `6d696d6f-b07d-4e13-9e88-000000000011`   |
| Control    | `6d696d6f-b07d-4e13-9e88-000000000012`   |

*The base `6d696d6f-…` prefix is ASCII "mimo".*

### Audio Up — device → phone (NOTIFY)

Device sends one notification per Opus frame.

```
+------+------+--- opus payload ---+
| seq (u16 LE) | variable bytes    |
+------+------+-------------------+
```

- `seq` increments per frame, wraps at 2^16. Used for loss detection.
- Payload is a single Opus frame (see Audio Format below). Never fragmented.

### Audio Down — phone → device (WRITE_WITHOUT_RESPONSE)

Same framing as Audio Up. Phone writes one Opus frame per operation.

### Control — bidirectional (NOTIFY + WRITE)

UTF-8 JSON, one message per GATT operation. Must fit in a single ATT MTU
payload (≤ 240 bytes after overhead). If a message would exceed that, add a
new compact field instead of fragmenting.

---

## Audio Format

| Field        | Value                   |
|--------------|-------------------------|
| Codec        | Opus (libopus)          |
| Application  | `OPUS_APPLICATION_VOIP` |
| Sample rate  | 16 000 Hz               |
| Channels     | 1 (mono)                |
| Frame size   | 20 ms (320 samples)     |
| Bitrate      | 24 kbps (target)        |
| FEC          | on                      |
| DTX          | off (v0)                |

A 24 kbps / 20 ms Opus frame is ≈ 60 bytes — one BLE notification per frame.
Expected throughput ≈ 3 kB/s + 2 B seq + BLE overhead; comfortable for a
connection interval of 15 ms.

---

## Control Messages

All messages are objects with a short `t` (type) field. Fields are minimal to
stay under the MTU.

### Device → Phone

```json
{"t":"hello","fw":"0.1.0","cap":["mic","spk"]}
{"t":"ptt","s":"down"}         // s = "down" | "up"
{"t":"batt","p":87}            // p = percent 0-100
{"t":"err","c":"opus_enc","m":"..."}
{"t":"end"}                    // user released PTT / VAD tail silence
```

### Phone → Device

```json
{"t":"cfg","sr":16000,"br":24000}   // configure audio (sent after hello)
{"t":"tts_start"}                   // phone is about to send audio_down
{"t":"tts_stop"}                    // phone stopped speaking (barge-in or end)
{"t":"led","r":0,"g":255,"b":0}     // optional status LED (v1+)
{"t":"ping"}                        // echo test
```

### Handshake

1. Phone connects, discovers service, subscribes to Audio-Up + Control.
2. Device sends `{"t":"hello",...}`.
3. Phone replies with `{"t":"cfg",...}` (even if just defaults).
4. Ready. Either side may now send audio.

---

## Turn-taking (v0 — push-to-talk)

1. User presses the button on the device → device sends `{"t":"ptt","s":"down"}`.
2. Device streams Audio Up frames.
3. User releases button → device sends `{"t":"ptt","s":"up"}` + `{"t":"end"}`.
4. Phone runs VAD + Whisper + LLM + TTS.
5. Phone sends `{"t":"tts_start"}`, streams Audio Down frames, ends with
   `{"t":"tts_stop"}`.

Full-duplex + barge-in land in v1 once AEC is in the firmware.

---

## Future extensions (not implemented)

- Wi-Fi SoftAP provisioning: ESP32 hosts AP only during setup; phone hands over
  home Wi-Fi creds over a provisioning characteristic; afterwards both join the
  LAN and audio moves to UDP for lower latency.
- LE Audio (LC3 + Auracast) once ESP-IDF support matures.
- Device OTA firmware updates over a dedicated characteristic.
- Servo/motor control messages.
