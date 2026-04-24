# Mimo Bot Firmware (ESP32-S3)

Firmware for the Mimo bot hardware. Connects to the phone over BLE and streams
audio in both directions. The phone does all the heavy lifting (Whisper, LLM,
TTS); the ESP32 just handles mic/speaker I/O, Opus codec, and BLE transport.

## Target hardware

- **MCU**: ESP32-S3 (any variant with ≥8 MB PSRAM recommended — Opus encoder
  wants a few hundred KB of RAM)
- **Mic**: I²S MEMS (INMP441 or similar), mono, 16 kHz sample rate
- **Speaker**: small 4/8 Ω driven by I²S DAC + Class-D amp (MAX98357A)
- **Button**: one GPIO for push-to-talk (active low, internal pull-up)
- **LED** (optional): NeoPixel for status

Suggested dev boards: Adafruit QT Py ESP32-S3, Seeed XIAO ESP32-S3 Sense
(has mic on board), ESP32-S3-BOX-3.

## Toolchain

- ESP-IDF v5.2 or newer
- Python 3.9+

```bash
# one-time:
git clone --recursive https://github.com/espressif/esp-idf.git ~/esp-idf
~/esp-idf/install.sh esp32s3

# each shell:
. ~/esp-idf/export.sh

# build + flash:
cd firmware
idf.py set-target esp32s3
idf.py menuconfig     # set PSRAM, partition table, NimBLE
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

## Components

- **main/ble_transport.c** — NimBLE GATT server exposing the Mimo service
  (see ../docs/mimobot-protocol.md)
- **main/audio_i2s.c** — I²S mic capture + speaker playback
- **main/opus_codec.c** — libopus encode/decode (16 kHz mono, 20 ms frames)
- **main/app_main.c** — wiring: VAD gate, PTT button, state machine

## Status

**Skeleton only.** None of the components build yet. Each .c file has `TODO(...)`
markers describing exactly what to fill in. The wire protocol is locked (see
`docs/mimobot-protocol.md`) so phone + firmware can be developed in parallel.
