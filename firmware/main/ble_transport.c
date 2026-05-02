// NimBLE GATT server exposing the Mimo service defined in
// docs/mimobot-protocol.md.
//
// NOT IMPLEMENTED YET.
//
// TODO(ble):
//   1. nimble_port_init(); ble_hs_cfg callbacks.
//   2. Advertise with name "mimobot-<hex4>" + service UUID MIMO_SVC_UUID_BYTES.
//   3. Register GATT service with three characteristics:
//        - AUDIO_UP   (NOTIFY)
//        - AUDIO_DOWN (WRITE_WITHOUT_RESPONSE)
//        - CONTROL    (NOTIFY + WRITE)
//   4. On subscribe, start mic_task. On unsubscribe / disconnect, stop it.
//   5. Expose:
//        int  ble_notify_audio_up(const uint8_t *opus, size_t len);
//        int  ble_notify_control (const char *json, size_t len);
//        void ble_on_audio_down_cb(void (*cb)(const uint8_t *, size_t));
//        void ble_on_control_cb   (void (*cb)(const char *, size_t));
//   6. MTU: negotiate 247 on connect.

#include "mimobot_ids.h"
