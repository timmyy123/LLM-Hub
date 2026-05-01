// Mimo bot — main wiring.
//
// Boots NVS, brings up BLE, starts I²S mic + speaker tasks, wires them to
// the Opus codec, and drives the PTT state machine.
//
// NOT IMPLEMENTED YET — skeleton only. See firmware/README.md.

#include "esp_log.h"

static const char *TAG = "mimobot";

void app_main(void) {
    ESP_LOGI(TAG, "boot: mimobot firmware skeleton");

    // TODO(boot):
    //   1. nvs_flash_init()
    //   2. ble_transport_init()   — see ble_transport.c
    //   3. opus_codec_init()      — see opus_codec.c
    //   4. audio_i2s_init()       — see audio_i2s.c
    //   5. Start FreeRTOS tasks:
    //        - mic_task:     I²S read → opus_encode → ble_notify(AUDIO_UP)
    //        - speaker_task: ble_audio_down_queue → opus_decode → I²S write
    //        - ptt_task:     debounce GPIO → ble_send_control({"t":"ptt",...})
    //   6. Loop forever / enter light sleep when idle.
}
