idf_component_register(
    SRCS
        "app_main.c"
        "ble_transport.c"
        "audio_i2s.c"
        "opus_codec.c"
    INCLUDE_DIRS "."
    REQUIRES
        # Add as implementations land:
        #   bt           (NimBLE host)
        #   nvs_flash
        #   driver       (I2S, GPIO)
        #   esp_timer
        # Opus is vendored as a managed component; add:
        #   opus
)
