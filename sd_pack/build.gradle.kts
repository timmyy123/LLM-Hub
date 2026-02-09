plugins {
    id("com.android.asset-pack")
}

assetPack {
    packName = "sd_pack"
    dynamicDelivery {
        deliveryType = "install-time"
    }
}
