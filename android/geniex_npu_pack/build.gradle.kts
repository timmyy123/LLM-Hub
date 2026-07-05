plugins {
    id("com.android.asset-pack")
}

assetPack {
    packName = "geniex_npu_pack"
    dynamicDelivery {
        deliveryType = "install-time"
    }
}
