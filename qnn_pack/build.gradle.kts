plugins {
    id("com.android.asset-pack")
}

assetPack {
    packName = "qnn_pack"
    dynamicDelivery {
        deliveryType = "install-time"
    }
}
