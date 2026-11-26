package com.llmhub.llmhub.imagegen

import android.os.Build
import android.util.Log
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.ModelData
import org.junit.Assert.assertTrue
import org.junit.Test

/** QNN image models and upscalers are offered on Snapdragon only; other chips cannot run their Hexagon packages. */
class NpuModelVisibilityDeviceTest {
    @Test fun qnnModelsOnlyOnSnapdragon() {
        val qnn = ModelData.models.filter { it.modelFormat == "qnn_npu" || it.modelFormat == "upscaler_bin" }
        Log.i(TAG, "${Build.MODEL} ${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}: snapdragon=${DeviceInfo.isSnapdragon()}, " +
            "qnn=${qnn.size}, image models=${ModelData.models.count { it.category == "image_generation" }}")
        if (DeviceInfo.isSnapdragon()) assertTrue("Snapdragon must list QNN models", qnn.isNotEmpty())
        else assertTrue("Non-Snapdragon lists QNN models: ${qnn.map { it.name }}", qnn.isEmpty())
    }

    private companion object { const val TAG = "NpuModelVisibility" }
}
