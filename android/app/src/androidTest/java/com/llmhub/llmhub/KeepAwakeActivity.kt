package com.llmhub.llmhub

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.view.WindowManager
import android.widget.TextView

/**
 * Keeps a test device's screen on while it stays in front. For devices where adb power commands are off
 * limits (Samsung Remote Test Lab drops the session on `svc power` and `settings put`). It lives in the test
 * APK, so it runs in the test package's own process and survives `am instrument` runs against the app:
 *
 *   adb shell am start -n com.llmhub.llmhub.preview.test/com.llmhub.llmhub.KeepAwakeActivity
 *
 * Press Back on the device, or reinstall the test APK, to end it.
 */
class KeepAwakeActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setTurnScreenOn(true)
        setShowWhenLocked(true)
        setContentView(TextView(this).apply {
            text = "LLM Hub device tests are running.\nThis screen keeps the display on."
            textSize = 20f
            gravity = Gravity.CENTER
        })
    }
}
