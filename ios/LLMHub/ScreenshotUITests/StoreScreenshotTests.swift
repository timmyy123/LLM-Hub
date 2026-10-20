import Darwin
import UIKit
import XCTest

/// App Store screenshots with real generated results.
///
/// Each test generates one result in English, then flips the app language live through
/// the Debug-only Darwin notification hook (`com.llmhub.screenshots.lang.<code>`) and
/// attaches `XCUIScreen.main.screenshot()` as `<lang>/<NN>_<feature>.png` for every app
/// language, without leaving the screen. Generated files are also attached as
/// `outputs/<file>` (fetched from the app's Debug-only loopback output server).
///
/// Test-runner environment (pass as `TEST_RUNNER_<NAME>` through xcodebuild):
/// - `SCREENSHOT_DUO_INJECT=1`: media tests show files already copied into the app's
///   Documents/screenshot_outputs/ instead of generating (iPhone Duo simulator).
/// - `SCREENSHOT_LLM_MODEL`: LLM for VibeVoice and Translator (and Vibecode unless set).
/// - `SCREENSHOT_VIBECODE_MODEL`: LLM for Vibecode (default Gemma 4 E2B (Q3_K_M)).
/// - `SCREENSHOT_LANGUAGES`: comma-separated subset of languages to capture.
/// - `SCREENSHOT_TRANSLATOR_TARGET`: translator target language code (default `es`).
/// - `SCREENSHOT_SETTLE_SECONDS`: wait after each language switch (default 1.5).
/// - `SCREENSHOT_ATTACH=1`: attach to an already running app instead of launching it.
/// - `SCREENSHOT_DUO_POSE=inner-landscape|outer-portrait`: iPhone Duo simulator pose. The
///   test rotates the device, captures the display of that pose (2853x2007 inner,
///   1398x2034 outer) and names files `<lang>/<NN>_<feature>_duo-<pose>.png`. Folding
///   or unfolding itself happens outside the test (Device Hub; a fresh boot is folded).
final class StoreScreenshotTests: XCTestCase {
    static let allLanguages = [
        "en", "ar", "da", "de", "es", "fa", "fr", "he", "hi", "id", "it",
        "ja", "ko", "nl", "pl", "pt", "ru", "th", "tr", "uk", "vi", "zh-TW",
    ]

    // Model catalog names (ios/LLMHub/Sources/LLMHub/ModelData.swift).
    static let imageModel = CatalogModel(name: "SDXL Turbo (8-bit)", id: "sd_xl_turbo_q6p_q8p.ckpt", categoryKey: "image_generation_models")
    static let videoModel = CatalogModel(name: "AnimateLCM SVD XT v1.1 (8-bit)", id: "animatelcm_svd_xt_v1.1_q6p_q8p.ckpt", categoryKey: "video_generation_models")
    static let musicModel = CatalogModel(name: "Magenta RealTime 2 Small", id: "magenta_realtime_2_small", categoryKey: "music_generation_models")
    static let upscaleModel = CatalogModel(name: "Real-ESRGAN X4+", id: "realesrgan_x4plus_f16.ckpt", categoryKey: "image_upscale_models")

    static let outputPort: UInt16 = 47_861
    static let vibecodePrompt = "Build a neon Snake game with a score counter, arrow-key and swipe controls, and a glowing grid background."
    static let musicPrompt = "Upbeat 80s synthwave with punchy drums, warm analog bass and a soaring lead"
    // Same scene as the Play Store screenshots.
    static let imagePrompt = "a cozy cabin in a snowy forest at night, warm lights, cinematic, highly detailed"
    static let videoPrompt = "Slow cinematic push-in, snow gently falling, warm window lights flickering"
    static let translatorInput = "Good morning! Could you recommend a quiet café nearby where I can work for a few hours?"
    static let vibeVoiceTranscript = "What are three fun things to do in Tokyo on a rainy day?"

    private var app: XCUIApplication!
    private let environment = ProcessInfo.processInfo.environment

    private var injectMode: Bool { environment["SCREENSHOT_DUO_INJECT"] == "1" }

    /// LLM for VibeVoice and Translator: small and fast (696 MB).
    private var llmModel: CatalogModel {
        Self.catalogLLM(environment["SCREENSHOT_LLM_MODEL"] ?? "LFM-2.5 1.2B Instruct (Q4_0)")
    }

    /// LLM for Vibecode: the 1.2B model writes broken HTML, so default to Gemma 4 E2B (2.5 GB).
    private var vibecodeModel: CatalogModel {
        Self.catalogLLM(environment["SCREENSHOT_VIBECODE_MODEL"] ?? environment["SCREENSHOT_LLM_MODEL"] ?? "Gemma 4 E2B (Q3_K_M)")
    }

    private static func catalogLLM(_ name: String) -> CatalogModel {
        // Gemma 4 GGUF builds are listed under Multimodal Models, LFM under Text Models.
        CatalogModel(name: name, id: nil, categoryKey: name.hasPrefix("Gemma") ? "vision_models" : "text_models")
    }

    /// llama.cpp's Metal path stalls in the iOS simulator, so LLM features run on the CPU
    /// there (the app reads `gpu_layers_<model id>` from UserDefaults).
    private func llmArguments(_ model: CatalogModel) -> [String] {
        #if targetEnvironment(simulator)
        return ["-gpu_layers_\(model.defaultId)", "0"]
        #else
        return []
        #endif
    }

    private var languages: [String] {
        guard let list = environment["SCREENSHOT_LANGUAGES"], !list.isEmpty else { return Self.allLanguages }
        return list.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    /// iPhone Duo pose for this run, if any.
    private var duoPose: DuoPose? {
        environment["SCREENSHOT_DUO_POSE"].flatMap(DuoPose.init(rawValue:))
    }

    private var settleSeconds: TimeInterval {
        environment["SCREENSHOT_SETTLE_SECONDS"].flatMap(TimeInterval.init) ?? 1.5
    }

    override func setUpWithError() throws {
        continueAfterFailure = false
        executionTimeAllowance = 4 * 60 * 60
        // No UI interruption monitor: it also fires for the app's own alerts (Vibecode's
        // "Create file" prompt) and invalidates their text fields. The one system prompt
        // the tests trigger (adding the generated image to Photos) is handled explicitly.
    }

    // MARK: - 00 Home

    func test00_home() throws {
        launch([])
        // Premium is unlocked (-premium_purchased YES / BYPASS_PREMIUM_UNLOCK), so the
        // grid shows no lock icons.
        XCTAssertTrue(element("home_feature_translator").waitForExistence(timeout: 20))
        pause(2)
        captureAllLanguages(number: "00", feature: "home")
    }

    // MARK: - 01 Vibecode

    func test01_vibecode() throws {
        let llm = llmModel
        launch([
            "-feature_vibecoder_model_name", llm.name,
            "-feature_vibecoder_enable_thinking", "NO",
            "-screenshot_vibecode_workspace", "LLMHubScreenshots",
            "-screenshot_vibecode_file", "snake.html",
            "-screenshot_vibecode_prompt", Self.vibecodePrompt,
        ] + llmArguments(llm))
        ensureDownloaded(llm)
        openFeature("vibe_coder")

        // Editor and chat box only: the app opens snake.html in an empty chat and prefills
        // the request (-screenshot_vibecode_file / -screenshot_vibecode_prompt); it is not sent.
        XCTAssertTrue(element("vibecode_chat_input").waitForExistence(timeout: 20))
        dismissKeyboardIfNeeded()
        sleep(1)

        captureAllLanguages(number: "01", feature: "vibecode")
    }

    // MARK: - 02 Image generation

    func test02_image() throws {
        var args = ["-sd_selected_model_id", Self.imageModel.id ?? "", "-screenshot_image_prompt", Self.imagePrompt]
        if injectMode { args += ["-screenshot_inject_image", "screenshot_outputs/image.png"] }
        launch(args)
        if !injectMode { ensureDownloaded(Self.imageModel) }
        openFeature("image_generator")

        typeIfEmpty("image_prompt", Self.imagePrompt)
        tap("image_generate_button")

        let result = element("image_result")
        XCTAssertTrue(waitForElement(result, timeout: 45 * 60), "No generated image")
        waitForIdleProgress()

        if !injectMode {
            attachOutputs(["image.png"])
        }
        reveal(result)

        captureAllLanguages(number: "02", feature: "image")
    }

    // MARK: - 03 Video generation

    func test03_video() throws {
        var args = ["-sd_video_model_id", Self.videoModel.id ?? "", "-sd_video_steps", "6", "-screenshot_video_prompt", Self.videoPrompt]
        if injectMode {
            args += [
                "-screenshot_inject_video", "screenshot_outputs/video.mp4",
                "-screenshot_inject_video_input", "screenshot_outputs/video_input.png",
            ]
        }
        // Real generation from the FLUX cabin shown in the Image Generator screenshot (passed
        // at launch; FLUX is too large to generate on the phone). The Photos picker is
        // unreliable on Device Farm.
        launch(args, environment: injectMode ? [:] : ["SCREENSHOT_INPUT_IMAGE_B64": FluxFixture.jpegBase64])
        if !injectMode { ensureDownloaded(Self.videoModel) }
        openFeature("video_generator")

        typeIfEmpty("video_prompt", Self.videoPrompt)
        let generate = element("video_generate_button")
        XCTAssertTrue(waitUntil(timeout: 30) { generate.isEnabled }, "Generate stayed disabled (no input image?)")
        generate.tap()

        let result = element("video_result")
        XCTAssertTrue(waitForElement(result, timeout: 90 * 60), "No generated video")
        if !injectMode { attachOutputs(["video.mp4", "video_input.png"]) }
        reveal(result)

        captureAllLanguages(number: "03", feature: "video", settle: max(settleSeconds, 2.5))
    }

    // MARK: - 04 Music generation

    func test04_music() throws {
        // Live generation: the track plays while it is generated, with no fixed length.
        var args = ["-feature_music_model_name", Self.musicModel.name, "-screenshot_music_prompt", Self.musicPrompt,
                    "-feature_music_live_generation", "YES", "-feature_music_unlimited_duration", "YES"]
        if injectMode { args += ["-screenshot_inject_music", "screenshot_outputs/music.wav"] }
        launch(args)
        if !injectMode { ensureDownloaded(Self.musicModel) }
        openFeature("music_generator")

        if typeIfEmpty("music_prompt", Self.musicPrompt) { dismissKeyboardIfNeeded() }
        tap("music_generate_button")
        // Record the screen shortly after the tap so a stalled run shows why (error, loading...).
        pause(20)
        let afterTap = XCTAttachment(screenshot: XCUIScreen.main.screenshot(), quality: .medium)
        afterTap.name = "debug/04_music_after_generate.png"; afterTap.lifetime = .keepAlways; add(afterTap)
        let error = app.staticTexts.containing(NSPredicate(format: "label CONTAINS[c] 'error'")).firstMatch
        if error.exists { XCTFail("Music generation failed: \(error.label)"); return }

        // While live, the Generate button shows "Stop · mm:ss". Capture it playing.
        let button = element("music_generate_button")
        let live = waitUntil(timeout: 10 * 60, poll: 2) { button.label.range(of: #"\d\d:\d\d"#, options: .regularExpression) != nil }
        XCTAssertTrue(live, "Live generation did not start (button: \(button.label))")
        if !injectMode { pause(10) }
        captureAllLanguages(number: "04", feature: "music")

        if !injectMode {
            // Stop, so the generated audio is saved for the Duo simulator screenshots.
            button.tap()
            XCTAssertTrue(waitForElement(element("music_track"), timeout: 5 * 60), "No track after stopping live generation")
            attachOutputs(["music.wav"])
        }
    }

    // MARK: - 05 Image upscale

    func test05_upscale() throws {
        var args = ["-selected_upscaler_model_id", Self.upscaleModel.id ?? ""]
        if injectMode {
            args += [
                "-screenshot_inject_upscale_input", "screenshot_outputs/upscale_input.png",
                "-screenshot_inject_upscale_output", "screenshot_outputs/upscale_output.png",
            ]
        }
        // Real upscale of the FLUX cabin shown in the Image Generator screenshot (passed at launch).
        launch(args, environment: injectMode ? [:] : ["SCREENSHOT_INPUT_IMAGE_B64": FluxFixture.jpegBase64])
        if !injectMode { ensureDownloaded(Self.upscaleModel) }
        openFeature("image_upscaler")
        let upscale = element("upscale_button")
        XCTAssertTrue(waitUntil(timeout: 30) { upscale.isEnabled }, "Upscale stayed disabled (no input image?)")
        upscale.tap()

        XCTAssertTrue(waitForElement(element("upscale_output"), timeout: 30 * 60), "No upscaled image")
        if !injectMode { attachOutputs(["upscale_input.png", "upscale_output.png"]) }

        captureAllLanguages(number: "05", feature: "upscale")
    }

    // MARK: - 06 VibeVoice

    func test06_vibevoice() throws {
        let llm = llmModel
        launch([
            "-feature_vibevoice_model_name", llm.name,
            "-screenshot_vibevoice_transcript", Self.vibeVoiceTranscript,
        ] + llmArguments(llm))
        ensureDownloaded(llm)
        openFeature("vibe_voice")

        // VibeVoice shows the globe only once the model is loaded.
        tap("feature_settings_button")
        tap("feature_settings_load_button")
        XCTAssertTrue(waitForElement(element("feature_settings_unload_button"), timeout: 15 * 60), "LLM did not load")
        tap("feature_settings_done_button")

        let globe = element("vibevoice_globe")
        XCTAssertTrue(globe.waitForExistence(timeout: 30))
        globe.tap()
        XCTAssertTrue(waitForElement(element("vibevoice_reply"), timeout: 15 * 60), "No VibeVoice reply")
        // Wait for the spoken reply to finish so the status stays the same in every language.
        XCTAssertTrue(waitUntil(timeout: 10 * 60, poll: 2) { (globe.value as? String) == "idle" }, "Reply never finished")

        captureAllLanguages(number: "06", feature: "vibevoice")
    }

    // MARK: - 07 Translator

    func test07_translator() throws {
        let llm = llmModel
        launch([
            "-feature_translator_model_name", llm.name,
            "-feature_translator_source_lang", "en",
            "-feature_translator_target_lang", environment["SCREENSHOT_TRANSLATOR_TARGET"] ?? "es",
            "-feature_translator_auto_detect", "NO",
            "-screenshot_translator_input", Self.translatorInput,
        ] + llmArguments(llm))
        ensureDownloaded(llm)
        openFeature("translator")

        // The app prefills the text (-screenshot_translator_input); type it only if that failed.
        if typeIfEmpty("translator_input", Self.translatorInput) {
            dismissKeyboardIfNeeded(tapping: "Text to translate")
        }
        tap("translator_translate_button")

        let output = element("translator_output")
        XCTAssertTrue(waitForElement(output, timeout: 15 * 60), "No translation")
        XCTAssertTrue(waitUntil(timeout: 15 * 60, poll: 2) { (output.value as? String) == "done" }, "Translation never finished")
        // The output scrolls to its end while streaming; show the translation from the start.
        output.swipeDown()
        pause(1)

        captureAllLanguages(number: "07", feature: "translator")
    }

    // MARK: - Launch and navigation

    private func launch(_ extraArguments: [String], environment launchEnv: [String: String] = [:]) {
        app = XCUIApplication()
        app.launchEnvironment.merge(launchEnv) { _, new in new }
        if let duoPose {
            // SCREENSHOT_DUO_ORIENTATION (portrait|landscapeLeft|landscapeRight|portraitUpsideDown)
            // overrides the pose's rotation: folded, the simulator's orientations are relative to
            // the inner display.
            let overrides: [String: UIDeviceOrientation] = [
                "portrait": .portrait, "landscapeLeft": .landscapeLeft,
                "landscapeRight": .landscapeRight, "portraitUpsideDown": .portraitUpsideDown,
            ]
            XCUIDevice.shared.orientation = environment["SCREENSHOT_DUO_ORIENTATION"].flatMap { overrides[$0] } ?? duoPose.orientation
        }
        app.launchArguments = [
            "-app_language", "en",
            "-AppleLanguages", "(en)",
            "-AppleLocale", "en_US",
            "-premium_purchased", "YES",
            "-screenshot_disable_auto_variation", "YES",
            "-screenshot_save_outputs", "YES",
            "-screenshot_output_port", "\(Self.outputPort)",
        ] + extraArguments
        if environment["SCREENSHOT_ATTACH"] == "1" {
            // Debugging: drive an app instance that was started separately (for example
            // with `simctl launch --console-pty` and the same arguments) to see its stdout.
            app.activate()
        } else {
            app.launch()
        }
        XCTAssertTrue(element("home_models_button").waitForExistence(timeout: 30), "Home screen did not appear")
    }

    private func openFeature(_ route: String) {
        let card = element("home_feature_\(route)")
        XCTAssertTrue(card.waitForExistence(timeout: 20), "Missing Home card \(route)")
        reveal(card)
        card.tap()
    }

    private func goBack() {
        tap("nav_back_button")
    }

    /// Downloads a catalog model through the AI Models screen unless it is already there.
    /// Selects a model category chip, scrolling the chip row as needed.
    private func selectCategory(_ key: String) {
        let category = element("model_category_\(key)")
        XCTAssertTrue(category.waitForExistence(timeout: 15), "Missing category \(key)")
        // The chip row is wider than the screen (iPhone and iPad), and asking an off-screen chip
        // for isHittable fails the test. Compare frames instead and drag the row by coordinates
        // along its own y until the chip is fully on screen.
        let window = app.windows.firstMatch.frame
        let rowY = category.frame.midY
        func chipOnScreen() -> Bool {
            let frame = category.frame
            return !frame.isEmpty && frame.minX >= window.minX && frame.maxX <= window.maxX
        }
        let origin = app.coordinate(withNormalizedOffset: .zero)
        var chipSwipes = 0
        while !chipOnScreen() && chipSwipes < 8 {
            // Drag exactly the distance needed (slow drag, then hold: no momentum).
            let frame = category.frame
            let shift = frame.maxX > window.maxX ? -(frame.maxX - window.maxX + 16) : (window.minX + 16 - frame.minX)
            let distance = max(-window.width * 0.7, min(window.width * 0.7, shift))
            let startX = distance < 0 ? window.width * 0.85 : window.width * 0.15
            let from = origin.withOffset(CGVector(dx: startX, dy: rowY))
            let to = origin.withOffset(CGVector(dx: startX + distance, dy: rowY))
            from.press(forDuration: 0.05, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.3)
            pause(0.5)
            chipSwipes += 1
        }
        XCTAssertTrue(chipOnScreen(), "Category \(key) chip stayed off screen")
        category.tap()
    }

    private func ensureDownloaded(_ model: CatalogModel) {
        // Built into the OS (e.g. the iPhone Duo simulator): nothing to download.
        if model.name == "Apple Foundation Model" { return }
        tap("home_models_button")
        XCTAssertTrue(app.navigationBars["AI Models"].waitForExistence(timeout: 15) || element("model_category_text_models").exists)

        selectCategory(model.categoryKey)

        let row = element("model_row_\(model.name)")
        if !row.waitForExistence(timeout: 3) {
            let family = element("model_family_\(model.familyTitle)")
            reveal(family)
            XCTAssertTrue(family.exists, "Missing model family \(model.familyTitle)")
            family.tap()
        }
        XCTAssertTrue(row.waitForExistence(timeout: 10), "Missing model row \(model.name)")
        reveal(row)

        if (row.value as? String) != "downloaded" {
            let download = element("model_download_\(model.name)")
            if !download.exists { row.tap() }
            XCTAssertTrue(download.waitForExistence(timeout: 10), "Missing download button for \(model.name)")
            reveal(download)
            download.tap()

            var retries = 0
            let finished = waitUntil(timeout: 2 * 60 * 60, poll: 10) {
                let state = row.value as? String
                if state == "error" || state == "paused", retries < 5, download.exists {
                    retries += 1
                    download.tap()
                }
                return state == "downloaded"
            }
            XCTAssertTrue(finished, "\(model.name) did not finish downloading (state \(String(describing: row.value)))")
        }
        goBack()
        XCTAssertTrue(element("home_models_button").waitForExistence(timeout: 15))
    }

    // MARK: - Capture

    private func captureAllLanguages(number: String, feature: String, settle: TimeInterval? = nil) {
        let wait = settle ?? settleSeconds
        for language in languages {
            postLanguage(language)
            pause(wait)
            let suffix = duoPose.map { "_duo-\($0.rawValue)" } ?? ""
            let name = "\(language)/\(number)_\(feature)\(suffix).png"
            if let hostDir = hostCaptureDirectory {
                requestHostCapture(name, in: hostDir)
                continue
            }
            let attachment = XCTAttachment(screenshot: captureScreen(), quality: .original)
            attachment.name = name
            attachment.lifetime = .keepAlways
            add(attachment)
        }
        postLanguage("en")
        pause(0.5)
    }

    /// `SCREENSHOT_HOST_CAPTURE_DIR`: XCUIScreen cannot capture the unfolded iPhone Duo's inner
    /// display, so the Mac captures it with `simctl io screenshot` (run_duo_simulator.sh). The
    /// test writes the file name to `<dir>/request` and waits for the same name in `<dir>/done`.
    private var hostCaptureDirectory: URL? {
        environment["SCREENSHOT_HOST_CAPTURE_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
    }

    private func requestHostCapture(_ name: String, in directory: URL) {
        let done = directory.appendingPathComponent("done")
        try? FileManager.default.removeItem(at: done)
        XCTAssertNoThrow(try name.write(to: directory.appendingPathComponent("request"), atomically: true, encoding: .utf8))
        let captured = waitUntil(timeout: 60, poll: 0.25) { (try? String(contentsOf: done, encoding: .utf8)) == name }
        XCTAssertTrue(captured, "Host did not capture \(name)")
    }

    /// The screen to capture. Real iPhones and iPads have one; the iPhone Duo simulator
    /// reports several, of which only the display of the current pose shows the app.
    private func captureScreen() -> XCUIScreenshot {
        // Device Farm iPhones list an extra screen that cannot be captured, so only the Duo
        // simulator (a pose is set) looks beyond the main screen.
        let screens = XCUIScreen.screens
        guard duoPose != nil, screens.count > 1 else { return XCUIScreen.main.screenshot() }
        let shots = screens.map { $0.screenshot() }
        if let expected = duoPose?.pixelSize {
            if let match = shots.first(where: { Self.pixelSize(of: $0) == expected }) { return match }
            let sizes = shots.map { "\(Self.pixelSize(of: $0))" }.joined(separator: ", ")
            XCTFail("No \(duoPose!.rawValue) display (\(expected)) is active; screens are \(sizes). Fold or unfold the iPhone Duo first.")
        }
        // Otherwise the screen that is lit.
        return shots.max { Self.brightness(of: $0) < Self.brightness(of: $1) } ?? XCUIScreen.main.screenshot()
    }

    private static func pixelSize(of shot: XCUIScreenshot) -> CGSize {
        let image = shot.image
        return CGSize(width: (image.size.width * image.scale).rounded(), height: (image.size.height * image.scale).rounded())
    }

    private static func brightness(of shot: XCUIScreenshot) -> Double {
        guard let cgImage = shot.image.cgImage else { return 0 }
        var pixels = [UInt8](repeating: 0, count: 16 * 16 * 4)
        guard let context = CGContext(data: &pixels, width: 16, height: 16, bitsPerComponent: 8, bytesPerRow: 64,
                                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return 0 }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: 16, height: 16))
        return Double(pixels.reduce(0) { $0 + Int($1) }) / Double(pixels.count)
    }

    private func postLanguage(_ code: String) {
        let name = CFNotificationName("com.llmhub.screenshots.lang.\(code)" as CFString)
        CFNotificationCenterPostNotification(CFNotificationCenterGetDarwinNotifyCenter(), name, nil, nil, true)
    }

    /// Attaches result files the app saved to Documents/screenshot_outputs/.
    private func attachOutputs(_ names: [String]) {
        for name in names {
            var data: Data?
            _ = waitUntil(timeout: 30, poll: 2) {
                data = Self.httpGet(port: Self.outputPort, path: "/\(name)")
                return data?.isEmpty == false
            }
            guard let data, !data.isEmpty else {
                XCTContext.runActivity(named: "Output \(name) unavailable") { _ in }
                continue
            }
            let type: String
            switch (name as NSString).pathExtension.lowercased() {
            case "png": type = "public.png"
            case "mp4": type = "public.mpeg-4"
            case "wav": type = "com.microsoft.waveform-audio"
            default: type = "public.data"
            }
            let attachment = XCTAttachment(data: data, uniformTypeIdentifier: type)
            attachment.name = "outputs/\(name)"
            attachment.lifetime = .keepAlways
            add(attachment)
        }
    }

    // MARK: - Element helpers

    private func element(_ identifier: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: identifier).firstMatch
    }

    private func tap(_ identifier: String, timeout: TimeInterval = 20) {
        let target = element(identifier)
        XCTAssertTrue(target.waitForExistence(timeout: timeout), "Missing element \(identifier)")
        reveal(target)
        target.tap()
    }

    /// Types `text` unless the app already prefilled it through a screenshot hook (typing is
    /// unreliable on Device Farm iPads). Returns true when it had to type.
    @discardableResult
    private func typeIfEmpty(_ identifier: String, _ text: String) -> Bool {
        let field = element(identifier)
        XCTAssertTrue(field.waitForExistence(timeout: 20), "Missing text input \(identifier)")
        if ((field.value as? String) ?? "").contains(text.prefix(12)) { return false }
        typeInto(identifier, text)
        return true
    }

    private func typeInto(_ identifier: String, _ text: String) {
        let field = element(identifier)
        XCTAssertTrue(field.waitForExistence(timeout: 20), "Missing text input \(identifier)")
        reveal(field)
        // A tap occasionally lands while the screen is still settling; retry until the
        // field owns the keyboard.
        for _ in 0..<4 {
            field.tap()
            if waitUntil(timeout: 3, { (field.value(forKey: "hasKeyboardFocus") as? Bool) == true }) { break }
        }
        field.typeText(text)
    }

    /// Scrolls the main scroll view until the element is fully visible (above the keyboard).
    private func reveal(_ target: XCUIElement, maxSwipes: Int = 10) {
        let windowElement = app.windows.firstMatch
        var swipes = 0
        while swipes < maxSwipes {
            let window = windowElement.frame
            let keyboard = app.keyboards.firstMatch
            let visibleBottom = keyboard.exists ? min(window.maxY, keyboard.frame.minY) : window.maxY
            let visible = CGRect(x: window.minX, y: window.minY, width: window.width, height: visibleBottom - window.minY)
            if target.exists, target.isHittable,
               visible.contains(target.frame) || target.frame.height > visible.height * 0.8 {
                return
            }
            let below = !target.exists || target.frame.maxY > visibleBottom - 8
            // Drag inside the visible area: content moves up to reveal what is below.
            let top = (visible.minY + visible.height * 0.3 - window.minY) / window.height
            let bottom = (visible.minY + visible.height * 0.7 - window.minY) / window.height
            let start = windowElement.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: below ? bottom : top))
            let end = windowElement.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: below ? top : bottom))
            start.press(forDuration: 0.05, thenDragTo: end)
            swipes += 1
        }
    }

    /// Hides the keyboard: the keyboard toolbar's Done (Vibecode), or a tap on a
    /// non-interactive label on screens that dismiss on tap (Translator).
    private func dismissKeyboardIfNeeded(tapping label: String? = nil) {
        guard app.keyboards.firstMatch.exists else { return }
        let done = app.toolbars.buttons["Done"]
        if done.exists {
            done.tap()
        } else if let label, app.staticTexts[label].exists {
            app.staticTexts[label].tap()
        }
        _ = waitUntil(timeout: 3) { !self.app.keyboards.firstMatch.exists }
    }

    /// Waits while a spinner (a generation in progress) is still on screen.
    private func waitForIdleProgress() {
        _ = waitUntil(timeout: 60, poll: 1) { !self.app.activityIndicators.firstMatch.exists }
    }

    /// Picks the newest photo in the system photo picker (the image the image test saved).
    private func pickMostRecentPhoto() {
        let photos = app.images.matching(NSPredicate(format: "label BEGINSWITH[c] 'Photo'"))
        let anyImage = app.scrollViews.images.firstMatch
        XCTAssertTrue(waitUntil(timeout: 30) { photos.firstMatch.exists || anyImage.exists }, "Photo picker showed no photos")
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US")
        formatter.dateFormat = "MMMM d"
        let today = formatter.string(from: Date())
        let fromToday = photos.matching(NSPredicate(format: "label CONTAINS %@", today))
        let choice = fromToday.count > 0 ? fromToday.element(boundBy: 0) : (photos.count > 0 ? photos.element(boundBy: 0) : anyImage)
        choice.tap()
        _ = waitUntil(timeout: 15) { !photos.firstMatch.exists }
        pause(2)
    }

    private func handleSystemAlerts(timeout: TimeInterval) {
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        _ = waitUntil(timeout: timeout, poll: 1) {
            let alert = springboard.alerts.firstMatch
            if alert.exists { return Self.tapAllowButton(in: alert) }
            return false
        }
    }

    @discardableResult
    private static func tapAllowButton(in alert: XCUIElement) -> Bool {
        for label in ["Allow Full Access", "Allow Access", "Allow", "OK", "Continue"] {
            let button = alert.buttons[label]
            if button.exists {
                button.tap()
                return true
            }
        }
        return false
    }

    // MARK: - Waiting

    private func pause(_ seconds: TimeInterval) {
        RunLoop.current.run(until: Date().addingTimeInterval(seconds))
    }

    /// Polls for an element without `waitForExistence`, which attaches a debug
    /// description to the result bundle on every check during long waits.
    private func waitForElement(_ target: XCUIElement, timeout: TimeInterval) -> Bool {
        waitUntil(timeout: timeout, poll: 3) { target.exists }
    }

    @discardableResult
    private func waitUntil(timeout: TimeInterval, poll: TimeInterval = 0.5, _ condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            pause(poll)
        }
        return condition()
    }

    // MARK: - Loopback HTTP (no ATS, no Network framework)

    static func httpGet(port: UInt16, path: String) -> Data? {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        var timeout = timeval(tv_sec: 20, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = UInt32(INADDR_LOOPBACK).bigEndian
        let connected = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connected == 0 else { return nil }
        let request = Array("GET \(path) HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".utf8)
        guard send(fd, request, request.count, 0) == request.count else { return nil }
        var response = Data()
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            let count = recv(fd, &buffer, buffer.count, 0)
            if count <= 0 { break }
            response.append(buffer, count: count)
        }
        guard let split = response.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let header = String(decoding: response[..<split.lowerBound], as: UTF8.self)
        guard header.hasPrefix("HTTP/1.1 200") else { return nil }
        return response[split.upperBound...]
    }
}

enum DuoPose: String {
    case innerLandscape = "inner-landscape"
    case outerPortrait = "outer-portrait"

    var orientation: UIDeviceOrientation {
        switch self {
        case .innerLandscape: return .landscapeLeft
        case .outerPortrait: return .portrait
        }
    }

    /// App Store pixel size of the display in this pose.
    var pixelSize: CGSize {
        switch self {
        case .innerLandscape: return CGSize(width: 2853, height: 2007)
        case .outerPortrait: return CGSize(width: 1398, height: 2034)
        }
    }
}

struct CatalogModel {
    let name: String
    let id: String?
    /// `ModelCategory.titleKey` of the model's AI Models tab.
    let categoryKey: String

    /// `AIModel.id` default: the lowercased name with spaces replaced by underscores.
    var defaultId: String { id ?? name.lowercased().replacingOccurrences(of: " ", with: "_") }

    /// Family header title, derived like `ModelDownloadViewModel.familyName(for:)`.
    var familyTitle: String {
        guard let open = name.lastIndex(of: "("), let close = name.lastIndex(of: ")"), open < close else { return name }
        let suffix = name[name.index(after: open)..<close].uppercased().replacingOccurrences(of: "-", with: "_")
        let markers = ["Q", "IQ", "F16", "BF16", "INT", "FP16", "FP8", "F32"]
        guard markers.contains(where: { suffix.contains($0) }) else { return name }
        return name[..<open].trimmingCharacters(in: .whitespaces)
    }
}
