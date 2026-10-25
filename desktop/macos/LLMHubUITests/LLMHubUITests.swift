import XCTest

/// End-to-end UI tests for LLM Hub on macOS, mirroring desktop/windows/LLMHub.UITests.
/// Needs Gemma 4 12B installed (run the app once with `--selftest gemma-4-12b`). Tests run in order and share state.
final class LLMHubUITests: XCTestCase {
    /// A word unique to this test run, so chats left by earlier runs never match.
    private static let run = "kiwi" + String(Int(Date().timeIntervalSince1970) % 1_000_000)
    private var run: String { Self.run }
    private var app: XCUIApplication!
    /// The app's real data folder. The test runner is sandboxed, so its own home directory is a container;
    /// the account's home comes from the password database instead (the runner may read any absolute path).
    private let support = URL(fileURLWithPath: String(cString: getpwuid(getuid())!.pointee.pw_dir))
        .appendingPathComponent("Library/Application Support/LLMHub")

    override func setUp() {
        continueAfterFailure = false
        launch()
    }

    override func tearDown() {
        // A test may have killed the app on purpose; there is no window to capture then.
        guard app.state == .runningForeground || app.state == .runningBackground else { return }
        let shot = XCTAttachment(screenshot: app.windows.firstMatch.screenshot())
        shot.name = name
        shot.lifetime = .keepAlways
        add(shot)
        app.terminate()
    }

    // MARK: Helpers

    private func launch(_ args: [String] = [], home: String? = nil) {
        app = XCUIApplication()
        app.launchArguments = args
        if let home { app.launchEnvironment["LLMHUB_HOME"] = home }
        app.launch()
        XCTAssertTrue(el("NewChatButton").waitForExistence(timeout: 30) || el("BackButton").waitForExistence(timeout: 5))
    }

    private func el(_ id: String) -> XCUIElement { app.descendants(matching: .any).matching(identifier: id).firstMatch }
    private func all(_ id: String) -> [XCUIElement] { app.descendants(matching: .any).matching(identifier: id).allElementsBoundByIndex }

    private func wait(_ timeout: TimeInterval, _ what: String, _ condition: () -> Bool) {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return }
            RunLoop.current.run(until: Date().addingTimeInterval(0.25))
        }
        XCTFail("Timed out waiting for \(what)")
    }

    private func text(_ e: XCUIElement) -> String { (e.value as? String).flatMap { $0.isEmpty ? nil : $0 } ?? e.label }

    /// The engine card is one button, so its texts are read from the button's combined label.
    private var engineStatus: String { el("SettingsButton").label }

    private var assistantText: String { all("AssistantMessage").last.map(text) ?? "" }

    @discardableResult
    private func send(_ message: String, timeout: TimeInterval = 300) -> String {
        let before = all("AssistantMessage").count
        let composer = el("Composer")
        composer.click()
        composer.typeText(message + "\r")
        wait(20, "reply bubble") { all("AssistantMessage").count > before }
        wait(timeout, "generation to finish") { !el("StopButton").exists }
        let reply = assistantText
        XCTAssertFalse(reply.isEmpty, "empty reply; status: \(all("MessageStatus").last.map(text) ?? "")")
        return reply
    }

    private func pickModel(_ part: String, exact: Bool = false) {
        let picker = el("ModelPicker")
        picker.click()
        let item = app.menuItems.matching(NSPredicate(format: exact ? "title == %@" : "title CONTAINS %@", part)).firstMatch
        XCTAssertTrue(item.waitForExistence(timeout: 5), "\(part) not in model picker")
        item.click()
    }

    /// Settings replace the app navigation in the single sidebar; Back returns to it.
    private func goToChat() {
        if el("BackButton").exists { el("BackButton").click() }
        el("NavChat").click()
    }

    private func openSettings(_ section: String) {
        el("SettingsButton").click()
        el(section).click()
    }

    private func readSettings() -> [String: Any] {
        guard let data = try? Data(contentsOf: support.appendingPathComponent("settings.json")) else { return [:] }
        return (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    private var serverRunning: Bool {
        FileManager.default.fileExists(atPath: support.appendingPathComponent("engines/llama.cpp/server.pid").path)
    }

    private func catalogRow(_ id: String) -> XCUIElement {
        let row = el("CatalogRow-\(id)")
        let list = el("CatalogList")
        for _ in 0..<30 where !row.isHittable {
            list.scroll(byDeltaX: 0, deltaY: -300)
        }
        XCTAssertTrue(row.waitForExistence(timeout: 30), "catalog row \(id)")
        return row
    }

    private func progress(_ row: XCUIElement) -> Double {
        let bar = row.descendants(matching: .any).matching(identifier: "DownloadProgress").firstMatch
        guard bar.exists else { return -1 }
        // Reported as a number or as text such as "45%" or "0.45".
        if let number = bar.value as? NSNumber { return number.doubleValue <= 1 ? number.doubleValue * 100 : number.doubleValue }
        let digits = String(describing: bar.value ?? "").filter { $0.isNumber || $0 == "." }
        guard let value = Double(digits) else { return -1 }
        return value <= 1 && !String(describing: bar.value ?? "").contains("%") ? value * 100 : value
    }

    private func attach(_ name: String) {
        let shot = XCTAttachment(screenshot: app.windows.firstMatch.screenshot())
        shot.name = name
        shot.lifetime = .keepAlways
        add(shot)
    }

    /// Real window screenshots (glass is composited by the window server, so only XCUITest/screen capture shows it).
    /// Uses a separate data folder so it never touches models or chats.
    func test00_Screenshots() {
        // Outside the runner's sandbox container, which the (unsandboxed) app could not share anyway.
        let home = "/tmp/llmhub-shots"
        for theme in ["dark", "light"] {
            app.terminate()
            launch(["--theme", theme], home: home)
            attach("\(theme)-chat")
            openSettings("SectionModels")
            Thread.sleep(forTimeInterval: 8)
            attach("\(theme)-models")
            el("SectionGeneral").click()
            attach("\(theme)-general")
        }
    }

    // MARK: Shell

    func test01_SidebarNavigation() {
        for id in ["NavChat", "NavAgent", "NavDesign", "NavStudio", "NewChatButton", "SettingsButton"] {
            XCTAssertTrue(el(id).exists, "\(id) missing")
        }
        for id in ["NavAgent", "NavDesign", "NavStudio"] {
            el(id).click()
            XCTAssertTrue(el("PlaceholderHeading").waitForExistence(timeout: 5), "\(id) page")
        }
        el("NavChat").click()
        XCTAssertTrue(el("Composer").waitForExistence(timeout: 5))
    }

    func test02_SingleSidebarSettings() {
        el("SettingsButton").click()
        XCTAssertTrue(el("BackButton").waitForExistence(timeout: 5), "Back button")
        XCTAssertFalse(el("NavAgent").exists, "app navigation should be replaced by settings")
        for id in ["SectionGeneral", "SectionModels", "SectionApi"] { XCTAssertTrue(el(id).exists, id) }
        el("BackButton").click()
        XCTAssertTrue(el("NavAgent").waitForExistence(timeout: 5), "back to app navigation")
    }

    // MARK: Chat

    func test03_ChatReplyWithReturnKey() {
        goToChat()
        el("NewChatButton").click()
        pickModel("Gemma 4 12B")
        let reply = send("\(run): reply with exactly one word: pineapple")
        XCTAssertTrue(reply.localizedCaseInsensitiveContains("pineapple"), reply)
        XCTAssertTrue(engineStatus.contains("Gemma"), engineStatus)
        XCTAssertTrue(all("ChatTitle").contains { text($0).contains(run) }, "chat in sidebar")
    }

    func test04_MultiTurnContext() {
        el("NewChatButton").click()
        pickModel("Gemma 4 12B")
        send("Remember this: my favourite colour is teal. Just answer OK.")
        let reply = send("What is my favourite colour? Answer with one word.")
        XCTAssertTrue(reply.localizedCaseInsensitiveContains("teal"), reply)
        // Follow-ups belong to the same chat (Windows once saved each one as a new chat).
        XCTAssertFalse(all("ChatTitle").contains { text($0).hasPrefix("What is my favourite") }, "follow-up became its own chat")
    }

    func test05_StopGeneration() {
        el("NewChatButton").click()
        pickModel("Gemma 4 12B")
        let composer = el("Composer")
        composer.click()
        composer.typeText("\(run) count from 1 to 400, one number per line, no other text.\r")
        XCTAssertTrue(el("StopButton").waitForExistence(timeout: 120))
        wait(120, "some output") { assistantText.count > 20 }
        el("StopButton").click()
        XCTAssertTrue(el("SendButton").waitForExistence(timeout: 10))
        let stopped = assistantText
        XCTAssertFalse(stopped.contains("400"), "did not stop early")
        Thread.sleep(forTimeInterval: 2)
        XCTAssertEqual(assistantText, stopped, "kept generating after Stop")
    }

    func test06_SearchAndOpenOldChat() {
        let search = app.searchFields.firstMatch
        search.click()
        search.typeText(run)
        // This run made two chats so far: the multi-turn one and the stopped "count" one.
        wait(5, "this run's two chats") { all("ChatTitle").count == 2 && all("ChatTitle").allSatisfy { text($0).contains(run) } }
        all("ChatTitle").first { text($0).hasPrefix("\(run):") }?.click()
        wait(10, "messages loaded") { all("UserMessage").count >= 1 }
        search.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: run.count))
    }

    func test07_ChatsPersistAndDelete() {
        XCTAssertTrue(all("ChatTitle").contains { text($0).contains(run) }, "chat survived restart")
        let count = all("ChatTitle").count
        let victim = all("ChatTitle").first { text($0).hasPrefix("\(run) count") }
        XCTAssertNotNil(victim)
        victim?.rightClick()
        el("DeleteChatMenuItem").click()
        wait(5, "chat removed") { all("ChatTitle").count == count - 1 }
        app.terminate()
        XCTAssertFalse(serverRunning, "llama-server still running after quit")
        launch()
        XCTAssertFalse(all("ChatTitle").contains { text($0).hasPrefix("\(run) count") }, "deleted chat came back")
    }

    // MARK: Models

    func test08_CatalogOffersOnlyFittingQuants() {
        openSettings("SectionModels")
        let row = catalogRow("gemma-4-12b")
        let picker = row.descendants(matching: .any).matching(identifier: "VariantPicker").firstMatch
        XCTAssertTrue(picker.waitForExistence(timeout: 60), "variants")
        picker.click()
        let items = app.menuItems.allElementsBoundByIndex.map(\.title).filter { !$0.isEmpty }
        app.typeKey(.escape, modifierFlags: [])
        XCTAssertFalse(items.isEmpty)
        XCTAssertFalse(items.contains { $0.contains("Too large") }, items.joined(separator: " | "))
        print("Gemma 4 12B variants on this Mac: \(items.joined(separator: " | "))")
    }

    func test09_DownloadPauseResume() {
        openSettings("SectionModels")
        let row = catalogRow("lfm2-5-1-2b-thinking")
        let download = row.descendants(matching: .any).matching(identifier: "DownloadButton").firstMatch
        let delete = row.descendants(matching: .any).matching(identifier: "DeleteModelButton").firstMatch
        wait(60, "row actions") { download.exists || delete.exists }
        if delete.exists { delete.click() } // Installed by an earlier run; start clean.
        XCTAssertTrue(download.waitForExistence(timeout: 30))
        download.click()
        wait(120, "15% progress") { progress(row) > 15 }
        let paused = progress(row)
        row.descendants(matching: .any).matching(identifier: "CancelDownloadButton").firstMatch.click()
        XCTAssertTrue(download.waitForExistence(timeout: 15))
        download.click()
        wait(30, "progress after resume") { progress(row) > 0 }
        let resumed = progress(row)
        XCTAssertGreaterThanOrEqual(resumed, paused - 1, "restarted instead of resuming")
        print("paused at \(paused)%, resumed at \(resumed)%")
        XCTAssertTrue(row.descendants(matching: .any).matching(identifier: "InstalledLabel").firstMatch.waitForExistence(timeout: 600))
    }

    func test10_GatedModelMessage() {
        openSettings("SectionModels")
        app.radioButtons["Hugging Face"].click()
        let query = el("HfQuery")
        query.click()
        query.typeText("google/gemma-3-1b-it-qat-q4_0-gguf\r")
        let download = el("DownloadButton")
        XCTAssertTrue(download.waitForExistence(timeout: 60))
        download.click()
        let error = el("DownloadError")
        XCTAssertTrue(error.waitForExistence(timeout: 60))
        XCTAssertTrue(text(error).localizedCaseInsensitiveContains("gated"), text(error))
    }

    func test11_HuggingFaceSearchDownloadDelete() {
        openSettings("SectionModels")
        app.radioButtons["Hugging Face"].click()
        let query = el("HfQuery")
        query.click()
        query.typeText("LFM2.5-1.2B-Instruct-GGUF\r")
        let detail = el("HfDetail")
        XCTAssertTrue(detail.waitForExistence(timeout: 30))
        // The file table has one row per quant, each with its own Download button.
        let q4 = detail.descendants(matching: .any).matching(identifier: "HfFileRow-Q4_0").firstMatch
        XCTAssertTrue(q4.waitForExistence(timeout: 30))
        q4.descendants(matching: .any).matching(identifier: "DownloadButton").firstMatch.click()
        XCTAssertTrue(q4.descendants(matching: .any).matching(identifier: "InstalledLabel").firstMatch.waitForExistence(timeout: 600))

        app.radioButtons.matching(NSPredicate(format: "label BEGINSWITH %@", "Installed")).firstMatch.click()
        wait(10, "3 installed") { all("InstalledName").count >= 3 }
        let before = all("InstalledName").count
        let row = all("InstalledRow").first { r in
            r.descendants(matching: .any).matching(identifier: "InstalledName").allElementsBoundByIndex.contains { text($0).hasPrefix("LFM2.5-1.2B-Instruct") && text($0).contains("Q4_0") }
        }
        XCTAssertNotNil(row)
        row?.descendants(matching: .any).matching(identifier: "DeleteInstalledButton").firstMatch.click()
        wait(10, "model removed") { all("InstalledName").count == before - 1 }
        let folders = (try? FileManager.default.contentsOfDirectory(atPath: support.appendingPathComponent("models").path)) ?? []
        XCTAssertFalse(folders.contains { $0.contains("lfm2.5-1.2b-instruct-gguf__q4_0") }, "folder still on disk")
    }

    func test12_SwitchModelAndThinking() {
        goToChat()
        el("NewChatButton").click()
        pickModel("LFM2.5 1.2B Thinking")
        let reply = send("What is 17 multiplied by 3? Answer with the number.")
        XCTAssertTrue(reply.contains("51"), reply)
        XCTAssertTrue(engineStatus.contains("LFM2.5"), engineStatus)
        XCTAssertTrue(el("ThinkingExpander").exists, "no thinking section")
    }

    // MARK: Settings

    func test13_SettingsAreSaved() {
        openSettings("SectionGeneral")
        el("ContextBox").click()
        app.menuItems["32K"].click()
        XCTAssertEqual(readSettings()["contextLength"] as? Int, 32768)
        el("ContextBox").click()
        app.menuItems["16K"].click()
        XCTAssertEqual(readSettings()["contextLength"] as? Int, 16384)

        let token = el("TokenBox")
        token.click()
        token.typeText("hf_uitest")
        wait(3, "token saved") { readSettings()["huggingFaceToken"] as? String == "hf_uitest" }
        token.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: 9))
        wait(3, "token cleared") { readSettings()["huggingFaceToken"] as? String == "" }

        // Start from the system language whatever an earlier run left behind, and end there too.
        let language = el("LanguageBox")
        language.click()
        app.menuItems["Use system language"].click()
        language.click()
        app.menuItems["Deutsch"].click()
        XCTAssertTrue(el("RestartHint").waitForExistence(timeout: 3))
        language.click()
        app.menuItems["Use system language"].click()
    }

    func test14_Translations() {
        let expected = ["es": "Nuevo chat", "pt-BR": "Nova conversa", "fr": "Nouvelle discussion", "de": "Neuer Chat",
                        "it": "Nuova chat", "ru": "Новый чат", "en": "New chat"]
        for (lang, label) in expected.sorted(by: { $0.key < $1.key }) {
            app.terminate()
            launch(["-AppleLanguages", "(\(lang))"])
            XCTAssertEqual(el("NewChatButton").label, label, lang)
            let shot = XCTAttachment(screenshot: app.windows.firstMatch.screenshot())
            shot.name = "lang-\(lang)"
            shot.lifetime = .keepAlways
            add(shot)
        }
    }

    func test15_DarkAndLightThemes() {
        for theme in ["dark", "light"] {
            app.terminate()
            launch(["--theme", theme])
            openSettings("SectionModels")
            Thread.sleep(forTimeInterval: 3)
            let shot = XCTAttachment(screenshot: app.windows.firstMatch.screenshot())
            shot.name = "theme-\(theme)"
            shot.lifetime = .keepAlways
            add(shot)
        }
    }

    // MARK: Engines and API

    func test17_AppleIntelligenceChat() {
        goToChat()
        el("NewChatButton").click()
        pickModel("Apple Intelligence")
        let reply = send("\(run) Name the capital of Japan in one word.", timeout: 60)
        XCTAssertTrue(reply.localizedCaseInsensitiveContains("tokyo"), reply)
        XCTAssertFalse(el("MtpToggle").exists, "MTP is only for LiteRT-LM models")
    }

    func test18_LiteRtChatWithSpeed() {
        goToChat()
        el("NewChatButton").click()
        pickModel("Gemma 4 E2B")
        let reply = send("\(run) Name the capital of France in one word.", timeout: 120)
        XCTAssertTrue(reply.localizedCaseInsensitiveContains("paris"), reply)
        // The footer appears once the speed arrives, just after the Stop button goes away.
        wait(10, "speed in the footer") { all("MessageFooter").last.map(text)?.range(of: #"\d tok\./?s|\d tok/s"#, options: [.regularExpression, .caseInsensitive]) != nil }
    }

    /// Needs both Gemma 4 E2B files: "LiteRT GPU" (test18 loads it; Google's GPU build has no MTP drafter) and the standard
    /// "LiteRT" file (`LLMHub --selftest gemma-4-e2b --format litertlm --quant LiteRT --exit` downloads it).
    func test19_MtpToggle() {
        goToChat()
        el("NewChatButton").click()
        pickModel("Gemma 4 E2B · LiteRT GPU", exact: true)
        XCTAssertFalse(el("MtpToggle").waitForExistence(timeout: 2), "no MTP toggle for a file without a drafter")
        pickModel("Gemma 4 E2B · LiteRT", exact: true)
        let toggle = el("MtpToggle")
        XCTAssertTrue(toggle.waitForExistence(timeout: 5), "MTP toggle for a LiteRT-LM model with a drafter")
        let id = "litert-community_gemma-4-e2b-it-litert-lm__litert"
        let wasOff = (readSettings()["mtpOff"] as? [String] ?? []).contains(id)
        toggle.click()
        wait(3, "MTP setting saved") { (readSettings()["mtpOff"] as? [String] ?? []).contains(id) != wasOff }
        // A reply after switching reloads the engine with the new setting.
        let reply = send("\(run) Reply with exactly one word: cherry", timeout: 120)
        XCTAssertTrue(reply.localizedCaseInsensitiveContains("cherry"), reply)
        toggle.click()
        wait(3, "MTP setting restored") { (readSettings()["mtpOff"] as? [String] ?? []).contains(id) == wasOff }
        pickModel("Gemma 4 12B")
        XCTAssertFalse(el("MtpToggle").exists, "MTP toggle hidden for GGUF models")
    }

    func test20_LocalApiToggle() {
        openSettings("SectionApi")
        let toggle = el("ApiToggle")
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        let port = readSettings()["apiPort"] as? Int ?? 8642
        func reachable() -> Bool {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/models")!)
            request.timeoutInterval = 2
            let done = DispatchSemaphore(value: 0)
            var code = 0
            URLSession.shared.dataTask(with: request) { _, response, _ in
                code = (response as? HTTPURLResponse)?.statusCode ?? 0
                done.signal()
            }.resume()
            done.wait()
            return code == 401 || code == 200
        }
        let wasOn = readSettings()["apiEnabled"] as? Bool ?? false
        if !wasOn { toggle.click() }
        wait(10, "API answering") { reachable() }
        XCTAssertFalse(el("ApiError").exists && !text(el("ApiError")).isEmpty, text(el("ApiError")))
        toggle.click()
        wait(10, "API stopped") { !reachable() }
        if wasOn { toggle.click() }
    }

    func test21_ServerStopsOnQuit() {
        goToChat()
        el("NewChatButton").click()
        pickModel("Gemma 4 12B")
        send("Say OK.")
        XCTAssertTrue(serverRunning)
        // terminate() kills the app without its normal shutdown, like a crash; the watchdog must stop the server.
        app.terminate()
        wait(10, "llama-server to stop after the app was killed") { !serverRunning }
    }

    // MARK: Studio

    /// Needs a picture in ~/Pictures/LLM Hub (any Studio image or upscale).
    func test22_StudioUseInChat() {
        goToChat()
        el("NavStudio").click()
        el("StudioTab_image").click()
        let pictures = NSPredicate(format: "NOT (label CONTAINS[c] 'ltxv') AND (label ENDSWITH[c] '.png' OR label ENDSWITH[c] '.jpg' OR label ENDSWITH[c] '.webp')")
        let tile = app.descendants(matching: .any).matching(identifier: "GalleryItem").matching(pictures).firstMatch
        XCTAssertTrue(tile.waitForExistence(timeout: 10), "a picture in the Studio gallery")
        let name = tile.label
        tile.click()
        let use = el("UseInChatButton")
        wait(5, "Use in Chat enabled for \(name)") { use.isEnabled }
        use.click()
        XCTAssertTrue(el("Composer").waitForExistence(timeout: 5), "Chat opened")
        let chip = all("AttachmentChip").first { $0.label == name }
        XCTAssertNotNil(chip, "\(name) waiting in the composer; chips: \(all("AttachmentChip").map(\.label))")
        attach("studio-use-in-chat")
        chip?.descendants(matching: .any).matching(identifier: "RemoveAttachmentButton").firstMatch.click()
        wait(5, "chip removed") { !all("AttachmentChip").contains { $0.label == name } }
    }
}
